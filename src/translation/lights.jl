# Lights translation for OmniverseMakie.
#
# Provides:
#   lights_usda(scene; cam_to_world)       — all scene lights as UsdLux USDA prim blocks
#   usda_light(l, index)                   — per-type USDA prim block
#   light_prim_path(light, index)          — SINGLE SOURCE for /World/<Type>_<index>
#   _direction_to_xform_matrix(dir)        — orientation matrix (Matrix{Float64})
#   sync_lights!(screen, scene)            — live diff: push changed intensity/color/xform
#   author_lights!(screen, scene)          — intent-named delegator to author_root_from_scene!
#
# Type mapping (fully tested: DirectionalLight, PointLight, AmbientLight;
# best-effort: RectLight, SpotLight, EnvironmentLight):
#   DirectionalLight → UsdLuxDistantLight  (intensity = 3000 × max-channel)
#   PointLight       → UsdLuxSphereLight   (intensity = 10000 × max-channel, radius = 1)
#   AmbientLight     → UsdLuxDomeLight     (intensity = 1000 × max-channel)
#   RectLight        → UsdLuxRectLight     [best-effort; orient + translate from fields]
#   SpotLight        → UsdLuxSphereLight   + ShapingAPI [best-effort]
#   EnvironmentLight → UsdLuxDomeLight     [best-effort; texture image deferred to M4]
#
# Open-stage live updates (M2.1):
#   `author_root_from_scene!` (usd.jl) BAKES camera + lights once at author time.
#   Afterwards, `sync_lights!` pushes minimal live writes for changed lights —
#   `inputs:intensity` (lanes=1), `inputs:color` (lanes=3), and `omni:xform` —
#   exactly as proven by the lights spike (clean A→B→A round-trips).  Light prim
#   paths are derived from the shared `light_prim_path` helper so the authored path
#   EQUALS the path written to.  Only the fully-tested light types get live sync;
#   exotic types remain baked (their live `_light_render_state` returns `nothing`).
#
# NOTE: included inside OmniverseMakie module, after camera.jl.
#       All of: OV, LibOVRTX, Makie, LinearAlgebra, camera_to_world, camera_intrinsics,
#       _usda_row_vector_matrix, _validate_camera_path, author_render_root! are in scope.

# ------------------------------------------------------------------
# light_prim_path — SINGLE SOURCE for a light's prim name + path
# ------------------------------------------------------------------

# A light's prim NAME is "<MakieType>_<index>" (DirectionalLight_0, PointLight_1,
# AmbientLight_0, ...), matching what `usda_light` authors; the prim is a direct
# child of /World.  BOTH the USDA authoring (`usda_light`) AND the live writer
# (`sync_lights!`) derive paths from here, so authored path == written path.
# `nameof(typeof(l))` yields exactly the legacy hard-coded names for every standard
# Makie light type (DirectionalLight/PointLight/AmbientLight/RectLight/SpotLight/
# EnvironmentLight).
light_prim_name(light, index::Int) = "$(nameof(typeof(light)))_$(index)"

"""
    light_prim_path(light, index) -> String

The `/World/<Type>_<index>` USD prim path for a Makie light.  Shared single source
of truth used by both `usda_light` authoring and `sync_lights!` live writes.
"""
light_prim_path(light, index::Int) = "/World/" * light_prim_name(light, index)

# Intensity scale factor (max-channel × scale) per Makie light type — kept in ONE
# place so `usda_light` (bake) and `_light_render_state` (live sync) never drift.
_light_intensity_scale(::Makie.DirectionalLight) = 3000.0
_light_intensity_scale(::Makie.PointLight)       = 10000.0
_light_intensity_scale(::Makie.AmbientLight)     = 1000.0
_light_intensity_scale(::Makie.RectLight)        = 3000.0
_light_intensity_scale(::Makie.SpotLight)        = 10000.0
_light_intensity_scale(_)                        = 1000.0

# ------------------------------------------------------------------
# _direction_to_xform — orientation matrix for directional lights
# ------------------------------------------------------------------

"""
    _direction_to_xform_matrix(dir) -> Matrix{Float64}

Build a 4×4 USD `matrix4d` (row-vector convention) that orients a light whose
emission axis is `dir`.  USD DistantLights (and RectLights) emit along local −Z,
so local +Z points opposite to the emission direction.  Returned as a plain
`Matrix{Float64}` so both the USDA emitter and the live `write_xform!` path can
consume it.
"""
function _direction_to_xform_matrix(dir)
    z = -normalize(Float64[dir[1], dir[2], dir[3]])   # local +Z = −emission direction
    a = abs(z[3]) < 0.99 ? Float64[0, 0, 1] : Float64[1, 0, 0]
    x = normalize(cross(a, z))
    y = cross(z, x)
    return Float64[x[1] x[2] x[3] 0.0
                   y[1] y[2] y[3] 0.0
                   z[1] z[2] z[3] 0.0
                   0.0  0.0  0.0  1.0]
end

"""
    _direction_to_xform(dir) -> String

USDA `matrix4d` literal (row-vector convention) for a directional light's
orientation — `_usda_row_vector_matrix ∘ _direction_to_xform_matrix`.
"""
_direction_to_xform(dir) = _usda_row_vector_matrix(_direction_to_xform_matrix(dir))

# 4×4 pure-translation matrix (USD row-vector convention: translation in the last ROW)
# for a light positioned at `p` (Sphere/Spot lights).
function _translation_xform(p)
    return Float64[1.0 0.0 0.0 0.0
                   0.0 1.0 0.0 0.0
                   0.0 0.0 1.0 0.0
                   Float64(p[1]) Float64(p[2]) Float64(p[3]) 1.0]
end

# ------------------------------------------------------------------
# _intensity_and_color — factor magnitude out into intensity; normalize hue
# ------------------------------------------------------------------

# Returns (intensity::Float64, r::Float64, g::Float64, b::Float64).
# intensity = scale × max(r,g,b); color = (r,g,b) / max(r,g,b) so max-channel = 1.
#
# NOTE: In Makie 0.24.x, AmbientLight.color is an Observable{RGBf} while all other
# light types store a plain RGBf.  This function accepts both forms.
function _intensity_and_color(color_or_obs, scale::Float64)
    c = color_or_obs isa Observable ? color_or_obs[] : color_or_obs
    r  = Float64(c.r)
    g  = Float64(c.g)
    b  = Float64(c.b)
    mag = max(r, g, b, 1e-10)
    return scale * mag, r / mag, g / mag, b / mag
end

# ------------------------------------------------------------------
# usda_light — per-type USDA prim block
# ------------------------------------------------------------------

"""
    usda_light(l, index::Int) -> String

Emit a UsdLux prim block for a single Makie light.  `index` is used to produce a
unique prim name when multiple lights of the same type exist (name = `<Type>_<index>`).

Each returned string starts with 4-space indent (direct child of `/World`) and ends
with `}\\n\\n` so blocks can be concatenated and placed directly in the `World` Xform.

Type mapping summary:
  DirectionalLight → UsdLuxDistantLight   intensity=3000×max
  PointLight       → UsdLuxSphereLight    intensity=10000×max, radius=1
  AmbientLight     → UsdLuxDomeLight      intensity=1000×max
  RectLight        → UsdLuxRectLight      [best-effort]
  SpotLight        → UsdLuxSphereLight    + ShapingAPI cone [best-effort]
  EnvironmentLight → UsdLuxDomeLight      (texture image deferred to M4) [best-effort]
"""
function usda_light(l::Makie.DirectionalLight, index::Int)
    intensity, cr, cg, cb = _intensity_and_color(l.color, _light_intensity_scale(l))
    xform = _direction_to_xform(l.direction)
    return """    def DistantLight "$(light_prim_name(l, index))" (
        prepend apiSchemas = ["ShapingAPI"]
    )
    {
        float inputs:angle = 1
        float inputs:intensity = $(intensity)
        color3f inputs:color = ($(cr), $(cg), $(cb))
        matrix4d xformOp:transform = $(xform)
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }

"""
end

function usda_light(l::Makie.PointLight, index::Int)
    intensity, cr, cg, cb = _intensity_and_color(l.color, _light_intensity_scale(l))
    p = l.position
    xform_str = _usda_row_vector_matrix(_translation_xform(p))
    return """    def SphereLight "$(light_prim_name(l, index))"
    {
        float inputs:radius = 1
        float inputs:intensity = $(intensity)
        color3f inputs:color = ($(cr), $(cg), $(cb))
        matrix4d xformOp:transform = $(xform_str)
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }

"""
end

function usda_light(l::Makie.AmbientLight, index::Int)
    intensity, cr, cg, cb = _intensity_and_color(l.color, _light_intensity_scale(l))
    return """    def DomeLight "$(light_prim_name(l, index))"
    {
        float inputs:intensity = $(intensity)
        color3f inputs:color = ($(cr), $(cg), $(cb))
    }

"""
end

function usda_light(l::Makie.RectLight, index::Int)
    # Best-effort: UsdLuxRectLight with width=norm(u1), height=norm(u2).
    # Orientation from direction; translation from position.
    intensity, cr, cg, cb = _intensity_and_color(l.color, _light_intensity_scale(l))
    w = norm(Float64[l.u1[1], l.u1[2], l.u1[3]])
    h = norm(Float64[l.u2[1], l.u2[2], l.u2[3]])
    # Build combined rotation+translation matrix.
    d = Float64[l.direction[1], l.direction[2], l.direction[3]]
    z = -normalize(d)
    a = abs(z[3]) < 0.99 ? Float64[0, 0, 1] : Float64[1, 0, 0]
    x_ax = normalize(cross(a, z))
    y_ax = cross(z, x_ax)
    p = l.position
    M = Float64[x_ax[1] x_ax[2] x_ax[3] 0.0
                y_ax[1] y_ax[2] y_ax[3] 0.0
                z[1]    z[2]    z[3]    0.0
                Float64(p[1]) Float64(p[2]) Float64(p[3]) 1.0]
    xform_str = _usda_row_vector_matrix(M)
    return """    def RectLight "$(light_prim_name(l, index))"
    {
        float inputs:width = $(w)
        float inputs:height = $(h)
        float inputs:intensity = $(intensity)
        color3f inputs:color = ($(cr), $(cg), $(cb))
        matrix4d xformOp:transform = $(xform_str)
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }

"""
end

function usda_light(l::Makie.SpotLight, index::Int)
    # Best-effort: UsdLuxSphereLight + ShapingAPI cone from l.angles.
    # l.angles = (inner_cutoff_radians, outer_cutoff_radians).
    intensity, cr, cg, cb = _intensity_and_color(l.color, _light_intensity_scale(l))
    p = l.position
    xform_str = _usda_row_vector_matrix(_translation_xform(p))
    inner_deg = Float64(l.angles[1]) * (180.0 / π)
    outer_deg = Float64(l.angles[2]) * (180.0 / π)
    softness  = max(0.0, (outer_deg - inner_deg) / max(outer_deg, 1e-6))
    return """    def SphereLight "$(light_prim_name(l, index))" (
        prepend apiSchemas = ["ShapingAPI"]
    )
    {
        float inputs:radius = 1
        float inputs:intensity = $(intensity)
        color3f inputs:color = ($(cr), $(cg), $(cb))
        float inputs:shaping:cone:angle = $(outer_deg)
        float inputs:shaping:cone:softness = $(softness)
        matrix4d xformOp:transform = $(xform_str)
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }

"""
end

function usda_light(l::Makie.EnvironmentLight, index::Int)
    # Best-effort: DomeLight with l.intensity scaled to USD range.
    # Texture image (l.image) is deferred to M4 — HDR environment map loading.
    intensity = Float64(l.intensity) * 1000.0
    return """    def DomeLight "$(light_prim_name(l, index))"
    {
        float inputs:intensity = $(intensity)
    }

"""
end

# Fallback for any future unknown light types — emit nothing, warn.
function usda_light(l, index::Int)
    @warn "OmniverseMakie: unsupported light type $(typeof(l)) at index $index — skipped"
    return ""
end

# Walk the scene lights (per-concrete-type 0-based counter) and return each light
# paired with its per-type index, in scene order.  Unsupported types still increment
# their counter so indices never desync between callers.  This is the SINGLE SOURCE
# for per-type light indexing; both `lights_usda` and `_light_paths` consume it.
function _enumerate_lights(lights)
    counts = Dict{DataType,Int}()
    out = Vector{Tuple{Any,Int}}(undef, length(lights))
    for (i, l) in enumerate(lights)
        T   = typeof(l)
        idx = get(counts, T, 0)
        counts[T] = idx + 1
        out[i] = (l, idx)
    end
    return out
end

# ------------------------------------------------------------------
# lights_usda — concatenate all scene lights into a USDA block
# ------------------------------------------------------------------

"""
    lights_usda(scene; cam_to_world = nothing) -> String

Convert `scene.compute[:lights][]` to a concatenated set of UsdLux prim blocks
for embedding inside the `/World` Xform in the root stage.

If the lights vector is empty, returns `_DEFAULT_LIGHTS_STR` (the M1.2 "Sun"
DistantLight) so renders stay lit.

`cam_to_world` (optional): the 4×4 camera-to-world matrix (USD row-vector
convention, from `camera_to_world`).  When provided, any `DirectionalLight` with
`camera_relative = true` has its direction transformed from camera space to world
space before emission.  If `nil`, camera-relative directions are treated as world-
space (a reasonable approximation when no camera matrix is available).
"""
function lights_usda(scene; cam_to_world = nothing)
    lights = scene.compute[:lights][]
    isempty(lights) && return _DEFAULT_LIGHTS_STR

    buf = IOBuffer()

    for (l, idx) in _enumerate_lights(lights)
        # Camera-relative DirectionalLight: transform direction to world space.
        l_emit = if l isa Makie.DirectionalLight && l.camera_relative && cam_to_world !== nothing
            d   = Float64[l.direction[1], l.direction[2], l.direction[3]]
            M33 = cam_to_world[1:3, 1:3]
            # Camera-space direction → world space (row-vector convention):
            # d_world[j] = Σᵢ d_cam[i] * M[i,j]  ≡  M33' * d_cam  (column-vector multiply)
            wd  = M33' * d
            Makie.DirectionalLight(l.color, Vec3f(wd[1], wd[2], wd[3]), false)
        else
            l
        end

        write(buf, usda_light(l_emit, idx))
    end

    result = String(take!(buf))
    isempty(result) && return _DEFAULT_LIGHTS_STR   # all lights were unsupported types
    return result
end

# ------------------------------------------------------------------
# Live light sync (M2.1) — push minimal writes for changed lights
# ------------------------------------------------------------------

# Per-light live render-state used for BOTH the snapshot and the writes:
#   (intensity::Float64, color::NTuple{3,Float64}, xform::Union{Nothing,Matrix{Float64}})
# Only the fully-tested light types get live sync; exotic types return `nothing`
# (they stay baked from author time).  Uses the SAME scale/intensity/xform math as
# `usda_light`, so the snapshot taken at author time matches the baked USDA exactly.
function _light_render_state(l::Makie.DirectionalLight)
    intensity, cr, cg, cb = _intensity_and_color(l.color, _light_intensity_scale(l))
    return (intensity = intensity, color = (cr, cg, cb),
            xform = _direction_to_xform_matrix(l.direction))
end
function _light_render_state(l::Makie.PointLight)
    intensity, cr, cg, cb = _intensity_and_color(l.color, _light_intensity_scale(l))
    return (intensity = intensity, color = (cr, cg, cb),
            xform = _translation_xform(l.position))
end
function _light_render_state(l::Makie.AmbientLight)
    intensity, cr, cg, cb = _intensity_and_color(l.color, _light_intensity_scale(l))
    return (intensity = intensity, color = (cr, cg, cb), xform = nothing)  # DomeLight: no xform
end
_light_render_state(_) = nothing   # exotic light type → no live sync (stays baked)

# Build (light, prim_path) pairs via the shared `_enumerate_lights` counter
# so each light's prim path here EQUALS the authored path.
function _light_paths(lights)
    return [(l, light_prim_path(l, idx)) for (l, idx) in _enumerate_lights(lights)]
end

# Snapshot of every light's (path, render-state); render-state may be `nothing`.
_lights_snapshot(lights) = Any[(path, _light_render_state(l)) for (l, path) in _light_paths(lights)]

# Scalar Float32 attribute write (e.g. inputs:intensity) — lanes=1.
function _write_light_intensity!(r, prim, val)
    dtype = LibOVRTX.DLDataType(UInt8(LibOVRTX.kDLFloat), UInt8(32), UInt16(1))
    OV._write_attribute!(r, prim, "inputs:intensity", dtype, false,
                         LibOVRTX.OVRTX_SEMANTIC_NONE, Float32[val], Int64[1])
    return nothing
end

# color3f attribute write (inputs:color) — one element with 3 float32 lanes.
function _write_light_color!(r, prim, color)
    dtype = LibOVRTX.DLDataType(UInt8(LibOVRTX.kDLFloat), UInt8(32), UInt16(3))
    OV._write_attribute!(r, prim, "inputs:color", dtype, false, LibOVRTX.OVRTX_SEMANTIC_NONE,
                         Float32[color[1], color[2], color[3]], Int64[1])
    return nothing
end

"""
    sync_lights!(screen, scene) -> Bool

Compare the scene's current lights to `screen.last_lights` (the snapshot last
written/baked) and push a MINIMAL live write for each changed attribute:
`inputs:intensity`, `inputs:color`, and/or `omni:xform` (per the proven lights
spike).  Updates `screen.last_lights` and returns `true` iff anything was written
(so `colorbuffer` knows to `OV.reset!` before rendering).

A structural change (the light *count* differs from the snapshot) is NOT a live
edit — it requires a stage re-open (a later-M2 concern); this function refreshes
the snapshot and returns `false` in that case.
"""
function sync_lights!(screen, scene)
    lights   = scene.compute[:lights][]
    new_snap = _lights_snapshot(lights)
    old_snap = screen.last_lights

    # Structural mismatch (or first call without a baked snapshot): don't try to
    # write to prims that may not exist — just record the current snapshot.
    if old_snap === nothing || length(old_snap) != length(new_snap)
        screen.last_lights = new_snap
        return false
    end

    changed = false
    r = screen.renderer
    for i in eachindex(new_snap)
        path, st  = new_snap[i]
        _, ost    = old_snap[i]
        st === nothing && continue          # exotic light: no live sync
        if ost === nothing || st.intensity != ost.intensity
            _write_light_intensity!(r, path, st.intensity); changed = true
        end
        if ost === nothing || st.color != ost.color
            _write_light_color!(r, path, st.color); changed = true
        end
        if st.xform !== nothing && (ost === nothing || st.xform != ost.xform)
            OV.write_xform!(r, path, st.xform); changed = true
        end
    end
    screen.last_lights = new_snap
    return changed
end

# ------------------------------------------------------------------
# author_lights! — intent-named delegator to author_root_from_scene!
# ------------------------------------------------------------------

"""
    author_lights!(screen, scene; camera_path="/World/Camera") -> Nothing

Bake `scene`'s lights (and camera) into the USD render-root via
`author_root_from_scene!` (usd.jl).  Camera and lights are baked together in one
`open_usd_string!`.  After the initial bake, live light changes go through
`sync_lights!`, not a re-bake.
"""
function author_lights!(screen, scene; camera_path::String = "/World/Camera")
    return author_root_from_scene!(screen, scene; camera_path = camera_path)
end
