# Lights translation for OmniverseMakie. Key functions: lights_usda (all scene lights
# as UsdLux blocks), usda_light (per-type block), light_prim_path (single source for
# /World/<Type>_<index>), sync_lights! (live diff), author_lights! (delegator).
#
# Type mapping (tested: Directional/Point/Ambient; best-effort: Rect/Spot/Environment).
# Intensity = scale × max-channel; scales live in `_light_intensity_scale`:
#   DirectionalLight → DistantLight (750)
#   PointLight       → SphereLight  (2500, radius 1)
#   AmbientLight     → DomeLight    (250)
#   RectLight        → RectLight    [orient+translate from fields]
#   SpotLight        → SphereLight  + ShapingAPI
#   EnvironmentLight → DomeLight    [texture deferred to M4]
#
# Open-stage live updates (M2.1): author_root_from_scene! BAKES camera + lights once;
# afterwards sync_lights! pushes minimal writes (inputs:intensity, inputs:color,
# omni:xform) for changed lights. Paths come from `light_prim_path` so authored path
# EQUALS written path. Only tested types get live sync; exotic stay baked.
#
# Included inside the OmniverseMakie module after camera.jl (OV, LibOVRTX, Makie,
# LinearAlgebra, camera_to_world, camera_intrinsics, _usda_row_vector_matrix,
# _validate_camera_path, author_render_root! in scope).

# ------------------------------------------------------------------
# light_prim_path — SINGLE SOURCE for a light's prim name + path
# ------------------------------------------------------------------

# A light's prim NAME is "<MakieType>_<index>" (DirectionalLight_0, ...), a direct
# child of /World. Both `usda_light` (authoring) and `sync_lights!` (live) derive
# paths from here, so authored path == written path. `nameof(typeof(l))` yields the
# standard Makie light type names.
light_prim_name(light, index::Int) = "$(nameof(typeof(light)))_$(index)"

"""
    light_prim_path(light, index) -> String

`/World/<Type>_<index>` prim path for a Makie light. Single source used by both
`usda_light` authoring and `sync_lights!` live writes.
"""
light_prim_path(light, index::Int) = "/World/" * light_prim_name(light, index)

# Intensity scale (max-channel × scale) per light type — kept in ONE place so
# `usda_light` (bake) and `_light_render_state` (live sync) never drift.
# CALIBRATION (2026-06-29): reduced 4× — the originals were too hot vs RPRMakie and
# washed lit surfaces to white. ovrtx's RT2 path does NOT honor camera exposure/
# auto-exposure (verified), so INPUT RADIANCE (this scale) is the only brightness
# lever; nothing auto-compensates.
_light_intensity_scale(::Makie.DirectionalLight) = 750.0
_light_intensity_scale(::Makie.PointLight)       = 2500.0
_light_intensity_scale(::Makie.AmbientLight)     = 250.0
_light_intensity_scale(::Makie.RectLight)        = 750.0
_light_intensity_scale(::Makie.SpotLight)        = 2500.0
_light_intensity_scale(_)                        = 250.0

# ------------------------------------------------------------------
# _direction_to_xform — orientation matrix for directional lights
# ------------------------------------------------------------------

"""
    _direction_to_xform_matrix(dir) -> Matrix{Float64}

4×4 USD orientation matrix (row-vector) for a light emitting along `dir`. USD
DistantLights/RectLights emit along local −Z, so local +Z = −`dir`. Plain
`Matrix{Float64}` so both the USDA emitter and live `write_xform!` consume it.
"""
function _direction_to_xform_matrix(dir)
    z = -normalize(Float64[dir[1], dir[2], dir[3]])   # local +Z = −emission direction
    ref_axis = abs(z[3]) < 0.99 ? Float64[0, 0, 1] : Float64[1, 0, 0]
    x = normalize(cross(ref_axis, z))
    y = cross(z, x)
    return Float64[x[1] x[2] x[3] 0.0
                   y[1] y[2] y[3] 0.0
                   z[1] z[2] z[3] 0.0
                   0.0  0.0  0.0  1.0]
end

"""
    _direction_to_xform(dir) -> String

USDA `matrix4d` literal for a directional light's orientation
(`_usda_row_vector_matrix ∘ _direction_to_xform_matrix`).
"""
_direction_to_xform(dir) = _usda_row_vector_matrix(_direction_to_xform_matrix(dir))

# 4×4 pure-translation matrix (USD row-vector: translation in the last ROW) for a
# light at `p` (Sphere/Spot lights).
function _translation_xform(p)
    return Float64[1.0 0.0 0.0 0.0
                   0.0 1.0 0.0 0.0
                   0.0 0.0 1.0 0.0
                   Float64(p[1]) Float64(p[2]) Float64(p[3]) 1.0]
end

# ------------------------------------------------------------------
# _intensity_and_color — factor magnitude out into intensity; normalize hue
# ------------------------------------------------------------------

# Returns (intensity, r, g, b)::Float64. intensity = scale × max(r,g,b); colour is
# normalized so max-channel = 1. Accepts a plain RGBf or an Observable{RGBf}
# (AmbientLight.color is observable in Makie 0.24.x; others are plain).
function _intensity_and_color(color_or_obs, scale::Float64)
    color = color_or_obs isa Observable ? color_or_obs[] : color_or_obs
    r  = Float64(color.r)
    g  = Float64(color.g)
    b  = Float64(color.b)
    mag = max(r, g, b, 1e-10)
    return scale * mag, r / mag, g / mag, b / mag
end

# ------------------------------------------------------------------
# usda_light — per-type USDA prim block
# ------------------------------------------------------------------

"""
    usda_light(l, index::Int) -> String

UsdLux prim block for one Makie light. `index` disambiguates prim names of the same
type (name = `<Type>_<index>`). Each block is 4-space indented (a `/World` child) and
ends with `}\\n\\n` so blocks concatenate directly into the `World` Xform. See the
file header for the per-type mapping and intensity scales.
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
    # Best-effort: RectLight, width=norm(u1), height=norm(u2); orient from direction,
    # translate from position.
    intensity, cr, cg, cb = _intensity_and_color(l.color, _light_intensity_scale(l))
    w = norm(Float64[l.u1[1], l.u1[2], l.u1[3]])
    h = norm(Float64[l.u2[1], l.u2[2], l.u2[3]])
    # Combined rotation + translation matrix.
    dir = Float64[l.direction[1], l.direction[2], l.direction[3]]
    z = -normalize(dir)
    ref_axis = abs(z[3]) < 0.99 ? Float64[0, 0, 1] : Float64[1, 0, 0]
    x_ax = normalize(cross(ref_axis, z))
    y_ax = cross(z, x_ax)
    p = l.position
    xform_matrix = Float64[x_ax[1] x_ax[2] x_ax[3] 0.0
                           y_ax[1] y_ax[2] y_ax[3] 0.0
                           z[1]    z[2]    z[3]    0.0
                           Float64(p[1]) Float64(p[2]) Float64(p[3]) 1.0]
    xform_str = _usda_row_vector_matrix(xform_matrix)
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
    # Best-effort: SphereLight + ShapingAPI cone from l.angles =
    # (inner_cutoff, outer_cutoff) radians.
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
    # Best-effort: DomeLight, l.intensity scaled to USD range. Texture (l.image)
    # deferred to M4 (HDR environment map).
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

# Pair each light with its per-type 0-based index, in scene order. Unsupported types
# still increment their counter so indices never desync. SINGLE SOURCE for per-type
# indexing; used by `lights_usda` and `_light_paths`.
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

Concatenated UsdLux prim blocks from `scene.compute[:lights][]`, for the root
stage's `/World` Xform. Empty lights → `_DEFAULT_LIGHTS_STR` (the "Sun" DistantLight)
so renders stay lit.

`cam_to_world` (4×4 camera-to-world, row-vector): when given, a `DirectionalLight`
with `camera_relative=true` has its direction transformed camera→world; `nothing`
treats such directions as world-space.
"""
function lights_usda(scene; cam_to_world = nothing)
    lights = scene.compute[:lights][]
    isempty(lights) && return _DEFAULT_LIGHTS_STR

    buf = IOBuffer()

    for (l, idx) in _enumerate_lights(lights)
        # Camera-relative DirectionalLight: rotate its direction camera→world.
        l_emit = if l isa Makie.DirectionalLight && l.camera_relative && cam_to_world !== nothing
            cam_dir = Float64[l.direction[1], l.direction[2], l.direction[3]]
            rot3x3  = cam_to_world[1:3, 1:3]
            # Row-vector convention: world[j] = Σᵢ cam[i]·M[i,j] ≡ rot3x3' * cam_dir.
            world_dir = rot3x3' * cam_dir
            Makie.DirectionalLight(l.color, Vec3f(world_dir[1], world_dir[2], world_dir[3]), false)
        else
            l
        end

        write(buf, usda_light(l_emit, idx))
    end

    result = String(take!(buf))
    isempty(result) && return _DEFAULT_LIGHTS_STR   # all lights unsupported → default
    return result
end

# ------------------------------------------------------------------
# Live light sync (M2.1) — push minimal writes for changed lights
# ------------------------------------------------------------------

# Per-light live render-state for both the snapshot and the writes:
#   (intensity, color::NTuple{3}, xform::Union{Nothing,Matrix{Float64}}).
# Same scale/intensity/xform math as `usda_light`, so the author-time snapshot matches
# the baked USDA. Only tested types sync; exotic types return `nothing` (stay baked).
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
_light_render_state(_) = nothing   # exotic type → no live sync (stays baked)

# (light, prim_path) pairs via the shared `_enumerate_lights` counter, so each path
# here EQUALS the authored path.
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

Diff the scene's lights against `screen.last_lights` and push a MINIMAL live write per
changed attribute (`inputs:intensity`, `inputs:color`, `omni:xform`). Updates
`screen.last_lights`; returns `true` iff anything was written (so `colorbuffer` knows
to `OV.reset!`).

A structural change (light *count* differs) needs a stage re-open, not a live edit:
this just refreshes the snapshot and returns `false`.
"""
function sync_lights!(screen, scene)
    lights   = scene.compute[:lights][]
    new_snap = _lights_snapshot(lights)
    old_snap = screen.last_lights

    # Structural mismatch (or first call, no baked snapshot): prims may not exist yet,
    # so just record the snapshot.
    if old_snap === nothing || length(old_snap) != length(new_snap)
        screen.last_lights = new_snap
        return false
    end

    changed = false
    r = screen.renderer
    for i in eachindex(new_snap)
        path, new_state = new_snap[i]
        _, old_state    = old_snap[i]
        new_state === nothing && continue          # exotic light: no live sync
        if old_state === nothing || new_state.intensity != old_state.intensity
            _write_light_intensity!(r, path, new_state.intensity); changed = true
        end
        if old_state === nothing || new_state.color != old_state.color
            _write_light_color!(r, path, new_state.color); changed = true
        end
        if new_state.xform !== nothing && (old_state === nothing || new_state.xform != old_state.xform)
            OV.write_xform!(r, path, new_state.xform); changed = true
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

Bake `scene`'s lights and camera into the render-root via `author_root_from_scene!`
(one `open_usd_string!`). After the bake, live light changes go through
`sync_lights!`, not a re-bake.
"""
function author_lights!(screen, scene; camera_path::String = "/World/Camera")
    return author_root_from_scene!(screen, scene; camera_path = camera_path)
end
