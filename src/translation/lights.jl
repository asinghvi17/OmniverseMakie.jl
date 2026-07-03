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
#   EnvironmentLight → textured DomeLight via a REMOVABLE reference (envlight.jl), NOT baked
#                      here — live-swappable via push_environment_image!
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
    _xform_matrix(t::NTuple{16,Float64}) -> Matrix{Float64}

Row-major 4×4 `Matrix{Float64}` from a flat 16-tuple in USD row-vector layout (row 1 =
`t[1:4]`, …). The stack-tuple form (`_direction_to_xform_tuple`/`_translation_xform_tuple`)
is what the per-frame light snapshot stores allocation-free; this rebuilds the `Matrix` the
USDA emitter and `OV.write_xform!` consume — called only at author time and on an ACTUAL
live xform change, never on an idle frame.
"""
function _xform_matrix(t::NTuple{16,Float64})
    return Float64[t[1]  t[2]  t[3]  t[4]
                   t[5]  t[6]  t[7]  t[8]
                   t[9]  t[10] t[11] t[12]
                   t[13] t[14] t[15] t[16]]
end

"""
    _direction_to_xform_tuple(dir) -> NTuple{16,Float64}

Flat 4×4 USD orientation matrix (row-vector, `_xform_matrix` layout) for a light emitting
along `dir`. USD DistantLights/RectLights emit along local −Z, so local +Z = −`dir`.
Computed on `Vec3d` (StaticArrays) math so it ALLOCATES NOTHING — the form the per-frame
snapshot stores. Byte-identical to the pre-L2 `Matrix` build (pinned by the RectLight golden
in `test/l1_lights_structural_test.jl`).
"""
function _direction_to_xform_tuple(dir)
    z = -normalize(Vec3d(dir[1], dir[2], dir[3]))     # local +Z = −emission direction
    ref_axis = abs(z[3]) < 0.99 ? Vec3d(0.0, 0.0, 1.0) : Vec3d(1.0, 0.0, 0.0)
    x = normalize(cross(ref_axis, z))
    y = cross(z, x)
    return (x[1], x[2], x[3], 0.0,
            y[1], y[2], y[3], 0.0,
            z[1], z[2], z[3], 0.0,
            0.0,  0.0,  0.0,  1.0)
end

"""
    _direction_to_xform_matrix(dir) -> Matrix{Float64}

4×4 USD orientation matrix (row-vector) for a light emitting along `dir`, as a plain
`Matrix{Float64}` the USDA emitter (`_direction_to_xform`) and RectLight authoring consume.
Thin `Matrix` wrapper over the allocation-free `_direction_to_xform_tuple` core, so the bake
and live-sync share ONE direction→orientation computation.
"""
_direction_to_xform_matrix(dir) = _xform_matrix(_direction_to_xform_tuple(dir))

"""
    _direction_to_xform(dir) -> String

USDA `matrix4d` literal for a directional light's orientation
(`_usda_row_vector_matrix ∘ _direction_to_xform_matrix`).
"""
_direction_to_xform(dir) = _usda_row_vector_matrix(_direction_to_xform_matrix(dir))

# Flat pure-translation matrix (USD row-vector: translation in the last ROW) for a light at
# `p` (Sphere/Spot lights), as an allocation-free `NTuple{16,Float64}` — the form the snapshot
# stores. `_translation_xform` is the `Matrix` wrapper the USDA emitter consumes.
function _translation_xform_tuple(p)
    return (1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            Float64(p[1]), Float64(p[2]), Float64(p[3]), 1.0)
end
_translation_xform(p) = _xform_matrix(_translation_xform_tuple(p))

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
    # Best-effort: RectLight, width=norm(u1), height=norm(u2); orient from direction
    # (shared with DistantLight), translate from position.
    intensity, cr, cg, cb = _intensity_and_color(l.color, _light_intensity_scale(l))
    w = norm(Float64[l.u1[1], l.u1[2], l.u1[3]])
    h = norm(Float64[l.u2[1], l.u2[2], l.u2[3]])
    # Orientation from the shared helper; drop the translation into the last row.
    xform_matrix = copy(_direction_to_xform_matrix(l.direction))
    xform_matrix[4, 1:3] .= l.position
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

# EnvironmentLight is NOT baked here: its textured DomeLight is authored as a REMOVABLE
# reference by `_author_env_light!` (envlight.jl, hooked in `_author_screen!`) so the
# environment map can be live-swapped via `push_environment_image!` — asset inputs are not
# FFI-writable and the root layer is not removable, so a root-baked dome would be frozen.
# Emitting nothing here keeps `_enumerate_lights` indices stable for the other types.
usda_light(l::Makie.EnvironmentLight, index::Int) = ""

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

"""
    LightState(intensity, color, xform)

Immutable per-light render state the live-sync snapshot compares frame-to-frame. Stores the
exact values `sync_lights!` writes — `Float32` intensity + colour (matching the attribute
writes) and the xform as a stack `NTuple{16,Float64}` (`nothing` for a DomeLight), converted
to the `Matrix` form (`_xform_matrix`) only AT WRITE TIME. Fields are compared individually in
`sync_lights!`, so an idle frame recomputes state without a `Dict`, fresh strings, or `Matrix`
temporaries.
"""
struct LightState
    intensity::Float32
    color::NTuple{3,Float32}
    xform::Union{Nothing,NTuple{16,Float64}}
end

# The per-frame light snapshot: (authored prim path, render-state-or-`nothing`) per light, a
# CONCRETE eltype so `sync_lights!`'s diff loop reads it without dynamic dispatch (vs the old
# `Any[]`). Named so `Screen.last_lights` (screen.jl) can be typed `Union{Nothing,LightSnapshot}`.
const LightSnapshot = Vector{Tuple{String,Union{LightState,Nothing}}}

# Per-light live render-state for both the snapshot and the writes. Same scale/intensity/xform
# math as `usda_light`, so the author-time snapshot matches the baked USDA (the `LightState`
# constructor narrows the Float64 intensity/colour to the Float32 that gets written). Only
# tested types sync; exotic types return `nothing` (stay baked).
function _light_render_state(l::Makie.DirectionalLight)
    intensity, cr, cg, cb = _intensity_and_color(l.color, _light_intensity_scale(l))
    return LightState(intensity, (cr, cg, cb), _direction_to_xform_tuple(l.direction))
end
function _light_render_state(l::Makie.PointLight)
    intensity, cr, cg, cb = _intensity_and_color(l.color, _light_intensity_scale(l))
    return LightState(intensity, (cr, cg, cb), _translation_xform_tuple(l.position))
end
function _light_render_state(l::Makie.AmbientLight)
    intensity, cr, cg, cb = _intensity_and_color(l.color, _light_intensity_scale(l))
    return LightState(intensity, (cr, cg, cb), nothing)   # DomeLight: no xform
end
_light_render_state(_) = nothing   # exotic type → no live sync (stays baked)

# (light, prim_path) pairs via the shared `_enumerate_lights` counter, so each path
# here EQUALS the authored path.
function _light_paths(lights)
    return [(l, light_prim_path(l, idx)) for (l, idx) in _enumerate_lights(lights)]
end

# Per-type authored-path PREFIX ("/World/<Type>_") cache. Lets the two-arg snapshot verify —
# allocation-free after a one-time warmup per type — that a reused path still names the SAME
# light type as the light now at that position. Keyed by concrete light type; the trailing "_"
# makes the prefix an unambiguous type tag (no light type name is a prefix of another followed
# by "_"). The `get(…, "")` + explicit store avoids the closure `get!` would allocate per call.
const _LIGHT_PATH_PREFIX = Dict{DataType,String}()
function _light_path_prefix(l)
    T = typeof(l)
    cached = get(_LIGHT_PATH_PREFIX, T, "")
    isempty(cached) || return cached
    prefix = "/World/$(nameof(T))_"
    _LIGHT_PATH_PREFIX[T] = prefix
    return prefix
end

# Snapshot of every light's (authored path, render-state); render-state may be `nothing`.
# The single-arg form (author-time seed) computes paths fresh via `_light_paths`. The two-arg
# form REUSES the previous snapshot's path strings only when the light count AND every position's
# light TYPE are unchanged (paths encode <Type>_<idx>) — so an idle-frame rebuild allocates
# neither the `_enumerate_lights` `Dict` nor fresh interpolated path strings. Any count OR
# positional-type change falls back to a fresh build with correctly recomputed paths.
function _lights_snapshot(lights)
    snap = LightSnapshot(undef, length(lights))
    for (i, (l, path)) in enumerate(_light_paths(lights))
        snap[i] = (path, _light_render_state(l))
    end
    return snap
end
function _lights_snapshot(lights, old::LightSnapshot)
    length(old) == length(lights) || return _lights_snapshot(lights)   # count changed → fresh
    # Reuse the authored path strings ONLY if every position's light TYPE still matches its old
    # path. Paths encode <Type>_<idx>, so a same-count type SWAP/permutation would otherwise
    # graft a stale path onto a new light's state → that state gets written to the WRONG prim,
    # silently corrupting both. On any positional-type mismatch, rebuild FRESH (recomputed paths
    # = exact pre-L2 semantics: a permutation writes to the correctly-named authored prim; a
    # same-count multiset change writes to a never-authored prim = silent no-op, as before L2).
    # Positional-type equality ⟺ identical per-type enumeration ⟺ identical paths, so a passing
    # check guarantees the reused paths EQUAL a fresh build's.
    for i in eachindex(lights)
        startswith(old[i][1], _light_path_prefix(lights[i])) || return _lights_snapshot(lights)
    end
    snap = LightSnapshot(undef, length(lights))
    for i in eachindex(lights)
        snap[i] = (old[i][1], _light_render_state(lights[i]))          # reuse the authored path
    end
    return snap
end

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

A structural change (light *count* differs) can't be applied live — it would need a
stage re-open. Such a change warns once (`maxlog=1`) and returns `false` WITHOUT
advancing the snapshot, so the mismatch stays detectable every frame instead of
silently corrupting the diff baseline (advancing it would diff the next edit against
never-authored prims). The added/removed light does not render; create a new Screen.
"""
function sync_lights!(screen, scene)
    lights   = scene.compute[:lights][]
    old_snap = screen.last_lights

    # First call (nothing baked yet): seed the snapshot so later diffs have a baseline.
    if old_snap === nothing
        screen.last_lights = _lights_snapshot(lights)
        return false
    end

    # Reuse the baked path strings (invariant unless the count changes) so an idle frame's
    # rebuild stays allocation-lean; a count change falls back to a fresh snapshot, caught next.
    new_snap = _lights_snapshot(lights, old_snap)

    # Structural change (light COUNT differs): a live edit can't add/remove prims (that
    # needs a stage re-open). Do NOT advance the snapshot — advancing it would diff the
    # next edit against never-authored prims. Leave it stale so the mismatch stays
    # detectable every frame; warn once.
    if length(old_snap) != length(new_snap)
        @warn "OmniverseMakie: adding/removing lights on a live Screen is not supported — create a new Screen" maxlog=1
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
            OV.write_xform!(r, path, _xform_matrix(new_state.xform)); changed = true   # NTuple → Matrix at write time
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
