# Environment-light image (IBL): a textured UsdLux DomeLight in its OWN removable reference
# layer, so the environment map is LIVE-swappable on the open stage.
#
# WHY a reference layer (not the root-baked lights_usda path): ovrtx asset inputs
# (`inputs:texture:file`) are NOT live-writable through the attribute FFI, and the root layer is
# not removable — so the ONLY way to swap an environment map mid-session is the volume-reload
# pattern: `remove_usd!` the dome's layer + `add_usd_reference!` a fresh one pointing at a FRESH
# temp texture path (ovrtx reads an asset path exactly once; re-writing an existing file corrupts
# the async read — the texture fresh-path lesson).
#
# The dome illuminates the scene always; with `ScreenConfig.background = :domelight` (settings.jl)
# it also shows as the visible background (`omni:rtx:background:source:type`).
#
# Included after materials.jl (uses `_usd_asset_path`, PNGFiles) and BEFORE screen.jl (the
# `Screen.env_light` field is typed `Union{Nothing,EnvLightState}`).  `screen` args are duck-typed
# (Screen is defined later), like volume.jl.

"""
    EnvLightState

Per-screen environment-light record (`Screen.env_light`; `nothing` until the first
`EnvironmentLight` / [`push_environment_image!`](@ref)).

- `handle`  — the dome layer's `add_usd_reference!` handle (`0` = no dome authored yet).
- `tmp`     — the screen-owned temp texture backing the CURRENT dome (`nothing` when the
              texture is a user-supplied file).  Deleted on the next push and at `close`.
- `pending` — a `(source, intensity, format)` push stashed BEFORE the stage was authored;
              applied by `_author_env_light!` (imperative pushes win over a scene
              `EnvironmentLight`).
"""
mutable struct EnvLightState
    handle::UInt64
    tmp::Union{Nothing,String}
    pending::Union{Nothing,Tuple{Any,Float64,String}}
end
EnvLightState() = EnvLightState(UInt64(0), nothing, nothing)

# One dome per screen, at a fixed prim path (not a plot; not in the lights index space).
const _ENV_LIGHT_PRIM = "/World/OVMakieEnvLight"

# PURE self-contained USDA layer: one DomeLight (the defaultPrim) with a latlong
# (equirectangular) environment texture.  `intensity` is in Makie `EnvironmentLight` units —
# 1.0 maps to USD 1000 (the same scale `usda_light(::EnvironmentLight)` used).
function _env_dome_usda(texture_path::AbstractString; intensity::Real = 1.0,
                        format::AbstractString = "latlong")
    usd_intensity = Float64(intensity) * 1000.0
    return """#usda 1.0
(
    defaultPrim = "EnvLight"
)
def DomeLight "EnvLight"
{
    float inputs:intensity = $(usd_intensity)
    asset inputs:texture:file = $(_usd_asset_path(texture_path; what = "environment texture"))
    token inputs:texture:format = "$(format)"
}
"""
end

# Resolve an environment-image source to an on-disk texture path.
# - `AbstractMatrix{<:Colorant}` → a FRESH temp PNG (LDR: components CLAMPED to [0,1] — N0f8
#   conversion of >1 values would throw; warn once pointing at the file-path route for true HDR).
# - `AbstractString` → an existing file (`.exr`/`.hdr`/`.png`/…), absolutized; ovrtx reads HDR
#   formats natively so this is the full-radiance route.
# Returns `(abs_path, is_temp)`.
function _env_texture_file(img::AbstractMatrix{<:Colorant})
    clamped = map(img) do c
        r, g, b = Float32(red(c)), Float32(green(c)), Float32(blue(c))
        (r > 1 || g > 1 || b > 1) &&
            @warn "OmniverseMakie: environment image has components > 1 — clamped to [0,1] for \
                   the PNG env map. Pass an .exr/.hdr FILE PATH to push_environment_image! for \
                   true HDR radiance." maxlog = 1
        RGBA{N0f8}(clamp(r, 0f0, 1f0), clamp(g, 0f0, 1f0), clamp(b, 0f0, 1f0), 1f0)
    end
    path = tempname() * ".png"
    PNGFiles.save(path, clamped)
    return (path, true)
end
function _env_texture_file(path::AbstractString)
    isfile(path) || throw(ArgumentError(
        "push_environment_image!: environment texture file not found: $(path)"))
    return (abspath(String(path)), false)
end

# Get-or-create the screen's EnvLightState.
_env_state!(screen) =
    screen.env_light === nothing ? (screen.env_light = EnvLightState()) : screen.env_light

# Swap the dome layer on the OPEN stage: resolve the texture, remove the old layer (if any),
# add the fresh one.  Teardown-safe like reload_volume_data!: a fresh temp is GC'd if the add
# throws (no leak); a failed add leaves handle = 0 so the NEXT push recovers with a plain add.
# Both remove and add are structural composition changes (`_note_composition_change!`) so
# accumulate-across-frames resets exactly once for the swap.
function _apply_environment!(screen, source, intensity::Real, format::AbstractString)
    st = _env_state!(screen)
    path, is_tmp = _env_texture_file(source)
    added = false
    try
        usda = _env_dome_usda(path; intensity, format)
        if st.handle != 0
            OV.remove_usd!(screen.renderer, st.handle)
            st.handle = 0                                # dangling until the add succeeds
        end
        st.handle = OV.add_usd_reference!(screen.renderer, usda, _ENV_LIGHT_PRIM)
        added = true
        _note_composition_change!(screen)                # dome swapped → structural reset
        screen.requires_update = true
    finally
        if added
            old = st.tmp                                 # GC the PRIOR temp (bounded: one on disk)
            st.tmp = is_tmp ? path : nothing
            old === nothing || old == path || rm(old; force = true)
        elseif is_tmp
            rm(path; force = true)                       # add threw → GC the orphan temp
        end
    end
    return nothing
end

"""
    push_environment_image!(screen, source; intensity = 1.0, format = "latlong") -> Nothing

Set (or live-replace) the scene's image-based environment light: a `UsdLux DomeLight` whose
latlong/equirectangular texture comes from `source` —

- an `AbstractMatrix{<:Colorant}` (e.g. Makie's `EnvironmentLight.image` `Matrix{RGBf}`):
  written to a fresh temp PNG (LDR — components clamped to `[0,1]`);
- an `AbstractString` path to an existing `.exr`/`.hdr`/`.png` file (the true-HDR route;
  ovrtx reads the file directly).

`intensity` is in Makie `EnvironmentLight` units (1.0 ≈ USD dome intensity 1000).  Pushing again
replaces the map live (remove + re-reference — the same proven pattern as volume data reloads);
in `accumulate_across_frames` mode the swap is a structural change, so it resets accumulation
exactly once.  Called BEFORE the screen's first render, the push is stashed and applied when the
stage is authored (and then wins over any scene `EnvironmentLight`).

The dome always ILLUMINATES; to also SHOW it as the visible background, create the screen with
`background = :domelight` (see `ScreenConfig`).  A scene-lights
`EnvironmentLight(intensity, image)` is authored through this same mechanism automatically.
"""
function push_environment_image!(screen, source; intensity::Real = 1.0,
                                 format::AbstractString = "latlong")
    if !screen.authored
        # Stage not open yet — stash; `_author_env_light!` applies it at author time.  Resolve a
        # MATRIX to a copy now (the caller may mutate it before display); a path stays a path.
        src = source isa AbstractMatrix ? copy(source) : source
        _env_state!(screen).pending = (src, Float64(intensity), String(format))
        return nothing
    end
    _apply_environment!(screen, source, intensity, format)
    return nothing
end

# Author-time hook (called by `_author_screen!` right after the root opens): apply a stashed
# pre-display push if there is one (imperative wins), else the FIRST scene `EnvironmentLight`
# (warn on extras — one dome per screen).  A failure degrades to a loud warn + skip so one bad
# environment image can't kill the whole figure (matches the usdplot stashed-binding policy).
function _author_env_light!(screen, cam_scene)
    st = screen.env_light
    if st !== nothing && st.pending !== nothing
        source, intensity, format = st.pending
        st.pending = nothing
        try
            _apply_environment!(screen, source, intensity, format)
        catch e
            @warn "OmniverseMakie: pending push_environment_image! failed at author time — \
                   skipped." exception = e
        end
        return nothing
    end
    envs = [l for l in cam_scene.compute[:lights][] if l isa Makie.EnvironmentLight]
    isempty(envs) && return nothing
    length(envs) > 1 &&
        @warn "OmniverseMakie: $(length(envs)) EnvironmentLights in the scene — only the first \
               is used (one environment dome per screen)."
    env = first(envs)
    try
        _apply_environment!(screen, env.image, Float64(env.intensity), "latlong")
    catch e
        @warn "OmniverseMakie: authoring the scene EnvironmentLight failed — skipped." exception = e
    end
    return nothing
end

# Teardown: GC the screen-owned temp texture (the dome layer itself dies with the renderer).
# Idempotent (`force = true`); called from `Base.close(::Screen)`.
function _destroy_env_light!(screen)
    st = screen.env_light
    st === nothing && return nothing
    st.tmp === nothing || rm(st.tmp; force = true)
    st.tmp = nothing
    st.handle = 0
    return nothing
end
