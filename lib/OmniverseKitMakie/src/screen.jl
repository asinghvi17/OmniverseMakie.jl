# KitScreen — a Makie screen whose backend is a Kit render runtime, giving
# full-color NVIDIA IndeX volume rendering (transfer-function colormaps) that
# the ovrtx-FFI `OmniverseMakie.Screen` cannot do.
#
# The runtime is reached through a `KitTransport` (transport.jl):
#   * :subprocess (DEFAULT) — a persistent headless `kit` subprocess (KitServer).
#   * :inprocess (opt-in)   — Kit hosted in this process via LibKitJL.
# The Julia-facing API below is identical across transports.
#
# Two modes:
#   * scene-backed (`KitScreen(scene)`): the scene's volume plots, lights and
#     camera are authored to a stage (authoring.jl) and opened; every
#     `Makie.colorbuffer(screen)` re-syncs the camera and renders.
#   * stage-backed (`KitScreen(server)`): render externally-authored stages
#     via `render_stage!` — the original spike surface, kept for probing.

"""
    KitScreen(scene::Makie.Scene; server=nothing, transport=nothing, frames=240, kwargs...) -> KitScreen
    KitScreen(server::KitServer; size=(1280, 720)) -> KitScreen

Scene-backed: author `scene` (volume plots + lights + camera; see
[`stage_usda`](@ref)) and open it.  `Makie.colorbuffer(screen)` syncs the
camera and renders `frames` convergence frames.  `close(screen)` shuts the
backend down only if the screen started it.

Transport selection (default `:subprocess`): pass `transport = :subprocess`
(persistent `kit` subprocess — proven, and the only path that coexists with
in-process ovrtx) or `:inprocess` (Kit hosted in this Julia process via
LibKitJL — opt-in, faster repeated renders, but a session may host only ONE
in-process backend; see [`InProcessTransport`](@ref)).  Overridable by the
`OMK_KIT_TRANSPORT` env var.  Passing an explicit `server::KitServer` forces
the subprocess transport on that server; passing a `transport::KitTransport`
instance reuses that transport (the caller owns it — the A/B pattern for the
single in-process app).  Extra `kwargs` go to the chosen transport's
constructor (`start_kit_server` / `InProcessTransport`), with `size(scene)` as
the default resolution.

Stage-backed: no scene; drive with [`render_stage!`](@ref) / [`set_attr!`](@ref).
"""
mutable struct KitScreen <: Makie.MakieScreen
    transport::KitTransport
    owns_transport::Bool                # close the transport on `close` only if we started it
    scene::Union{Nothing, Makie.Scene}
    size::Tuple{Int, Int}
    workdir::String
    stage_path::Union{Nothing, String}  # currently open stage
    framecount::Int                     # for default output naming
    frames::Int                         # convergence frames per colorbuffer
end

KitScreen(server::KitServer; size::Tuple{<:Integer, <:Integer} = (1280, 720)) =
    KitScreen(SubprocessTransport(server, false), false, nothing,
              (Int(size[1]), Int(size[2])), server.workdir, nothing, 0, 240)

# Volume payloads for a LIVE screen: raw column-major Float32 file -> Kit
# converts to classic OpenVDB .vdb via omni.volume's pyopenvdb (the composite
# importer cannot fetch NanoVDB data — see kit_server.py's write_vdb / the
# embedded helper's _omk_h_write_vdb).  Samples are node-centered: N samples
# span [origin, origin+extent], so the voxel size is extent/(N-1).
function _vdb_volume_writer(tr::KitTransport, workdir::AbstractString)
    return function (i, scalars, origin, extent)
        raw = joinpath(workdir, "volume_$(i).f32")
        write(raw, vec(scalars))
        out = joinpath(workdir, "volume_$(i).vdb")
        n = size(scalars)
        vs = ntuple(d -> Float64(extent[d]) / max(n[d] - 1, 1), 3)
        _t_write_vdb(tr; raw, shape = collect(n),
                     origin = Float64.(collect(origin)), voxel_size = collect(vs),
                     out, name = "density")
        return out
    end
end

# Resolve the transport kind: explicit kwarg wins, else OMK_KIT_TRANSPORT env,
# else :subprocess (the proven default + the ovrtx-coexistence path).
function _resolve_transport_kind(transport::Union{Nothing, Symbol, AbstractString})
    raw = transport === nothing ? get(ENV, "OMK_KIT_TRANSPORT", "subprocess") : String(transport)
    kind = Symbol(lowercase(raw))
    kind in (:subprocess, :inprocess) ||
        error("KitScreen: unknown transport $(repr(raw)) (expected :subprocess or :inprocess)")
    return kind
end

function KitScreen(scene::Makie.Scene; server::Union{Nothing, KitServer} = nothing,
                   transport::Union{Nothing, Symbol, AbstractString, KitTransport} = nothing,
                   frames::Integer = 240, kwargs...)
    sz = Base.size(scene)
    if server !== nothing
        tr = SubprocessTransport(server, false)
        owns = false
    elseif transport isa KitTransport
        tr = transport                       # caller-owned (A/B: reuse the one in-process app)
        owns = false
    else
        kind = _resolve_transport_kind(transport)
        if kind === :inprocess
            tr = InProcessTransport(; width = sz[1], height = sz[2], kwargs...)
        else
            tr = SubprocessTransport(start_kit_server(; width = sz[1], height = sz[2], kwargs...), true)
        end
        owns = true
    end
    screen = KitScreen(tr, owns, scene, (Int(sz[1]), Int(sz[2])),
                       mktempdir(; prefix = "omk_screen_", cleanup = false),
                       nothing, 0, Int(frames))
    path = joinpath(screen.workdir, "stage.usda")
    write(path, stage_usda(scene; workdir = screen.workdir,
                           volume_writer = _vdb_volume_writer(tr, screen.workdir)))
    open_stage!(screen, path)
    return screen
end

Base.size(screen::KitScreen) = screen.size
Base.isopen(screen::KitScreen) = _t_isopen(screen.transport)
Base.close(screen::KitScreen) = (screen.owns_transport && _t_close(screen.transport); nothing)

function Base.show(io::IO, screen::KitScreen)
    print(io, "KitScreen($(screen.size[1])×$(screen.size[2]), ",
          isopen(screen) ? "open" : "closed",
          screen.stage_path === nothing ? "" : ", stage=$(screen.stage_path)", ")")
end

"""
    open_stage!(screen::KitScreen, usda_path) -> KitScreen

Open a USD stage in the backend (synchronous; includes post-open settle
frames so the RTX/IndeX pipeline has rebuilt when this returns).
"""
function open_stage!(screen::KitScreen, usda_path::AbstractString; timeout_s::Real = 600)
    path = abspath(usda_path)
    _t_open_stage!(screen.transport, path; timeout_s)
    screen.stage_path = path
    return screen
end

"""
    render!(screen::KitScreen; frames=screen.frames, out=...) -> image matrix

Converge the open stage for `frames` update frames, capture the viewport to
`out` (PNG), and load it (via `Makie.FileIO`).
"""
function render!(screen::KitScreen; frames::Integer = screen.frames,
        out::AbstractString = joinpath(screen.workdir,
                                       "frame_$(screen.framecount + 1).png"),
        timeout_s::Real = 600)
    screen.stage_path === nothing && error("render!: no stage open; call open_stage! first")
    bytes = _t_render(screen.transport; frames, out = abspath(out), timeout_s)
    screen.framecount += 1
    bytes > 0 || error("render: capture reported 0 bytes for $out")
    return Makie.FileIO.load(abspath(out))
end

"""
    render_stage!(screen::KitScreen, usda_path; kwargs...) -> image matrix

`open_stage!` + `render!` in one call — the stage-backed entry point.
"""
function render_stage!(screen::KitScreen, usda_path::AbstractString; kwargs...)
    open_stage!(screen, usda_path)
    return render!(screen; kwargs...)
end

"""
    set_attr!(screen::KitScreen, prim, attr, value; usd_type=nothing) -> Nothing

Typed attribute write on the open stage.  Scalars coerce to the attribute's
existing type; pass `usd_type = "matrix4d"` with 4 rows of 4 numbers (or
`"double3"` with 3 numbers) for the pxr `Gf` types.  Throws if the prim/attr
does not exist or the write fails.
"""
function set_attr!(screen::KitScreen, prim::AbstractString, attr::AbstractString, value;
                   usd_type::Union{Nothing, AbstractString} = nothing)
    _t_set_attr!(screen.transport, prim, attr, value; usd_type)
    return nothing
end

# Push the CURRENT Makie camera to the bound USD camera (matrix4d rows).
function _sync_camera!(screen::KitScreen)
    scene = screen.scene
    scene === nothing && return nothing
    m = permutedims(_camera_to_world(scene))       # row-vector layout
    rows = [[m[i, j] for j in 1:4] for i in 1:4]
    set_attr!(screen, "/World/Camera", "xformOp:transform", rows; usd_type = "matrix4d")
    return nothing
end

"""
    Makie.colorbuffer(screen::KitScreen) -> image matrix

Scene-backed render: re-sync the camera from the Makie scene, converge, and
return the captured frame.
"""
function Makie.colorbuffer(screen::KitScreen)
    _sync_camera!(screen)
    return render!(screen)
end
