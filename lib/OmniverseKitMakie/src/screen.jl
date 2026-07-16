# KitScreen — a Makie screen whose backend is a persistent Kit render
# server, giving full-color NVIDIA IndeX volume rendering (transfer-function
# colormaps) that the ovrtx-FFI `OmniverseMakie.Screen` cannot do.
#
# Two modes:
#   * scene-backed (`KitScreen(scene)`): the scene's volume plots, lights and
#     camera are authored to a stage (authoring.jl) and opened; every
#     `Makie.colorbuffer(screen)` re-syncs the camera and renders.
#   * stage-backed (`KitScreen(server)`): render externally-authored stages
#     via `render_stage!` — the original spike surface, kept for probing.

"""
    KitScreen(scene::Makie.Scene; server=nothing, frames=240, kwargs...) -> KitScreen
    KitScreen(server::KitServer; size=(1280, 720)) -> KitScreen

Scene-backed: author `scene` (volume plots + lights + camera; see
[`stage_usda`](@ref)) and open it on `server` (started on demand with
`size(scene)` as the resolution when not supplied; extra `kwargs` go to
[`start_kit_server`](@ref)).  `Makie.colorbuffer(screen)` syncs the camera
and renders `frames` convergence frames.  `close(screen)` shuts the server
down only if the screen started it.

Stage-backed: no scene; drive with [`render_stage!`](@ref) / [`set_attr!`](@ref).
"""
mutable struct KitScreen <: Makie.MakieScreen
    server::KitServer
    owns_server::Bool
    scene::Union{Nothing, Makie.Scene}
    size::Tuple{Int, Int}
    workdir::String
    stage_path::Union{Nothing, String}  # currently open stage
    framecount::Int                     # for default output naming
    frames::Int                         # convergence frames per colorbuffer
    # authored volume prims (plot, prim, shape, origin, extent) — the
    # in-plane's plot→prim map (gpu_plane.jl / gpu_update_volume!)
    volumes::Vector{NamedTuple}
    gpu::Dict{Any, Any}                 # ext-owned GPU-plane state (IPC wraps)
end

KitScreen(server::KitServer; size::Tuple{<:Integer, <:Integer} = (1280, 720)) =
    KitScreen(server, false, nothing, (Int(size[1]), Int(size[2])),
              server.workdir, nothing, 0, 240, NamedTuple[], Dict{Any, Any}())

# Volume payloads for a LIVE screen: raw column-major Float32 file -> server
# converts to classic OpenVDB .vdb via omni.volume's pyopenvdb (the composite
# importer cannot fetch NanoVDB data — see kit_server.py's write_vdb).
# Samples are node-centered: N samples span [origin, origin+extent], so the
# voxel size is extent/(N-1).
function _vdb_volume_writer(srv::KitServer, workdir::AbstractString)
    return function (i, scalars, origin, extent)
        raw = joinpath(workdir, "volume_$(i).f32")
        write(raw, vec(scalars))
        out = joinpath(workdir, "volume_$(i).vdb")
        n = size(scalars)
        vs = ntuple(d -> Float64(extent[d]) / max(n[d] - 1, 1), 3)
        _check(rpc(srv, "write_vdb"; raw, shape = collect(n),
                   origin = Float64.(collect(origin)), voxel_size = collect(vs),
                   out, name = "density"),
               "write_vdb(volume_$(i))")
        return out
    end
end

function KitScreen(scene::Makie.Scene; server::Union{Nothing, KitServer} = nothing,
                   frames::Integer = 240, kwargs...)
    sz = Base.size(scene)
    owns = server === nothing
    srv = owns ? start_kit_server(; width = sz[1], height = sz[2], kwargs...) : server
    screen = KitScreen(srv, owns, scene, (Int(sz[1]), Int(sz[2])),
                       mktempdir(; prefix = "omk_screen_", cleanup = false),
                       nothing, 0, Int(frames), NamedTuple[], Dict{Any, Any}())
    # Track plot→prim for the in-plane: plots in _collect_plots! order match
    # the writer's index (and the authored "/World/Volume\$i" prim names).
    vols = Makie.Volume[]
    _collect_plots!(vols, Symbol[], scene.plots)
    inner_writer = _vdb_volume_writer(srv, screen.workdir)
    function recording_writer(i, scalars, origin, extent)
        vol_path = inner_writer(i, scalars, origin, extent)
        push!(screen.volumes, (; plot = vols[i], prim = "/World/Volume$(i)",
                                 shape = size(scalars), origin, extent))
        return vol_path
    end
    path = joinpath(screen.workdir, "stage.usda")
    write(path, stage_usda(scene; workdir = screen.workdir,
                           volume_writer = recording_writer))
    open_stage!(screen, path)
    return screen
end

Base.size(screen::KitScreen) = screen.size
Base.isopen(screen::KitScreen) = isopen(screen.server)
Base.close(screen::KitScreen) = (screen.owns_server && close(screen.server); nothing)

function Base.show(io::IO, screen::KitScreen)
    print(io, "KitScreen($(screen.size[1])×$(screen.size[2]), ",
          isopen(screen) ? "open" : "closed",
          screen.stage_path === nothing ? "" : ", stage=$(screen.stage_path)", ")")
end

"""
    open_stage!(screen::KitScreen, usda_path) -> KitScreen

Open a USD stage in the server (synchronous; includes post-open settle
frames so the RTX/IndeX pipeline has rebuilt when this returns).
"""
function open_stage!(screen::KitScreen, usda_path::AbstractString; timeout_s::Real = 600)
    path = abspath(usda_path)
    _check(rpc(screen.server, "open_stage"; path, timeout_s), "open_stage($path)")
    screen.stage_path = path
    return screen
end

"""
    render!(screen::KitScreen; frames=screen.frames, device=:auto, out=...) -> image

Converge the open stage for `frames` update frames and capture the viewport.
`device` selects the out-plane (see gpu_plane.jl): `:auto` (best CPU path —
shared memory when the server supports it, else PNG), `:cpu`/`:shm`
(zero-disk shared memory), `:png` (file round-trip via `out`), or `:cuda`
(device-resident `CuArray`; requires CUDA.jl loaded).
"""
function render!(screen::KitScreen; frames::Integer = screen.frames,
        device::Symbol = :auto,
        out::AbstractString = joinpath(screen.workdir,
                                       "frame_$(screen.framecount + 1).png"),
        timeout_s::Real = 600)
    screen.stage_path === nothing && error("render!: no stage open; call open_stage! first")
    dev = _resolve_device(screen, device)
    if dev === :png
        rsp = _check(rpc(screen.server, "render"; frames, device = "png",
                         out = abspath(out), timeout_s), "render(png)")
        screen.framecount += 1
        rsp.bytes > 0 || error("render: capture reported 0 bytes for $out")
        return Makie.FileIO.load(abspath(out))
    elseif dev === :cpu
        rsp = _check(rpc(screen.server, "render"; frames, device = "shm", timeout_s),
                     "render(shm)")
        screen.framecount += 1
        return _shm_frame(rsp)
    elseif dev === :cuda
        return _cuda_render(screen; frames, timeout_s)
    end
    error("render!: unknown device $(repr(device))")
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
    set_attr!(screen::KitScreen, prim, attr, value; usd_type=nothing) -> NamedTuple

Typed attribute write on the open stage.  Scalars coerce to the attribute's
existing type; pass `usd_type = "matrix4d"` with 4 rows of 4 numbers (or
`"double3"` with 3 numbers) for the pxr `Gf` types.  Throws if the prim/attr
does not exist or the write fails.
"""
function set_attr!(screen::KitScreen, prim::AbstractString, attr::AbstractString, value;
                   usd_type::Union{Nothing, AbstractString} = nothing)
    kwargs = usd_type === nothing ? (;) : (; usd_type = String(usd_type))
    return _check(rpc(screen.server, "set_attr";
                      prim = String(prim), attr = String(attr), value, kwargs...),
                  "set_attr($prim.$attr)")
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
