# GPU data plane, Julia core (spec:
# docs/superpowers/specs/2026-07-16-kit-gpu-data-plane-design.md).
#
# Out-plane devices for `render!`:
#   :png  — the original file round-trip (compatibility floor).
#   :cpu  — zero-disk fast path: the server memmoves the RGBA8 capture into
#           POSIX shared memory; Julia mmaps it.  No PNG encode, no disk.
#   :cuda — server D2D-copies the LdrColor render var into a cudaMalloc'd
#           buffer exported over CUDA IPC; Julia wraps it as a CuArray.
#           Implemented by OmniverseKitMakieCUDAExt (requires CUDA.jl loaded).
#   :auto — best CPU-returning path (:cpu when the server has shm_out, else
#           :png); `Makie.colorbuffer` rides this, keeping its CPU contract.
#
# In-plane: `gpu_update_volume!` (ext) stages a sim CuArray into a server-
# owned IPC buffer, has the server write a FRESH .vdb (IndeX's importer is
# file-based — proven), and swaps the volume prim's filePath.

using FixedPointNumbers: N0f8
using ColorTypes: RGBA
import Mmap

"""
    gpu_caps(screen::KitScreen) -> Dict{Symbol, Any}

The server's GPU-plane capabilities from its ready message:
`:shm_out` (zero-disk CPU capture), `:cuda_ipc` (IPC buffers available),
`:cuda_out` (device-resident frames), `:cuda_device`.
"""
gpu_caps(screen::KitScreen) = screen.server.caps

# Resolve :auto against server capabilities (CPU-returning paths only —
# :cuda is always explicit; a colorbuffer that sometimes returns a CuArray
# would break Makie's contract).
_resolve_device(caps::AbstractDict, device::Symbol) =
    device === :auto ? (get(caps, :shm_out, false) ? :cpu : :png) :
    device === :shm ? :cpu : device
_resolve_device(screen::KitScreen, device::Symbol) =
    _resolve_device(screen.server.caps, device)

# mmap the server's shm segment and convert the tightly-packed RGBA8 rows
# (top-down, row-major — TextureFormat.RGBA8_UNORM, probed 2026-07-16) into
# the (height, width) image matrix orientation the PNG path returns.
function _shm_frame(rsp::NamedTuple)
    width = Int(rsp.width); height = Int(rsp.height)
    nbytes = Int(rsp.nbytes)
    nbytes == width * height * 4 ||
        error("shm frame: nbytes=$(nbytes) != $(width)×$(height)×4 " *
              "(format=$(get(rsp, :format, "?")))")
    raw = Mmap.mmap(String(rsp.shm_path), Vector{UInt8}, nbytes)
    px = reinterpret(RGBA{N0f8}, raw)                # row-major W×H, top-down
    img = permutedims(reshape(px, (width, height)), (2, 1))
    return collect(img)   # detach from the mmap (server reuses the segment)
end

# :cuda hooks, installed by OmniverseKitMakieCUDAExt at load time.  Plain
# Refs (not method overloads) so the ext neither pirates nor redefines.
const _CUDA_RENDER = Ref{Any}(nothing)
const _CUDA_UPDATE_VOLUME = Ref{Any}(nothing)

_cuda_render(screen; kwargs...) =
    _CUDA_RENDER[] === nothing ?
    error("render!(…; device = :cuda) needs CUDA.jl loaded (OmniverseKitMakieCUDAExt)") :
    _CUDA_RENDER[](screen; kwargs...)

"""
    gpu_update_volume!(screen::KitScreen, plot; data, colorrange = nothing) -> Nothing

Live volume update from a `CuArray{Float32,3}` with no Julia-side host copy:
device→device into a server-owned CUDA-IPC staging buffer, server-side
device→host + fresh `.vdb` write (pyopenvdb), and a `filePath` swap on the
plot's Volume prim.  `data` must match the authored volume's grid size.
Requires CUDA.jl loaded (OmniverseKitMakieCUDAExt); the server must report
`cuda_ipc` in [`gpu_caps`](@ref).  The colormap/colorrange stay as authored
(the transfer function lives in the stage, not the payload).
"""
gpu_update_volume!(screen::KitScreen, plot; kwargs...) =
    _CUDA_UPDATE_VOLUME[] === nothing ?
    error("gpu_update_volume! needs CUDA.jl loaded (OmniverseKitMakieCUDAExt)") :
    _CUDA_UPDATE_VOLUME[](screen, plot; kwargs...)

# Find the authored-volume record for `plot` (populated by KitScreen(scene)).
function _volume_record(screen::KitScreen, plot)
    for rec in screen.volumes
        rec.plot === plot && return rec
    end
    error("plot has no authored Volume prim on this KitScreen " *
          "($(length(screen.volumes)) volume(s) tracked)")
end
