module OmniverseKitMakieCUDAExt

# CUDA side of the Kit GPU data plane (spec:
# docs/superpowers/specs/2026-07-16-kit-gpu-data-plane-design.md).
#
# The Kit server owns every IPC buffer (raw cudaMalloc, exported via
# cudaIpcGetMemHandle); this extension opens the handles once per screen and
# wraps them as CuArrays:
#   frames OUT : render!(screen; device = :cuda) -> CuMatrix{RGBA{N0f8}}
#   volumes IN : gpu_update_volume!(screen, plot; data::CuArray{Float32,3})
# Installed through the plain-function Refs in gpu_plane.jl (no piracy, no
# method overwrites); the core package keeps working without CUDA loaded.

import OmniverseKitMakie as OMK
import CUDA
using CUDA: CuArray, CuPtr, CuVector
using CUDA.CUDACore: cuIpcOpenMemHandle_v2, CUipcMemHandle, CUdeviceptr,
    CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
using Base64: base64decode
using FixedPointNumbers: N0f8
using ColorTypes: RGBA

# Open a server-exported IPC handle (base64 of the 64-byte CUipcMemHandle).
# Legacy IPC handles only work across processes on the SAME device — the
# caller checks the server's device id against ours first.
function _open_ipc(handle_b64::AbstractString)
    bytes = base64decode(String(handle_b64))
    length(bytes) == 64 || error("IPC handle must be 64 bytes, got $(length(bytes))")
    handle = CUipcMemHandle(ntuple(i -> bytes[i] % Int8, 64))
    CUDA.context()  # ensure a current CUDA context on this task
    dptr = Ref{CUdeviceptr}(0)
    cuIpcOpenMemHandle_v2(dptr, handle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS)
    return dptr[]
end

function _assert_same_device(server_device)
    ours = CUDA.deviceid(CUDA.device())
    Int(server_device) == ours ||
        error("CUDA IPC needs the same GPU on both sides: server device " *
              "$(server_device), Julia device $(ours)")
end

# ---------------------------------------------------------------------------
# Frames out: render!(screen; device = :cuda)
# ---------------------------------------------------------------------------
function _cuda_render(screen::OMK.KitScreen; frames::Integer = screen.frames,
                      timeout_s::Real = 600)
    srv = screen.server
    get(srv.caps, :cuda_out, false) ||
        error("server lacks the CUDA out-plane (caps = $(srv.caps)); " *
              "is omni.syntheticdata enabled/resolvable?")
    w, h = screen.size
    fr = get(screen.gpu, :frame, nothing)
    if fr === nothing || fr.width != w || fr.height != h
        rsp = OMK._check(OMK.rpc(srv, "gpu_frame_setup"; width = w, height = h),
                         "gpu_frame_setup")
        _assert_same_device(rsp.device)
        ptr = _open_ipc(rsp.handle_b64)
        arr = unsafe_wrap(CuArray, CuPtr{UInt8}(UInt(ptr)), Int(rsp.nbytes))
        fr = (; arr, width = Int(rsp.width), height = Int(rsp.height))
        screen.gpu[:frame] = fr
    end
    rsp = OMK._check(OMK.rpc(srv, "render"; frames, device = "cuda", timeout_s),
                     "render(cuda)")
    screen.framecount += 1
    (Int(rsp.width) == fr.width && Int(rsp.height) == fr.height) ||
        error("render(cuda): frame $(rsp.width)×$(rsp.height) != buffer " *
              "$(fr.width)×$(fr.height) (resolution changed? recreate the screen)")
    # Tight RGBA8 rows, top-down (row-major W×H) -> (H, W) image orientation.
    # permutedims materializes a NEW CuArray, detaching the result from the
    # shared buffer (which the next render overwrites).
    px = reinterpret(RGBA{N0f8}, fr.arr)
    return permutedims(reshape(px, (fr.width, fr.height)), (2, 1))
end

# ---------------------------------------------------------------------------
# Volumes in: gpu_update_volume!(screen, plot; data)
# ---------------------------------------------------------------------------
function _gpu_update_volume!(screen::OMK.KitScreen, plot;
                             data::CuArray{Float32, 3})
    srv = screen.server
    get(srv.caps, :cuda_ipc, false) ||
        error("server lacks CUDA IPC (caps = $(srv.caps))")
    rec = OMK._volume_record(screen, plot)
    size(data) == rec.shape ||
        error("data is $(size(data)), the authored volume grid is $(rec.shape) " *
              "(the payload size is frozen at author time)")
    key = (:volume, rec.shape)
    staging = get(screen.gpu, key, nothing)
    if staging === nothing
        rsp = OMK._check(OMK.rpc(srv, "gpu_volume_setup"; shape = collect(rec.shape)),
                         "gpu_volume_setup")
        _assert_same_device(rsp.device)
        ptr = _open_ipc(rsp.handle_b64)
        staging = unsafe_wrap(CuArray, CuPtr{Float32}(UInt(ptr)), prod(rec.shape))
        screen.gpu[key] = staging
    end
    copyto!(staging, vec(data))          # device -> device
    CUDA.synchronize()                   # staged before the server reads it
    seq = get(screen.gpu, :volume_seq, 0) + 1
    screen.gpu[:volume_seq] = seq
    out = joinpath(screen.workdir, "volume_live_$(seq).vdb")
    vs = ntuple(d -> Float64(rec.extent[d]) / max(rec.shape[d] - 1, 1), 3)
    OMK._check(OMK.rpc(srv, "gpu_write_vdb"; shape = collect(rec.shape),
                       voxel_size = collect(vs),
                       origin = Float64.(collect(rec.origin)), out,
                       name = "density"),
               "gpu_write_vdb")
    # Fresh-file swap (asset inputs are not hot-reloaded in place — the same
    # constraint as ovrtx textures); the previous live file stays on disk
    # until the screen's workdir is cleaned.
    OMK.set_attr!(screen, rec.prim * "/density", "filePath", out;
                  usd_type = "asset")
    return nothing
end

function __init__()
    OMK._CUDA_RENDER[] = _cuda_render
    OMK._CUDA_UPDATE_VOLUME[] = _gpu_update_volume!
    return nothing
end

end # module OmniverseKitMakieCUDAExt
