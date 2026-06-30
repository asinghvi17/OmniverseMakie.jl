module OmniverseMakieCUDAExt

# M6.A Task 4 — GPU-direct present!: map ovrtx HdrColor as linear float16 CUDA device
# memory (OV.map_cuda, Task 3), tonemap float16→RGBA8 on-device with the SHARED scalar
# `tonemap` (Task 2), and copy straight into the GLMakie image! plot's GL texture via
# CUDA↔GL interop — no CPU roundtrip.  Auto-selected by `_pick_blitter` (GLMakie ext) when
# CUDA is functional; degrades gracefully to the CPU blit on any GPU-setup failure.
#
# CUDA-GL call sequence + handle types REPL-verified on an RTX A5000 (CUDA.jl 6.2.0):
# see references/notes/cuda-gl-interop.md §1-§4 and .superpowers/sdd/task-4-report.md.

using OmniverseMakie, CUDA, GLMakie
using OmniverseMakie: tonemap, OV, RGBA, N0f8
import OmniverseMakie: present!
const CC = CUDA.CUDACore

# Bounded per-step timeout (mirrors the GLMakie ext's _M5_STEP_TIMEOUT_NS): 10 s.
const _M6_STEP_TIMEOUT_NS = UInt64(10_000_000_000)

# ------------------------------------------------------------------
# _cuda_functional — lets the GLMakie ext's _pick_blitter decide :gpu vs :cpu
# ------------------------------------------------------------------
OmniverseMakie._cuda_functional() = CUDA.functional()

# ------------------------------------------------------------------
# GPUBlitState — per-session GPU-direct state (cached on session.gpu_state)
# ------------------------------------------------------------------
"""
    GPUBlitState

Per-session CUDA→GL interop state: the registered GL-texture CUDA resource, the
texture id it was registered for (re-register on resize), a reusable `copy_done`
event (gates ovrtx's `unmap_cuda` so the buffer is not reclaimed mid-copy), and
the GL context (for `gl_switch_context!`).
"""
mutable struct GPUBlitState
    res::Base.RefValue{CC.CUgraphicsResource}   # registered GL texture resource
    tex_id::GLMakie.GLAbstraction.GLuint        # the GL texture id res was registered for
    registered::Bool
    copy_done::CUDA.CuEvent                      # records copy completion → ovrtx unmap gate
    context                                      # GL context for gl_switch_context!
end

# ------------------------------------------------------------------
# Device tonemap kernel — reuses the SHARED scalar `tonemap` (Task 2)
# ------------------------------------------------------------------
# One thread per output pixel (i,j) of the [H,W] display matrix; reads the
# channel-fastest HDR buffer hdr[1:3, j, i] (float16 or float32) and writes
# out[i,j] = tonemap((r,g,b), exposure).  Identical indexing to the host
# `tonemap_frame`, so host and device agree pixel-for-pixel.
function _tonemap_kernel!(out, hdr, exposure::Float32, H::Int, W::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= H * W
        i = (idx - 1) % H + 1
        j = (idx - 1) ÷ H + 1
        @inbounds out[i, j] = tonemap(
            (Float32(hdr[1, j, i]), Float32(hdr[2, j, i]), Float32(hdr[3, j, i])), exposure)
    end
    return nothing
end

# Tonemap an [C,W,H] HDR CuArray → an [H,W] RGBA{N0f8} CuArray (matches tonemap_frame).
function _tonemap_dev(hdr, exposure::Float32)
    C, W, H = size(hdr)
    out = CuArray{RGBA{N0f8}}(undef, H, W)
    n = H * W
    threads = 256
    @cuda threads = threads blocks = cld(n, threads) _tonemap_kernel!(out, hdr, exposure, H, W)
    return out
end

# Device-tonemap + the SAME orientation the CPU path uploads: _orient_for_display =
# reverse(permutedims([H,W]), dims=2) → a [W,H] contiguous RGBA8 CuArray ready for cuMemcpy2D.
function _tonemap_oriented(hdr, exposure::Float32)
    reverse(permutedims(_tonemap_dev(hdr, exposure)), dims = 2)   # [W, H]
end

"""
    tonemap_kernel_to_matrix(hdr::CuArray{<:Real,3}, exposure::Float32) -> Matrix{RGBA{N0f8}}

Test-only helper: device-tonemap an `[C,W,H]` HDR CuArray and copy the result to a host
`[H,W]` matrix (same layout/indexing as the host `tonemap_frame`).  Used by the host-vs-
kernel agreement test to prove the CUDA tonemap kernel matches the host tonemap exactly.
"""
tonemap_kernel_to_matrix(hdr::CuArray{<:Real,3}, exposure::Float32) =
    Array(_tonemap_dev(hdr, exposure))

# ------------------------------------------------------------------
# GL texture access + registration
# ------------------------------------------------------------------
# The image! plot's GL texture: plot2robjs(glscreen, plot) → [RenderObject] (one atomic
# plot) → robj.uniforms[:image]::Texture (RGBA8, 4 bytes/texel — matches our RGBA{N0f8}).
function _image_texture(session)
    robjs = GLMakie.plot2robjs(session.glscreen, session.image_plot)
    robj  = only(robjs)
    return robj.uniforms[:image]::GLMakie.GLAbstraction.Texture
end

function _register!(st::GPUBlitState, tex)
    CC.cuGraphicsGLRegisterImage(st.res, tex.id, tex.texturetype,
                                 CC.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD)
    st.tex_id     = tex.id
    st.registered = true
    return nothing
end

function _unregister!(st::GPUBlitState)
    if st.registered
        try
            CC.cuGraphicsUnregisterResource(st.res[])
        catch e
            @warn "M6: cuGraphicsUnregisterResource failed" exception = e maxlog = 1
        end
        st.registered = false
    end
    return nothing
end

# close(::ViewportSession) calls this (duck-typed via OmniverseMakie._gpu_teardown!).
function OmniverseMakie._gpu_teardown!(st::GPUBlitState)
    try
        st.context === nothing || GLMakie.GLAbstraction.gl_switch_context!(st.context)
    catch
    end
    _unregister!(st)
    return nothing
end

# ------------------------------------------------------------------
# present!(session, ::Val{:gpu}) — the GPU-direct blit
# ------------------------------------------------------------------
"""
    OmniverseMakie.present!(session, ::Val{:gpu}) -> Nothing

GPU-direct blit (M6.A): step ovrtx, map `HdrColor` as linear float16 CUDA device memory,
tonemap float16→RGBA8 on-device (SHARED `tonemap`), and copy straight into the GLMakie
image! plot's GL texture — no CPU roundtrip.

Lazily registers the GL texture on first call (the texture must already be realized — the
GLMakie ext seeds it via a CPU frame before any GPU present!).  On ANY GPU-setup/run
failure, warns once, flips `session.blitter` to `:cpu`, and falls back to the CPU blit for
this and all subsequent frames (the window stays alive).
"""
function OmniverseMakie.present!(session, ::Val{:gpu})
    try
        _gpu_present!(session)
    catch e
        @warn "M6: GPU-direct present! failed — falling back to the CPU blit for this session" exception = (e, catch_backtrace()) maxlog = 1
        session.blitter = :cpu
        OmniverseMakie.present!(session, Val(:cpu))   # still update this frame via CPU
    end
    return nothing
end

function _gpu_present!(session)
    screen = session.screen

    # --- lazy setup / resize re-registration (GL context current first) ---
    tex = _image_texture(session)
    GLMakie.GLAbstraction.gl_switch_context!(tex.context)
    st = session.gpu_state
    if st === nothing
        st = GPUBlitState(Ref{CC.CUgraphicsResource}(), tex.id, false, CUDA.CuEvent(), tex.context)
        _register!(st, tex)
        session.gpu_state = st
    elseif st.tex_id != tex.id            # texture recreated (e.g. resize) — re-register
        _unregister!(st)
        st.context = tex.context
        _register!(st, tex)
    end

    # --- step ovrtx; keep the FINAL StepResult alive to map its HdrColor on-device ---
    for _ in 1:(session.steps_per_tick - 1)
        s0 = OV.step!(screen.renderer, screen.product; timeout_ns = _M6_STEP_TIMEOUT_NS)
        close(s0)
    end
    sr = OV.step!(screen.renderer, screen.product; timeout_ns = _M6_STEP_TIMEOUT_NS)
    try
        data, W, H, C, mh, wait_event = OV.map_cuda(sr, "HdrColor")
        s = Base.unsafe_convert(CC.CUstream, CUDA.stream())

        # Wait for ovrtx's buffer-ready event BEFORE the tonemap kernel reads it (the
        # kernel consumes `data`, so the wait must precede it — refines spike §4 ordering).
        if wait_event != Csize_t(0)
            CC.cuStreamWaitEvent(s, CC.CUevent(UInt(wait_event)), Cuint(0))
        end

        # Wrap the linear float16 CUdeviceptr as a CuArray [C,W,H]; tonemap + orient on-device.
        hdr = unsafe_wrap(CuArray, reinterpret(CUDA.CuPtr{Float16}, data), (C, W, H))
        oriented = _tonemap_oriented(hdr, session.exposure)   # [W,H] RGBA8, contiguous

        GC.@preserve oriented begin
            CC.cuGraphicsMapResources(1, st.res, s)
            dst = Ref{CC.CUarray}()
            CC.cuGraphicsSubResourceGetMappedArray(dst, st.res[], 0, 0)
            cp = Ref(CC.CUDA_MEMCPY2D(
                UInt64(0), UInt64(0), CC.CU_MEMORYTYPE_DEVICE, Ptr{Nothing}(0),
                CC.CUdeviceptr(UInt(pointer(oriented))), CC.CUarray(0), UInt64(W * 4),
                UInt64(0), UInt64(0), CC.CU_MEMORYTYPE_ARRAY, Ptr{Nothing}(0),
                CC.CUdeviceptr(0), dst[], UInt64(0),
                UInt64(W * 4), UInt64(H),
            ))
            CC.cuMemcpy2DAsync_v2(cp, s)
            CC.cuGraphicsUnmapResources(1, st.res, s)
            # Record copy completion; gate ovrtx's unmap on it so it can't reclaim mid-copy.
            evh = Base.unsafe_convert(CC.CUevent, st.copy_done)
            CC.cuEventRecord(evh, s)
            OV.unmap_cuda(sr, mh; stream = Csize_t(UInt(s)), done_event = Csize_t(UInt(evh)))
            CC.cuStreamSynchronize(s)                          # v1 GL sync before GLMakie samples
        end
    finally
        close(sr)
    end
    return nothing
end

end # module OmniverseMakieCUDAExt
