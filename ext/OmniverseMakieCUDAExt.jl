module OmniverseMakieCUDAExt

# M6.A Task 4 — GPU-direct present!: map ovrtx HdrColor as LINEAR float16 CUDA
# device memory (OV.map_cuda), tonemap float16→RGBA8 on-device with the SHARED
# scalar `tonemap`, and copy straight into the GLMakie image! plot's GL texture
# via CUDA↔GL interop — no CPU roundtrip.  Auto-selected by `_pick_blitter` when
# CUDA is functional; degrades gracefully to the CPU blit on GPU-setup failure.
#
# CUDA-GL call sequence + handle types REPL-verified on an RTX A5000 (CUDA.jl
# 6.2.0): see references/notes/cuda-gl-interop.md §1-§4 and
# .superpowers/sdd/task-4-report.md.

using OmniverseMakie, CUDA, GLMakie
using OmniverseMakie: tonemap, OV, RGBA, N0f8
import OmniverseMakie: present!
import CUDA.CUDACore as CC

# Bounded per-step timeout (mirrors GLMakie ext's _M5_STEP_TIMEOUT_NS): 10 s.
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
# One thread per output pixel (i,j) of the [H,W] matrix; reads channel-fastest
# hdr[1:3, j, i] (float16/float32) → out[i,j] = tonemap((r,g,b), exposure).
# Same indexing as host `tonemap_frame`, so host and device agree pixelwise.
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

# [C,W,H] HDR CuArray → [H,W] RGBA{N0f8} CuArray (matches tonemap_frame).
function _tonemap_dev(hdr, exposure::Float32)
    C, W, H = size(hdr)
    out = CuArray{RGBA{N0f8}}(undef, H, W)
    n = H * W
    threads = 256
    @cuda threads = threads blocks = cld(n, threads) _tonemap_kernel!(out, hdr, exposure, H, W)
    return out
end

# Device-tonemap + the SAME CPU-path orientation (_orient_for_display =
# reverse(permutedims([H,W]), dims=2)) → [W,H] contiguous RGBA8, for cuMemcpy2D.
function _tonemap_oriented(hdr, exposure::Float32)
    reverse(permutedims(_tonemap_dev(hdr, exposure)), dims = 2)   # [W, H]
end

"""
    tonemap_kernel_to_matrix(hdr::CuArray{<:Real,3}, exposure::Float32) -> Matrix{RGBA{N0f8}}

Test-only: device-tonemap an `[C,W,H]` HDR CuArray to a host `[H,W]` matrix (same
layout/indexing as `tonemap_frame`), proving the CUDA kernel matches the host tonemap.
"""
tonemap_kernel_to_matrix(hdr::CuArray{<:Real,3}, exposure::Float32) =
    Array(_tonemap_dev(hdr, exposure))

# ------------------------------------------------------------------
# GL texture access + registration
# ------------------------------------------------------------------
# The image! plot's GL texture: plot2robjs(glscreen, plot) → [RenderObject] (one
# atomic plot) → robj.uniforms[:image]::Texture (RGBA8, 4 bytes/texel = our
# RGBA{N0f8}).
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

# close(::ViewportSession) calls this (duck-typed via _gpu_teardown!).
function OmniverseMakie._gpu_teardown!(st::GPUBlitState)
    try
        st.context === nothing || GLMakie.GLAbstraction.gl_switch_context!(st.context)
    catch
    end
    _unregister!(st)
    return nothing
end

# M6.A Task 5: resize re-registration hook (duck-typed via gpu_unregister!).
# resize_viewport! recreates the image! plot, and GL may RECYCLE the freed
# texture id — so `tex_id != tex.id` alone can miss a recreated same-id texture
# (A-2).  Unregister the OLD resource while its texture is still alive and clear
# `registered` (keep the GPUBlitState), so the next GPU present! re-registers
# cleanly (its guard tests `!st.registered`).  No-op when the session never used
# the GPU path (gpu_state === nothing).
#
# Threading: this fires from the window_area listener on GLMakie's RENDER task —
# the SAME task that registered it, so cuGraphicsUnregisterResource runs on the
# registering task (no cross-task hazard).  gl_switch_context! makes the GL
# context current first (redundant-but-safe on the render task, REQUIRED when
# driven off it, e.g. the benchmark).
function OmniverseMakie.gpu_unregister!(session)
    st = session.gpu_state
    st === nothing && return nothing
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

GPU-direct blit (M6.A): step ovrtx, map `HdrColor` as linear float16 CUDA memory, tonemap
float16→RGBA8 on-device (SHARED `tonemap`), copy straight into the image! plot's GL texture
— no CPU roundtrip.  Lazily registers the GL texture on first call (it must already be
realized — the GLMakie ext seeds it via a CPU frame first).  On ANY failure: warn once, flip
`session.blitter` to `:cpu`, fall back to the CPU blit for this and all later frames.
"""
function OmniverseMakie.present!(session, ::Val{:gpu})
    try
        _gpu_present!(session)
    catch e
        # Forced GPU (gpu_direct=true): rethrow to surface the real GPU bug
        # (on_render_tick!'s try/catch logs it, window stays alive) instead
        # of silently switching to CPU.
        # Auto/false: degrade gracefully to the CPU blit.
        if session.gpu_forced
            rethrow()
        else
            @warn "M6: GPU-direct present! failed — falling back to the CPU blit for this session" exception = (e, catch_backtrace()) maxlog = 1
            session.blitter = :cpu
            OmniverseMakie.present!(session, Val(:cpu))   # still update this frame via CPU
        end
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
    elseif !st.registered || st.tex_id != tex.id   # explicit unregister (resize) OR recreated/recycled-id texture — re-register
        _unregister!(st)                            # no-op if already unregistered (registered=false)
        st.context = tex.context
        _register!(st, tex)
    end

    # --- step ovrtx; keep FINAL StepResult to map its HdrColor on-device ---
    for _ in 1:(session.steps_per_tick - 1)
        sr_drop = OV.step!(screen.renderer, screen.product; timeout_ns = _M6_STEP_TIMEOUT_NS)
        close(sr_drop)
    end
    sr = OV.step!(screen.renderer, screen.product; timeout_ns = _M6_STEP_TIMEOUT_NS)
    try
        data, W, H, C, mh, wait_event = OV.map_cuda(sr, "HdrColor")
        stream = Base.unsafe_convert(CC.CUstream, CUDA.stream())

        # Wait ovrtx's buffer-ready event BEFORE the tonemap kernel reads `data`
        # (kernel consumes it, so the wait must precede — spike §4 ordering).
        if wait_event != Csize_t(0)
            CC.cuStreamWaitEvent(stream, CC.CUevent(UInt(wait_event)), Cuint(0))
        end

        # Wrap the linear float16 CUdeviceptr as CuArray [C,W,H]; tonemap+orient
        # on-device.
        hdr = unsafe_wrap(CuArray, reinterpret(CUDA.CuPtr{Float16}, data), (C, W, H))
        oriented = _tonemap_oriented(hdr, session.exposure)   # [W,H] RGBA8, contiguous

        GC.@preserve oriented begin
            CC.cuGraphicsMapResources(1, st.res, stream)
            dst = Ref{CC.CUarray}()
            CC.cuGraphicsSubResourceGetMappedArray(dst, st.res[], 0, 0)
            cp = Ref(CC.CUDA_MEMCPY2D(
                UInt64(0), UInt64(0), CC.CU_MEMORYTYPE_DEVICE, Ptr{Nothing}(0),
                CC.CUdeviceptr(UInt(pointer(oriented))), CC.CUarray(0), UInt64(W * 4),
                UInt64(0), UInt64(0), CC.CU_MEMORYTYPE_ARRAY, Ptr{Nothing}(0),
                CC.CUdeviceptr(0), dst[], UInt64(0),
                UInt64(W * 4), UInt64(H),
            ))
            CC.cuMemcpy2DAsync_v2(cp, stream)
            CC.cuGraphicsUnmapResources(1, st.res, stream)
            # Record copy done; gate ovrtx unmap on it (can't reclaim mid-copy).
            evh = Base.unsafe_convert(CC.CUevent, st.copy_done)
            CC.cuEventRecord(evh, stream)
            OV.unmap_cuda(sr, mh; stream = Csize_t(UInt(stream)), done_event = Csize_t(UInt(evh)))
            CC.cuStreamSynchronize(stream)                     # v1 GL sync before GLMakie samples
        end
    finally
        close(sr)
    end
    return nothing
end

end # module OmniverseMakieCUDAExt
