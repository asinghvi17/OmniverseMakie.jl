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
event (gates ovrtx's `unmap_cuda` so the buffer is not reclaimed mid-copy), the GL
context (for `gl_switch_context!`), the cached `[W,H]` fused-tonemap+orient device
buffer (`oriented`, reused every frame — allocated lazily / on size change, freed on
resize + teardown), and the cached resolved GL `Texture` (`tex`, invalidated in
`_unregister!` so the `tex_id != tex.id` re-register guard keeps working).
"""
mutable struct GPUBlitState
    res::Base.RefValue{CC.CUgraphicsResource}   # registered GL texture resource
    tex_id::GLMakie.GLAbstraction.GLuint        # the GL texture id res was registered for
    registered::Bool
    copy_done::CUDA.CuEvent                      # records copy completion → ovrtx unmap gate
    context                                      # GL context for gl_switch_context!
    oriented                                     # cached [W,H] fused tonemap+orient CuArray{RGBA{N0f8}} (lazy) or nothing
    tex                                          # cached resolved GL Texture (invalidated in _unregister!) or nothing
end

# ------------------------------------------------------------------
# Device tonemap kernel — reuses the SHARED scalar `tonemap` (Task 2)
# ------------------------------------------------------------------
# FUSED tonemap + display-orient kernel (C3): one thread per HDR pixel (i in 1:H, j in 1:W)
# writes the y-flipped/transposed output out[j, H+1-i] straight into a [W,H] RGBA8 buffer —
# ONE kernel replacing _tonemap_dev + permutedims + reverse (3 kernels / 3 device allocations).
# Device twin of the host `_tonemap_orient!` (GLMakie ext), same out[j, H+1-i] indexing, so the
# two agree pixel-for-pixel.  The @inbounds output write is guarded by an EXPLICIT size(out)
# check (C2 review: do NOT derive the output bounds from `hdr` alone).
function _tonemap_orient_kernel!(out, hdr, scale::Float32, H::Int, W::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= H * W
        i  = (idx - 1) % H + 1        # 1:H  (hdr fastest spatial index)
        j  = (idx - 1) ÷ H + 1        # 1:W
        oj = H + 1 - i                # 1:H  (y-flip → output column)
        if j <= size(out, 1) && oj <= size(out, 2)
            @inbounds out[j, oj] = tonemap(
                (Float32(hdr[1, j, i]), Float32(hdr[2, j, i]), Float32(hdr[3, j, i])), scale)
        end
    end
    return nothing
end

# Launch the fused oriented kernel into a caller-owned [W,H] buffer `out` (no allocation here).
function _tonemap_orient_dev!(out, hdr, exposure::Float32)
    C, W, H = size(hdr)
    n = H * W
    threads = 256
    scale = exp2(exposure)                          # once per launch (host-side), not per thread
    @cuda threads = threads blocks = cld(n, threads) _tonemap_orient_kernel!(out, hdr, scale, H, W)
    return out
end

# Free the cached oriented device buffer (deterministic GPU-memory release on resize / teardown;
# the CuArray finalizer would eventually reclaim it — this frees eagerly).  Safe: every present!
# ends with cuStreamSynchronize, so no copy still reads the buffer.  No-op if unset.
function _free_oriented!(st::GPUBlitState)
    st.oriented === nothing && return nothing
    CUDA.unsafe_free!(st.oriented)
    st.oriented = nothing
    return nothing
end

# Fused tonemap+orient into the state's cached [W,H] buffer: lazily allocate (or free+realloc on
# size change), then launch — so a steady frame does ZERO device allocations.  Replaces the old
# _tonemap_oriented (_tonemap_dev + permutedims + reverse = 3 kernels / 3 device allocations).
function _tonemap_orient_cached!(st::GPUBlitState, hdr, exposure::Float32)
    C, W, H = size(hdr)
    out = st.oriented
    if out === nothing || size(out) != (W, H)
        _free_oriented!(st)                         # drop any wrong-size buffer (no-op if nothing)
        out = CuArray{RGBA{N0f8}}(undef, W, H)
        st.oriented = out
    end
    return _tonemap_orient_dev!(out, hdr, exposure)
end

"""
    tonemap_oriented_kernel_to_matrix(hdr::CuArray{<:Real,3}, exposure::Float32) -> Matrix{RGBA{N0f8}}

Test-only: run the FUSED oriented device kernel over an `[C,W,H]` HDR CuArray into a fresh
`[W,H]` buffer and copy to host — proving it byte-equals host
`reverse(permutedims(tonemap_frame(hdr, exposure)), dims=2)` (== the `_tonemap_orient!` twin).
"""
function tonemap_oriented_kernel_to_matrix(hdr::CuArray{<:Real,3}, exposure::Float32)
    C, W, H = size(hdr)
    out = CuArray{RGBA{N0f8}}(undef, W, H)
    _tonemap_orient_dev!(out, hdr, exposure)
    return Array(out)
end

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
    st.tex        = tex                              # cache the resolved Texture (invalidated in _unregister!)
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
    st.tex = nothing                                 # invalidate the cached Texture (resize recreates it)
    return nothing
end

# close(::ViewportSession) calls this (duck-typed via _gpu_teardown!).
function OmniverseMakie._gpu_teardown!(st::GPUBlitState)
    try
        st.context === nothing || GLMakie.GLAbstraction.gl_switch_context!(st.context)
    catch
    end
    _unregister!(st)
    _free_oriented!(st)                              # release the cached oriented device buffer
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
    _free_oriented!(st)                              # release the cached oriented device buffer (resize)
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
    # Resolve the image! plot's GL texture ONCE and cache it on the state (st.tex); steady
    # state reuses the cached Texture — no per-frame plot2robjs + uniforms Dict lookup.  The
    # cache is invalidated in _unregister! (resize / teardown), so the next present re-resolves
    # the freshly-created texture and the guard below re-registers it.
    st  = session.gpu_state
    tex = (st !== nothing && st.tex !== nothing) ? st.tex : _image_texture(session)
    GLMakie.GLAbstraction.gl_switch_context!(tex.context)
    if st === nothing
        st = GPUBlitState(Ref{CC.CUgraphicsResource}(), tex.id, false, CUDA.CuEvent(),
                          tex.context, nothing, nothing)
        _register!(st, tex)
        session.gpu_state = st
    elseif !st.registered || st.tex_id != tex.id   # explicit unregister (resize) OR recreated/recycled-id texture — re-register
        _unregister!(st)                            # no-op if already unregistered (registered=false); invalidates st.tex
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
        oriented = _tonemap_orient_cached!(st, hdr, session.exposure)  # [W,H] RGBA8, cached — one kernel

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
