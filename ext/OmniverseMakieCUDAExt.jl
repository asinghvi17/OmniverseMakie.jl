module OmniverseMakieCUDAExt

# GPU-direct present!: map ovrtx HdrColor as linear float16 CUDA device
# memory (OV.map_cuda), tonemap float16→RGBA8 on-device with the shared
# scalar `tonemap`, and copy straight into the GLMakie image! plot's GL
# texture via CUDA↔GL interop — no CPU roundtrip.  Auto-selected by
# `_pick_blitter` when CUDA is functional; degrades to CPU on setup failure.
# CUDA↔GL call sequence: see references/notes/cuda-gl-interop.md.

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

Per-session CUDA→GL interop state, cached on `session.gpu_state`; see the
field comments.
"""
mutable struct GPUBlitState
    res::Base.RefValue{CC.CUgraphicsResource} # registered GL-texture resource
    tex_id::GLMakie.GLAbstraction.GLuint      # texture id `res` registered for
    registered::Bool
    copy_done::CUDA.CuEvent  # reusable event; gates ovrtx unmap on copy done
    context                  # GL context for gl_switch_context!
    oriented                 # cached [W,H] tonemap+orient CuArray (lazy)
    tex                      # resolved GL Texture; invalidated in _unregister!
end

# ------------------------------------------------------------------
# Device tonemap kernel — reuses the shared scalar `tonemap`
# ------------------------------------------------------------------
# Fused tonemap + display-orient kernel: one thread per HDR pixel (i in 1:H,
# j in 1:W) writes the y-flipped/transposed output out[j, H+1-i] straight
# into a [W,H] RGBA8 buffer.  Device twin of the host `_tonemap_orient!`
# (GLMakie ext), same indexing, so the two agree pixel-for-pixel.  The
# @inbounds write is guarded by an explicit size(out) check — never derive
# the output bounds from `hdr` alone.
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

# Launch the fused kernel into a caller-owned [W,H] buffer (no allocation).
function _tonemap_orient_dev!(out, hdr, exposure::Float32)
    C, W, H = size(hdr)
    n = H * W
    threads = 256
    scale = exp2(exposure)  # once per launch (host-side), not per thread
    @cuda threads = threads blocks = cld(n, threads) _tonemap_orient_kernel!(out, hdr, scale, H, W)
    return out
end

# Free the cached oriented device buffer eagerly (resize / teardown) instead
# of waiting on the CuArray finalizer.  Safe: every present! ends with
# cuStreamSynchronize, so no copy still reads it.  No-op if unset.
function _free_oriented!(st::GPUBlitState)
    st.oriented === nothing && return nothing
    CUDA.unsafe_free!(st.oriented)
    st.oriented = nothing
    return nothing
end

# Fused tonemap+orient into the state's cached [W,H] buffer: lazily allocate
# (or free+realloc on size change), then launch — a steady frame does zero
# device allocations.
function _tonemap_orient_cached!(st::GPUBlitState, hdr, exposure::Float32)
    C, W, H = size(hdr)
    out = st.oriented
    if out === nothing || size(out) != (W, H)
        _free_oriented!(st)  # drop any wrong-size buffer (no-op if nothing)
        out = CuArray{RGBA{N0f8}}(undef, W, H)
        st.oriented = out
    end
    return _tonemap_orient_dev!(out, hdr, exposure)
end

"""
    tonemap_oriented_kernel_to_matrix(hdr::CuArray{<:Real,3},
                                      exposure::Float32) -> Matrix{RGBA{N0f8}}

Test-only: run the fused oriented device kernel over a `[C,W,H]` HDR CuArray
into a fresh `[W,H]` buffer and copy to host — for byte-equality checks
against `reverse(permutedims(tonemap_frame(hdr, exposure)), dims=2)` (the
`_tonemap_orient!` twin).
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
# The image! plot's GL texture: plot2robjs(glscreen, plot) → [RenderObject]
# (one atomic plot) → robj.uniforms[:image]::Texture (RGBA8, 4 bytes/texel =
# our RGBA{N0f8}).
function _image_texture(session)
    robjs = GLMakie.plot2robjs(session.glscreen, session.image_plot)
    robj  = only(robjs)
    return robj.uniforms[:image]::GLMakie.GLAbstraction.Texture
end

function _register!(st::GPUBlitState, tex)
    CC.cuGraphicsGLRegisterImage(st.res, tex.id, tex.texturetype,
                                 CC.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD)
    st.tex_id     = tex.id
    st.tex        = tex  # cached; invalidated in _unregister!
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
    st.tex = nothing  # invalidate the cached Texture (resize recreates it)
    return nothing
end

# close(::ViewportSession) calls this (duck-typed via _gpu_teardown!).
function OmniverseMakie._gpu_teardown!(st::GPUBlitState)
    try
        st.context === nothing || GLMakie.GLAbstraction.gl_switch_context!(st.context)
    catch
    end
    _unregister!(st)
    _free_oriented!(st)  # release the cached oriented device buffer
    return nothing
end

# Resize re-registration hook (duck-typed via gpu_unregister!).
# resize_viewport! recreates the image! plot, and GL may recycle the freed
# texture id — `tex_id != tex.id` alone can miss a recreated same-id texture.
# Unregister the old resource while its texture is still alive and clear
# `registered` (keep the GPUBlitState); the next GPU present! re-registers
# cleanly via its `!st.registered` guard.  No-op when the session never used
# the GPU path (gpu_state === nothing).
#
# Threading: this fires from the window_area listener on GLMakie's render
# task — the same task that registered it, so cuGraphicsUnregisterResource
# runs on the registering task.  gl_switch_context! makes the GL context
# current first (redundant on the render task, required when driven off it).
function OmniverseMakie.gpu_unregister!(session)
    st = session.gpu_state
    st === nothing && return nothing
    try
        st.context === nothing || GLMakie.GLAbstraction.gl_switch_context!(st.context)
    catch
    end
    _unregister!(st)
    _free_oriented!(st)  # release the cached oriented device buffer (resize)
    return nothing
end

# ------------------------------------------------------------------
# present!(session, ::Val{:gpu}) — the GPU-direct blit
# ------------------------------------------------------------------
"""
    OmniverseMakie.present!(session, ::Val{:gpu}) -> Nothing

GPU-direct blit: step ovrtx, map `HdrColor` as linear float16 CUDA memory,
tonemap float16→RGBA8 on-device (shared `tonemap`), copy straight into the
image! plot's GL texture — no CPU roundtrip.  Lazily registers the GL texture
on first call (it must already be realized — the GLMakie ext seeds it via a
CPU frame first).  On ANY failure: warn once, flip `session.blitter` to
`:cpu`, fall back to the CPU blit for this and all later frames.
"""
function OmniverseMakie.present!(session, ::Val{:gpu})
    try
        _gpu_present!(session)
    catch e
        # Forced GPU (gpu_direct=true): rethrow to surface the real GPU bug
        # (on_render_tick!'s try/catch logs it; the window stays alive).
        # Auto/false: degrade gracefully to the CPU blit.
        if session.gpu_forced
            rethrow()
        else
            @warn "M6: GPU-direct present! failed — falling back to the CPU blit for this session" exception = (e, catch_backtrace()) maxlog = 1
            session.blitter = :cpu
            # still update this frame via the CPU path
            OmniverseMakie.present!(session, Val(:cpu))
        end
    end
    return nothing
end

function _gpu_present!(session)
    screen = session.screen

    # --- lazy setup / resize re-registration (GL context current first) ---
    # Resolve the image! plot's GL texture once and cache it on the state;
    # steady state skips the per-frame plot2robjs + uniforms lookup.  The
    # cache is invalidated in _unregister! (resize / teardown), so the next
    # present re-resolves the fresh texture and the guard below re-registers.
    st  = session.gpu_state
    tex = (st !== nothing && st.tex !== nothing) ? st.tex : _image_texture(session)
    GLMakie.GLAbstraction.gl_switch_context!(tex.context)
    if st === nothing
        st = GPUBlitState(Ref{CC.CUgraphicsResource}(), tex.id, false, CUDA.CuEvent(),
                          tex.context, nothing, nothing)
        _register!(st, tex)
        session.gpu_state = st
    # Explicit unregister (resize) or a recreated/recycled-id texture —
    # re-register.
    elseif !st.registered || st.tex_id != tex.id
        # no-op if already unregistered (registered=false); invalidates st.tex
        _unregister!(st)
        st.context = tex.context
        _register!(st, tex)
    end

    # --- step ovrtx; keep the final StepResult to map its HdrColor ---
    for _ in 1:(session.steps_per_tick - 1)
        sr_drop = OV.step!(screen.renderer, screen.product; timeout_ns = _M6_STEP_TIMEOUT_NS)
        close(sr_drop)
    end
    sr = OV.step!(screen.renderer, screen.product; timeout_ns = _M6_STEP_TIMEOUT_NS)
    # Track the ovrtx mapping so the outer finally can release it on any
    # unwind. Cleared once the happy path unmaps it.
    mapping = nothing
    try
        mapping = OV.map_cuda(sr, "HdrColor")
        W = mapping.width
        H = mapping.height
        C = mapping.channels
        stream = Base.unsafe_convert(CC.CUstream, CUDA.stream())

        # Wait ovrtx's buffer-ready event before the kernel reads `data`.
        if mapping.wait_event != Csize_t(0)
            CC.cuStreamWaitEvent(stream, CC.CUevent(UInt(mapping.wait_event)), Cuint(0))
        end

        # Wrap the linear float16 CUdeviceptr as a CuArray [C,W,H];
        # tonemap+orient on-device into the cached [W,H] RGBA8 buffer.
        hdr = unsafe_wrap(CuArray, reinterpret(CUDA.CuPtr{Float16}, mapping.data), (C, W, H))
        oriented = _tonemap_orient_cached!(st, hdr, session.exposure)

        GC.@preserve oriented begin
            CC.cuGraphicsMapResources(1, st.res, stream)
            # A throw between map and unmap would leave the GL texture
            # CUDA-mapped (GLMakie then samples a mapped texture; the next
            # unregister errors) — the finally unmaps it on unwind.
            gl_mapped = true
            try
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
                gl_mapped = false
                # Record copy done; gate ovrtx unmap on it (no reclaim mid-copy).
                evh = Base.unsafe_convert(CC.CUevent, st.copy_done)
                CC.cuEventRecord(evh, stream)
                OV.unmap_cuda(mapping; stream = Csize_t(UInt(stream)), done_event = Csize_t(UInt(evh)))
                CC.cuStreamSynchronize(stream)  # GL sync before GLMakie samples
            finally
                # Swallow secondary errors so the primary exception surfaces.
                gl_mapped && try
                    CC.cuGraphicsUnmapResources(1, st.res, stream)
                catch e
                    @warn "M6: cuGraphicsUnmapResources failed on unwind" exception = e maxlog = 1
                end
            end
        end
    finally
        # Release the ovrtx mapping (synchronous NOSYNC unmap) if the happy
        # path did not; then always reap the StepResult.
        mapping !== nothing && mapping.open && try
            close(mapping)
        catch e
            @warn "M6: ovrtx unmap_cuda failed on unwind" exception = e maxlog = 1
        end
        close(sr)
    end
    return nothing
end

end # module OmniverseMakieCUDAExt
