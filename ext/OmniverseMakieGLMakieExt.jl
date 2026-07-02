module OmniverseMakieGLMakieExt

using OmniverseMakie, GLMakie
using OmniverseMakie: Makie, RGBA, N0f8, ColorTypes
# main-module internals used by the moved src/interactive code (bare names):
import OmniverseMakie: Screen, OV, _author_screen!, _sync_and_needs_reset!,
    _scene_for_camera, interactive_display, present!, on_render_tick!, cpu_blit!,
    attach_picking!, detach_picking!, _pick_at!   # M6.B Task 5: picking
using OmniverseMakie: tonemap
using Makie: Consume, MouseButtonEvent  # event types for the input forwarders

# ===== moved verbatim from src/interactive/blit.jl =====
# CPU blit (M5): set the image! plot's data Observable from a frame; GLMakie
# re-uploads the texture on the change (idiomatic Makie path).
#
# Orientation: the ovrtx frame is Matrix{RGBA{N0f8}}[H,W], row 1 = top.  Makie
# image! is [x, y] with y increasing UPWARD, so map frame→display via
# reverse(permutedims(frame), dims=2): permutedims → [W,H]; reverse dim 2 → y-up
# (data[col,H]=frame[1,col] at TOP, data[col,1]=frame[H,col] at BOTTOM).
# image! data is Observable [3] (x=[1], y=[2]) — REPL-verified.

# The one host-HDR→display orientation is the fused tonemap+orient loop
# `_tonemap_orient!` below: present!, interactive_display, and resize_viewport! all
# fill their session-owned [W,H] display buffer through it — reproducing
# reverse(permutedims(tonemap_frame([C,W,H] hdr, exposure)), dims=2) via out[j, H+1-i]
# (the SAME fused indexing the CUDA ext's oriented copy uses).  `_orient_for_display`
# / `cpu_blit!` are the equivalent orientation for an ALREADY-tonemapped [H,W] frame
# (the standalone M5 blit helper + its test); keep them pixel-consistent with the loop.
_orient_for_display(frame) = reverse(permutedims(frame), dims = 2)

# Fused host tonemap + display-orient: write the [C,W,H] linear-HDR `hdr` INTO the
# pre-oriented [W,H] RGBA{N0f8} buffer `out` in place — one pass, no intermediate
# frame / permutedims / reverse.  out[j, H+1-i] reproduces
# reverse(permutedims(tonemap_frame(hdr, exposure)), dims=2) exactly.  scale =
# exp2(exposure) is hoisted out of the W×H loop (once per tick); the per-pixel
# `tonemap` is the SHARED scalar (src/tonemap.jl), so host and CUDA kernel agree.
# `hdr` is eltype-generic (`<:Real`) with a per-element `Float32(...)` promotion, EXACTLY
# like the CUDA oriented kernel (C3, `_tonemap_orient_kernel!`): it accepts either the Float32
# [C,W,H] from render_hdr_to_array (interactive_display / resize_viewport! warmups) OR — for the
# INT-2 zero-copy present! — the still-mapped Float16 HdrColor view DIRECTLY (no Float32 transient).
# The Float32 path stays byte-identical (`Float32(::Float32)` is identity, goldens/C2/C3 pin it);
# Float16→Float32 widening is exact, so the zero-copy Float16 path matches the widen-first Float32
# path pixel-for-pixel.
function _tonemap_orient!(out::AbstractMatrix{RGBA{N0f8}},
                          hdr::AbstractArray{<:Real,3}, exposure::Float32)
    C, W, H = size(hdr)
    # The loop bounds come from `hdr` and the writes are `@inbounds`, so `out` MUST be the
    # oriented [W,H] — guard it explicitly (the CUDA twin `_tonemap_orient_kernel!` carries the
    # same guard per-thread).  Callers that only @warn on a size mismatch (resize_viewport!)
    # would otherwise turn a mismatch into an out-of-bounds write instead of a clean error.
    size(out) == (W, H) ||
        throw(DimensionMismatch("_tonemap_orient!: out is $(size(out)), need ($W, $H) for the [C=$C,W,H] hdr"))
    scale = exp2(exposure)                          # once per tick, hoisted out of the W×H loop
    @inbounds for j in 1:W, i in 1:H
        out[j, H + 1 - i] = tonemap((Float32(hdr[1, j, i]), Float32(hdr[2, j, i]), Float32(hdr[3, j, i])), scale)
    end
    return out
end

"""
    cpu_blit!(image_plot, frame::AbstractMatrix{RGBA{N0f8}}) -> Nothing

Set the GLMakie `image!` plot's data Observable from a `[H,W]` host frame (row 1 =
top), triggering a texture re-upload (CPU blit).  `reverse(permutedims(frame),
dims=2)` orients it right-side-up in the window.
"""
function cpu_blit!(image_plot, frame::AbstractMatrix{RGBA{N0f8}})
    image_plot[3][] = _orient_for_display(frame)   # [3] = data Observable
    return nothing
end

# ===== moved verbatim from src/interactive/viewport.jl =====
# Viewport session: a GLMakie window showing the live ovrtx RTX frame.
#
# Display (M5): one bare `campixel!` Scene holding a full-viewport `image!` (the
# blit target).  No Axis (its 2-D zoom/pan/limits are meaningless for a raw
# framebuffer).  The path tracer renders OFFSCREEN; each tick uploads its frame
# into the image texture (`present!`).
#
# Interaction (M5): campixel! has no 3-D camera, so FORWARD window input (mouse
# button/position/scroll/keyboard) into `cam_scene.events`.  cam_scene owns the
# `Camera3D` that `on_render_tick!` reads, so drag orbits / scroll zooms live.

"""
    mutable struct ViewportSession

All state for an interactive RTX viewport window (M5).

Fields:
  - `screen::Screen`        — open-stage ovrtx Screen (stage, renderer, plots)
  - `glscreen`              — GLMakie.Screen (the host window)
  - `glscene::Makie.Scene`  — the campixel! display Scene (image + input source)
  - `image_plot`            — full-viewport image! plot (present! target)
  - `present_buf`           — `[W,H]` pre-oriented `RGBA{N0f8}` display buffer; IS the
                              image! data array (CPU `present!` writes it in place + notify)
  - `cam_scene::Makie.Scene`— scene whose Camera3D drives the ovrtx view
  - `steps_per_tick::Int`   — ovrtx accumulation steps per render tick
  - `samples::Int`          — RT2 accumulation counter (0 on reset, += steps_per_tick)
  - `tick_listener`         — the render_tick per-frame hook
  - `input_listeners::Vector`— glscene→cam_scene forwarders (orbit/zoom)
  - `resize_listener`       — window-resize hook, or `nothing`
  - `exposure::Float32`     — EV exposure stops (0 = none) for the HDR tonemap
  - `blitter::Symbol`       — `:cpu` or `:gpu` — the present! strategy
  - `gpu_state`             — CUDA-ext GPUBlitState (lazy), or `nothing`; reset to
                              `:cpu` on GPU-setup failure (graceful CPU fallback)
  - `gpu_forced::Bool`      — gpu_direct=true was forced; surface GPU errors, no fallback
"""
mutable struct ViewportSession
    screen::Screen                  # the open-stage ovrtx Screen
    glscreen                        # GLMakie.Screen (the window)
    glscene::Makie.Scene            # the campixel! display Scene
    image_plot                      # full-viewport image! plot (blit target)
    present_buf::Matrix{RGBA{N0f8}} # [W,H] pre-oriented display buffer = image_plot's data array
    cam_scene::Makie.Scene          # the scene whose Camera3D drives the view
    steps_per_tick::Int
    samples::Int
    tick_listener                   # render_tick listener
    input_listeners::Vector         # glscene→cam_scene forwarding listeners
    resize_listener                 # window-resize hook, or nothing
    exposure::Float32               # EV stops for ACES tonemap
    blitter::Symbol                 # :cpu or :gpu — present! strategy
    gpu_state                       # CUDA-ext GPUBlitState (lazy) or nothing
    gpu_forced::Bool                # gpu_direct=true forced; no CPU fallback
end

# Input events forwarded from the display window into the camera scene, so the
# existing Camera3D orbits/zooms in response to the window's mouse/keyboard.
const _M5_FORWARDED_EVENTS = (:mousebutton, :mouseposition, :scroll, :keyboardbutton)

"""
    _pick_blitter(gpu_direct::Symbol_or_Bool) -> :gpu | :cpu

Resolve the per-frame blit strategy for `interactive_display`.  GPU-direct is
available only when the CUDA package extension is loaded AND `CUDA.functional()`.

  - `:auto`  → `:gpu` if CUDA is functional, else `:cpu` (the default).
  - `true`   → `:gpu` if CUDA is functional, else `error` (the caller demanded GPU).
  - `false`  → `:cpu` (always).
"""
function _pick_blitter(gpu_direct)
    cuda_ready = false
    try
        cuda_ready = Base.get_extension(OmniverseMakie, :OmniverseMakieCUDAExt) !== nothing &&
                     Base.invokelatest(OmniverseMakie._cuda_functional)
    catch
        cuda_ready = false
    end
    gpu_direct === :auto ? (cuda_ready ? :gpu : :cpu) :
    gpu_direct === true  ? (cuda_ready ? :gpu : error("gpu_direct=true but CUDA is unavailable (load `using CUDA` and ensure CUDA.functional())")) :
    :cpu
end

"""
    interactive_display(fig_or_scene; size=(800,600), steps_per_tick=2,
                        exposure=0f0, gpu_direct=:auto, selection_outline=false) -> ViewportSession

Open an orbit-able GLMakie window showing the live ovrtx RTX render of `fig_or_scene`.

Authors the open ovrtx stage once (mirrors `colorbuffer`'s open-once path), renders a
warmup frame into a single `campixel!` Scene as a full-viewport `image!` (no Axis), and
forwards window input to the scene's `Camera3D` so drag orbits / scroll zooms the live
view; a `render_tick` hook re-accumulates each frame (`on_render_tick!`).  Render
resolution is set to `size` by resizing the input scene's root before building `Screen`.

`gpu_direct` picks the blit path (M6.A): `:auto` = GPU-direct when the CUDA ext is
loaded and functional else CPU; `true` = force GPU-direct (errors if unavailable);
`false` = CPU.  GPU-setup failure degrades gracefully to `:cpu`.
`selection_outline=true` is NOT yet supported (the HdrColor viewport can't carry the
LdrColor-only outline) — it throws; use offscreen `select!` for outline images.

Calls `GLMakie.activate!()`; call `OmniverseMakie.activate!()` afterwards to restore the
ovrtx backend for offscreen `save`/`colorbuffer`.
"""
function interactive_display(fig_or_scene::Union{Makie.Figure,Makie.Scene}; size = (800, 600), steps_per_tick = 2, exposure = 0f0, gpu_direct = :auto, selection_outline::Bool = false)
    # M6.B (scope): the viewport presents HdrColor, but a selection-outline
    # Screen is LdrColor-only (no HdrColor AOV) — both present paths map
    # HdrColor and would throw.  Rather than build an LdrColor live-present
    # now, refuse selection_outline=true with an actionable error (offscreen
    # select! + render_to_matrix/colorbuffer give outline images).  false
    # (default) keeps a normal HdrColor Screen — byte-identical to before.
    selection_outline && throw(ArgumentError("interactive_display(; selection_outline=true) is not yet supported: the live viewport presents via the HdrColor path, but the selection outline requires an LdrColor-only Screen. Use offscreen `select!` + `Makie.colorbuffer`/`render_to_matrix` for outline images, or pass selection_outline=false. (LdrColor live-present is a planned follow-up.)"))
    scene     = fig_or_scene isa Makie.Figure ? fig_or_scene.scene : fig_or_scene
    cam_scene = something(_scene_for_camera(scene), scene)

    # 1. Resize root so Screen() (size(Makie.root(scene))) renders at `size`.
    Makie.resize!(Makie.root(cam_scene), size...)

    # 2. Build + author open-stage ovrtx Screen (colorbuffer open-once path).
    screen = Screen(cam_scene)
    screen.fb_size == size ||
        error("interactive_display: expected fb_size $(size), got $(screen.fb_size)")
    _author_screen!(screen, cam_scene, scene)

    # 3. First ovrtx frame (full warmup for a clean initial image) via HDR path.
    warmup_hdr = OV.render_hdr_to_array(screen.renderer, screen.product;
                                        warmup = screen.config.warmup)
    # The eager insertplots! flipped requires_update; the warmup already drew
    # that geometry, so consume the flag (as colorbuffer does) — else the first
    # on_render_tick! would redundantly OV.reset! and discard this clean frame.
    screen.requires_update = false

    # 4. Display: one campixel! Scene with a full-viewport image! whose data array IS the
    #    session-owned present buffer, filled by the fused tonemap+orient loop (the one
    #    orientation → frame row 1 = top).  present! writes this buffer in place each tick.
    GLMakie.activate!()
    W, H = size
    glscene = Makie.Scene(size = size)
    Makie.campixel!(glscene)
    present_buf = Matrix{RGBA{N0f8}}(undef, W, H)
    _tonemap_orient!(present_buf, warmup_hdr, exposure)
    img = image!(glscene, 0 .. W, 0 .. H, present_buf; interpolate = false)
    glscr = GLMakie.Screen()
    display(glscr, glscene)

    # 5. Forward window input to cam_scene so its Camera3D orbits/zooms
    #    (campixel! has no 3-D camera).  One-way; listeners observe (never
    #    Consume), so GLMakie still processes events.
    input_listeners = [on(getproperty(glscene.events, f)) do v
        getproperty(cam_scene.events, f)[] = v
        return Makie.Consume(false)
    end for f in _M5_FORWARDED_EVENTS]

    # Blit strategy: :gpu when CUDA loaded + functional, else :cpu.
    blitter = _pick_blitter(gpu_direct)
    # Only literal `true` is forced; :auto/false allow graceful CPU fallback.
    gpu_forced = (gpu_direct === true)

    session = ViewportSession(screen, glscr, glscene, img, present_buf, cam_scene,
                              steps_per_tick, screen.config.warmup,
                              nothing, input_listeners, nothing, exposure,
                              blitter, nothing, gpu_forced)

    # 6. Per-frame hook on GLMakie's render task. Must NOT Consume(true).
    session.tick_listener = on(glscr.render_tick) do _
        on_render_tick!(session)
        return Makie.Consume(false)
    end

    # 7. Window-resize hook (resize_viewport!): rebuild the ovrtx renderer +
    #    image! at the new size.  Fires on glscene.events.window_area (GLFW size
    #    callback); the initial fire carries fb_size and is a no-op.
    session.resize_listener = on(glscene.events.window_area) do area
        w, h = Int.(widths(area))
        (w, h) == session.screen.fb_size && return   # no change — skip
        w > 0 && h > 0                  || return   # skip degenerate sizes
        resize_viewport!(session, (w, h))
        return nothing
    end

    return session
end

# ------------------------------------------------------------------
# Teardown
# ------------------------------------------------------------------

"""
    Base.close(session::ViewportSession) -> Nothing

Tear down the interactive viewport: detach all listeners, close the GLMakie window,
then close the underlying ovrtx `Screen` (renderer + bindings).

Idempotent: a second `close` is a safe no-op (listener guards + `renderer.alive`
check in `Base.close(::Screen)` absorb repeated calls).
"""
function Base.close(session::ViewportSession)
    session.tick_listener   === nothing || off(session.tick_listener)
    session.resize_listener === nothing || off(session.resize_listener)
    for l in session.input_listeners; off(l); end
    empty!(session.input_listeners)
    session.tick_listener   = nothing
    session.resize_listener = nothing
    # M6.A: STOP the render loop and JOIN the render task BEFORE any teardown.
    # The task fires render_tick→on_render_tick!→present!, which for GPU-direct
    # runs raw CUDA→GL interop on ovrtx/CUDA resources — tearing those down
    # mid-tick is a use-after-free (segfault).  stop_renderloop! joins the task
    # (called off it), so no present! races the teardown below.
    try
        isopen(session.glscreen) && GLMakie.stop_renderloop!(session.glscreen)
    catch e
        @warn "M6: error stopping GLMakie render loop" exception=e
    end
    # M6.A: unregister the GPU-direct GL texture resource (texture still alive,
    # loop stopped) — MUST run BEFORE close(glscreen) destroys the texture.
    if session.gpu_state !== nothing
        try
            Base.invokelatest(OmniverseMakie._gpu_teardown!, session.gpu_state)
        catch e
            @warn "M6: error tearing down GPU-direct blit state" exception=e
        end
        session.gpu_state = nothing
    end
    try
        isopen(session.glscreen) && close(session.glscreen)
    catch e
        @warn "M5: error closing GLMakie screen" exception=e
    end
    Base.close(session.screen)   # ovrtx Screen LAST (bindings→renderer; M1)
    return nothing
end

# ------------------------------------------------------------------
# Window resize
# ------------------------------------------------------------------

"""
    resize_viewport!(session::ViewportSession, (W, H)::Tuple{Int,Int}) -> Nothing

Rebuild the ovrtx renderer at `(W, H)` and refresh the displayed image.  Called by
the `resize_listener` on `glscene.events.window_area`.

Resize root → build+author a new ovrtx `Screen` + warmup → swap it in, then close the
OLD `Screen` (avoid a GPU leak) → replace the `image!` plot at the new size (delete +
recreate, avoiding image! range-change edge cases).
"""
function resize_viewport!(session::ViewportSession, (W, H)::Tuple{Int,Int})
    cam_scene = session.cam_scene
    root_scene = Makie.root(cam_scene)  # root scene passed to insertplots!

    # 1. Resize root so Screen() picks up the new resolution.
    Makie.resize!(root_scene, W, H)

    # 2. Build + author the new ovrtx Screen (same as interactive_display).
    new_screen = Screen(cam_scene)
    if new_screen.fb_size != (W, H)
        @warn "M5 resize: expected fb_size $((W, H)), got $(new_screen.fb_size)"
    end
    _author_screen!(new_screen, cam_scene, root_scene)

    # 3. Render a warmup frame at the new size via HDR path.
    warmup_hdr = OV.render_hdr_to_array(new_screen.renderer, new_screen.product;
                                        warmup = new_screen.config.warmup)

    # 4. Swap in the new Screen FIRST (session stays self-consistent), then
    #    free the OLD ovrtx renderer (avoid a GPU leak).  Reset the counter.
    old_screen      = session.screen
    session.screen  = new_screen
    session.samples = new_screen.config.warmup
    # Consume the warmup flag (as interactive_display does) so the first tick
    # does not redundantly reset and discard this clean frame.
    new_screen.requires_update = false
    Base.close(old_screen)

    # 5. Replace the image!: delete the wrong-size one, allocate a fresh [W,H] present
    #    buffer filled by the fused tonemap+orient loop (the one orientation), and add an
    #    image! backed by it spanning 0..W,0..H (campixel! coords cover that after resize)
    #    — so the next present! writes it in place (its size guard passes).
    # M6.A: unregister the OLD CUDA-GL resource BEFORE delete! (texture still
    # alive → no leak).  GL may RECYCLE the freed texture id, so explicit
    # unregister + the present! `!st.registered` re-register guard make resize
    # id-recycle-proof (A-2).  No-op for CPU / no-CUDA (gpu_state === nothing).
    session.gpu_state === nothing || Base.invokelatest(OmniverseMakie.gpu_unregister!, session)
    delete!(session.glscene, session.image_plot)
    new_buf = Matrix{RGBA{N0f8}}(undef, W, H)
    _tonemap_orient!(new_buf, warmup_hdr, session.exposure)
    new_img = image!(session.glscene, 0 .. W, 0 .. H, new_buf; interpolate = false)
    session.image_plot  = new_img
    session.present_buf = new_buf

    return nothing
end

# ===== moved verbatim from src/interactive/camera_loop.jl =====
# Camera loop (M5 Task 3): per-frame hook on GLMakie's render task — push live
# camera/light/plot deltas, reset RT2 accumulation only on a change, step, blit.
# Mirrors colorbuffer's sync/reset (screen.jl:342-358) but bounded-steps, blits
# (not returns).  Must NOT Consume (registered as an observer, not a handler).
#
# Accumulation state machine:
#   - idle tick (no delta): samples += steps_per_tick (RT2 keeps refining)
#   - change tick (sync_camera!/sync_lights!/pull_ovrtx_nodes!): OV.reset!,
#     samples = 0, then += steps_per_tick

"""Bounded step timeout: 10 s — long enough for a normal RT2 step, short enough not to hang."""
const _M5_STEP_TIMEOUT_NS = UInt64(10_000_000_000)

# ------------------------------------------------------------------
# present! — CPU HDR blit: HdrColor → host tonemap → image! data
# ------------------------------------------------------------------

"""
    OmniverseMakie.present!(session::ViewportSession, ::Val{:cpu}) -> Nothing

`:cpu` blit strategy: step ovrtx `steps_per_tick` times (host twin of the CUDA ext's
`_gpu_present!` step loop — drop each intermediate `StepResult`, keep the final `sr`), then map
its `HdrColor` (float16) and in ONE fused pass tonemap (ACES + sRGB + `session.exposure`) AND
orient it STRAIGHT from the still-mapped Float16 view into the session's cached `[W,H]` display
buffer (`present_buf`) IN PLACE, then `notify` the image! plot's data Observable so GLMakie
re-uploads the texture.  Steady state = ZERO full-frame allocations: the buffer is reused and
(INT-2) the tonemap reads the mapped Float16 directly, so the Float32 `[C,W,H]` HDR transient that
`render_hdr_to_array` used to materialize is gone.  Everything touching the mapped view stays
INSIDE the `with_mapped_hdr` closure (the mapping dies on unmap at its return); the loop fully
materializes into `present_buf` (owned — NOT a view over the mapping), so nothing lazy escapes.  A
size guard reallocates + re-seats the buffer after a resize (Screen rebuilt).  The `:gpu` strategy
(on-device tonemap + CUDA→GL copy, no host roundtrip) lives in the CUDA extension; both are selected
via `present!(session, Val(session.blitter))`.
"""
function OmniverseMakie.present!(session::ViewportSession, ::Val{:cpu})
    screen = session.screen
    # Host twin of the CUDA ext's _gpu_present! step loop: run `steps_per_tick` bounded RT2 steps,
    # closing each dropped StepResult; KEEP the final `sr` to map its HdrColor.  (Was
    # OV.render_hdr_to_array, which ADDITIONALLY materialized a Float32 [C,W,H] HDR copy per tick —
    # the transient INT-2 removes; render_hdr_to_array itself stays for colorbuffer/HDR callers.)
    for _ in 1:(session.steps_per_tick - 1)
        sr_drop = OV.step!(screen.renderer, screen.product; timeout_ns = _M5_STEP_TIMEOUT_NS)
        close(sr_drop)
    end
    sr = OV.step!(screen.renderer, screen.product; timeout_ns = _M5_STEP_TIMEOUT_NS)
    try
        # INT-2 zero-copy present: tonemap+orient STRAIGHT from the still-mapped Float16 HdrColor
        # view into the cached buffer — no Float32 HDR transient.  CORRECTNESS: everything that
        # touches `raw16` stays INSIDE this closure (the mapping dies on unmap at with_mapped_hdr's
        # return); the loop fully materializes into `buf` (== session.present_buf, owned — NOT a
        # view over raw16), so nothing lazy over the mapping escapes.  The closure captures only
        # `session` (read + field-mutated, never rebound → not boxed); `buf` is closure-local.
        OV.with_mapped_hdr(sr) do raw16, W, H
            # Size guard: a resize rebuilds the Screen (+ image!) at a new size — reallocate the
            # cached buffer and re-seat it AS the image Observable's data array when it no longer
            # fits (interactive_display / resize_viewport! seed it; this is belt-and-suspenders).
            buf = session.present_buf
            if size(buf) != (W, H)
                buf = Matrix{RGBA{N0f8}}(undef, W, H)
                session.present_buf = buf
                session.image_plot[3][] = buf
            end
            # Fuse tonemap+orient straight into the cached buffer (in place), then notify so GLMakie
            # re-uploads the texture from that SAME array — zero steady-state display garbage.
            _tonemap_orient!(buf, raw16, session.exposure)
            Makie.notify(session.image_plot[3])
            return nothing
        end
    finally
        close(sr)
    end
    return nothing
end

"""
    _on_render_tick_impl!(session::ViewportSession) -> Nothing

Core of the per-frame live camera loop (M5 Task 3), called from `on_render_tick!`.
Pushes live camera/light/plot deltas, resets RT2 accumulation only on a change (else
keeps accumulating), runs `steps_per_tick` bounded steps, and blits to the image plot.
"""
function _on_render_tick_impl!(session::ViewportSession)
    screen    = session.screen
    cam_scene = session.cam_scene

    # Push live camera/light/plot deltas; reset RT2 accumulation on any change.
    if _sync_and_needs_reset!(screen, cam_scene)
        OV.reset!(screen.renderer)
        session.samples = 0
    end

    # HDR step + tonemap + blit via the selected strategy (:cpu / :gpu).  The
    # CUDA ext's GPU present! falls back to :cpu (session.blitter) on failure.
    present!(session, Val(session.blitter))
    session.samples += session.steps_per_tick
    return nothing
end

"""
    on_render_tick!(session::ViewportSession) -> Nothing

Per-frame live camera loop (M5 Task 3), fired on every GLMakie `render_tick`.  Wraps
`_on_render_tick_impl!` in a try/catch (a bad frame warns, `maxlog=5`, but does NOT
crash the window).  Must NOT Consume — the caller returns `Makie.Consume(false)`.
"""
function OmniverseMakie.on_render_tick!(session::ViewportSession)
    try
        _on_render_tick_impl!(session)
    catch e
        @warn "M5: render-tick frame failed (window kept alive)" exception=(e, catch_backtrace()) maxlog=5
    end
    return nothing
end

# ===== M6.B Task 5 — attachable picking interaction =====
# Wire a viewport click to a native AOV pick.  The pick CORE is in the main
# module (`pick_hit`, no GLMakie/CUDA dep); this ext only adds input wiring.
# Data flow: click → `_pick_at!` → `pick_hit(session.screen, xy)` →
# `PickHandle.selected[]` + `on_hit`.  Works on the HdrColor viewport.
#
# In-viewport OUTLINE is DEFERRED (see interactive_display's selection_outline
# guard): the viewport is always a normal HdrColor Screen
# (config.selection_outline == false), so attach_picking!(outline=true) degrades
# (@warn maxlog=1 → outline=false).  The _pick_at! outline branch is kept (gated
# on config.selection_outline) for the LdrColor follow-up.

"""
    mutable struct PickHandle

Handle from [`attach_picking!`](@ref): the click listener + picking state.

Fields:
  - `session`      — the `ViewportSession` this handle picks on
  - `listener`     — the `glscene.events.mousebutton` listener (or `nothing` once detached)
  - `on_hit`       — user callback `hit -> …` run after each pick (or `nothing`)
  - `outline::Bool`— draw a hit outline (degrades to `false` unless the Screen was built
                     with `selection_outline=true` — deferred)
  - `selected::Observable{Any}` — last hit `(; plot, index, world_position, normal)`, or
                     `nothing` over background; observe it to react to picks
  - `last_plot`    — currently-outlined plot (or `nothing`); cleared on next pick / detach
"""
mutable struct PickHandle
    session
    listener
    on_hit
    outline::Bool
    selected::Observable{Any}
    last_plot
end

# Max press→release travel (px) still counted as a CLICK, not a drag.  A
# left-drag orbits the Camera3D (M5 forwarding); a pick fires only on a click.
const _PICK_CLICK_PX = 5.0

# Cursor position on scene as (Float64, Float64).  `mouseposition` may hold a
# Point2 or a tuple (M5 forwarders assign tuples); indexing works for both.
_mouse_xy(session) = let mp = session.glscene.events.mouseposition[]
    (Float64(mp[1]), Float64(mp[2]))
end

# Did press→release travel far enough to be a drag (orbit) rather than a click?
# `press_pos` is a Ref captured by the attach_picking! listener (picking state
# lives on the handle/closure, off the ViewportSession).  NaN press ⇒ a click.
function _was_dragging(session, press_pos::Ref)
    px, py = press_pos[]
    isnan(px) && return false
    rx, ry = _mouse_xy(session)
    return hypot(rx - px, ry - py) > _PICK_CLICK_PX
end

"""
    _pick_at!(session, h::PickHandle, xy) -> hit | nothing

Pick at display pixel `xy` and publish: set `h.selected[]`, invoke `h.on_hit`.  Also the
test entry point (drive synchronously to assert wiring without the flaky GLFW event thread).
`pick_hit` is a renderer query, safe from the click listener (GLMakie runs handlers on the
render task, same as `present!`).  The outline branch is a no-op on the current HdrColor
viewport (`config.selection_outline == false`); kept for the deferred LdrColor path.
"""
function _pick_at!(session, h::PickHandle, xy)
    hit = OmniverseMakie.pick_hit(session.screen, xy)
    if h.outline && session.screen.config.selection_outline
        h.last_plot === nothing || OmniverseMakie.clear_selection!(session.screen, h.last_plot)
        h.last_plot = hit === nothing ? nothing : hit.plot
        hit === nothing || OmniverseMakie.select!(session.screen, hit.plot)
    end
    h.selected[] = hit
    h.on_hit === nothing || h.on_hit(hit)
    return hit
end

"""
    attach_picking!(session; on_hit=nothing, outline=false, button=Makie.Mouse.left) -> PickHandle

Attach click-to-pick to a live viewport `session`.  A click (press+release of `button`
without a drag, so it does not fight the M5 left-drag orbit) runs a native AOV pick and
publishes the hit on the handle's `selected` Observable, also invoking `on_hit(hit)`.
`hit` is `(; plot, index, world_position, normal)` or `nothing` over background.

`outline=true` DEGRADES (warns once): the HdrColor viewport can't carry the LdrColor-only
outline, so no highlight is drawn (pick data still works) — for outline IMAGES use offscreen
`select!` + `render_to_matrix`/`colorbuffer`.

[`detach_picking!`](@ref) removes it (also torn down on window close — the listener lives on
`glscene.events`).
"""
function attach_picking!(session; on_hit = nothing, outline::Bool = false, button = Makie.Mouse.left)
    if outline && !session.screen.config.selection_outline
        @warn "attach_picking!(outline=true) but the viewport was built without selection_outline=true; \
               no highlight will be drawn (the live in-viewport outline is a deferred follow-up — use \
               offscreen select! + render_to_matrix/colorbuffer for outline images)" maxlog=1
        outline = false
    end
    h = PickHandle(session, nothing, on_hit, outline, Observable{Any}(nothing), nothing)
    # Click = press+release without a drag.  Track press pos in a closure Ref
    # so a moved release (orbit drag) does not fire a pick.  Never Consume.
    press_pos = Ref((NaN, NaN))
    h.listener = on(session.glscene.events.mousebutton) do ev
        if ev.button == button
            if ev.action == Makie.Mouse.press
                press_pos[] = _mouse_xy(session)
            elseif ev.action == Makie.Mouse.release && !_was_dragging(session, press_pos)
                _pick_at!(session, h, _mouse_xy(session))
            end
        end
        return Makie.Consume(false)
    end
    return h
end

"""
    detach_picking!(h::PickHandle) -> Nothing

Remove the click listener attached by [`attach_picking!`](@ref) and clear any selection outline
the handle drew.  Idempotent (a second call is a safe no-op).
"""
function detach_picking!(h::PickHandle)
    h.listener === nothing || off(h.listener)
    h.listener = nothing
    h.last_plot === nothing || OmniverseMakie.clear_selection!(h.session.screen, h.last_plot)
    h.last_plot = nothing
    return nothing
end

end # module OmniverseMakieGLMakieExt
