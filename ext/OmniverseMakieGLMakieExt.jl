module OmniverseMakieGLMakieExt

using OmniverseMakie, GLMakie
using OmniverseMakie: Makie, RGBA, N0f8, ColorTypes
# main-module internals the moved code uses (bare names in the original src/interactive/*.jl):
import OmniverseMakie: Screen, OV, _author_screen!, _sync_and_needs_reset!,
    _scene_for_camera, interactive_display, present!, on_render_tick!, cpu_blit!
using OmniverseMakie: tonemap, tonemap_frame
using Makie: Consume, MouseButtonEvent  # event types used by the input forwarders

# ===== moved verbatim from src/interactive/blit.jl =====
# CPU blit (M5): update the image! plot's data Observable from a host frame.
# GLMakie re-uploads the texture on the data change (spike §5 — idiomatic Makie path).
#
# Orientation: our ovrtx frame is Matrix{RGBA{N0f8}}[H,W] with row 1 = top (right-side-up).
# Makie's image! plots with first dimension = x (horizontal) and second dimension = y
# (vertical, y increases upward in the default Axis convention).
#
# Transform needed to display frame rows as vertical top-to-bottom:
#   - permutedims(frame): [W,H] — swaps dims so frame rows become y-axis (second dim)
#   - reverse(..., dims=2): flip second dim to match y-up — data[col, k] = frame[H-k+1, col]
# Result: data[col, H] (high y = TOP) = frame[1, col] (red); data[col, 1] (low y = BOTTOM) = frame[H, col] (blue).
#
# Verified in Step 1 REPL:
#   img[1][] = EndPoints (x range), img[2][] = EndPoints (y range), img[3][] = Matrix{RGBA{N0f8}}
#   Data Observable index = [3].  `image!(ax, frame)` — x=img[1], y=img[2], data=img[3].

# single source for the host-frame → Makie-image orientation; cpu_blit!, interactive_display, and resize_viewport! must all use it
_orient_for_display(frame) = reverse(permutedims(frame), dims = 2)

"""
    cpu_blit!(image_plot, frame::AbstractMatrix{RGBA{N0f8}}) -> Nothing

Update the GLMakie `image!` plot's data Observable from a host frame, triggering a
texture re-upload (CPU blit, M5 §5).

The host frame is `[H, W]` top-left origin (row 1 = top).  The transform
`reverse(permutedims(frame), dims=2)` maps frame rows to Makie's y-axis so the
image appears right-side-up in the GLMakie window.
"""
function cpu_blit!(image_plot, frame::AbstractMatrix{RGBA{N0f8}})
    # [3] = data Observable (x=img[1], y=img[2], data=img[3]; verified Step 1 REPL)
    image_plot[3][] = _orient_for_display(frame)
    return nothing
end

# ===== moved verbatim from src/interactive/viewport.jl =====
# Viewport session: opens a real GLMakie window displaying the live ovrtx RTX frame.
#
# Display model (M5): the window is a SINGLE bare Scene with a pixel camera
# (`campixel!`) holding one full-viewport `image!` plot — the blit target.  No
# `Axis` (an Axis would add 2-D zoom/pan/limits that have no meaning for a raw
# framebuffer).  The path tracer renders OFFSCREEN; we upload its frames into the
# image plot's texture each tick (`cpu_blit!`).
#
# Interaction model (M5): a `campixel!` scene has no 3-D camera, so we FORWARD the
# window's input events (mouse button / position / scroll / keyboard) into
# `cam_scene.events`.  `cam_scene` already owns the `Camera3D` that `on_render_tick!`
# reads, so dragging orbits / scrolling zooms it live and the RTX view reframes.

"""
    mutable struct ViewportSession

Holds all state for an interactive RTX viewport window (M5).

Fields:
  - `screen::Screen`        — open-stage ovrtx Screen (USD stage, renderer, plots)
  - `glscreen`              — GLMakie.Screen (the host window)
  - `glscene::Makie.Scene`  — the `campixel!` display Scene (image target + input source)
  - `image_plot`            — the full-viewport `image!` plot (blit target for cpu_blit!)
  - `cam_scene::Makie.Scene`— the scene whose `Camera3D` drives the ovrtx view
  - `steps_per_tick::Int`   — ovrtx accumulation steps per render tick
  - `samples::Int`          — live RT2 accumulation counter (0 on reset, += steps_per_tick)
  - `tick_listener`         — the `render_tick` per-frame hook (Observables listener)
  - `input_listeners::Vector`— glscene→cam_scene event-forwarding listeners (orbit/zoom)
  - `resize_listener`       — window-resize hook (set in Task 4); `nothing` otherwise
  - `exposure::Float32`    — EV exposure in stops (0 = no change); used by the HDR tonemap path
  - `blitter::Symbol`      — `:cpu` or `:gpu` — selects the `present!` strategy (M6.A Task 4)
  - `gpu_state`            — CUDA-ext GPU-direct blit state (a `GPUBlitState`), or `nothing`
                             until the GPU `present!` lazily registers; reset to `:cpu` on
                             GPU-setup failure (graceful CPU fallback)
  - `gpu_forced::Bool`     — true when gpu_direct=true was forced; surfaces GPU errors instead of falling back
"""
mutable struct ViewportSession
    screen::Screen                  # the open-stage ovrtx Screen
    glscreen                        # GLMakie.Screen (the window)
    glscene::Makie.Scene            # the campixel! display Scene
    image_plot                      # the full-viewport image! plot (blit target)
    cam_scene::Makie.Scene          # the scene whose Camera3D drives the view
    steps_per_tick::Int
    samples::Int
    tick_listener                   # render_tick listener
    input_listeners::Vector         # glscene→cam_scene forwarding listeners
    resize_listener                 # set in Task 4; nothing here
    exposure::Float32               # EV stops for ACES tonemap (Task 2 HDR path)
    blitter::Symbol                 # :cpu or :gpu — present! strategy (M6.A Task 4)
    gpu_state                       # CUDA-ext GPUBlitState (lazy) or nothing
    gpu_forced::Bool                # true when gpu_direct=true was forced; surfaces GPU errors instead of falling back
end

# Input events forwarded from the display window into the 3-D camera scene so the
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
    interactive_display(fig_or_scene; size=(800,600), steps_per_tick=2, exposure=0f0, gpu_direct=:auto) -> ViewportSession

Open an interactive, orbit-able GLMakie window showing the live ovrtx RTX render of
`fig_or_scene`.

`gpu_direct` selects the per-frame blit path (M6.A): `:auto` uses the GPU-direct CUDA
path when the CUDA extension is loaded and `CUDA.functional()` (else the CPU host
tonemap path); `true` forces GPU-direct (errors if CUDA is unavailable); `false`
forces the CPU path.  On GPU-setup failure the session degrades gracefully to `:cpu`.

Authors the open ovrtx stage once (mirrors `colorbuffer`'s open-once path), renders a
warmup frame, and shows it in a single `campixel!` Scene as a full-viewport `image!`
(pixel-perfect, no Axis).  The window's input events are forwarded to the scene's
`Camera3D`, so dragging orbits and scrolling zooms the live RTX view; a `render_tick`
hook reframes and re-accumulates each frame (`on_render_tick!`).

The ovrtx render resolution is set to `size` by resizing the input scene's root before
constructing the OmniverseMakie `Screen`.

Calls `GLMakie.activate!()` (switches the active Makie backend to GLMakie for the
window); call `OmniverseMakie.activate!()` afterwards to restore the ovrtx backend for
offscreen `save`/`colorbuffer` use.
"""
function interactive_display(fig_or_scene::Union{Makie.Figure,Makie.Scene}; size = (800, 600), steps_per_tick = 2, exposure = 0f0, gpu_direct = :auto)
    scene     = fig_or_scene isa Makie.Figure ? fig_or_scene.scene : fig_or_scene
    cam_scene = something(_scene_for_camera(scene), scene)

    # 1. Resize the root scene to the requested size so the ovrtx renderer renders at
    #    that resolution (the Screen constructor uses size(Makie.root(scene))).
    Makie.resize!(Makie.root(cam_scene), size...)

    # 2. Build + author the open-stage ovrtx Screen (mirrors colorbuffer's open-once path).
    screen = Screen(cam_scene)
    screen.fb_size == size ||
        error("interactive_display: expected fb_size $(size), got $(screen.fb_size)")
    _author_screen!(screen, cam_scene, scene)

    # 3. First ovrtx frame (full warmup for a clean initial image) via HDR path.
    frame = tonemap_frame(OV.render_hdr_to_array(screen.renderer, screen.product;
                          warmup = screen.config.warmup), exposure)
    # The eager `insertplots!` build flipped `screen.requires_update`; the warmup render
    # above already drew that built geometry, so consume the flag here (mirrors how
    # `colorbuffer` consumes it).  Otherwise the FIRST `on_render_tick!` would treat it as a
    # pending change and redundantly `OV.reset!`, discarding this clean warmup frame for a
    # noisy `steps_per_tick`-sample one before idle ticks refine it back.
    screen.requires_update = false

    # 4. Display: a single pixel-perfect Scene (campixel!) holding one full-viewport
    #    image! of the RTX frame.  The orientation transform matches cpu_blit!
    #    (reverse(permutedims) → row 1 of the frame = top of the window).
    GLMakie.activate!()
    W, H = size
    glscene = Makie.Scene(size = size)
    Makie.campixel!(glscene)
    img = image!(glscene, 0 .. W, 0 .. H, _orient_for_display(frame); interpolate = false)
    glscr = GLMakie.Screen()
    display(glscr, glscene)

    # 5. Forward the window's input into cam_scene so its Camera3D orbits/zooms live
    #    (campixel! has no 3-D camera of its own).  One-way glscene→cam_scene; the
    #    listeners observe (never Consume), so GLMakie still processes events normally.
    input_listeners = [on(getproperty(glscene.events, f)) do v
        getproperty(cam_scene.events, f)[] = v
        return Makie.Consume(false)
    end for f in _M5_FORWARDED_EVENTS]

    # Resolve the per-frame blit strategy (:gpu when CUDA is loaded + functional, else :cpu).
    blitter = _pick_blitter(gpu_direct)
    # Only the literal `true` counts as forced; :auto and false both allow graceful CPU fallback.
    gpu_forced = (gpu_direct === true)

    session = ViewportSession(screen, glscr, glscene, img, cam_scene,
                              steps_per_tick, screen.config.warmup,
                              nothing, input_listeners, nothing, exposure,
                              blitter, nothing, gpu_forced)

    # 6. Per-frame hook on GLMakie's render task. Must NOT Consume(true).
    session.tick_listener = on(glscr.render_tick) do _
        on_render_tick!(session)
        return Makie.Consume(false)
    end

    # 7. Window-resize hook: when the GLMakie window is resized, rebuild the ovrtx
    #    renderer at the new resolution (resize_viewport!) and update the image! plot.
    #    Fires on glscene.events.window_area (Rect2), which GLMakie updates via the
    #    GLFW window-size callback whenever the window is resized.  The initial fire
    #    at display time carries the current size (== fb_size) and is a no-op.
    session.resize_listener = on(glscene.events.window_area) do area
        w, h = Int.(widths(area))
        (w, h) == session.screen.fb_size && return   # no change — skip
        w > 0 && h > 0                  || return   # guard against degenerate sizes
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
    # M6.A: STOP the GLMakie render loop and WAIT for the render task to finish BEFORE any
    # teardown.  The background render task fires `render_tick` → `on_render_tick!` →
    # `present!`; for the GPU-direct path that runs raw CUDA→GL interop on ovrtx/CUDA
    # resources.  Tearing those down while a tick is in flight is a use-after-free
    # (segfault).  `stop_renderloop!` joins the task (when called off the render task), so
    # no `present!` can race the teardown below.  Harmless for the CPU path.
    try
        isopen(session.glscreen) && GLMakie.stop_renderloop!(session.glscreen)
    catch e
        @warn "M6: error stopping GLMakie render loop" exception=e
    end
    # M6.A: unregister the GPU-direct GL texture resource (texture still alive here, render
    # loop stopped) — must run BEFORE close(glscreen) destroys the texture.
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
    Base.close(session.screen)   # ovrtx Screen LAST (bindings then renderer — M1 teardown order)
    return nothing
end

# ------------------------------------------------------------------
# Window resize
# ------------------------------------------------------------------

"""
    resize_viewport!(session::ViewportSession, (W, H)::Tuple{Int,Int}) -> Nothing

Rebuild the ovrtx renderer at the new size `(W, H)` and refresh the displayed
image.  Called by the `resize_listener` on `glscene.events.window_area`.

Steps:
1. Resize the root cam scene so the new `Screen` picks up the new resolution.
2. Build + author a new open-stage ovrtx `Screen`, render a warmup frame.
3. Close the OLD ovrtx `Screen` (free the GPU renderer — avoid a leak).
4. Swap in the new screen and update the full-viewport `image!` plot to the
   new dimensions (delete old, create new — avoids image! range-change edge cases).
"""
function resize_viewport!(session::ViewportSession, (W, H)::Tuple{Int,Int})
    cam_scene = session.cam_scene
    root_scene = Makie.root(cam_scene)  # the figure/root scene passed to insertplots!

    # 1. Resize the root scene so Screen() picks up the new resolution.
    Makie.resize!(root_scene, W, H)

    # 2. Build + author the new ovrtx Screen (same sequence as interactive_display).
    new_screen = Screen(cam_scene)
    if new_screen.fb_size != (W, H)
        @warn "M5 resize: expected fb_size $((W, H)), got $(new_screen.fb_size)"
    end
    _author_screen!(new_screen, cam_scene, root_scene)

    # 3. Render a warmup frame at the new size via HDR path.
    frame = tonemap_frame(OV.render_hdr_to_array(new_screen.renderer, new_screen.product;
                          warmup = new_screen.config.warmup), session.exposure)

    # 4. Swap in the new Screen FIRST (session stays self-consistent at every point —
    #    defensive even if a future resize path is not synchronous), then free the OLD
    #    ovrtx renderer (avoid a GPU leak).  Reset the accumulation counter for the new frame.
    old_screen      = session.screen
    session.screen  = new_screen
    session.samples = new_screen.config.warmup
    # Consume the warmup render's flag (mirrors interactive_display's post-warmup reset):
    # the warmup already drew the new geometry, so the first tick should not redundantly
    # reset and discard this clean frame.
    new_screen.requires_update = false
    Base.close(old_screen)

    # 5. Update the displayed image: delete the old image! (wrong size) and add a new
    #    one spanning the full new viewport.  The campixel! coordinate system already
    #    covers 0..W, 0..H after the GLMakie window resize.
    # M6.A Task 5: drop the OLD GPU-direct CUDA-GL registration BEFORE deleting the image!
    # plot — its GL texture is still alive here (no unregister-of-a-dead-texture, no leak).
    # GL may recycle the freed texture id, so the explicit unregister + the present!
    # `!st.registered` re-register guard make resize id-recycle-proof (review A-2).  Guarded
    # on `gpu_state` (the CUDA ext sets it) so this is a no-op for CPU sessions / no-CUDA.
    session.gpu_state === nothing || Base.invokelatest(OmniverseMakie.gpu_unregister!, session)
    delete!(session.glscene, session.image_plot)
    new_img = image!(session.glscene, 0 .. W, 0 .. H,
                     _orient_for_display(frame); interpolate = false)
    session.image_plot = new_img

    return nothing
end

# ===== moved verbatim from src/interactive/camera_loop.jl =====
# Camera loop (M5 Task 3): live render tick — camera sync + ovrtx step + cpu_blit!.
#
# Per-frame hook on GLMakie's render task: push live camera/light/plot deltas, reset
# RT2 accumulation only when something changed (else keep accumulating — progressive
# refinement), do ONE bounded accumulation step, and blit.
#
# Mirrors `colorbuffer`'s per-frame sync/reset logic (screen.jl:342-358) but steps a
# bounded number of times and blits instead of returning the matrix.  Must NOT Consume
# events (registered as an observer, not a handler).
#
# Accumulation state machine (M5.3):
#   - idle tick (no camera/light/plot change): session.samples += steps_per_tick
#     (RT2 keeps accumulating — progressive refinement)
#   - change tick (sync_camera!/sync_lights!/pull_ovrtx_nodes! saw a delta):
#     OV.reset! restarts RT2, session.samples = 0, then += steps_per_tick → steps_per_tick

"""Bounded step timeout: 10 s — long enough for a normal RT2 step, short enough not to hang."""
const _M5_STEP_TIMEOUT_NS = UInt64(10_000_000_000)

# ------------------------------------------------------------------
# present! — CPU HDR blit: HdrColor → host tonemap → image! data
# ------------------------------------------------------------------

"""
    OmniverseMakie.present!(session::ViewportSession, ::Val{:cpu}) -> Nothing

CPU blit: render `HdrColor` (float16) from the open ovrtx stage, tonemap to
`RGBA{N0f8}` via ACES + sRGB + `session.exposure`, and update the GLMakie
image! plot's data Observable (triggering a texture re-upload).

This is the `:cpu` blit strategy.  The `:gpu` strategy (an on-device tonemap +
CUDA→GL copy, no host roundtrip) is defined in the CUDA package extension and
selected via `present!(session, Val(session.blitter))` in `_on_render_tick_impl!`.
"""
function OmniverseMakie.present!(session::ViewportSession, ::Val{:cpu})
    hdr = OV.render_hdr_to_array(session.screen.renderer, session.screen.product;
                                  warmup = session.steps_per_tick,
                                  timeout_ns = _M5_STEP_TIMEOUT_NS)
    frame = tonemap_frame(hdr, session.exposure)   # [H, W] RGBA{N0f8}
    cpu_blit!(session.image_plot, frame)
    return nothing
end

"""
    _on_render_tick_impl!(session::ViewportSession) -> Nothing

Core implementation of the per-frame live camera loop (M5 Task 3).  Called from
`on_render_tick!`, which wraps this in a try/catch to keep the window alive even
on transient render errors.

Pushes live camera/light/plot deltas to the open ovrtx stage, resets RT2
accumulation only when something changed (else keeps accumulating), runs
`steps_per_tick` bounded accumulation steps, and blits the result to the
GLMakie image plot.
"""
function _on_render_tick_impl!(session::ViewportSession)
    screen    = session.screen
    cam_scene = session.cam_scene

    # Push live camera/light/plot deltas; reset RT2 accumulation if anything changed.
    if _sync_and_needs_reset!(screen, cam_scene)
        OV.reset!(screen.renderer)
        session.samples = 0
    end

    # HDR accumulation step + tonemap + blit via the selected strategy (:cpu / :gpu).
    # The CUDA ext's GPU present! falls back to :cpu (sets session.blitter) on setup failure.
    present!(session, Val(session.blitter))
    session.samples += session.steps_per_tick
    return nothing
end

"""
    on_render_tick!(session::ViewportSession) -> Nothing

Per-frame live camera loop (M5 Task 3).  Called on every GLMakie render tick via
the `render_tick` Observable listener registered in `interactive_display`.

Wraps `_on_render_tick_impl!` in a try/catch so that a single bad frame does NOT
crash the window.  Up to `maxlog=5` warnings are printed; the window stays alive.

Must NOT Consume events — the caller wraps this in `on(glscr.render_tick) do _`
and returns `Makie.Consume(false)`.
"""
function OmniverseMakie.on_render_tick!(session::ViewportSession)
    try
        _on_render_tick_impl!(session)
    catch e
        @warn "M5: render-tick frame failed (window kept alive)" exception=(e, catch_backtrace()) maxlog=5
    end
    return nothing
end

end # module OmniverseMakieGLMakieExt
