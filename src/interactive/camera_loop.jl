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

    # Bounded accumulation step (timeout_ns prevents hanging on a slow step).
    frame = OV.render_to_matrix(screen.renderer, screen.product;
                                warmup = session.steps_per_tick,
                                timeout_ns = _M5_STEP_TIMEOUT_NS)
    session.samples += session.steps_per_tick

    # CPU blit: update the GLMakie image! plot's data Observable → texture re-upload.
    cpu_blit!(session.image_plot, frame)
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
function on_render_tick!(session::ViewportSession)
    try
        _on_render_tick_impl!(session)
    catch e
        @warn "M5: render-tick frame failed (window kept alive)" exception=(e, catch_backtrace()) maxlog=5
    end
    return nothing
end
