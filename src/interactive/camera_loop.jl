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
    on_render_tick!(session::ViewportSession) -> Nothing

Per-frame live camera loop (M5 Task 3).  Called on every GLMakie render tick via
the `render_tick` Observable listener registered in `interactive_display`.

Pushes live camera/light/plot deltas to the open ovrtx stage, resets RT2
accumulation only when something changed (else keeps accumulating), runs
`steps_per_tick` bounded accumulation steps, and blits the result to the
GLMakie image plot.

Must NOT Consume events — the caller wraps this in `on(glscr.render_tick) do _`
and returns `Makie.Consume(false)`.
"""
function on_render_tick!(session::ViewportSession)
    screen    = session.screen
    cam_scene = session.cam_scene

    # Push live render-config deltas (each a no-op when unchanged).
    cam_changed   = sync_camera!(screen, cam_scene)
    light_changed = sync_lights!(screen, cam_scene)

    # Pull every plot's :ovrtx_renderobject diff node — mirrors colorbuffer's per-frame
    # logic.  Capture `pending` (a requires_update set before this tick, e.g. from
    # delete!) and clear the flag so it never carries over.
    pending = screen.requires_update
    screen.requires_update = false
    pull_ovrtx_nodes!(screen, cam_scene)
    need_reset = cam_changed || light_changed || screen.requires_update || pending
    screen.requires_update = false

    # Reset RT2 accumulation if anything changed; otherwise keep accumulating.
    if need_reset
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
