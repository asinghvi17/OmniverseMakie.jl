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
end

# Input events forwarded from the display window into the 3-D camera scene so the
# existing Camera3D orbits/zooms in response to the window's mouse/keyboard.
const _M5_FORWARDED_EVENTS = (:mousebutton, :mouseposition, :scroll, :keyboardbutton)

"""
    interactive_display(fig_or_scene; size=(800,600), steps_per_tick=2) -> ViewportSession

Open an interactive, orbit-able GLMakie window showing the live ovrtx RTX render of
`fig_or_scene`.

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
function interactive_display(fig_or_scene; size = (800, 600), steps_per_tick = 2)
    scene     = fig_or_scene isa Makie.Figure ? fig_or_scene.scene : fig_or_scene
    cam_scene = something(_scene_for_camera(scene), scene)

    # 1. Resize the root scene to the requested size so the ovrtx renderer renders at
    #    that resolution (the Screen constructor uses size(Makie.root(scene))).
    Makie.resize!(Makie.root(cam_scene), size...)

    # 2. Build + author the open-stage ovrtx Screen (mirrors colorbuffer's open-once path).
    screen = Screen(cam_scene)
    screen.fb_size == size ||
        error("interactive_display: expected fb_size $(size), got $(screen.fb_size)")
    author_root_from_scene!(screen, cam_scene; resolution = screen.fb_size)
    screen.last_camera = _camera_snapshot(cam_scene)
    screen.last_lights = _lights_snapshot(cam_scene.compute[:lights][])
    screen.authored = true
    Makie.insertplots!(screen, scene)

    # 3. First ovrtx frame (full warmup for a clean initial image).
    frame = OV.render_to_matrix(screen.renderer, screen.product; warmup = screen.config.warmup)
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
    img = image!(glscene, 0 .. W, 0 .. H, reverse(permutedims(frame), dims = 2); interpolate = false)
    glscr = GLMakie.Screen()
    display(glscr, glscene)

    # 5. Forward the window's input into cam_scene so its Camera3D orbits/zooms live
    #    (campixel! has no 3-D camera of its own).  One-way glscene→cam_scene; the
    #    listeners observe (never Consume), so GLMakie still processes events normally.
    input_listeners = Any[]
    for f in _M5_FORWARDED_EVENTS
        push!(input_listeners, on(getproperty(glscene.events, f)) do v
            getproperty(cam_scene.events, f)[] = v
            return Makie.Consume(false)
        end)
    end

    session = ViewportSession(screen, glscr, glscene, img, cam_scene,
                              steps_per_tick, screen.config.warmup,
                              nothing, input_listeners, nothing)

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
    author_root_from_scene!(new_screen, cam_scene; resolution = new_screen.fb_size)
    new_screen.last_camera = _camera_snapshot(cam_scene)
    new_screen.last_lights = _lights_snapshot(cam_scene.compute[:lights][])
    new_screen.authored    = true
    Makie.insertplots!(new_screen, root_scene)

    # 3. Render a warmup frame at the new size.
    frame = OV.render_to_matrix(new_screen.renderer, new_screen.product;
                                warmup = new_screen.config.warmup)

    # 4. Swap in the new Screen FIRST (session stays self-consistent at every point —
    #    defensive even if a future resize path is not synchronous), then free the OLD
    #    ovrtx renderer (avoid a GPU leak).  Reset the accumulation counter for the new frame.
    old_screen      = session.screen
    session.screen  = new_screen
    session.samples = new_screen.config.warmup
    Base.close(old_screen)

    # 5. Update the displayed image: delete the old image! (wrong size) and add a new
    #    one spanning the full new viewport.  The campixel! coordinate system already
    #    covers 0..W, 0..H after the GLMakie window resize.
    delete!(session.glscene, session.image_plot)
    new_img = image!(session.glscene, 0 .. W, 0 .. H,
                     reverse(permutedims(frame), dims = 2); interpolate = false)
    session.image_plot = new_img

    return nothing
end
