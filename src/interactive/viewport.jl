# Viewport session: opens a real GLMakie window displaying the ovrtx RTX frame.
#
# Design (M5 Task 2):
#   - ViewportSession holds both an OmniverseMakie Screen (the open ovrtx stage) and
#     a GLMakie Screen (the host window), plus the image! plot used as the blit target.
#   - interactive_display authors the ovrtx stage, renders one warmup frame, and
#     shows it in a GLMakie window via a full-viewport image! plot.
#   - Task 3 fills the camera loop (tick_listener); it is nothing here.

"""
    mutable struct ViewportSession

Holds all state for an interactive RTX viewport window (M5).

Fields:
  - `screen::Screen`        — open-stage ovrtx Screen (USD stage, renderer, plots)
  - `glscreen`              — GLMakie.Screen (the host window)
  - `image_plot`            — the full-viewport `image!` plot (blit target for cpu_blit!)
  - `cam_scene::Makie.Scene`— the scene whose Camera3D drives the ovrtx view
  - `steps_per_tick::Int`   — number of ovrtx steps per render tick (used in Task 3)
  - `samples::Int`          — warmup frame count (= screen.config.warmup)
  - `tick_listener`         — Task 3 on_render_tick! registration; `nothing` here
"""
mutable struct ViewportSession
    screen::Screen                  # the open-stage ovrtx Screen
    glscreen                        # GLMakie.Screen (the window)
    image_plot                      # the full-viewport image! plot (blit target)
    cam_scene::Makie.Scene          # the scene whose Camera3D drives the view
    steps_per_tick::Int
    samples::Int
    tick_listener                   # set in Task 3; nothing here
end

"""
    interactive_display(fig_or_scene; size=(800,600), steps_per_tick=2) -> ViewportSession

Open an interactive GLMakie window showing the first ovrtx RTX frame of `fig_or_scene`.

Performs the full open-stage authoring sequence (mirrors `colorbuffer`'s open-once
path), renders one warmup frame, then creates a GLMakie window containing a
full-viewport `image!` plot seeded with that frame.  The static first frame is
immediately visible.  Task 3 wires the live camera loop.

The ovrtx render resolution is set to `size` by resizing the input scene's root
before constructing the OmniverseMakie Screen.
"""
function interactive_display(fig_or_scene; size = (800, 600), steps_per_tick = 2)
    scene     = fig_or_scene isa Makie.Figure ? fig_or_scene.scene : fig_or_scene
    cam_scene = something(_scene_for_camera(scene), scene)

    # 1. Resize the root scene to the requested size so the ovrtx renderer renders
    #    at that resolution (Screen constructor uses size(Makie.root(scene))).
    Makie.resize!(Makie.root(cam_scene), size...)

    # 2. Build the open-stage ovrtx Screen at the window size (no size= kwarg;
    #    fb_size comes from root scene after resize above).
    screen = Screen(cam_scene)
    @assert screen.fb_size == size "expected fb_size $(size), got $(screen.fb_size)"

    # 3. Author the root stage ONCE (mirrors colorbuffer's open-once path).
    author_root_from_scene!(screen, cam_scene; resolution = screen.fb_size)
    screen.last_camera = _camera_snapshot(cam_scene)
    screen.last_lights = _lights_snapshot(cam_scene.compute[:lights][])
    screen.authored = true
    Makie.insertplots!(screen, scene)

    # 4. Render the first ovrtx frame with full warmup for a clean initial image.
    frame = OV.render_to_matrix(screen.renderer, screen.product; warmup = screen.config.warmup)

    # 5. GLMakie host window: an Axis filling the figure with one image! of `frame`.
    #    Calling GLMakie.activate!() switches the active backend to GLMakie for the window;
    #    subsequent OmniverseMakie rendering is unaffected (the Screen is already open).
    GLMakie.activate!()
    glf = Figure(; size = size)
    ax  = Makie.Axis(glf[1, 1])
    Makie.hidedecorations!(ax)
    Makie.hidespines!(ax)
    # Apply the same orientation transform as cpu_blit! (reverse(permutedims)) so the
    # initial static frame and future blits both display right-side-up.
    img = image!(ax, reverse(permutedims(frame), dims=2))
    glscr = GLMakie.Screen(); display(glscr, glf)   # real visible window

    return ViewportSession(screen, glscr, img, cam_scene, steps_per_tick, screen.config.warmup, nothing)
end
