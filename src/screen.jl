# Screen — the Makie backend screen that owns an OV.Renderer.
#
# Open-stage model: the USD stage is authored ONCE (lazily, on the first
# `colorbuffer`) and stays open across frames.  Later `colorbuffer` calls push
# MINIMAL live edits — camera + light writes via `sync_camera!`/`sync_lights!`, and
# per-plot attribute writes via the M2.2 `:ovrtx_renderobject` diff nodes
# (`pull_ovrtx_nodes!` → `push_to_ovrtx!`) — instead of re-authoring the whole stage.

mutable struct Screen <: Makie.MakieScreen
    renderer::OV.Renderer
    fb_size::Tuple{Int,Int}
    product::String                          # RenderProduct prim path; authored in M1.2
    config::ScreenConfig
    scene::Union{Nothing,Makie.Scene}
    plot2robj::Dict{UInt64,OvrtxRObj}        # objectid(plot) => render object (path + handle)
    scene2scope::Dict{UInt64,String}         # objectid(scene) => USD scope path (idempotency)
    scene_listeners::Dict{UInt64,Vector}     # objectid(scene) => redraw listeners (M2.4 teardown)
    requires_update::Bool                    # M2.2 diff-node signal (unused in M2.1)
    authored::Bool                           # true once the root stage has been opened
    last_camera::Any                         # snapshot (eye,target,up,fov) last WRITTEN — change detect
    last_lights::Any                         # snapshot of per-light render-state last WRITTEN
end

# ------------------------------------------------------------------
# Core constructor: build the OV.Renderer and capture scene dimensions.
# ------------------------------------------------------------------

function Screen(scene::Makie.Scene, config::ScreenConfig)
    renderer = OV.Renderer()
    # Render at the ROOT scene size.  `Makie.colorbuffer(scene)` crops a NON-root
    # (e.g. LScene) scene out of the full figure via `get_sub_picture`, indexing
    # with the scene's viewport in ROOT pixel coordinates — so the rendered image
    # must be root-sized for that crop to be in-bounds.  For a root scene,
    # `Makie.root(scene) === scene`, so this is unchanged.
    fb_size  = size(Makie.root(scene))   # (w, h)
    product  = "/Render/OVMakie/RenderProduct"
    return Screen(
        renderer,
        fb_size,
        product,
        config,
        scene,
        Dict{UInt64,OvrtxRObj}(),
        Dict{UInt64,String}(),
        Dict{UInt64,Vector}(),
        false,     # requires_update
        false,     # authored
        nothing,   # last_camera
        nothing,   # last_lights
    )
end

# kwargs entry-point: merge caller overrides with the registered defaults, then
# delegate to the core constructor.
function Screen(scene::Makie.Scene; screen_config...)
    config = Makie.merge_screen_config(ScreenConfig, Dict{Symbol,Any}(screen_config))
    return Screen(scene, config)
end

# Offscreen pass-throughs — image path / video recording.
# These signatures are required by Makie.getscreen(backend, scene, config, args...) dispatch.

function Screen(
        scene::Makie.Scene, config::ScreenConfig,
        ::Union{Nothing,String,IO}, ::MIME,
    )
    return Screen(scene, config)
end

function Screen(scene::Makie.Scene, config::ScreenConfig, ::Makie.ImageStorageFormat)
    return Screen(scene, config)
end

# ------------------------------------------------------------------
# Standard MakieScreen interface
# ------------------------------------------------------------------

Base.size(s::Screen)   = s.fb_size
Base.isopen(s::Screen) = s.renderer.alive

"""
    close(screen::Screen)

Close the renderer.  In M2.1 there are no per-frame `StepResult` handles to drain
(`render_to_matrix` closes each step internally) and no persistent bindings yet
(`OV.destroy!` of `OvrtxRObj.bindings` is an M2.3 concern), so teardown is just
closing the renderer.
"""
function Base.close(s::Screen)
    close(s.renderer)
    return
end

# ------------------------------------------------------------------
# Scene tree → camera scene
# ------------------------------------------------------------------

# Depth-first search for the first descendant scene with a 3-D camera controller.
# `colorbuffer(ax.scene)` passes the LScene scene directly (it IS a Camera3D scene);
# `save(fig)` passes the figure ROOT (a 2-D PixelCamera) whose 3-D content lives in a
# descendant scene, so we walk the tree to find the camera to author/sync from.
function _scene_for_camera(scene::Makie.Scene)
    Makie.cameracontrols(scene) isa Makie.Camera3D && return scene
    for child in scene.children
        s = _scene_for_camera(child)
        s === nothing || return s
    end
    return nothing
end

# ------------------------------------------------------------------
# add_scene! / insert! / insertplots! — imperative open-stage authoring
# ------------------------------------------------------------------

"""
    add_scene!(screen::Screen, scene::Scene) -> String

Register `scene` with `screen` (idempotently) and return its scope path.

M2.1 is bookkeeping-only: plots attach directly at `/World/plot_<objectid(plot)>`
(children of the root `/World` Xform), so a per-scene USD `Scope` prim is not
required for them to render — authoring one would be an orphan prim on the open
stage.  We record the scene (so `insert!`/`insertplots!` are idempotent) and
reserve a `scene_listeners` slot for M2.4 teardown.  Camera/light changes are
detected by snapshot-compare in `colorbuffer` (not observable listeners), so no
redraw listeners are registered here.
"""
function add_scene!(screen::Screen, scene::Makie.Scene)
    return get!(screen.scene2scope, objectid(scene)) do
        screen.scene_listeners[objectid(scene)] = Any[]
        "/World"   # logical scope: where this scene's plot references live
    end
end

"""
    Base.insert!(screen::Screen, scene::Scene, plot::Plot) -> Screen

Open-stage plot insert.  Registers `scene` (`add_scene!`) first.  Idempotent via
`screen.plot2robj`.  An atomic plot (`isempty(plot.plots)`) is registered via
`register_ovrtx_robj!` (its `:ovrtx_renderobject` diff node builds the USD reference
at `/World/plot_<id>` and records the `OvrtxRObj`); a composite plot recurses over
its children.

Before the stage is authored (open), this is a no-op: the plot is already in
`scene.plots` and will be added by `insertplots!` at the first `colorbuffer`
(adding a USD reference requires an open stage).  After the stage is open, a live
`plot!` on the displayed scene is authored immediately (Makie calls this via
`push!(scene, plot)` → `insert!(screen, scene, plot)`).
"""
function Base.insert!(screen::Screen, scene::Makie.Scene, plot::Makie.Plot)
    add_scene!(screen, scene)
    screen.authored || return screen   # defer to insertplots! at first colorbuffer
    haskey(screen.plot2robj, objectid(plot)) && return screen
    if isempty(plot.plots)
        # M2.2: register the :ovrtx_renderobject diff node, which builds the USD
        # reference (author_usd_prim!) on first resolve and records plot2robj.  The
        # prim path is owned by register_ovrtx_robj!/plot_prim_path (single source).
        register_ovrtx_robj!(screen, scene, plot)
    else
        foreach(p -> insert!(screen, scene, p), plot.plots)
    end
    return screen
end

"""
    Makie.insertplots!(screen::Screen, scene::Scene) -> Screen

Add every plot of `scene` (and its child scenes) to the open stage.  Registers
each scene via `add_scene!`, then `insert!`s each plot, then recurses into
`scene.children`.  Called once from `colorbuffer` right after the root is authored.
"""
function Makie.insertplots!(screen::Screen, scene::Makie.Scene)
    add_scene!(screen, scene)
    for plot in scene.plots
        insert!(screen, scene, plot)
    end
    foreach(child -> Makie.insertplots!(screen, child), scene.children)
    return screen
end

# ------------------------------------------------------------------
# colorbuffer — open-once + live render-config sync + RT2 render
# ------------------------------------------------------------------

"""
    Makie.colorbuffer(screen::Screen; kw...) -> Matrix{RGBA{N0f8}}

Render `screen`'s scene and return the LdrColor framebuffer.

Open-stage model: on the FIRST call, author the root stage ONCE
(`author_root_from_scene!`, which bakes the camera + lights) and add every plot's
USD reference (`insertplots!` → `register_ovrtx_robj!`, building each plot's
`:ovrtx_renderobject` diff node); the camera/light snapshots are seeded so the
immediately-following sync is a no-op.  On EVERY call, `sync_camera!`/`sync_lights!`
push minimal live writes for any pose/intensity/color/transform change, and
`pull_ovrtx_nodes!` resolves every plot's diff node — pushing one minimal C write
per changed plot attribute (`push_to_ovrtx!`).  If anything was written (camera,
light, or geometry), `OV.reset!` restarts RT2 accumulation once before rendering.  A
static scene therefore keeps accumulating across frames (no re-open, no reset); a
camera orbit reframes via `write_xform!` and a `plot.color`/`translate!` edit writes
displayColor/`omni:xform` in place (NOT a re-author).

The matrix is returned EXACTLY as `OV.render_to_matrix` produces it — top-left
origin (right-side-up, verified by `test/m1_orientation_test.jl`), 4-channel
`RGBA{N0f8}`, with NO flip / alpha-drop / conversion.
"""
function Makie.colorbuffer(screen::Screen; kw...)
    scene = screen.scene
    scene === nothing && error("OmniverseMakie.colorbuffer: screen has no scene")
    cam_scene = something(_scene_for_camera(scene), scene)

    if !screen.authored
        # 1. Author the root ONCE (camera + lights baked) — this opens the stage.
        author_root_from_scene!(screen, cam_scene; resolution = screen.fb_size)
        # 2. Seed snapshots to the just-baked state so the first sync is a no-op.
        screen.last_camera = _camera_snapshot(cam_scene)
        screen.last_lights = _lights_snapshot(cam_scene.compute[:lights][])
        screen.authored = true
        # 3. Add every plot's USD reference on the fresh stage.
        Makie.insertplots!(screen, scene)
    end

    # Push live render-config deltas (each a no-op when unchanged).
    cam_changed   = sync_camera!(screen, cam_scene)
    light_changed = sync_lights!(screen, cam_scene)

    # Pull every plot's :ovrtx_renderobject diff node (M2.2): a clean graph is a
    # no-op; any changed attribute pushes one minimal C write and flips
    # `requires_update`.  Fold that into the single per-frame RT2 reset.
    screen.requires_update = false
    pull_ovrtx_nodes!(screen, scene)
    (cam_changed || light_changed || screen.requires_update) && OV.reset!(screen.renderer)

    return OV.render_to_matrix(screen.renderer, screen.product; warmup = screen.config.warmup)
end

# PNG showability: `save(fig, "x.png")` / `Base.show(io, MIME"image/png", fig)` and
# Jupyter auto-display both check `backend_showable` → Makie's generic `backend_show`
# → our `colorbuffer`.
Makie.backend_showable(::Type{Screen}, ::MIME"image/png")  = true
# JPEG showability: `Base.showable(MIME"image/jpeg", fig) = true` so Jupyter and
# `Base.show(io, MIME"image/jpeg", fig)` work.
Makie.backend_showable(::Type{Screen}, ::MIME"image/jpeg") = true

# ------------------------------------------------------------------
# activate! — register OmniverseMakie as the current Makie backend
# ------------------------------------------------------------------

"""
    OmniverseMakie.activate!(; screen_config...)

Set OmniverseMakie as the active Makie backend and optionally update screen
configuration.  Accepted keys match `ScreenConfig` field names: `mode`,
`samples`, `warmup`, `max_bounces`.
"""
function activate!(; screen_config...)
    Makie.set_screen_config!(OmniverseMakie, screen_config)
    Makie.set_active_backend!(OmniverseMakie)
    return
end
