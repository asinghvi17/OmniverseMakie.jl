# Screen — the Makie backend screen that owns an OV.Renderer.
# Mirrors the GLMakie/RPRMakie pattern: core constructor + offscreen pass-throughs.

mutable struct Screen <: Makie.MakieScreen
    renderer::OV.Renderer
    fb_size::Tuple{Int,Int}
    product::String                      # RenderProduct prim path; authored in M1.2
    config::ScreenConfig
    scene::Union{Nothing,Makie.Scene}
    plot2usd::Dict{UInt64,UInt64}        # objectid(plot) => ovrtx_usd_handle_t (M1.5 fills)
    open_results::Vector{OV.StepResult}  # closed before renderer (teardown order)
    setup::Bool                          # lazy USD-author flag; M1.2 sets to true
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
        Dict{UInt64,UInt64}(),
        OV.StepResult[],
        false,
    )
end

# kwargs entry-point: merge caller overrides with the registered defaults, then
# delegate to the core constructor.
function Screen(scene::Makie.Scene; screen_config...)
    config = Makie.merge_screen_config(ScreenConfig, Dict{Symbol,Any}(screen_config))
    return Screen(scene, config)
end

# Offscreen pass-throughs — image path / video recording (M1.5/M1.6 fill the real impl).
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

Close all open `StepResult` handles BEFORE closing the renderer — deterministic
GPU teardown ensures render buffers are released before the device is torn down.
"""
function Base.close(s::Screen)
    for sr in s.open_results
        close(sr)
    end
    empty!(s.open_results)
    close(s.renderer)
    return
end

# ------------------------------------------------------------------
# Scene → USD setup (lazy; runs once on the first colorbuffer)
# ------------------------------------------------------------------

# Depth-first search for the first descendant scene with a 3-D camera controller.
# `colorbuffer(ax.scene)` passes the LScene scene directly (it IS a Camera3D scene);
# `save(fig)` passes the figure ROOT (a 2-D PixelCamera) whose 3-D content lives in a
# descendant scene, so we walk the tree to find the camera to author from.
function _scene_for_camera(scene::Makie.Scene)
    Makie.cameracontrols(scene) isa Makie.Camera3D && return scene
    for child in scene.children
        s = _scene_for_camera(child)
        s === nothing || return s
    end
    return nothing
end

"""
    setup_scene!(screen::Screen) -> Screen

Lazily author the full USD stage for `screen.scene` and insert every plot's USD
geometry.  Runs once, on the first `colorbuffer`.

ORDER IS CRITICAL: `author_root_from_scene!` re-opens the stage (wiping any added
references), so the camera + lights root MUST be authored BEFORE any plot reference
is added.  The camera pose is read from the first descendant `Camera3D` scene; the
stage is rendered at `screen.fb_size` (the root scene size).
"""
function setup_scene!(screen::Screen)
    scene = screen.scene
    scene === nothing && error("OmniverseMakie.setup_scene!: screen has no scene")
    cam_scene = something(_scene_for_camera(scene), scene)
    # 1. author root (camera + lights) in ONE open_usd_string! — this wipes refs.
    author_root_from_scene!(screen, cam_scene; resolution = screen.fb_size)
    # 2. THEN add each plot's USD reference (recurse over scene.plots + children).
    Makie.insertplots!(screen, scene)
    screen.setup = true
    return screen
end

"""
    Base.insert!(screen::Screen, scene::Scene, plot::Plot) -> Screen

Insert a plot's USD geometry into `screen`, recursing build-once into composite
plots.  An atomic plot (`isempty(plot.plots)`) is authored via `to_ovrtx_object`
and recorded in `screen.plot2usd`; a composite plot recurses over its children.
Idempotent: a plot already in `screen.plot2usd` is skipped.  `to_ovrtx_object`
returns `nothing` for unsupported plot types, which are then silently skipped.
"""
function Base.insert!(screen::Screen, scene::Makie.Scene, plot::Makie.Plot)
    haskey(screen.plot2usd, objectid(plot)) && return screen
    if isempty(plot.plots)
        h = to_ovrtx_object(screen, scene, plot)
        h === nothing || (screen.plot2usd[objectid(plot)] = h)
    else
        foreach(p -> insert!(screen, scene, p), plot.plots)
    end
    return screen
end

# ------------------------------------------------------------------
# colorbuffer — lazy setup + RT2 render → Matrix{RGBA{N0f8}}
# ------------------------------------------------------------------

"""
    Makie.colorbuffer(screen::Screen; kw...) -> Matrix{RGBA{N0f8}}

Render `screen`'s scene and return the LdrColor framebuffer.

On the first call this lazily authors the USD stage and inserts all plots via
`setup_scene!`.  The matrix is returned EXACTLY as `OV.render_to_matrix` produces
it — top-left origin (right-side-up, per the M0 readback), 4-channel `RGBA{N0f8}`,
with NO flip / alpha-drop / conversion.

This is the kwargs form (it tolerates the `figure` keyword that Makie's
`backend_show` → `save` path passes).  `ImageStorageFormat` dispatch
(`JuliaNative` / `GLNative`) is handled by Makie's generic
`colorbuffer(::MakieScreen, ::ImageStorageFormat)`, which calls this method.
"""
function Makie.colorbuffer(screen::Screen; kw...)
    screen.setup || setup_scene!(screen)
    return OV.render_to_matrix(screen.renderer, screen.product; warmup = screen.config.warmup)
end

# Minimal PNG showability so `save(fig, "x.png")` / `show(io, MIME"image/png", fig)`
# route through Makie's generic backend_show → our colorbuffer.  M1.6 generalizes
# (jpeg + record + the offscreen constructors).
Makie.backend_showable(::Type{Screen}, ::MIME"image/png") = true

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
