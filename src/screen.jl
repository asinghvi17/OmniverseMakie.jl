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
    path2plot::Dict{String,UInt64}           # M6.B: prim_path => objectid(plot); reverse of plot2robj (pick resolution)
    scene2scope::Dict{UInt64,String}         # objectid(scene) => USD scope path (idempotency)
    scene_listeners::Dict{UInt64,Vector}     # objectid(scene) => redraw listeners (M2.4 teardown)
    requires_update::Bool                    # M2.2 diff-node signal (unused in M2.1)
    authored::Bool                           # true once the root stage has been opened
    last_camera::Any                         # snapshot (eye,target,up,fov) last WRITTEN — change detect
    last_lights::Any                         # snapshot of per-light render-state last WRITTEN
    path_resolver::Union{Nothing,OV.PathResolver}  # M6.B: cached path-dictionary resolver (lazy, once per renderer)
    _outline_styled::Bool                    # M6.B: true once the default selection-outline style was installed (once per Screen)
end

# ------------------------------------------------------------------
# Core constructor: build the OV.Renderer and capture scene dimensions.
# ------------------------------------------------------------------

function Screen(scene::Makie.Scene, config::ScreenConfig)
    # M6.B: the selection-outline feature is a creation-time renderer config (it lives
    # on `ScreenConfig` so `resize_viewport!` — which rebuilds the Screen — preserves
    # it).  Default false ⇒ an empty config, identical to the pre-M6.B path.
    renderer = OV.Renderer(; selection_outline = config.selection_outline)
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
        Dict{String,UInt64}(),   # path2plot (reverse of plot2robj)
        Dict{UInt64,String}(),
        Dict{UInt64,Vector}(),
        false,     # requires_update
        false,     # authored
        nothing,   # last_camera
        nothing,   # last_lights
        nothing,   # path_resolver (M6.B: built lazily on first pick)
        false,     # _outline_styled (M6.B: default style installed lazily on first select!)
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

Tear down the screen.  First destroy every plot's persistent hot-path attribute
bindings (M2.4 — `destroy_bindings!` per `OvrtxRObj`), which must happen while the
Renderer is still alive (the bindings are GPU resources owned by it), THEN close the
Renderer.  `render_to_matrix` closes each per-frame `StepResult` internally, so there
are no step handles left to drain here.
"""
function Base.close(s::Screen)
    for robj in values(s.plot2robj)
        destroy_bindings!(robj)
    end
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

Register `scene` with `screen` (idempotently) and return its USD scope path.

M2.3: `scene2scope` is authored authoritatively by `author_root_from_scene!` (it
emits a nested `def Scope "Scene_<id>"` per subscene INTO the root layer and records
`objectid(scene) => scope path`).  So this returns that scene's scope path —
`/World` for the root scene, `/World/Scene_<id>/…` for a subscene — and a plot
inserted under `scene` nests at `<scope>/plot_<id>` (`plot_prim_path`).  A scene not
in the map (e.g. a subscene added live AFTER authoring) falls back to `/World` so its
plots still render flat.  We also reserve a `scene_listeners` slot for M2.4 teardown.
"""
function add_scene!(screen::Screen, scene::Makie.Scene)
    haskey(screen.scene_listeners, objectid(scene)) ||
        (screen.scene_listeners[objectid(scene)] = Any[])
    return get(screen.scene2scope, objectid(scene), "/World")
end

"""
    Base.insert!(screen::Screen, scene::Scene, plot::Plot) -> Screen

Open-stage plot insert.  Registers `scene` (`add_scene!`) first.  Idempotent via
`screen.plot2robj`.  An atomic plot (`isempty(plot.plots)`) is registered via
`register_ovrtx_robj!` (its `:ovrtx_renderobject` diff node builds the USD reference
at the scene's nested scope path `/World/Scene_<id>/plot_<id>` — M2.3 — and records
the `OvrtxRObj`); a composite plot recurses over its children, threading the same
`scene` so all its atomic pieces nest together.

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
# delete! / delete!(scene) / empty! — leak-free imperative teardown (M2.5)
# ------------------------------------------------------------------

# Tear down ONE atomic plot's render object, leaving zero GPU/USD/graph residue.
# Order mirrors `close(Screen)` (screen.jl): destroy the persistent hot-path
# bindings FIRST (they are GPU resources owned by the live Renderer — M2.4), THEN
# remove the USD reference (this closes the M1.6 per-reference handle leak), THEN
# drop `plot2robj`.  Finally drop the `:ovrtx_renderobject` diff node: per Spike B,
# `empty!(graph)` does NOT clear nodes, so without an explicit `delete!` the node
# both leaks and would re-fire on a later re-add.  Deleting the node is gated on
# presence so a node-less plot (empty geometry, or an unknown type) is a no-op.
function _delete_atomic_plot!(screen::Screen, plot::Makie.AbstractPlot)
    id   = objectid(plot)
    robj = get(screen.plot2robj, id, nothing)
    if robj !== nothing
        destroy_bindings!(robj)
        screen.renderer.alive && OV.remove_usd!(screen.renderer, robj.usd_handle)
        delete!(screen.plot2robj, id)
        delete!(screen.path2plot, robj.prim_path)   # M6.B: drop the reverse entry in lockstep
    end
    attr = plot.attributes
    haskey(attr, :ovrtx_renderobject) && delete!(attr, :ovrtx_renderobject)
    return
end

"""
    delete!(screen::Screen, scene::Scene, plot::AbstractPlot) -> Screen

Imperatively remove `plot` from the OPEN stage, leaving zero residual GPU bindings,
USD reference, or diff node.  A composite plot is flattened to its atomic children
(`plot.plots`, exactly like `insert!`); each atomic is torn down via
`_delete_atomic_plot!`.  Sets `requires_update` so the next `colorbuffer` issues one
`OV.reset!` (the live stage no longer contains the prim).

Typed signature (Spike B): an UNtyped `delete!(screen, scene, plot)` would be
ambiguous with Makie's no-op `delete!(::MakieScreen, ::Scene, ::AbstractPlot)`
fallback; `::Screen` is strictly more specific, so this wins dispatch cleanly.
"""
function Base.delete!(screen::Screen, scene::Makie.Scene, plot::Makie.AbstractPlot)
    if isempty(plot.plots)
        _delete_atomic_plot!(screen, plot)
    else
        foreach(p -> delete!(screen, scene, p), plot.plots)
    end
    screen.requires_update = true
    return screen
end

"""
    delete!(screen::Screen, scene::Scene) -> Screen

Remove an entire (sub)scene from the OPEN stage: recurse `scene.children`, delete
every plot in `scene.plots`, detach every redraw listener (`Observables.off` on each
`scene_listeners[objectid(scene)]` entry), then drop the scene's `scene_listeners` and
`scene2scope` entries.  The listener loop is vestigial TODAY (camera/lights use
snapshot-compare, not observable listeners, so the vector is empty) but is kept
correct for when listeners are wired.

Scope-prim design call (M2.5): a subscene's `def Scope` is authored INTO the root
layer by `author_root_from_scene!` (`scene_scopes_usda`), NOT as a removable
`add_usd_reference!`, so there is no `remove_usd!` handle for it.  Removing all the
scene's plot references + bindings + nodes is the leak-free core; the leftover EMPTY
`def Scope` prim renders nothing and holds no GPU resource, so it is left in place.  A
structural re-author to physically drop the scope is the heavier alternative and is
unnecessary for leak-freedom.
"""
function Base.delete!(screen::Screen, scene::Makie.Scene)
    foreach(child -> delete!(screen, child), scene.children)
    for plot in scene.plots
        delete!(screen, scene, plot)
    end
    id = objectid(scene)
    for l in get(screen.scene_listeners, id, ())
        Makie.Observables.off(l)
    end
    delete!(screen.scene_listeners, id)
    delete!(screen.scene2scope, id)
    screen.requires_update = true
    return screen
end

"""
    empty!(screen::Screen) -> Screen

Tear down every plot and subscene the screen has authored, leaving `plot2robj`,
`scene2scope`, and `scene_listeners` all empty and every persistent binding destroyed
— on the OPEN stage (no re-author).  Walks `screen.scene` via `delete!(screen, scene)`
to reach each plot OBJECT (needed to drop its `:ovrtx_renderobject` node — `plot2robj`
alone is keyed by `objectid` and cannot recover it), then sweeps any orphaned render
object (a plot no longer in the tree) and force-clears the three registries as a final
guarantee.

Structural-re-open carry (M2.2 Minor #5): `empty!` does targeted `remove_usd!` per
plot — it does NOT re-open the stage — so the `:ovrtx_renderobject` nodes are never
left stale against a wiped stage.  A plot added afterwards builds a fresh reference on
the still-open stage and renders correctly; the `requires_update` set here makes the
next `colorbuffer` reset accumulation once.
"""
function Base.empty!(screen::Screen)
    screen.scene === nothing || delete!(screen, screen.scene)
    # Belt-and-suspenders: tear down any robj still cached (a plot no longer reachable
    # from screen.scene) so no GPU binding or USD handle leaks, then guarantee empty.
    for (id, robj) in collect(screen.plot2robj)
        destroy_bindings!(robj)
        screen.renderer.alive && OV.remove_usd!(screen.renderer, robj.usd_handle)
        delete!(screen.plot2robj, id)
        delete!(screen.path2plot, robj.prim_path)   # M6.B: drop the reverse entry in lockstep
    end
    empty!(screen.plot2robj)
    empty!(screen.path2plot)                         # M6.B: reverse map cleared alongside plot2robj
    empty!(screen.scene2scope)
    empty!(screen.scene_listeners)
    screen.requires_update = true
    return screen
end

# ------------------------------------------------------------------
# Shared authoring + sync helpers (colorbuffer + M5 interactive paths)
# ------------------------------------------------------------------

# Author the open ovrtx stage from a scene + seed the camera/light snapshots so the next
# sync is a no-op, then add every plot.  Shared by colorbuffer (offscreen) + the M5
# interactive paths so the snapshot-before-sync invariant cannot drift between them.
function _author_screen!(screen::Screen, cam_scene, plot_scene)
    author_root_from_scene!(screen, cam_scene; resolution = screen.fb_size)
    screen.last_camera = _camera_snapshot(cam_scene)
    screen.last_lights = _lights_snapshot(cam_scene.compute[:lights][])
    screen.authored    = true
    Makie.insertplots!(screen, plot_scene)
    return screen
end

# Push live camera/light/plot deltas to the open stage; return whether RT2 must restart this
# frame (camera/light moved, or a plot diff node flipped requires_update).  Clears
# requires_update on both the pre-pull capture and the post-pull read.  Shared by colorbuffer
# and the M5 render tick so the subtle two-write reset logic stays in one place.
function _sync_and_needs_reset!(screen::Screen, cam_scene)::Bool
    cam_changed   = sync_camera!(screen, cam_scene)
    light_changed = sync_lights!(screen, cam_scene)
    pending = screen.requires_update
    screen.requires_update = false
    pull_ovrtx_nodes!(screen, screen.scene)
    need_reset = cam_changed || light_changed || screen.requires_update || pending
    screen.requires_update = false
    return need_reset
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
        # Author the root ONCE (camera + lights baked, snapshots seeded, plots added).
        _author_screen!(screen, cam_scene, scene)
    end

    # Push live render-config deltas; reset RT2 if anything changed.
    need_reset = _sync_and_needs_reset!(screen, cam_scene)
    need_reset && OV.reset!(screen.renderer)

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
# M6.B — Makie pick protocol over the native ray-query AOV pick
#
# `pick_hit` enqueues a 1-pixel ovrtx pick at a Makie pixel, steps the renderer, decodes
# the hit, resolves its prim path to the owning plot, and computes the Makie element index.
# `Makie.pick`/`pick_closest`/`pick_sorted` are the standard backend overrides on top of it
# (so DataInspector and `pick(fig, xy)` compose).  Picking core has NO GLMakie/CUDA dep.
# ------------------------------------------------------------------

# Pick step timeout (ns).  The pick consumes one render step (after which RT2 accumulation
# restarts — expected); 10 s is generous for the 1-pixel query + decode.
const _PICK_TIMEOUT_NS = UInt64(10_000_000_000)

# Map a Makie pixel (BOTTOM-left origin, Float64, over `fb_size`) to an ovrtx RenderProduct
# pixel (TOP-left origin, Int, left/top-inclusive).
#
# VERIFIED EMPIRICALLY (M6.B Task 3, off-center marker): x maps straight through; y is
# FLIPPED (`H - y`).  A marker placed off the vertical center was located three ways that
# agree — `Makie.project` (Makie bottom-left), the LdrColor framebuffer row (top-left), and
# the pick FFI's reported pixel — and the ovrtx row that returned the marker was
# `H - round(y)`, never `round(y)` (that row was background).  Getting this flip wrong would
# silently pick the wrong pixel, so it is also locked by an off-center assertion in
# `test/m6b_pick_test.jl`.
function _to_ovrtx_pixel(xy, fb_size)
    W, H = fb_size
    px = clamp(round(Int, xy[1]), 0, W - 1)
    py = clamp(H - round(Int, xy[2]), 0, H - 1)   # ← verified y-flip
    return (px, py)
end

# instance_id (ovrtx `geometryInstanceId`) → Makie element index.  Scatter/MeshScatter map
# the 0-based instance to a 1-based point index; every other plot kind (Mesh / Lines /
# LineSegments / Surface / materialized-merged scatter) is plot-level → index 0.
#
# VERIFIED LIMITATION (M6.B Task 3): ovrtx's USD PointInstancer pick collapses to the
# prototype — EVERY instance reports `geometryInstanceId == 0` (and `worldPosition ==
# (0,0,0)`), even with `OVRTX_PICK_FLAG_INCLUDE_TRACKED_INFO` — so the per-point index is
# NOT recoverable today and a multi-point scatter pick yields index 1 for any point.  The
# `Int(instance_id)+1` formula is kept because it is exact for a single-point marker (the
# acceptance case) and forward-compatible should a future ovrtx surface real instance ids.
function _element_index(plot, instance_id::UInt64)::Int
    return (plot isa Makie.Scatter || plot isa Makie.MeshScatter) ? Int(instance_id) + 1 : 0
end

# Build + cache the renderer's PathResolver ONCE per Screen.  The path dictionary is a LIVE
# renderer handle (its context pointer stays valid for the renderer's life and reflects
# stage edits), so one resolver serves every pick — not rebuilt per query.  A rebuilt Screen
# (resize) starts with `path_resolver === nothing` and lazily builds its own.
function path_resolver_for(screen::Screen)
    pr = screen.path_resolver
    pr === nothing || return pr
    pr = OV.path_resolver(screen.renderer)
    screen.path_resolver = pr
    return pr
end

# Resolve a hit prim path to the owning plot's `objectid` by walking UP the path until a
# `path2plot` entry matches.  A Mesh hit resolves to the plot prim exactly (`/World/plot_<id>`),
# but a Scatter/MeshScatter hit resolves to the PointInstancer PROTOTYPE child
# (`/World/plot_<id>/proto`, verified) — stripping trailing components recovers the plot.
# `path2plot` holds only plot prims (never scopes/scenes/cameras), and plot prims are siblings
# (never nested in one another), so the FIRST ancestor that matches is unambiguously the plot.
# Also handles a subscene plot (`/World/Scene_<sid>/plot_<pid>/…`).
function _path_to_oid(screen::Screen, path::AbstractString)
    isempty(path) && return nothing
    p = String(path)
    while true
        oid = get(screen.path2plot, p, nothing)
        oid === nothing || return oid
        i = findlast('/', p)
        (i === nothing || i <= 1) && return nothing   # reached the root with no plot ancestor
        p = p[1:(i - 1)]
    end
end

# objectid(plot) → the `Plot` object, via the `plot` reference stored on its `OvrtxRObj`
# (set at the `plot2robj` insert sites — so this rides that map's lifecycle with no extra
# registry to keep in lockstep).  `nothing` if the objectid is unknown or unset.
function _plot_for_objectid(screen::Screen, oid::UInt64)
    robj = get(screen.plot2robj, oid, nothing)
    return robj === nothing ? nothing : robj.plot
end

"""
    pick_hit(screen::Screen, xy) -> Union{Nothing, NamedTuple{(:plot,:index,:world_position,:normal)}}

Enqueue a 1-pixel native ray-query pick at Makie pixel `xy` (bottom-left origin, over
`screen.fb_size`), step the renderer once, and decode the closest hit:

- map `xy` to the ovrtx RenderProduct pixel (`_to_ovrtx_pixel` — verified y-flip),
- `OV.enqueue_pick_query` → `OV.step!` → `OV.read_pick_hit`,
- resolve the hit's `primpath_id` to a prim path (cached `PathResolver`), walk it to the
  owning plot's `objectid` (`_path_to_oid`), then to the `Plot` object (`_plot_for_objectid`),
- compute the Makie element index (`_element_index`).

Returns `nothing` over background (no hit), or when the hit prim is not a registered plot
(camera / light / looks).  The pick consumes one step, after which RT2 accumulation restarts
(expected).
"""
function pick_hit(screen::Screen, xy)
    isopen(screen) || return nothing
    px, py = _to_ovrtx_pixel(xy, screen.fb_size)
    OV.enqueue_pick_query(screen.renderer, screen.product, (px, py, px + 1, py + 1))
    sr = OV.step!(screen.renderer, screen.product; timeout_ns = _PICK_TIMEOUT_NS)
    hits = try
        OV.read_pick_hit(sr)
    finally
        close(sr)
    end
    isempty(hits) && return nothing
    hit  = hits[1]
    pr   = path_resolver_for(screen)
    path = OV.resolve_prim_path(pr, hit.primpath_id)
    oid  = _path_to_oid(screen, path)
    oid === nothing && return nothing                 # camera/light/looks prim, not a plot
    plot = _plot_for_objectid(screen, oid)
    plot === nothing && return nothing
    return (plot = plot, index = _element_index(plot, hit.instance_id),
            world_position = hit.world_position, normal = hit.normal)
end

"""
    Makie.pick(scene::Scene, screen::Screen, xy::Vec{2,Float64}) -> (plot_or_nothing, index)

Backend pick: the plot + element index under Makie pixel `xy`, or `(nothing, 0)` over
background.  Delegates to `pick_hit`.
"""
function Makie.pick(::Makie.Scene, screen::Screen, xy::Makie.Vec{2,Float64})
    h = pick_hit(screen, xy)
    return h === nothing ? (nothing, 0) : (h.plot, h.index)
end

# The native ray-query already returns the closest hit at the pixel, so `range` is advisory:
# pick the pixel directly (more specific than Makie's `pick_closest(::SceneLike, screen, …)`,
# which would otherwise fan out a Rect2i region pick we don't implement).
Makie.pick_closest(scene::Makie.Scene, screen::Screen, xy, range) =
    Makie.pick(scene, screen, Makie.Vec{2,Float64}(xy))

# One native hit at `xy`, wrapped as the distance-sorted `(plot,index)` list DataInspector
# expects (empty over background).
function Makie.pick_sorted(scene::Makie.Scene, screen::Screen, xy, range)
    h = pick_hit(screen, Makie.Vec{2,Float64}(xy))
    return h === nothing ? Tuple{Makie.AbstractPlot,Int}[] : [(h.plot, h.index)]
end

# ------------------------------------------------------------------
# M6.B — select! / clear_selection! selection-outline API
#
# `select!(screen, plot; group)` assigns a plot's prim to a selection-outline group (default 1)
# so the RTX outline pipeline draws a high-contrast ring around it; `clear_selection!` removes it
# (group 0).  The drawn highlight needs the creation-time `selection_outline=true` renderer config
# (`ScreenConfig.selection_outline`); on a Screen built without it, `select!` warns ONCE and is a
# no-op (pick DATA still works regardless — only the highlight needs the flag).  A default orange
# style is installed ONCE per Screen the first time a selection is made.
# ------------------------------------------------------------------

# Default high-contrast outline style: orange edge (matches the ovrtx C reference
# `DEFAULT_SELECTION_STYLE`, test_picking_selection.cpp:54) + transparent fill (the default
# fill mode is EDGE_ONLY, so only the outline color is drawn).
const _OUTLINE_ORANGE = (1.0f0, 0.6f0, 0.0f0, 1.0f0)

# Install the default group-1 orange outline style ONCE per Screen (idempotent via
# `_outline_styled`).  Returns whether this Screen can draw an outline at all — i.e. it was
# created with `selection_outline=true` — which is the select!/clear gate.
function _ensure_outline_style!(screen::Screen)
    screen.config.selection_outline || return false
    if !screen._outline_styled                      # one-time per Screen
        OV.set_selection_group_styles!(screen.renderer, UInt8[0x01],
            [LibOVRTX.ovrtx_selection_group_style_t(_OUTLINE_ORANGE, (0f0, 0f0, 0f0, 0f0))])
        screen._outline_styled = true
    end
    return true
end

"""
    select!(screen::Screen, plot; group::UInt8 = 0x01)

Highlight `plot`'s prim with a selection outline (group `group`, default 1), installing the
default orange outline style once per Screen.  No-op (warns once, `maxlog=1`) on a Screen built
without `selection_outline=true`, or when `plot` is not a registered render object.  Returns
`nothing`.
"""
function select!(screen::Screen, plot; group::UInt8 = 0x01)
    if !_ensure_outline_style!(screen)
        @warn "select!: Screen built without selection_outline=true; no highlight drawn" maxlog=1
        return nothing
    end
    robj = get(screen.plot2robj, objectid(plot), nothing)
    robj === nothing && return nothing
    OV.set_selection_outline_group!(screen.renderer, [robj.prim_path], UInt8[group])
    return nothing
end

"""
    clear_selection!(screen::Screen[, plot])

Remove the selection outline from `plot` (group 0), or — with no `plot` — from every
currently-tracked plot.  Returns `nothing`.
"""
function clear_selection!(screen::Screen, plot)
    robj = get(screen.plot2robj, objectid(plot), nothing)
    robj === nothing && return nothing
    OV.set_selection_outline_group!(screen.renderer, [robj.prim_path], UInt8[0x00])
    return nothing
end
clear_selection!(screen::Screen) =                # clear ALL currently-tracked plots
    for robj in values(screen.plot2robj)
        OV.set_selection_outline_group!(screen.renderer, [robj.prim_path], UInt8[0x00])
    end

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
