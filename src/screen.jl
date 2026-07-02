# Screen — the Makie backend screen that owns an OV.Renderer.
#
# Open-stage model: the USD stage is authored ONCE (lazily, on the first `colorbuffer`) and
# stays open across frames.  Later `colorbuffer` calls push MINIMAL live edits — camera + light
# writes (sync_camera!/sync_lights!) and per-plot attribute writes via the M2.2
# `:ovrtx_renderobject` diff nodes (pull_ovrtx_nodes! → push_to_ovrtx!) — not a re-author.

mutable struct Screen <: Makie.MakieScreen
    renderer::OV.Renderer
    fb_size::Tuple{Int,Int}
    product::String                          # RenderProduct prim path; authored in M1.2
    config::ScreenConfig
    scene::Union{Nothing,Makie.Scene}
    plot2robj::Dict{UInt64,OvrtxRObj}        # objectid(plot) => render object (path + handle)
    path2plot::Dict{String,UInt64}           # M6.B: prim_path => objectid(plot); reverse of plot2robj (picks)
    scene2scope::Dict{UInt64,String}         # objectid(scene) => USD scope path (idempotency)
    scene_listeners::Dict{UInt64,Vector}     # objectid(scene) => redraw listeners (M2.4 teardown)
    requires_update::Bool                    # M2.2 diff-node signal (unused in M2.1)
    authored::Bool                           # true once the root stage has been opened
    last_camera::Union{Nothing,NamedTuple{(:eye,:target,:up,:fov),Tuple{Vec3d,Vec3d,Vec3d,Float64}}}  # camera pose+FOV last WRITTEN — change detect (_camera_snapshot)
    last_lights::Union{Nothing,LightSnapshot}  # per-light render-state last WRITTEN — change detect (_lights_snapshot)
    path_resolver::Union{Nothing,OV.PathResolver}  # M6.B: cached path resolver (lazy, once/renderer)
    _outline_styled::Bool                    # M6.B: true once default outline style installed (once/Screen)
    structural_dirty::Bool                   # accumulate mode: a USD reference was added/removed → reset once
    preroll_done::Bool                       # accumulate mode: first-frame warm-up already folded in
    # usdplot bind_usd! writes, coalesced per target (key = (full_prim_path, attr|nothing)); flushed
    # once per frame in `_sync_and_needs_reset!`.  Value is `(USDBinding, value)` — typed `Any` because
    # `USDBinding` (usdplot.jl) is defined AFTER this struct.
    pending_usd_writes::Dict{Tuple{String,Union{Nothing,String}},Any}
end

# ------------------------------------------------------------------
# Core constructor: build the OV.Renderer and capture scene dimensions.
# ------------------------------------------------------------------

function Screen(scene::Makie.Scene, config::ScreenConfig;
               fb_size::Tuple{Int,Int} = size(Makie.root(scene)))
    # M6.B: selection-outline is a creation-time renderer config (on `ScreenConfig` so
    # `resize_viewport!`, which rebuilds the Screen, preserves it).  Default false ⇒ empty
    # config, identical to the pre-M6.B path.
    renderer = OV.Renderer(; selection_outline = config.selection_outline)
    # Default fb_size is the ROOT scene size: `Makie.colorbuffer(scene)` crops a NON-root (e.g.
    # LScene) scene out of the full figure via `get_sub_picture`, indexing with the scene's
    # viewport in ROOT pixel coords — so the image must be root-sized for that crop to be
    # in-bounds.  For a root scene `Makie.root(scene) === scene`, so this is unchanged.  The
    # `fb_size` override lets `replace_scene!` render an EMBEDDED sub-scene at its OWN viewport
    # size (blitted into that scene's rectangle, no crop).
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
        false,     # structural_dirty (accumulate mode: no composition change yet)
        false,     # preroll_done (accumulate mode: first-frame warm-up not yet folded in)
        Dict{Tuple{String,Union{Nothing,String}},Any}(),  # pending_usd_writes (usdplot bind_usd!)
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

Tear down the screen: FIRST destroy every plot's persistent hot-path bindings
(`destroy_bindings!` per `OvrtxRObj`) — which must happen while the Renderer is still alive (the
bindings are GPU resources it owns) — THEN close the Renderer.  `render_to_matrix` closes each
per-frame `StepResult` internally, so no step handles remain to drain.
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

# DFS for the first descendant scene with a 3-D camera controller.  `colorbuffer(ax.scene)`
# passes the LScene scene directly (it IS a Camera3D scene); `save(fig)` passes the figure ROOT
# (a 2-D PixelCamera) whose 3-D content lives in a descendant, so we walk the tree to find it.
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

Register `scene` (idempotently) and return its USD scope path.  `scene2scope` is authored by
`author_root_from_scene!` (a nested `def Scope "Scene_<id>"` per subscene), so this returns
`/World` for the root scene or `/World/Scene_<id>/…` for a subscene; a plot under `scene` nests
at `<scope>/plot_<id>` (`plot_prim_path`).  A scene not in the map (subscene added live AFTER
authoring) falls back to `/World` (renders flat).  Also reserves a `scene_listeners` slot (M2.4).
"""
function add_scene!(screen::Screen, scene::Makie.Scene)
    haskey(screen.scene_listeners, objectid(scene)) ||
        (screen.scene_listeners[objectid(scene)] = Any[])
    return get(screen.scene2scope, objectid(scene), "/World")
end

"""
    Base.insert!(screen::Screen, scene::Scene, plot::Plot) -> Screen

Open-stage plot insert; registers `scene` first, idempotent via `screen.plot2robj`.  An atomic
plot (`isempty(plot.plots)`) goes through `register_ovrtx_robj!` (its diff node builds the USD
reference at the nested scope path `/World/Scene_<id>/plot_<id>`); a composite recurses over its
children, threading the same `scene` so its atomic pieces nest together.

Before the stage is authored, this is a NO-OP: the plot is already in `scene.plots` and gets
added by `insertplots!` at the first `colorbuffer` (a USD reference needs an open stage).  After
the stage is open, a live `plot!` is authored immediately (Makie calls `push!(scene, plot)` →
`insert!(screen, scene, plot)`).
"""
function Base.insert!(screen::Screen, scene::Makie.Scene, plot::Makie.Plot)
    add_scene!(screen, scene)
    screen.authored || return screen   # defer to insertplots! at first colorbuffer
    haskey(screen.plot2robj, objectid(plot)) && return screen
    if isempty(plot.plots)
        # M2.2: register the :ovrtx_renderobject diff node, which builds the USD reference
        # (author_usd_prim!) on first resolve and records plot2robj.  The prim path is owned
        # by register_ovrtx_robj!/plot_prim_path (single source).
        register_ovrtx_robj!(screen, scene, plot)
    else
        foreach(p -> insert!(screen, scene, p), plot.plots)
    end
    return screen
end

"""
    Makie.insertplots!(screen::Screen, scene::Scene) -> Screen

Add every plot of `scene` + child scenes to the open stage: `add_scene!` each scene, `insert!`
each plot, recurse `scene.children`.  Called once from `colorbuffer` after the root is authored.
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

# Remove `robj`'s USD reference IFF the renderer is live AND the handle is valid.  A volume whose
# live-data reload FAILED at `add_usd_reference!` (old layer already removed) marks
# `robj.meta[:usd_handle_valid] = false` (see `reload_volume_data!`); some ovrtx builds throw an
# `OVRTXError` on `remove_usd!` of that stale handle, which would escape delete!/empty!/close and
# abort teardown mid-way.  The flag guard makes teardown safe; non-volume plots never set it →
# default valid → behaviour unchanged.
function _teardown_usd_reference!(screen::Screen, robj::OvrtxRObj)
    if screen.renderer.alive && get(robj.meta, :usd_handle_valid, true)
        OV.remove_usd!(screen.renderer, robj.usd_handle)
        _note_composition_change!(screen)           # reference removed → drop resolver + flag reset
    end
    return nothing
end

# Tear down ONE atomic plot's render object, leaving zero GPU/USD/graph residue.  Order mirrors
# `close(Screen)`: destroy the persistent hot-path bindings FIRST (GPU resources owned by the live
# Renderer), THEN remove the USD reference (closes the M1.6 per-reference handle leak), THEN drop
# `plot2robj`.  Finally drop the `:ovrtx_renderobject` node: per Spike B, `empty!(graph)` does NOT
# clear nodes, so without an explicit `delete!` it leaks and would re-fire on a later re-add.
# Gated on presence so a node-less plot (empty geometry / unknown type) is a no-op.
function _delete_atomic_plot!(screen::Screen, plot::Makie.AbstractPlot)
    id   = objectid(plot)
    robj = get(screen.plot2robj, id, nothing)
    if robj !== nothing
        destroy_bindings!(robj)
        _teardown_usd_reference!(screen, robj)      # guarded: a stale volume handle can't abort teardown
        delete!(screen.plot2robj, id)
        delete!(screen.path2plot, robj.prim_path)   # M6.B: drop the reverse entry in lockstep
    end
    attr = plot.attributes
    haskey(attr, :ovrtx_renderobject) && delete!(attr, :ovrtx_renderobject)
    return
end

"""
    delete!(screen::Screen, scene::Scene, plot::AbstractPlot) -> Screen

Imperatively remove `plot` from the OPEN stage, leaving zero residual GPU bindings, USD
reference, or diff node.  A composite is flattened to its atomic children (like `insert!`); each
atomic is torn down via `_delete_atomic_plot!`.  Sets `requires_update` so the next `colorbuffer`
issues one `OV.reset!`.

Typed signature (Spike B): an UNtyped `delete!` would be ambiguous with Makie's no-op
`delete!(::MakieScreen, ::Scene, ::AbstractPlot)` fallback; `::Screen` is strictly more specific,
so this wins dispatch cleanly.
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

Remove an entire (sub)scene from the OPEN stage: recurse `scene.children`, delete every plot,
detach every redraw listener (`Observables.off` on each `scene_listeners` entry), then drop the
scene's `scene_listeners` + `scene2scope` entries.  The listener loop is vestigial TODAY
(camera/lights use snapshot-compare, so the vector is empty) but kept correct for when wired.

Scope-prim design call (M2.5): a subscene's `def Scope` is authored INTO the root layer, NOT as a
removable `add_usd_reference!`, so it has no `remove_usd!` handle.  Removing the scene's plot
references + bindings + nodes is the leak-free core; the leftover EMPTY `def Scope` renders
nothing and holds no GPU resource, so it is left in place (a structural re-author to drop it is
unnecessary for leak-freedom).
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

Tear down every plot + subscene the screen authored, leaving `plot2robj`, `scene2scope`, and
`scene_listeners` empty and every persistent binding destroyed — on the OPEN stage (no re-author).
Walks `screen.scene` via `delete!(screen, scene)` to reach each plot OBJECT (needed to drop its
`:ovrtx_renderobject` node — `plot2robj` is keyed by `objectid` and can't recover it), then
sweeps any orphaned render object and force-clears the three registries as a final guarantee.

`empty!` does targeted `remove_usd!` per plot — it does NOT re-open the stage — so the nodes are
never left stale against a wiped stage.  A plot added afterwards builds a fresh reference on the
still-open stage; the `requires_update` set here makes the next `colorbuffer` reset once.
"""
function Base.empty!(screen::Screen)
    screen.scene === nothing || delete!(screen, screen.scene)
    # Belt-and-suspenders: tear down any robj still cached (a plot no longer reachable
    # from screen.scene) so no GPU binding or USD handle leaks, then guarantee empty.
    for (id, robj) in collect(screen.plot2robj)
        destroy_bindings!(robj)
        _teardown_usd_reference!(screen, robj)      # guarded: a stale volume handle can't abort teardown
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

# Author the open ovrtx stage from a scene, seed the camera/light snapshots so the next sync is a
# no-op, then add every plot.  Shared by colorbuffer + the M5 interactive paths so the
# snapshot-before-sync invariant cannot drift between them.
function _author_screen!(screen::Screen, cam_scene, plot_scene)
    author_root_from_scene!(screen, cam_scene; resolution = screen.fb_size)
    screen.last_camera = _camera_snapshot(cam_scene)
    screen.last_lights = _lights_snapshot(cam_scene.compute[:lights][])
    screen.authored    = true
    Makie.insertplots!(screen, plot_scene)
    return screen
end

# Push live camera/light/plot deltas to the open stage; return whether RT2 must restart this frame
# (camera/light moved, or a plot diff node flipped requires_update).  Clears requires_update on both
# the pre-pull capture and post-pull read.  Shared by colorbuffer + the M5 tick (one place for the
# two-write reset logic).
function _sync_and_needs_reset!(screen::Screen, cam_scene)::Bool
    cam_changed   = sync_camera!(screen, cam_scene)
    light_changed = sync_lights!(screen, cam_scene)
    pending = screen.requires_update
    screen.requires_update = false
    pull_ovrtx_nodes!(screen, screen.scene)
    # usdplot bind_usd! writes (coalesced per target) flush here, BEFORE the accumulate gate: the
    # OR into need_reset makes default mode reconverge (like a camera/light edit), while the gate
    # below drops it in accumulate mode (bound writes are non-structural — RT2 reprojection keeps
    # the history).
    usd_wrote = _flush_pending_usd_writes!(screen)
    need_reset = cam_changed || light_changed || screen.requires_update || pending || usd_wrote
    screen.requires_update = false
    # Accumulate-across-frames: keep RT2 history across frames (realtime-style), so camera/light/
    # attribute changes do NOT reset — only a STRUCTURAL change (a USD reference added/removed:
    # plot insert/delete/empty→fill/volume reload, flagged via `_note_composition_change!`) does,
    # because RT2 reprojection has no history for a prim that just appeared or vanished.
    screen.config.accumulate_across_frames && (need_reset = screen.structural_dirty)
    screen.structural_dirty = false
    return need_reset
end

# ------------------------------------------------------------------
# colorbuffer — open-once + live render-config sync + RT2 render
# ------------------------------------------------------------------

"""
    Makie.colorbuffer(screen::Screen; kw...) -> Matrix{RGBA{N0f8}}

Render `screen`'s scene and return the LdrColor framebuffer.

Open-stage model: on the FIRST call, author the root ONCE (`author_root_from_scene!`, baking
camera + lights) and add every plot's USD reference (`insertplots!` → `register_ovrtx_robj!`);
the camera/light snapshots are seeded so the immediately-following sync is a no-op.  On EVERY
call, `sync_camera!`/`sync_lights!` push minimal writes for any pose/intensity/color/transform
change and `pull_ovrtx_nodes!` resolves every diff node (one minimal C write per changed
attribute).  If anything was written, `OV.reset!` restarts RT2 accumulation once before rendering
— so a static scene keeps accumulating (no re-open/reset), a camera orbit reframes via
`write_xform!`, and a `plot.color`/`translate!` edit writes displayColor/`omni:xform` in place.

The matrix is returned EXACTLY as `OV.render_to_matrix` produces it — top-left origin
(right-side-up, verified by `test/m1_orientation_test.jl`), 4-channel `RGBA{N0f8}`, NO
flip/alpha-drop/conversion.
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

    # Accumulate mode: fold the one-time pre-roll into the FIRST frame's warm-up so frame 1 lands
    # converged (a cold RT2 = noisy) instead of paying a separate discarded readback.
    warmup = screen.config.warmup
    if screen.config.accumulate_across_frames && !screen.preroll_done
        warmup += screen.config.accumulation_preroll
        screen.preroll_done = true
    end
    return OV.render_to_matrix(screen.renderer, screen.product; warmup = warmup)
end

# PNG showability: `save(fig, "x.png")` / `show(io, MIME"image/png", fig)` and Jupyter
# auto-display all check `backend_showable` → Makie's `backend_show` → our `colorbuffer`.
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

# Map a Makie pixel (BOTTOM-left origin, Float64, over `fb_size`) to an ovrtx RenderProduct pixel
# (TOP-left origin, Int, left/top-inclusive).
#
# VERIFIED EMPIRICALLY (off-center marker): x maps straight through; y is FLIPPED (`H - y`).  The
# ovrtx row returning the marker was `H - round(y)`, never `round(y)` (that row was background),
# agreeing across `Makie.project`, the framebuffer row, and the pick FFI.  Getting this flip wrong
# silently picks the wrong pixel — locked by an off-center assertion in test/m6b_pick_test.jl.
function _to_ovrtx_pixel(xy, fb_size)
    W, H = fb_size
    px = clamp(round(Int, xy[1]), 0, W - 1)
    py = clamp(H - round(Int, xy[2]), 0, H - 1)   # ← verified y-flip
    return (px, py)
end

# instance_id (ovrtx `geometryInstanceId`) → Makie element index.  Scatter/MeshScatter map the
# 0-based instance to a 1-based point index; every other plot kind is plot-level → index 0.
#
# VERIFIED LIMITATION: ovrtx's USD PointInstancer pick collapses to the prototype — EVERY instance
# reports `geometryInstanceId == 0` (and `worldPosition == (0,0,0)`), even with
# `OVRTX_PICK_FLAG_INCLUDE_TRACKED_INFO` — so the per-point index is NOT recoverable today (a
# multi-point scatter pick yields index 1 for any point).  `Int(instance_id)+1` is kept because it
# is exact for a single-point marker (the acceptance case) and forward-compatible.
function _element_index(plot, instance_id::UInt64)::Int
    return (plot isa Makie.Scatter || plot isa Makie.MeshScatter) ? Int(instance_id) + 1 : 0
end

# Lazily build + cache the renderer's PathResolver for picks.  COMPOSITION-SCOPED: the cached
# resolver captures the path dictionary as of the CURRENT stage composition, so it is dropped
# (`_invalidate_path_resolver!`) at every composition change — plot insert/delete, empty!, volume
# reload — and rebuilt here on the next pick.  A rebuilt Screen (resize) also starts at
# `path_resolver === nothing` and builds its own.
function path_resolver_for(screen::Screen)
    pr = screen.path_resolver
    pr === nothing || return pr
    pr = OV.path_resolver(screen.renderer)
    screen.path_resolver = pr
    return pr
end

# Drop the cached PathResolver: the dictionary it captured is valid only for the stage
# composition at build time, so any add_usd_reference!/remove_usd! (plot insert/delete, empty!,
# volume reload) must invalidate it.  Rebuilt lazily on the next pick (path_resolver_for).
_invalidate_path_resolver!(screen::Screen) = (screen.path_resolver = nothing; nothing)

# Every stage-composition change (a USD reference added OR removed) calls THIS one helper, so the
# two composition-triggered concerns can never drift: (1) drop the pick resolver cache, (2) flag
# `structural_dirty` so accumulate-across-frames mode resets RT2 once (its reprojection has no
# history for a prim that just appeared/disappeared).  Call sites: `_register_robj_maps!`
# (compute.jl — insert / Surface / empty→fill late build), `_teardown_usd_reference!` (delete /
# empty!), `reload_volume_data!` (volume.jl — remove+re-reference).
_note_composition_change!(screen::Screen) =
    (screen.path_resolver = nothing; screen.structural_dirty = true; nothing)

"""
    reset_accumulation!(screen::Screen) -> Nothing

Force one RT2 accumulation reset (restart the path-tracer fresh).  Only useful with
`accumulate_across_frames = true`, where per-frame resets are suppressed: call this if a change
the composition funnel does not cover (or a fast camera move) leaves visible ghosting.  A no-op
kind of insurance in the default per-frame-reconverge mode.  Not exported.
"""
reset_accumulation!(screen::Screen) = OV.reset!(screen.renderer)

# Resolve a hit prim path to the owning plot's `objectid` by walking UP the path until a
# `path2plot` entry matches.  A Mesh hit hits the plot prim exactly; a Scatter/MeshScatter hit
# hits the PointInstancer PROTOTYPE child (`/World/plot_<id>/proto`) — stripping trailing
# components recovers the plot.  `path2plot` holds only plot prims (never scopes/scenes/cameras)
# and plot prims are siblings, so the FIRST matching ancestor is unambiguously the plot.  Also
# handles a subscene plot (`/World/Scene_<sid>/plot_<pid>/…`).
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

# objectid(plot) → the `Plot` object via the `plot` reference on its `OvrtxRObj` (set at the
# `plot2robj` insert sites, so it rides that map's lifecycle).  `nothing` if unknown or unset.
function _plot_for_objectid(screen::Screen, oid::UInt64)
    robj = get(screen.plot2robj, oid, nothing)
    return robj === nothing ? nothing : robj.plot
end

"""
    pick_hit(screen::Screen, xy) -> Union{Nothing, NamedTuple{(:plot,:index,:world_position,:normal)}}

Enqueue a 1-pixel native ray-query pick at Makie pixel `xy` (bottom-left origin, over
`screen.fb_size`), step the renderer once, and decode the closest hit: map `xy` to the ovrtx
pixel (`_to_ovrtx_pixel`, verified y-flip), `enqueue_pick_query` → `step!` → `read_pick_hit`,
resolve `primpath_id` to a prim path (cached `PathResolver`), walk it to the owning plot's
`objectid` (`_path_to_oid`) then to the `Plot` (`_plot_for_objectid`), and compute the element
index (`_element_index`).

Returns `nothing` over background, or when the hit prim is not a registered plot (camera/light/
looks).  The pick consumes one step, after which RT2 accumulation restarts (expected).

!!! note "world_position / normal are not populated by this ovrtx"
    The returned `world_position` and `normal` read `(0,0,0)` — ovrtx does not populate the
    pick-hit world-position AOV for our render product (a verified renderer constraint, not a
    decode bug). Do not rely on them; `Makie.pick` uses only `plot` + `index`. The `index` is
    exact for a single-point scatter marker and plot-level (`0`) otherwise (ovrtx collapses a
    `UsdGeomPointInstancer` pick to the prototype). Both upgrade for free if a future ovrtx
    surfaces this data.
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
# `select!(screen, plot; group)` assigns a plot's prim to a selection-outline group (default 1) so
# the RTX outline pipeline draws a ring around it; `clear_selection!` removes it (group 0).  The
# highlight needs the creation-time `selection_outline=true` config; without it `select!` warns ONCE
# and is a no-op (pick DATA still works — only the highlight needs the flag).  A default orange
# style is installed ONCE per Screen on the first selection.
# ------------------------------------------------------------------

# Default outline style: orange edge (matches the ovrtx C reference `DEFAULT_SELECTION_STYLE`,
# test_picking_selection.cpp:54) + transparent fill (default fill mode is EDGE_ONLY, so only the
# outline colour draws).
const _OUTLINE_ORANGE = (1.0f0, 0.6f0, 0.0f0, 1.0f0)

# Install the default group-1 orange outline style ONCE per Screen (idempotent via
# `_outline_styled`).  Returns whether this Screen can draw an outline at all (created with
# `selection_outline=true`) — the select!/clear gate.
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

Set OmniverseMakie as the active Makie backend and optionally update screen configuration.
Accepted keys match `ScreenConfig` field names: `mode`, `samples`, `warmup`, `max_bounces`,
`selection_outline`.
"""
function activate!(; screen_config...)
    Makie.set_screen_config!(OmniverseMakie, screen_config)
    Makie.set_active_backend!(OmniverseMakie)
    return
end
