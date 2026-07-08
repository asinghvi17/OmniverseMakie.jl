# Screen — the Makie backend screen that owns an OV.Renderer.
#
# Open-stage model: the USD stage is authored once (lazily, on the first
# `colorbuffer`) and stays open across frames.  Later `colorbuffer` calls
# push minimal live edits — camera + light writes plus per-plot attribute
# writes via the `:ovrtx_renderobject` diff nodes — not a re-author.

mutable struct Screen <: Makie.MakieScreen
    renderer::OV.Renderer
    fb_size::Tuple{Int,Int}
    product::String                          # RenderProduct prim path
    config::ScreenConfig
    scene::Union{Nothing,Makie.Scene}
    plot2robj::Dict{UInt64,OvrtxRObj}        # objectid(plot) => render object
    path2plot::Dict{String,UInt64}           # prim_path => objectid(plot)
    scene2scope::Dict{UInt64,String}         # objectid(scene) => USD scope path
    requires_update::Bool                    # diff-node change signal
    authored::Bool                           # true once the root stage is open
    # camera pose+FOV last written — change detect (_camera_snapshot)
    last_camera::Union{Nothing,NamedTuple{(:eye,:target,:up,:fov),Tuple{Vec3d,Vec3d,Vec3d,Float64}}}
    # per-light render-state last written — change detect (_lights_snapshot)
    last_lights::Union{Nothing,LightSnapshot}
    path_resolver::Union{Nothing,OV.PathResolver}  # cached pick resolver (lazy)
    _outline_styled::Bool                    # default outline style installed
    structural_dirty::Bool                   # USD ref add/remove → reset once
    preroll_done::Bool                       # first-frame warmup folded in
    # usdplot bind_usd! writes, coalesced per (full_prim_path, attr|nothing);
    # flushed once per frame in `_sync_and_needs_reset!`.  Values are
    # `(USDBinding, value)` — `Any`: `USDBinding` is defined after this struct.
    pending_usd_writes::Dict{Tuple{String,Union{Nothing,String}},Any}
    # Environment-light (IBL) record: removable dome-layer handle +
    # screen-owned temp texture; `nothing` until an EnvironmentLight / push.
    env_light::Union{Nothing,EnvLightState}
    # Serializes the render-tick read-and-clear of requires_update /
    # structural_dirty / pending_usd_writes against user-task listener SETs, so
    # a threaded renderloop cannot drop an edit or corrupt the pending Dict.
    edit_lock::ReentrantLock
end

# ------------------------------------------------------------------
# Core constructor: build the OV.Renderer and capture scene dimensions.
# ------------------------------------------------------------------

# Sensor detection for the constructor's motion-BVH auto-enable.  The
# generic is `false` for every plot; sensors.jl specializes it for
# Lidar/Radar.
_is_sensor_plot(::Any) = false
# A composite recipe can nest a sensor among its child plots (`insert!`
# recurses `plot.plots`); mirror that recursion so auto-detection sees a
# wrapped sensor, not just a top-level one.
_plot_contains_sensor(plot::Makie.AbstractPlot) =
    _is_sensor_plot(plot) || any(_plot_contains_sensor, plot.plots)
function _scene_contains_sensors(scene::Makie.Scene)
    any(_plot_contains_sensor, scene.plots) && return true
    return any(_scene_contains_sensors, scene.children)
end

function Screen(scene::Makie.Scene, config::ScreenConfig;
               fb_size::Tuple{Int,Int} = size(Makie.root(scene)))
    # Both renderer flags are creation-frozen: selection-outline lives on
    # ScreenConfig so resize_viewport!'s rebuilt Screen preserves it; motion
    # BVH (needed for correct MOVING-object lidar/radar returns) auto-enables
    # when the scene already holds sensor plots, or is forced by
    # config.sensors (the post-display lidar!/radar! workflow).
    renderer = OV.Renderer(; selection_outline = config.selection_outline,
                             enable_motion_bvh = config.sensors || _scene_contains_sensors(scene))
    # Default fb_size = the ROOT scene size: Makie.colorbuffer(scene) crops a
    # non-root scene out of the full figure via get_sub_picture in root pixel
    # coords, so the image must be root-sized for the crop to be in-bounds.
    # The override lets replace_scene! render an embedded sub-scene at its
    # own viewport size (blitted into that scene's rectangle, no crop).
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
        false,     # requires_update
        false,     # authored
        nothing,   # last_camera
        nothing,   # last_lights
        nothing,   # path_resolver (built lazily on first pick)
        false,     # _outline_styled (style installed on first select!)
        false,     # structural_dirty (no composition change yet)
        false,     # preroll_done (first-frame warmup not folded in)
        Dict{Tuple{String,Union{Nothing,String}},Any}(),  # pending_usd_writes
        nothing,   # env_light (IBL dome state)
        ReentrantLock(),  # edit_lock (render-tick vs listener handoff)
    )
end

# kwargs entry-point: merge caller overrides with the registered defaults, then
# delegate to the core constructor.
function Screen(scene::Makie.Scene; screen_config...)
    config = Makie.merge_screen_config(ScreenConfig, Dict{Symbol,Any}(screen_config))
    return Screen(scene, config)
end

# Offscreen pass-throughs (image path / video recording): signatures
# required by Makie.getscreen(backend, scene, config, args...) dispatch.

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

Tear down the screen: first destroy every plot's persistent hot-path
bindings (`destroy_bindings!` per `OvrtxRObj`) while the Renderer is still
alive (the bindings are GPU resources it owns), then close the Renderer.
`render_to_matrix` closes each per-frame `StepResult` internally, so no step
handles remain to drain.
"""
function Base.close(s::Screen)
    for robj in values(s.plot2robj)
        destroy_bindings!(robj)
    end
    _destroy_env_light!(s)       # GC the env dome's temp texture (idempotent)
    close(s.renderer)
    return
end

# ------------------------------------------------------------------
# Scene tree → camera scene
# ------------------------------------------------------------------

# DFS for the first descendant scene with a 3-D camera:
# colorbuffer(ax.scene) passes a Camera3D scene directly; save(fig) passes
# the figure ROOT (2-D PixelCamera) whose 3-D content lives in a descendant.
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

Register `scene` (idempotently) and return its USD scope path: `/World` for
the root scene or `/World/Scene_<id>/…` for a subscene (`scene2scope` is
authored by `author_root_from_scene!`); a plot under `scene` nests at
`<scope>/plot_<id>` (`plot_prim_path`).  A scene not in the map (added live
AFTER authoring) falls back to `/World` (renders flat).
"""
function add_scene!(screen::Screen, scene::Makie.Scene)
    return get(screen.scene2scope, objectid(scene), "/World")
end

"""
    Base.insert!(screen::Screen, scene::Scene, plot::Plot) -> Screen

Open-stage plot insert; registers `scene` first, idempotent via
`screen.plot2robj`.  An atomic plot (`isempty(plot.plots)`) goes through
`register_ovrtx_robj!` (its diff node builds the USD reference at the nested
scope path); a composite recurses over its children with the same `scene`.

Before the stage is authored this is a NO-OP: the plot is already in
`scene.plots` and gets added by `insertplots!` at the first `colorbuffer` (a
USD reference needs an open stage).  After the stage is open, a live `plot!`
is authored immediately (Makie calls `push!(scene, plot)` → `insert!`).
"""
function Base.insert!(screen::Screen, scene::Makie.Scene, plot::Makie.Plot)
    add_scene!(screen, scene)
    screen.authored || return screen   # deferred to insertplots!
    haskey(screen.plot2robj, objectid(plot)) && return screen
    if isempty(plot.plots)
        # Register the :ovrtx_renderobject diff node; it builds the USD
        # reference on first resolve and records plot2robj.  The prim path is
        # owned by register_ovrtx_robj!/plot_prim_path (single source).
        register_ovrtx_robj!(screen, scene, plot)
    else
        foreach(p -> insert!(screen, scene, p), plot.plots)
    end
    return screen
end

"""
    Makie.insertplots!(screen::Screen, scene::Scene) -> Screen

Add every plot of `scene` + child scenes to the open stage: `add_scene!`
each scene, `insert!` each plot, recurse `scene.children`.  Called once from
`colorbuffer` after the root is authored.
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
# delete! / delete!(scene) / empty! — leak-free imperative teardown
# ------------------------------------------------------------------

# Remove `robj`'s USD reference IFF the renderer is live AND the handle is
# valid: a volume whose live-data reload failed at `add_usd_reference!` marks
# `robj.meta[:usd_handle_valid] = false` (see `reload_volume_data!`), and
# some ovrtx builds throw an `OVRTXError` on `remove_usd!` of that stale
# handle, which would abort teardown mid-way.  Non-volume plots never set
# the flag.
function _teardown_usd_reference!(screen::Screen, robj::OvrtxRObj)
    if screen.renderer.alive && get(robj.meta, :usd_handle_valid, true)
        OV.remove_usd!(screen.renderer, robj.usd_handle)
        _note_composition_change!(screen)   # reference removed
    end
    return nothing
end

# Tear down ONE atomic plot's render object, leaving zero GPU/USD/graph
# residue.  Order mirrors close(Screen): destroy the persistent bindings
# first (GPU resources owned by the live Renderer), then remove the USD
# reference, then drop plot2robj, then delete the :ovrtx_renderobject node —
# empty!(graph) does NOT clear nodes; an undeleted node leaks and re-fires
# on a later re-add.  Gated so a node-less plot is a no-op.
function _delete_atomic_plot!(screen::Screen, plot::Makie.AbstractPlot)
    id   = objectid(plot)
    robj = get(screen.plot2robj, id, nothing)
    if robj !== nothing
        destroy_bindings!(robj)
        _teardown_usd_reference!(screen, robj)      # stale-volume-handle safe
        delete!(screen.plot2robj, id)
        delete!(screen.path2plot, robj.prim_path)   # reverse entry in lockstep
    end
    attr = plot.attributes
    haskey(attr, :ovrtx_renderobject) && delete!(attr, :ovrtx_renderobject)
    return
end

"""
    delete!(screen::Screen, scene::Scene, plot::AbstractPlot) -> Screen

Imperatively remove `plot` from the OPEN stage, leaving zero residual GPU
bindings, USD reference, or diff node.  A composite is flattened to its
atomic children (like `insert!`).  Sets `requires_update` so the next
`colorbuffer` issues one `OV.reset!`.

The typed signature keeps dispatch away from Makie's no-op
`delete!(::MakieScreen, ::Scene, ::AbstractPlot)` fallback (`::Screen` is
strictly more specific).
"""
function Base.delete!(screen::Screen, scene::Makie.Scene, plot::Makie.AbstractPlot)
    if isempty(plot.plots)
        _delete_atomic_plot!(screen, plot)
    else
        foreach(p -> delete!(screen, scene, p), plot.plots)
    end
    _set_requires_update!(screen)
    return screen
end

"""
    delete!(screen::Screen, scene::Scene) -> Screen

Remove an entire (sub)scene from the OPEN stage: recurse `scene.children`,
delete every plot, then drop the scene's `scene2scope` entry.

A subscene's `def Scope` is authored INTO the root layer, not as a removable
reference, so it has no `remove_usd!` handle; the leftover empty scope
renders nothing and holds no GPU resource, so it is left in place.
"""
function Base.delete!(screen::Screen, scene::Makie.Scene)
    foreach(child -> delete!(screen, child), scene.children)
    for plot in scene.plots
        delete!(screen, scene, plot)
    end
    delete!(screen.scene2scope, objectid(scene))
    _set_requires_update!(screen)
    return screen
end

"""
    empty!(screen::Screen) -> Screen

Tear down every plot + subscene the screen authored, leaving `plot2robj`
and `scene2scope` empty and every persistent binding destroyed — on the OPEN
stage (no re-author).  Walks `screen.scene` via
`delete!(screen, scene)` to reach each plot OBJECT (needed to drop its
`:ovrtx_renderobject` node — `plot2robj` is keyed by `objectid` and can't
recover it), then sweeps any orphaned render object and force-clears the
registries as a final guarantee.

`empty!` does targeted `remove_usd!` per plot — it does NOT re-open the
stage — so the nodes are never stale against a wiped stage.  A plot added
afterwards builds a fresh reference on the still-open stage; the
`requires_update` set here makes the next `colorbuffer` reset once.
"""
function Base.empty!(screen::Screen)
    screen.scene === nothing || delete!(screen, screen.scene)
    # Belt-and-suspenders: tear down any robj still cached (a plot no longer
    # reachable from screen.scene), then guarantee empty.
    for (id, robj) in collect(screen.plot2robj)
        destroy_bindings!(robj)
        _teardown_usd_reference!(screen, robj)      # stale-volume-handle safe
        delete!(screen.plot2robj, id)
        delete!(screen.path2plot, robj.prim_path)   # reverse entry in lockstep
    end
    empty!(screen.plot2robj)
    empty!(screen.path2plot)   # reverse map cleared alongside plot2robj
    empty!(screen.scene2scope)
    _set_requires_update!(screen)
    return screen
end

# ------------------------------------------------------------------
# Shared authoring + sync helpers (colorbuffer + interactive paths)
# ------------------------------------------------------------------

# Author the open ovrtx stage from a scene, seed the camera/light snapshots
# so the next sync is a no-op, then add every plot.  Shared by colorbuffer +
# the interactive paths so the snapshot-before-sync invariant cannot drift.
function _author_screen!(screen::Screen, cam_scene, plot_scene)
    author_root_from_scene!(screen, cam_scene; resolution = screen.fb_size)
    screen.last_camera = _camera_snapshot(cam_scene)
    screen.last_lights = _lights_snapshot(cam_scene.compute[:lights][])
    screen.authored    = true
    # Environment dome (IBL): a stashed push_environment_image! or the
    # scene's EnvironmentLight, authored as a REMOVABLE reference so it can
    # be live-swapped later (asset inputs are not FFI-writable).
    _author_env_light!(screen, cam_scene)
    Makie.insertplots!(screen, plot_scene)
    return screen
end

# Push live camera/light/plot deltas to the open stage; return whether RT2
# must restart this frame.  Atomically read-and-clears requires_update on both
# the pre-pull and post-pull windows (edit_lock, so a concurrent listener SET
# is never lost).  Shared by colorbuffer + the interactive tick.
function _sync_and_needs_reset!(screen::Screen, cam_scene)::Bool
    cam_changed   = sync_camera!(screen, cam_scene)
    light_changed = sync_lights!(screen, cam_scene)
    pending = _take_requires_update!(screen)   # edits landed BEFORE the pull
    pull_ovrtx_nodes!(screen, screen.scene)
    # usdplot bind_usd! writes flush here, BEFORE the accumulate gate: the OR
    # makes default mode reconverge; the gate drops it in accumulate mode
    # (bound writes are non-structural — reprojection keeps the history).
    usd_wrote = _flush_pending_usd_writes!(screen)
    # Take (read-and-clear) unconditionally — NOT inside the short-circuiting
    # OR — so the flag is cleared even when the camera already forced a reset.
    post_pull  = _take_requires_update!(screen)  # diff-node + landed edits
    need_reset = cam_changed || light_changed || post_pull || pending || usd_wrote
    # Accumulate mode: keep RT2 history, so only a STRUCTURAL change (a USD
    # reference added/removed, via `_note_composition_change!`) resets —
    # reprojection has no history for a prim that just appeared/vanished.
    structural = _take_structural_dirty!(screen)
    screen.config.accumulate_across_frames && (need_reset = structural)
    return need_reset
end

# ------------------------------------------------------------------
# colorbuffer — open-once + live render-config sync + RT2 render
# ------------------------------------------------------------------

"""
    Makie.colorbuffer(screen::Screen; kw...) -> Matrix{RGBA{N0f8}}

Render `screen`'s scene and return the LdrColor framebuffer.

Open-stage model: on the FIRST call, author the root once
(`author_root_from_scene!`, baking camera + lights) and add every plot's USD
reference (`insertplots!` → `register_ovrtx_robj!`); the camera/light
snapshots are seeded so the immediately-following sync is a no-op.  On EVERY
call, `sync_camera!`/`sync_lights!` push minimal writes for any change and
`pull_ovrtx_nodes!` resolves every diff node (one minimal C write per
changed attribute).  If anything was written, `OV.reset!` restarts RT2
accumulation once before rendering — a static scene keeps accumulating.

The matrix is returned exactly as `OV.render_to_matrix` produces it —
top-left origin (right-side-up), 4-channel `RGBA{N0f8}`, no
flip/alpha-drop/conversion.
"""
function Makie.colorbuffer(screen::Screen; kw...)
    scene = screen.scene
    scene === nothing && error("OmniverseMakie.colorbuffer: screen has no scene")
    cam_scene = something(_scene_for_camera(scene), scene)

    if !screen.authored
        # author the root once (camera + lights baked, plots added)
        _author_screen!(screen, cam_scene, scene)
    end

    # Push live render-config deltas; reset RT2 if anything changed.
    need_reset = _sync_and_needs_reset!(screen, cam_scene)
    need_reset && OV.reset!(screen.renderer)

    # Accumulate mode: fold the one-time pre-roll into the FIRST frame's
    # warm-up so frame 1 lands converged (a cold RT2 = noisy).
    warmup = screen.config.warmup
    if screen.config.accumulate_across_frames && !screen.preroll_done
        warmup += screen.config.accumulation_preroll
        screen.preroll_done = true
    end
    return OV.render_to_matrix(screen.renderer, screen.product; warmup = warmup)
end

# PNG showability: `save(fig, "x.png")` / `show(io, MIME"image/png", fig)`
# and Jupyter auto-display check `backend_showable` → `backend_show` →
# `colorbuffer`.
Makie.backend_showable(::Type{Screen}, ::MIME"image/png")  = true
# JPEG showability: Jupyter and `Base.show(io, MIME"image/jpeg", fig)`.
Makie.backend_showable(::Type{Screen}, ::MIME"image/jpeg") = true

# ------------------------------------------------------------------
# Makie pick protocol over the native ray-query AOV pick
#
# `pick_hit` enqueues a 1-pixel ovrtx pick at a Makie pixel, steps the
# renderer, decodes the hit, resolves its prim path to the owning plot, and
# computes the element index.  `Makie.pick`/`pick_closest`/`pick_sorted` are
# the standard backend overrides on top (no GLMakie/CUDA dep).
# ------------------------------------------------------------------

# Pick step timeout (ns): the pick consumes one render step (after which RT2
# accumulation restarts — expected); 10 s is generous for a 1-pixel query.
const _PICK_TIMEOUT_NS = UInt64(10_000_000_000)

# Map a Makie pixel (BOTTOM-left origin, Float64, over `fb_size`) to an
# ovrtx RenderProduct pixel (TOP-left origin, Int, left/top-inclusive):
# x maps straight through; y is FLIPPED (`H - y`).  Getting the flip wrong
# silently picks the wrong pixel.
function _to_ovrtx_pixel(xy, fb_size)
    W, H = fb_size
    px = clamp(round(Int, xy[1]), 0, W - 1)
    py = clamp(H - round(Int, xy[2]), 0, H - 1)
    return (px, py)
end

# instance_id (ovrtx `geometryInstanceId`) → Makie element index:
# Scatter/MeshScatter map the 0-based instance to a 1-based point index;
# every other plot kind is plot-level → index 0.  ovrtx's PointInstancer
# pick collapses to the prototype (every instance reports
# `geometryInstanceId == 0`), so the per-point index is NOT recoverable —
# `Int(instance_id)+1` is exact for a single-point marker only.
function _element_index(plot, instance_id::UInt64)::Int
    return (plot isa Makie.Scatter || plot isa Makie.MeshScatter) ? Int(instance_id) + 1 : 0
end

# Lazily build + cache the renderer's PathResolver for picks.  The cache is
# composition-scoped: `_note_composition_change!` drops it on every plot
# insert/delete, empty!, or volume reload; the next pick rebuilds it here.
function path_resolver_for(screen::Screen)
    pr = screen.path_resolver
    pr === nothing || return pr
    pr = OV.path_resolver(screen.renderer)
    screen.path_resolver = pr
    return pr
end

# ------------------------------------------------------------------
# Edit-signal / pending-write concurrency helpers
#
# requires_update + structural_dirty (Bool flags) and pending_usd_writes
# (Dict) are SET by user-task Observable listeners and READ-then-CLEARED by
# the render tick; all access funnels through `edit_lock` so a threaded
# renderloop cannot lose a set landing between a read and its clear, nor
# mutate the Dict mid-iteration.  Duck-typed so a fake can exercise them.
# ------------------------------------------------------------------

# Raise the diff-node change signal (a listener / teardown SET site).
_set_requires_update!(screen) =
    (lock(() -> (screen.requires_update = true), screen.edit_lock); nothing)

# Consume the change signal after the render path has already drawn it
# (interactive/hybrid setup + resize warmup); used by the GLMakie ext.
_clear_requires_update!(screen) =
    (lock(() -> (screen.requires_update = false), screen.edit_lock); nothing)

# Atomically read-and-clear requires_update (render tick): one critical
# section, so a concurrent SET is never lost between the read and the clear.
function _take_requires_update!(screen)::Bool
    lock(screen.edit_lock) do
        v = screen.requires_update
        screen.requires_update = false
        return v
    end
end

# Atomically read-and-clear structural_dirty (accumulate-mode reset gate).
function _take_structural_dirty!(screen)::Bool
    lock(screen.edit_lock) do
        v = screen.structural_dirty
        screen.structural_dirty = false
        return v
    end
end

# Enqueue one coalesced usdplot bound write + raise the change signal, under
# edit_lock (a bind_usd! listener SET site); latest value per key wins.
_enqueue_usd_write!(screen, key, value) =
    lock(screen.edit_lock) do
        screen.pending_usd_writes[key] = value
        screen.requires_update = true
        return nothing
    end

# Snapshot-and-swap the pending-write Dict under edit_lock: install a fresh
# empty Dict and return the old one (or `nothing` when empty).  The caller
# issues the FFI writes on the returned Dict OUTSIDE the lock (holding it
# across enqueue+wait ccalls is a latency/deadlock hazard).
function _swap_pending_usd_writes!(screen)
    lock(screen.edit_lock) do
        isempty(screen.pending_usd_writes) && return nothing
        old = screen.pending_usd_writes
        screen.pending_usd_writes = empty(old)
        return old
    end
end

# Every stage-composition change (a USD reference added OR removed) calls
# this one helper, so the concerns never drift: (1) drop the pick resolver
# cache, (2) flag `structural_dirty` under edit_lock so accumulate mode resets
# RT2 once.  Call sites: `_register_robj_maps!` (compute.jl),
# `_teardown_usd_reference!`, `reload_volume_data!` (volume.jl).
_note_composition_change!(screen) =
    lock(screen.edit_lock) do
        screen.path_resolver    = nothing
        screen.structural_dirty = true
        return nothing
    end

"""
    reset_accumulation!(screen::Screen) -> Nothing

Force one RT2 accumulation reset (restart the path-tracer fresh).  Only
useful with `accumulate_across_frames = true`, where per-frame resets are
suppressed: call this if a change the composition funnel does not cover (or
a fast camera move) leaves visible ghosting.  Not exported.
"""
reset_accumulation!(screen::Screen) = OV.reset!(screen.renderer)

# Resolve a hit prim path to the owning plot's `objectid` by walking UP the
# path until a `path2plot` entry matches (a Scatter/MeshScatter hit lands on
# the PointInstancer prototype child, `/World/plot_<id>/proto`).  `path2plot`
# holds only plot prims, and plot prims are siblings, so the first matching
# ancestor is unambiguously the plot.
function _path_to_oid(screen::Screen, path::AbstractString)
    isempty(path) && return nothing
    p = String(path)
    while true
        oid = get(screen.path2plot, p, nothing)
        oid === nothing || return oid
        i = findlast('/', p)
        (i === nothing || i <= 1) && return nothing   # no plot ancestor
        p = p[1:(i - 1)]
    end
end

# objectid(plot) → the `Plot` via the `plot` reference on its `OvrtxRObj`
# (set at the `plot2robj` insert sites).  `nothing` if unknown or unset.
function _plot_for_objectid(screen::Screen, oid::UInt64)
    robj = get(screen.plot2robj, oid, nothing)
    return robj === nothing ? nothing : robj.plot
end

"""
    pick_hit(screen::Screen, xy)
        -> Union{Nothing, NamedTuple{(:plot,:index,:world_position,:normal)}}

Enqueue a 1-pixel native ray-query pick at Makie pixel `xy` (bottom-left
origin, over `screen.fb_size`), step the renderer once, and decode the
closest hit: map `xy` to the ovrtx pixel (`_to_ovrtx_pixel`),
`enqueue_pick_query` → `step!` → `read_pick_hit`, resolve `primpath_id` to a
prim path (cached `PathResolver`), walk it to the owning plot
(`_path_to_oid` → `_plot_for_objectid`), and compute the element index
(`_element_index`).

Returns `nothing` over background, or when the hit prim is not a registered
plot (camera/light/looks).  The pick consumes one step, after which RT2
accumulation restarts (expected).

!!! note "world_position / normal are not populated by this ovrtx"
    The returned `world_position` and `normal` read `(0,0,0)` — ovrtx does
    not populate the pick-hit world-position AOV for our render product.
    Do not rely on them; `Makie.pick` uses only `plot` + `index`.  The
    `index` is exact for a single-point scatter marker and plot-level (`0`)
    otherwise (ovrtx collapses a `UsdGeomPointInstancer` pick to the
    prototype).  Both upgrade for free if a future ovrtx surfaces this data.
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
    oid === nothing && return nothing   # camera/light/looks, not a plot
    plot = _plot_for_objectid(screen, oid)
    plot === nothing && return nothing
    return (plot = plot, index = _element_index(plot, hit.instance_id),
            world_position = hit.world_position, normal = hit.normal)
end

"""
    Makie.pick(scene::Scene, screen::Screen, xy::Vec{2,Float64}) -> (plot_or_nothing, index)

Backend pick: the plot + element index under Makie pixel `xy`, or
`(nothing, 0)` over background.  Delegates to `pick_hit`.
"""
function Makie.pick(::Makie.Scene, screen::Screen, xy::Makie.Vec{2,Float64})
    h = pick_hit(screen, xy)
    return h === nothing ? (nothing, 0) : (h.plot, h.index)
end

# The native ray-query already returns the closest hit, so `range` is
# advisory: pick the pixel directly (more specific than Makie's SceneLike
# method, which would fan out a Rect2i region pick we don't implement).
Makie.pick_closest(scene::Makie.Scene, screen::Screen, xy, range) =
    Makie.pick(scene, screen, Makie.Vec{2,Float64}(xy))

# One native hit at `xy`, wrapped as the distance-sorted `(plot,index)` list
# DataInspector expects (empty over background).
function Makie.pick_sorted(scene::Makie.Scene, screen::Screen, xy, range)
    h = pick_hit(screen, Makie.Vec{2,Float64}(xy))
    return h === nothing ? Tuple{Makie.AbstractPlot,Int}[] : [(h.plot, h.index)]
end

# ------------------------------------------------------------------
# select! / clear_selection! — selection-outline API
#
# `select!(screen, plot; group)` assigns a plot's prim to an outline group
# (default 1) so the RTX outline pipeline draws a ring around it;
# `clear_selection!` removes it (group 0).  The highlight needs the
# creation-time `selection_outline=true` config; without it `select!` warns
# once and is a no-op (pick data still works).  A default orange style is
# installed once per Screen on the first selection.
# ------------------------------------------------------------------

# Default outline style: orange edge (the ovrtx C reference default) +
# transparent fill (default fill mode is EDGE_ONLY, so only the outline
# colour draws).
const _OUTLINE_ORANGE = (1.0f0, 0.6f0, 0.0f0, 1.0f0)

# Install the default group-1 orange outline style once per Screen
# (idempotent via `_outline_styled`).  Returns whether this Screen can draw
# an outline at all (created with `selection_outline=true`).
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

Highlight `plot`'s prim with a selection outline (group `group`, default 1),
installing the default orange outline style once per Screen.  Warns once and
is a no-op on a Screen built without `selection_outline=true`; silently a
no-op when `plot` is not a registered render object.  Returns `nothing`.
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

Remove the selection outline from `plot` (group 0), or — with no `plot` —
from every currently-tracked plot.  Returns `nothing`.
"""
function clear_selection!(screen::Screen, plot)
    robj = get(screen.plot2robj, objectid(plot), nothing)
    robj === nothing && return nothing
    OV.set_selection_outline_group!(screen.renderer, [robj.prim_path], UInt8[0x00])
    return nothing
end
clear_selection!(screen::Screen) =   # clear all currently-tracked plots
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
`samples`, `warmup`, `max_bounces`, `selection_outline`,
`accumulate_across_frames`, `accumulation_preroll`, `background`, `sensors`.

`mode` picks the RTX render mode (see `rtx_settings_usda`):
  * `:rt2` (default) — the realtime accumulating path tracer + OptiX
    denoiser.  `samples` is inert.
  * `:pathtracing` — the OFFLINE path tracer for final-quality stills;
    `samples` sets the SPP per still (converged over the `warmup` step
    loop).  Slower, higher quality.
`:minimal` is not a valid `mode` (throws) — in standalone ovrtx it is an
exact RT2 fallback.

`sensors = true` forces the renderer's motion BVH on (correct MOVING-object
`lidar!`/`radar!` returns) — only needed when sensors are added AFTER
display; a scene that already contains sensor plots auto-enables it.
"""
function activate!(; screen_config...)
    Makie.set_screen_config!(OmniverseMakie, screen_config)
    Makie.set_active_backend!(OmniverseMakie)
    return
end
