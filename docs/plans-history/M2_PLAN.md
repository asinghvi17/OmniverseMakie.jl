# Milestone M2 — `ComputePipeline` diff path + dynamic add/delete (bite-sized plan)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes. This expands IMPLEMENTATION_PLAN.md's task-level M2 into TDD steps, **reconciled against what M1 actually built** (not the pre-M1 assumptions in ARCHITECTURE.md §6).

**Goal:** Plots, **the camera, and lights** update **live** on an **already-open** USD stage — a changed transform / color / point set / camera pose / light intensity pushes a **minimal C write** (no re-author) — and plots and subscenes can be **added and deleted at runtime, leak-free**. The hot path is benchmarked to interactive rates.

**Approach:** M1 is *build-once*: every `colorbuffer` re-opens the stage, re-bakes camera+lights, and re-adds all plot references. M2 shifts to a **persistent open stage**: author once, then each frame **pull** each plot's `:ovrtx_renderobject` node, which uses ComputePipeline's `changed` mask to push only the changed outputs through targeted ovrtx writes (xform / points / color). **The camera and lights are live-diffed peers, not baked** — a camera move drives `write_xform!("/World/Camera", …)`, a light change drives `write_xform!` / `inputs:intensity` / `inputs:color` on its prim (all spike-proven on an open stage, clean & reversible) — so they update without re-opening. Makie drives add/delete **imperatively** (`insert!`/`delete!`), so `add_usd_reference!`/`remove_usd!` + node registration/teardown ride those calls. `Screen.plot2usd` (M1) evolves into `plot2robj::Dict{UInt64,OvrtxRObj}` (handle + persistent bindings + the node).

---

## Global Constraints (M2)

Inherit **all** M0 + M1 Global Constraints (Pkg-managed pinned deps; generated bindings verbatim; `GC.@preserve` on every FFI path; carb `SignalGuard` intact; subprocess-isolated renderer tests + `timedwait` watchdog; `Pkg.test()` runner; reference layers omit `upAxis`; `upAxis="Z"`; `colorbuffer` returns `Matrix{RGBA{N0f8}}` **right-side-up, NO flip** — orientation is locked by `test/m1_orientation_test.jl`). Plus M2-specific:

- **Open-stage model (the core shift).** The USD stage is opened **once** (lazily, first `colorbuffer`/`display`) and stays open across frames. `colorbuffer` no longer re-authors per call — it pushes minimal edits (plot nodes + camera/light writes) then renders. The ONLY full `open_usd_string!` re-open is a **structural** change: `empty!(screen)`, or adding/removing a light (the prim *set* changes). Camera/light **attribute** changes (pose, intensity, color, direction) are **live writes, not re-opens** (spike-proven).
- **Camera + lights are live-diffed peers, NOT baked.** ✅ **ARCHITECTURE.md §6's camera row ("camera → `omni:xform` → `map_attribute`, updated cheaply") is VINDICATED. M1.3's "`omni:xform`/`write_xform!` ignored on camera/light prims" was a BUG, now DISPROVEN by spikes** — camera: an A→B→A round-trip returned to the original within **8/270000 px**; lights: intensity, transform, AND color all update live, round-trips clean within **3/160000 px**. The proven writes (apply after each change, then `OV.reset!`):
  - **Camera pose:** `OV.write_xform!(r, "/World/Camera", camera_to_world(eye,target,up))` (4×4 row-vector `omni:xform`, `SEMANTIC_XFORM_MAT4x4`).
  - **Light transform** (Distant direction / Sphere position): `OV.write_xform!(r, light_prim_path, xform)`.
  - **Light intensity:** `OV._write_attribute!(r, light_prim_path, "inputs:intensity", DLDataType(kDLFloat,32,1), false, OVRTX_SEMANTIC_NONE, Float32[v], Int64[1])`.
  - **Light color:** same, attr `"inputs:color"`, `lanes=3`, `Float32[r,g,b]`.
  ⚠️ **Light prim paths must be derived the SAME way `lights.jl` authors them** (a spike saw `/World/DirectionalLight_0`) — extract a shared `light_prim_path` helper; do NOT hardcode the index base. `reset!` is required after every such write. fov→focalLength is a rare `float focalLength` scalar write; orbit/pan/zoom (the common case) is `write_xform!`. **Do NOT re-bake the root on a camera/light *attribute* change.**
- **Live geometry writes target the REFERENCED prim path** `/World/plot_<objectid(plot)>` (the prim the M1 reference was added at). ⚠️ **Open assumption (validate FIRST, M2.2):** M0 proved `write_xform!`/array writes on a **root** prim (`/World/Torus`); whether the same writes on a **referenced** prim path are honored by ovrtx is UNPROVEN. If not honored, the per-plot fallback is **targeted re-reference** (`remove_usd!(handle)` + `add_usd_reference!` with the new data) — still O(1)-per-changed-plot, slower than a binding but correct and leak-free.
- **Source update data from RESOLVED COMPUTE OUTPUTS** (`args` in the node), never directly from the plot — so `:model_f32c` (the **composed world transform**, which fixes M1's scene-transform gap), `:positions_transformed_f32c`, `:scaled_color`, `:visible` drive **both** the build and update branches from one data source.
- **`plot2usd` → `plot2robj::Dict{UInt64,OvrtxRObj}`**; `remove_usd!` on every delete (this **kills the M1.6 per-frame handle leak**). Typed `delete!` signatures only (untyped `delete!(screen,scene,plot)` is ambiguous with Makie's default — Spike B).
- **Fix `OV.write_array_attribute!` for `Vector{Point3f}`/`Vec3f`** (M0 forward-carry): `reinterpret(Float32, …)` + a 3-lane / shaped DLTensor + copy-elision. The hot path (M2.3) needs point arrays.
- **No new external deps.** `ComputePipeline` is already a pinned dep (`=0.1.8`). Benchmark uses the existing harness.

**M1 reconciliations folded into M2 (do them in the named task):**
- `setup_scene!`/`colorbuffer` (re-author-per-call) → **open-once + pull nodes** (M2.1).
- `Screen` struct gains `plot2robj`, `scene2scope`, `scene_listeners`, `requires_update`; `plot2usd` removed/renamed (M2.1).
- `to_ovrtx_object(screen,scene,plot)` (reads plot directly) → `author_usd_prim!(screen, plot, args)` (reads compute outputs); the M1.5/M1.7 USDA emitters are reused but fed from `args` (M2.2).
- **`Screen.open_results` (vestigial — final-review finding):** wire it (if M2.3 retains a mapped `StepResult`) **or delete it** + simplify `close(Screen)` (M2.1).
- Relocate `author_root_from_scene!` (currently in `lights.jl`) → `usd.jl` or a new `composition.jl` (M2.1 cleanup).

---

## File structure (M2 adds / modifies)

```
src/
  compute.jl        # NEW: OvrtxRObj, register_ovrtx_robj!, push_to_ovrtx!, consumed_inputs, author_usd_prim!, bind_hot_attributes!
  screen.jl         # MODIFY: open-stage colorbuffer; live camera/lights push (sync_camera!/sync_lights!); insert!/insertplots!/add_scene!/delete!/empty!; Screen struct; drop open_results
  binding/OV.jl     # MODIFY: create_binding/map_binding/unmap!/destroy! wrappers; write_array_attribute! Point3f/Vec3f fix
  translation/usd.jl (or composition.jl)  # MODIFY: relocate author_root_from_scene! (M2.1); emit def Scope hierarchy via scene_scopes_usda (M2.3)
  translation/camera.jl  # MODIFY (M2.1): correct the false M1.3 "ignored on camera" comment; document the live write_xform! path
  translation/lights.jl  # MODIFY (M2.1): shared light_prim_path helper (single source for /World/<Type>Light_<i>); live attr writes
bench/
  hot_path.jl       # NEW (M2.6)
  RESULTS.md        # NEW (M2.6)
test/
  m2_openstage_test.jl  m2_insert_test.jl  m2_rendercfg_test.jl  m2_diffnode_test.jl
  m2_subscene_test.jl   m2_binding_test.jl    m2_delete_test.jl
```

---

## Task M2.1 — Open-stage Screen + live render-config (camera + lights) + `insert!`/`insertplots!`/`add_scene!` (foundational refactor) ★

**Files:** `src/screen.jl`, `src/compute.jl` (the `OvrtxRObj` struct), `src/translation/camera.jl` (fix the false M1.3 comment), `src/translation/lights.jl` (shared `light_prim_path` helper). Test: `test/m2_openstage_test.jl`, `test/m2_insert_test.jl`, `test/m2_rendercfg_test.jl`.

**Interfaces — Produces:**
```julia
mutable struct OvrtxRObj                 # in compute.jl
    prim_path::String                    # /World/plot_<id>
    usd_handle::UInt64                   # ovrtx_usd_handle_t from add_usd_reference!
    bindings::Dict{Symbol,Any}           # M2.3 fills (attr name => persistent binding); empty in M2.1
end
# Screen (evolved): drop `plot2usd` + the vestigial `open_results`/`setup`; add:
#   plot2robj::Dict{UInt64,OvrtxRObj}, scene2scope::Dict{UInt64,String},
#   scene_listeners::Dict{UInt64,Vector}, requires_update::Bool, authored::Bool,
#   last_camera::Any   # snapshot (eye,lookat,up,fov) last WRITTEN — for change detection
#   last_lights::Any   # snapshot of scene.compute[:lights][] last WRITTEN
Base.insert!(screen::Screen, scene::Scene, plot::Plot)       # open-stage: idempotent via plot2robj; add_scene! first; recurse plot.plots
Makie.insertplots!(screen::Screen, scene::Scene)             # loop scene.plots + recurse scene.children
add_scene!(screen::Screen, scene::Scene) -> String           # idempotent get! on scene2scope; author a USD scope; register+store redraw listeners
light_prim_path(light, index) -> String                      # SHARED with lights.jl authoring — single source for /World/<Type>Light_<i>
sync_camera!(screen, scene) -> Bool                          # if pose ≠ last_camera: write_xform!("/World/Camera", …); update snapshot; return changed
sync_lights!(screen, scene) -> Bool                          # per changed light: write_xform! + inputs:intensity + inputs:color; update snapshot; return changed
```

- **Key reconciliation:** M1's `setup_scene!` does `empty!(plot2usd)` + re-author every call. **Replace** with: `colorbuffer` authors the root **once** (guard on `screen.authored`), then on later calls **pushes deltas** — camera/light writes now (M2.1), plot nodes later (M2.2) — instead of re-opening. A static scene renders identically to M1; a **camera orbit** (the existing `m1_save_record_test.jl`) now reframes via `write_xform!`, **NOT** a re-author.
- **Camera + lights live via snapshot-compare in `colorbuffer`** (NOT observable listeners — keeps teardown trivial). Each render, `sync_camera!`/`sync_lights!` compare the current pose/lights to `screen.last_camera`/`last_lights`; on a diff, push the minimal write(s) (exact calls in Global Constraints), update the snapshot, and signal that an `OV.reset!` is needed. Unchanged ⇒ no write, no reset (so a static scene keeps accumulating). **Derive light prim paths via the shared `light_prim_path` helper** (a spike saw `/World/DirectionalLight_0` — don't hardcode the index base).
- [ ] **Step 1 (failing test, openstage):** subprocess — `Screen` for a 1-mesh `LScene`; `colorbuffer` twice with the camera UNCHANGED; assert (a) non-black both times, (b) the stage was authored exactly **once** (assert `screen.authored` true + the `plot2robj` handle is stable across both; optionally an `open_usd_string!` counter == 1). RED (M1 re-authors → counter == 2).
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3a (struct + teardown + false-finding fix):** evolve the `Screen` struct (drop `plot2usd` + the vestigial `open_results`; `setup` was already removed in M1.6; add the M2 fields incl. `authored`, `last_camera`, `last_lights`); simplify `close(Screen)` (close the renderer; no `open_results` loop; `OV.destroy!` bindings is an M2.3 concern). Relocate `author_root_from_scene!` to `usd.jl`/`composition.jl`. **Correct the false M1.3 finding in `camera.jl`** (the "write_xform!/omni:xform is ignored on camera prims" comment block, lines ~9–17) — it is DISPROVEN; document the live `write_xform!("/World/Camera", …)` path instead.
- [ ] **Step 3b (open-once colorbuffer + render-config sync):** rewrite `colorbuffer`: if `!authored`, author root once (`author_root_from_scene!`) + `insertplots!`; set `authored=true`. Then ALWAYS call `sync_camera!`/`sync_lights!` (each a no-op when unchanged); if either returns `true`, `OV.reset!` before rendering. Render. (M2.2 inserts the plot-node pull here too.)
- [ ] **Step 3c (live writes):** implement `sync_camera!` (`write_xform!("/World/Camera", camera_to_world(eye,target,up))`), `sync_lights!` (per changed light: `write_xform!` for transform + `_write_attribute!` for `inputs:intensity` / `inputs:color`), and the shared `light_prim_path(light, index)` helper used by BOTH `sync_lights!` and `lights.jl`'s authoring (refactor the authoring to call it — single source of truth).
- [ ] **Step 3d (imperative add):** `insert!`/`insertplots!`/`add_scene!` per the interfaces (idempotent via `plot2robj`; `add_scene!` authors a `def Scope` per subscene and stores its redraw listeners in `scene_listeners` for M2.4 teardown). `insert!` records `OvrtxRObj(path, handle, Dict())`.
- [ ] **Step 4a (insert test):** subprocess — `Scene` + `Makie.push_screen!(scene, screen)`; `scatter!(scene, …)` after attach → `plot2robj` grows by one and renders; a recipe (`poly!`) registers its atomic children. PASS.
- [ ] **Step 4b (render-config test, `m2_rendercfg_test.jl`):** subprocess — ONE open stage; (i) orbit the camera 180° between two `colorbuffer`s → the content reframes (red/blue centroid swap, shift ≥20px) with the stage authored ONCE; (ii) scale a light's intensity ~0.2× / ~5× → mean luminance over lit pixels shifts clearly; (iii) swap a light `inputs:color` → hue shifts; each with a **round-trip** back to baseline (returns within RT2 noise). PASS. (This is the same mechanism the spikes proved; it must also keep `m1_save_record_test.jl`'s camera-orbit assertion green via the write path.)
- [ ] **Step 5:** commit `feat(M2.1): open-stage Screen + live camera/lights + imperative insert!`.

**Acceptance:** the stage is authored ONCE and reused; a static scene renders as in M1; a camera orbit reframes via `write_xform!` (no re-author — the M1.6 record test passes on the open stage); light intensity/color/position update live and round-trip cleanly; live `plot!` on a displayed scene authors a USD reference + records an `OvrtxRObj`. The false M1.3 camera/light finding is corrected in `camera.jl`.

---

## Task M2.2 — The `:ovrtx_renderobject` compute node + `push_to_ovrtx!` (the diff driver) ★

**Files:** `src/compute.jl`. Test: `test/m2_diffnode_test.jl`.

**Interfaces — Produces:**
```julia
consumed_inputs(plot)::Vector{Symbol}                         # per-type Makie compute-output list
author_usd_prim!(screen, plot, args) -> OvrtxRObj             # build branch: M1 USDA emitter fed from `args`
register_ovrtx_robj!(screen, scene, plot) -> OvrtxRObj        # register node, force first resolve, record plot2robj
push_to_ovrtx!(screen, robj, name::Symbol, value)            # route ONE changed output to the right ovrtx write
```

- **Key code** (Spike-B contract — `args`/`last` hold deref'd values; `last===nothing` on first build with `changed` all-true):
  ```julia
  function register_ovrtx_robj!(screen, scene, plot)
      attr   = plot.attributes
      inputs = consumed_inputs(plot)
      ComputePipeline.register_computation!(attr, inputs, [:ovrtx_renderobject]) do args, changed, last
          if isnothing(last)
              robj = author_usd_prim!(screen, plot, args)     # build (reuses M1.5/M1.7 emitters, fed from args)
          else
              robj = last.ovrtx_renderobject
              for name in keys(args)
                  changed[name] || continue                   # ← minimal-delta gate
                  push_to_ovrtx!(screen, robj, name, args[name])
              end
          end
          screen.requires_update = true
          return (robj,)
      end
      robj = attr[:ovrtx_renderobject][]                       # force first resolve
      screen.plot2robj[objectid(plot)] = robj
      return robj
  end
  ```
  `push_to_ovrtx!` routing (per `ARCHITECTURE.md §6`, **geometry only** — NO camera row): `:model_f32c`→`OV.write_xform!` on `robj.prim_path` (Float32→Float64); `:positions_transformed_f32c`→`OV.write_array_attribute!`(points); `:faces`→`write_array_attribute!`; `:scaled_color`→`displayColor` write; `:visible`→`visibility` write. `consumed_inputs(::Mesh)=[:positions_transformed_f32c,:model_f32c,:faces,:normals,:scaled_color,:visible]`; `::Scatter=[:positions_transformed_f32c,:model_f32c,:quad_scale,:quad_offset,:converted_rotation,:scaled_color,:visible]`; `::Lines=[:positions_transformed_f32c,:scaled_color,:model_f32c,:linewidth,:visible]`. (Spike B verified these exist by default; GL-only nodes like `:gl_miter_limit` do NOT — skip them or `Makie.add_computation!(attr, Val(:computed_color))` for a packed color.)
- **`colorbuffer` open-stage pull:** before render, for each `robj`, `plot.attributes[:ovrtx_renderobject][]` (try/catch → `ComputePipeline.mark_resolved!`); a clean graph is a no-op. Any geometry change ⇒ `OV.reset!` (restart RT2 accumulation).
- ⚠️ **Validate FIRST (the referenced-prim assumption):** Step 1 asserts a `write_xform!` on `/World/plot_<id>` (a referenced prim) actually MOVES the geometry. If it does NOT (only noise), switch `push_to_ovrtx!` to the **re-reference fallback** (`remove_usd!` + `add_usd_reference!` with new data, update `robj.usd_handle`) and record it.
- [ ] **Step 1 (failing test):** subprocess — insert a mesh; pull the node (build); `plot.color[] = :red` (or `plot.color = …`); pull again; assert (a) **only** the color write fired (instrument `push_to_ovrtx!` with a counter per `name`), (b) the render changed (red now dominant). Also a transform sub-test: `translate!`/`plot.model` change → only `:model_f32c` fires → geometry moves (validates the referenced-prim write). RED.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** implement `consumed_inputs`, `author_usd_prim!` (wrap M1 emitters, source from `args`), `register_ovrtx_robj!`, `push_to_ovrtx!` (+ the re-reference fallback if Step 1 shows referenced writes are ignored); wire the per-frame pull into `colorbuffer`; call `register_ovrtx_robj!` from `insert!` (replacing M2.1's bare `add_usd_reference!`).
- [ ] **Step 4:** run → PASS (one minimal write per edit; visible change; `reset!` fires).
- [ ] **Step 5:** commit `feat(M2.2): :ovrtx_renderobject diff node + push_to_ovrtx! minimal edits`.

**Acceptance:** an attribute edit triggers **exactly one** minimal C write and a visible change on the open stage; `:model_f32c` carries the composed world transform (M1 scene-transform gap closed).

---

## Task M2.3 — USD subscene grouping (`def Scope` hierarchy mirroring the Makie scene tree) ★

> Inserted 2026-06-28 (user-requested), overriding the M2.1 `add_scene!` deferral. Full brief: `.superpowers/sdd/briefs/m2.3-subscene-brief.md`. **Renumber note:** the former M2.3/M2.4/M2.5 (bindings / delete / benchmark) are now **M2.4 / M2.5 / M2.6**; a few inline `(M2.x)` cross-references in earlier task bodies may still show the old numbers — the task HEADERS are authoritative.

**Files:** `src/translation/usd.jl` (root authoring emits the scope skeleton), `src/screen.jl` (`add_scene!`; thread scene through `insert!`/`insertplots!`), `src/compute.jl` (`plot_prim_path` scope-aware; thread scene through `author_usd_prim!`/`register_ovrtx_robj!`/node callback), `src/translation/{meshes,primitives}.jl` (remove the now-DEAD `to_ovrtx_object(::Mesh/::Scatter/::MeshScatter/::Lines/::LineSegments)` — M2.2 Minor #1; keep `::Surface`). Test: `test/m2_subscene_test.jl`.

**Goal:** author a `def Scope` per Makie subscene, nested to mirror the scene tree, so each plot's USD reference lives at `/World/Scene_<objectid(scene)>/plot_<objectid(plot)>`. The open stage's prim hierarchy then mirrors the Makie `Scene` graph (correct representation + future USD export + clean subscene deletion in M2.5). **Render stays pixel-equivalent** (plots keep `:model_f32c` composed-world; scopes carry NO transform — transform-decomposition onto `Xform` scopes is a future refinement).

**Interfaces — Produces:** `scene_scopes_usda(root) -> (usda, scene2scope)` (pure, unit-testable); `plot_prim_path(scene, plot)`; `add_scene!(screen, scene) -> scope_path`; `author_usd_prim!(screen, scene, plot, args)`.

**Approach:** root authoring embeds nested `def Scope "Scene_<objectid>"` inside `def Xform "World"` (scopes exist before plots reference under them) + populates `screen.scene2scope`; `plot_prim_path = scene2scope[objectid(scene)] * "/plot_<id>"` (objectid-derived ⇒ stable across screens / the M2.2 `:ovrtx_screen` rebuild); scene threaded through build/register/node. **Validate early (Step 2):** a reference nested under a root-authored Scope renders (report if not). Handle the 2-D figure root (root→`/World`, 3-D child→a `Scene_` scope) so `m1_save_record`/`save(fig)` stay green.

- [ ] **Step 1:** failing unit test — `scene_scopes_usda` nesting + `scene2scope` paths (no render). RED.
- [ ] **Step 2:** failing render test (subprocess) — Figure with a subscene renders, paths nested, `ROOT_OPENS==1`. RED.
- [ ] **Step 3:** implement + consolidate the dead `to_ovrtx_object` methods.
- [ ] **Step 4:** GREEN; full `Pkg.test` (incl. `m1_save_record`).
- [ ] **Step 5:** commit `feat(M2.3): USD subscene grouping (def Scope hierarchy)`.

**Acceptance:** USD prim hierarchy mirrors the scene tree; render pixel-equivalent within RT2 noise; dead build paths removed; `plot_prim_path` scope-aware + single-sourced.

---

## Task M2.4 — Persistent hot-path bindings (`map_attribute` / `bind_array_attribute`) + Point3f write fix

**Files:** `src/compute.jl`, `src/binding/OV.jl`. Test: `test/m2_binding_test.jl`.

**Interfaces — Produces:**
```julia
OV.create_binding(r, prim, name, dtype, shape; array::Bool) -> Binding   # locks prim+attr+type; OPTIMIZE flag on the hot one
OV.map_binding(b; device=CPU) -> ptr ;  OV.unmap!(b) ;  OV.destroy!(b)
OV.write_binding!(b, data)                                                # array path: bind + write()
bind_hot_attributes!(screen, robj, args)                                 # create persistent bindings, store in robj.bindings
# plus: OV.write_array_attribute! gains a Point3f/Vec3f reinterpret path (M0 carry)
```
- **Key code:** `ARCHITECTURE.md §6` hot-path tiers + `references/notes/ovrtx-api.md §4` (`bind_attribute`/`map_attribute`, `BindingFlag.OPTIMIZE`). Fixed-size (`omni:xform`) → **`map_attribute`** zero-copy (create once, reuse, `Float32→Float64`). Arrays (`points`/instancer `positions`) → **`bind_array_attribute`+`write()`** (CPU DLPack now; GPU-resident is M3/M4). `push_to_ovrtx!` for a bound attr writes through `robj.bindings[name]` instead of re-authoring.
- ⚠️ **Validate (binding-API assumption):** if `map_attribute`/`bind_attribute` don't behave as the C examples suggest, fall back to the M0 `write_attribute` path (correct, no zero-copy) and note it for the M2.5 benchmark.
- [ ] **Step 1 (failing test):** subprocess — animate a prim's `omni:xform` for 100 frames through a **mapped** binding; assert each frame moves the content AND no per-frame USDA authoring occurs (binding object identity stable across frames; `open_usd_string!` counter unchanged). Also a Point3f test: `write_array_attribute!(r, prim, "points", ::Vector{Point3f})` succeeds (M0 carry). RED.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** add the OV binding wrappers (mirror M0 `_write_attribute!` GC discipline) + the `Point3f`/`Vec3f` reinterpret in `write_array_attribute!`; `bind_hot_attributes!` creates+stores bindings; route `push_to_ovrtx!` hot attrs through them.
- [ ] **Step 4:** run → PASS (persistent bindings; no re-author).
- [ ] **Step 5:** commit `feat(M2.4): persistent map/bind hot-path bindings + Point3f array writes`.

**Acceptance:** transforms/points update through persistent bindings created once, no re-authoring per frame.

---

## Task M2.5 — `delete!` / `delete!(scene)` / `empty!` — leak-free teardown

**Files:** `src/screen.jl`. Test: `test/m2_delete_test.jl`.

**Interfaces — Produces** (typed — untyped is ambiguous with Makie's default, Spike B):
```julia
Base.delete!(screen::Screen, scene::Makie.Scene, plot::Makie.AbstractPlot)
Base.delete!(screen::Screen, scene::Makie.Scene)
Base.empty!(screen::Screen)
```
- **Key code** (`ARCHITECTURE.md §7`): `delete!(screen,scene,plot)` — flatten atomic plots; look up `OvrtxRObj` by `objectid`; `OV.destroy!` its bindings + **`OV.remove_usd!(robj.usd_handle)`**; drop from `plot2robj`; `delete!(plot.attributes, :ovrtx_renderobject)` (Spike B: `empty!(graph)` does NOT clear nodes — required); `requires_update=true`. `delete!(screen,scene)` — recurse children+plots, **`Observables.off` every `scene_listeners[objectid(scene)]`** (the GLMakie subscene leak we fix), remove the subscene's USD scope, drop from `scene2scope`. `empty!` — delete every cached plot, `delete!(screen, screen.scene)`, assert registries empty.
- [ ] **Step 1 (failing test):** subprocess — add then `delete!(scene, plot)` → the prim is gone from the render (non-black drops to background), `plot2robj` empty, node dropped; add/remove a subscene **50×** → `scene_listeners` returns to empty (no accumulation). RED.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** implement the three typed methods + the listener/scope/binding teardown.
- [ ] **Step 4:** run → PASS (no residual prims, bindings, nodes, or listeners).
- [ ] **Step 5:** commit `feat(M2.5): leak-free delete!/delete!(scene)/empty! teardown`.

**Acceptance:** add/delete of plots and subscenes leaves zero residual prims, bindings, nodes, or listeners; `remove_usd!` closes the M1.6 handle leak.

---

## Task M2.6 — Hot-path benchmark (de-risk gate)

**Files:** `bench/hot_path.jl`, `bench/RESULTS.md`. Test: a threshold assertion in `test/` (or a `bench` target).

- **Interfaces — Produces:** a benchmark animating **N transforms** (mapped) and **N points** (bind+write) per frame, reporting updates/sec + frame time, on the A5000.
- [ ] **Step 1:** write the benchmark (reuse the subprocess harness; create the renderer once; loop M frames over N objects).
- [ ] **Step 2:** run; record updates/sec + frame time to `bench/RESULTS.md`.
- [ ] **Step 3:** a test asserting the measured rate ≥ target (**≥30 Hz for ~10⁴ instance transforms OR ~10⁵ points**). If below target → the result documents the gap and escalates GPU-resident DLPack writes to M3 (do NOT silently pass).
- [ ] **Step 4:** commit `bench(M2.6): hot-path throughput benchmark + results`.

**Acceptance:** map/bind throughput sustains interactive rates on the A5000 (or the shortfall is measured + escalated).

---

**M2 GATE:** live attribute/transform/color edits — **including camera pose and light intensity/color/position** — push minimal C writes on an **open** stage (no per-frame re-author); plot + subscene add/delete is leak-free (`remove_usd!` + listener deregistration); the hot-path benchmark meets target (or escalates). ✅ → M3 (interactive **window** + event loop that drives the now-proven live-camera/lights writes — no camera-mechanism research needed).

---

## Open assumptions this plan validates early (flag, validate in the named task — with fallbacks)
1. **Live writes on a REFERENCED prim** (`/World/plot_<id>`) are honored by ovrtx (M2.2 Step 1). M0 only proved root-prim writes. Fallback: targeted re-reference (`remove_usd!`+`add_usd_reference!`).
2. **`map_attribute`/`bind_array_attribute`** behave as the C API suggests (M2.3). Fallback: M0 `write_attribute` (correct, no zero-copy) — feeds the M2.5 benchmark verdict.
3. **Makie compute outputs** (`:positions_transformed_f32c`, `:model_f32c`, `:scaled_color`, …) exist + resolve for our plots (Spike B verified; re-confirm on the real plots in M2.2). Fallback: source from the plot directly (M1 style) + `add_computation!` for packed color.
4. **Hot-path throughput** meets interactive rates (M2.5 — the explicit de-risk gate).
5. **RESOLVED — now IN M2, not deferred to M3.** Live CAMERA + LIGHT updates are PROVEN: `write_xform!` (camera & light transforms), `inputs:intensity`, and `inputs:color` are all honored on an open stage, clean and reversible (spikes; round-trips within RT2 noise — camera 8/270000 px, lights 3/160000 px). M1.3's "ignored on camera/light prims" was a bug; ARCHITECTURE §6 is vindicated. M3 now owns only the interactive *window* + event loop that DRIVES these writes — not the write mechanism (M2.1 builds it).
