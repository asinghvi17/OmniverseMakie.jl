# Milestone M2 — `ComputePipeline` diff path + dynamic add/delete (bite-sized plan)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes. This expands IMPLEMENTATION_PLAN.md's task-level M2 into TDD steps, **reconciled against what M1 actually built** (not the pre-M1 assumptions in ARCHITECTURE.md §6).

**Goal:** Plots update **live** via a per-plot `:ovrtx_renderobject` ComputePipeline node — a changed transform / color / point set pushes a **minimal C write** into an **already-open** USD stage (no re-author) — and plots and subscenes can be **added and deleted at runtime, leak-free**. The hot path is benchmarked to interactive rates.

**Approach:** M1 is *build-once*: every `colorbuffer` re-opens the stage, re-bakes camera+lights, and re-adds all plot references. M2 shifts to a **persistent open stage**: author once, then each frame **pull** each plot's `:ovrtx_renderobject` node, which uses ComputePipeline's `changed` mask to push only the changed outputs through targeted ovrtx writes (xform / points / color). Makie drives add/delete **imperatively** (`insert!`/`delete!`), so `add_usd_reference!`/`remove_usd!` + node registration/teardown ride those calls. `Screen.plot2usd` (M1) evolves into `plot2robj::Dict{UInt64,OvrtxRObj}` (handle + persistent bindings + the node).

---

## Global Constraints (M2)

Inherit **all** M0 + M1 Global Constraints (Pkg-managed pinned deps; generated bindings verbatim; `GC.@preserve` on every FFI path; carb `SignalGuard` intact; subprocess-isolated renderer tests + `timedwait` watchdog; `Pkg.test()` runner; reference layers omit `upAxis`; `upAxis="Z"`; `colorbuffer` returns `Matrix{RGBA{N0f8}}` **right-side-up, NO flip** — orientation is locked by `test/m1_orientation_test.jl`). Plus M2-specific:

- **Open-stage model (the core shift).** The USD stage is opened **once** (lazily, first `colorbuffer`/`display`) and stays open across frames. `colorbuffer` no longer re-authors per call — it pulls the compute nodes (minimal edits) then renders. The ONLY full `open_usd_string!` re-open is an explicit full rebuild (`empty!(screen)` / camera-or-lights change).
- **M2 diffs GEOMETRY plots only; camera + lights stay BAKED.** ⚠️ **The ARCHITECTURE.md §6 row "camera view/proj → camera `omni:xform` → map_attribute" is SUPERSEDED.** M1.3 PROVED `write_xform!`/`omni:xform` is **ignored on camera and light prims** (~7177 px = RT2 noise; the working path was re-baking the root). So in M2 the camera/lights remain baked into the root and are updated only by a **full rebuild** when they change (acceptable — M2's tests animate plot attributes, not the camera). **Live camera updates are an M3 research item** (needs its own spike for a mechanism ovrtx honors; do NOT attempt camera live-update via `write_xform!`).
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
  screen.jl         # MODIFY: open-stage colorbuffer; insert!/insertplots!/add_scene!/delete!/empty!; Screen struct fields; drop open_results
  binding/OV.jl     # MODIFY: create_binding/map_binding/unmap!/destroy! wrappers; write_array_attribute! Point3f/Vec3f fix
  translation/usd.jl (or composition.jl)  # MODIFY: relocate author_root_from_scene!
bench/
  hot_path.jl       # NEW (M2.5)
  RESULTS.md        # NEW (M2.5)
test/
  m2_openstage_test.jl  m2_insert_test.jl  m2_diffnode_test.jl
  m2_binding_test.jl    m2_delete_test.jl
```

---

## Task M2.1 — Open-stage Screen + `insert!`/`insertplots!`/`add_scene!` (foundational refactor)

**Files:** `src/screen.jl`, `src/compute.jl` (the `OvrtxRObj` struct). Test: `test/m2_openstage_test.jl`, `test/m2_insert_test.jl`.

**Interfaces — Produces:**
```julia
mutable struct OvrtxRObj                 # in compute.jl
    prim_path::String                    # /World/plot_<id>
    usd_handle::UInt64                   # ovrtx_usd_handle_t from add_usd_reference!
    bindings::Dict{Symbol,Any}           # M2.3 fills (attr name => persistent binding); empty in M2.1
end
# Screen (evolved): drop `plot2usd` + the vestigial `open_results`/`setup`; add:
#   plot2robj::Dict{UInt64,OvrtxRObj}, scene2scope::Dict{UInt64,String},
#   scene_listeners::Dict{UInt64,Vector}, requires_update::Bool, authored::Bool
Base.insert!(screen::Screen, scene::Scene, plot::Plot)       # open-stage: idempotent via plot2robj; add_scene! first; recurse plot.plots
Makie.insertplots!(screen::Screen, scene::Scene)             # loop scene.plots + recurse scene.children
add_scene!(screen::Screen, scene::Scene) -> String           # idempotent get! on scene2scope; author a USD scope; register+store redraw listeners
```

- **Key reconciliation:** M1's `setup_scene!` does `empty!(plot2usd)` + re-author every call. **Replace** with: `colorbuffer` authors the root **once** (guard on `screen.authored`), then on later calls **pulls nodes** (M2.2) instead of re-opening. In M2.1 (before the node exists) `insert!` just authors the reference once and records an `OvrtxRObj` (handle, empty bindings). A static scene must render identically to M1.
- [ ] **Step 1 (failing test, openstage):** subprocess — build a `Screen` for a 1-mesh `LScene`; `colorbuffer` twice; assert (a) the image is non-black both times, (b) `open_usd_string!` was called exactly **once** across both (instrument a counter / assert `screen.authored` and that the handle in `plot2robj` is stable). RED (M1 re-authors → counter=2).
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3a:** evolve the `Screen` struct (drop `plot2usd` + the vestigial `open_results`; `setup` was already removed in M1.6; add the M2 fields incl. `authored::Bool`); update `close(Screen)` (no `open_results` loop — close the renderer; `OV.destroy!` any bindings in M2.3). Relocate `author_root_from_scene!` to `usd.jl`/`composition.jl`.
- [ ] **Step 3b:** rewrite `colorbuffer`: if `!screen.authored`, author root once (`author_root_from_scene!`) + `insertplots!`; set `authored=true`. Else (M2.1) no-op the re-author. Render. (M2.2 adds the pull-nodes step here.)
- [ ] **Step 3c:** `insert!`/`insertplots!`/`add_scene!` per the interfaces (idempotent via `plot2robj`; `add_scene!` authors a `def Scope` per subscene and stores its redraw listeners in `scene_listeners` for M2.4 teardown). `insert!` records `OvrtxRObj(path, handle, Dict())`.
- [ ] **Step 4 (insert test):** subprocess — `Scene` + `Makie.push_screen!(scene, screen)`; `scatter!(scene, …)` after attach → `plot2robj` grows by one and renders; a recipe (`poly!`) registers its atomic children. PASS.
- [ ] **Step 5:** commit `feat(M2.1): open-stage Screen + imperative insert!/insertplots!/add_scene!`.

**Acceptance:** the stage is authored once and reused; a static scene renders as in M1; live `plot!` on a displayed scene authors a USD reference + records an `OvrtxRObj`.

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

## Task M2.3 — Persistent hot-path bindings (`map_attribute` / `bind_array_attribute`) + Point3f write fix

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
- [ ] **Step 5:** commit `feat(M2.3): persistent map/bind hot-path bindings + Point3f array writes`.

**Acceptance:** transforms/points update through persistent bindings created once, no re-authoring per frame.

---

## Task M2.4 — `delete!` / `delete!(scene)` / `empty!` — leak-free teardown

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
- [ ] **Step 5:** commit `feat(M2.4): leak-free delete!/delete!(scene)/empty! teardown`.

**Acceptance:** add/delete of plots and subscenes leaves zero residual prims, bindings, nodes, or listeners; `remove_usd!` closes the M1.6 handle leak.

---

## Task M2.5 — Hot-path benchmark (de-risk gate)

**Files:** `bench/hot_path.jl`, `bench/RESULTS.md`. Test: a threshold assertion in `test/` (or a `bench` target).

- **Interfaces — Produces:** a benchmark animating **N transforms** (mapped) and **N points** (bind+write) per frame, reporting updates/sec + frame time, on the A5000.
- [ ] **Step 1:** write the benchmark (reuse the subprocess harness; create the renderer once; loop M frames over N objects).
- [ ] **Step 2:** run; record updates/sec + frame time to `bench/RESULTS.md`.
- [ ] **Step 3:** a test asserting the measured rate ≥ target (**≥30 Hz for ~10⁴ instance transforms OR ~10⁵ points**). If below target → the result documents the gap and escalates GPU-resident DLPack writes to M3 (do NOT silently pass).
- [ ] **Step 4:** commit `bench(M2.5): hot-path throughput benchmark + results`.

**Acceptance:** map/bind throughput sustains interactive rates on the A5000 (or the shortfall is measured + escalated).

---

**M2 GATE:** live attribute/transform/color edits push minimal C writes on an **open** stage (no per-frame re-author); plot + subscene add/delete is leak-free (`remove_usd!` + listener deregistration); the hot-path benchmark meets target (or escalates). ✅ → M3 (interactive viewport + live camera — the deferred camera-live-update research lands there).

---

## Open assumptions this plan validates early (flag, validate in the named task — with fallbacks)
1. **Live writes on a REFERENCED prim** (`/World/plot_<id>`) are honored by ovrtx (M2.2 Step 1). M0 only proved root-prim writes. Fallback: targeted re-reference (`remove_usd!`+`add_usd_reference!`).
2. **`map_attribute`/`bind_array_attribute`** behave as the C API suggests (M2.3). Fallback: M0 `write_attribute` (correct, no zero-copy) — feeds the M2.5 benchmark verdict.
3. **Makie compute outputs** (`:positions_transformed_f32c`, `:model_f32c`, `:scaled_color`, …) exist + resolve for our plots (Spike B verified; re-confirm on the real plots in M2.2). Fallback: source from the plot directly (M1 style) + `add_computation!` for packed color.
4. **Hot-path throughput** meets interactive rates (M2.5 — the explicit de-risk gate).
5. **(carried to M3, NOT M2)** Live CAMERA/LIGHT updates need a mechanism ovrtx honors — `omni:xform` is PROVEN ignored on those prims (M1.3). M2 keeps them baked; M3 spikes the live path (e.g. re-bake-on-camera-change vs a different attribute/API).
