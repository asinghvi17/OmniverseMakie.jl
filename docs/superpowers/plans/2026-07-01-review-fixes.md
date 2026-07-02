# Review-Fixes Implementation Plan (design-doc style)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development, one
> executor per TRACK (tracks run in PARALLEL worktrees; tasks within a track are SEQUENTIAL).
> This is deliberately a DESIGN DOC, not a code-level plan: each task states the problem
> (with file:line evidence), the locked design decisions, and the acceptance tests — the
> implementing subagent owns the code. Checkboxes track task completion.

**Goal:** Fix every finding from the 2026-07-01 whole-library review (correctness, perf,
duplication, hygiene) with behavior-preserving refactors verified by tests.

**Architecture:** Seven parallel, file-disjoint tracks (A=FFI, B=translation, L=lights,
C=present paths, D=NanoVDBWriter, E=test harness, F=docs), each a branch off `main@89a672b`
in its own worktree. Controller merges in a fixed order with full-suite gates.

**Tech stack:** Julia, Makie/ComputePipeline, ovrtx via ccall (generated LibOVRTX), OpenUSD
text authoring, CUDA.jl + GLMakie weakdep exts, NanoVDB binary writer.

## Global constraints (apply to every task)

- House import style: `import X as Y` / `using X: name` — never `const Y = X`.
- Deps via Pkg.jl operations only; never hand-edit Project.toml `[deps]`.
- Do NOT touch generated `lib/LibOVRTX/src/libovrtx_api.jl`; NanoVDBWriter's attribution
  header is a LICENSE requirement — immutable.
- GPU tests follow the house pattern: env-gated subprocess progs (`test/helpers.jl`
  `run_ovrtx_subprocess`), skip-if-absent, `OK_*` sentinel + regex-parsed hard `@test`s,
  retry loop for the known intermittent `GeometryGroup::attachToContext` startup crash.
- Volume test data must be spatially GRADED — IndeX Direct renders uniform density fully
  transparent (M2 lesson). Volume colors are grayscale-only by design — never a bug.
- Prefer PURE tests (USDA string emission, snapshot structs, writer bytes) over GPU tests;
  run only targeted testsets inside a task. NEVER run two GPU testsets concurrently across
  worktrees (single GPU + the startup-crash flake). Full `Pkg.test()` happens only at
  integration gates, run by the controller.
- Behavior-preserving refactors need a regression anchor captured BEFORE the change
  (golden bytes/strings), not after.
- Commit per task, message style `fix(review): <what>` / `perf(review): <what>` /
  `test(review): <what>`, trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Controller (not subagents) owns merges and pushes; verify `git status` after every
  subagent.

## Track / conflict map

| Track | Owns (exclusive unless noted) |
|---|---|
| A | `src/binding/OV.jl`, `lib/LibOVRTX/src/LibOVRTX.jl`, `src/binding/dlpack.jl`; `src/screen.jl` ONLY the resolver-cache hunks (~:417-426 + invalidation call sites) |
| B | `src/compute.jl`, `src/translation/{primitives,materials,usd,camera,volume,meshes}.jl`; `test/volumes_live_test.jl` ONLY the all-zero testset |
| L | `src/translation/lights.jl`; `src/screen.jl` ONLY the `last_camera`/`last_lights` field types (:20-21) |
| C | `ext/OmniverseMakieGLMakieExt.jl`, `ext/OmniverseMakieCUDAExt.jl`, `src/tonemap.jl` |
| D | `lib/NanoVDBWriter/` (all) |
| E | `test/helpers.jl`, `test/runtests.jl` include order, `test/m0_*`, retry/const adoption edits across existing test files (NOT the volumes all-zero testset B owns) |
| F | `README.md`, `ARCHITECTURE.md`, `.gitignore`, root `M*_PLAN.md` moves |

`src/screen.jl` is touched by A and L in disjoint hunks — flagged so merges stay trivial.
New test files: name them per track (`test/review_a_*.jl` etc.) to avoid collisions; add
to `runtests.jl` at merge time (controller resolves the one-line include conflicts).

**Merge order (controller):** B → A (full suite + triage — see INT-1) → L → C → D → E → F,
full suite again at the end. Rationale: B removes the bogus instancer `"points"` write
BEFORE A starts throwing on op errors, so A's gate measures real regressions.

---

## Track A — FFI hardening (`src/binding/OV.jl` + `LibOVRTX.jl`)

### Task A1: op-error propagation + alive-check ordering in `enqueue_wait` [H]
- [ ] Done

**Problem.** `enqueue_wait` (OV.jl:78-85) checks only the wait's own status; per-op failures
arrive in `ovrtx_op_wait_result_t.error_op_ids`/`num_error_ops` (generated struct, and the
header says a missing USD file is reported there) — zero readers exist, so `open_usd!` of a
missing file and failed attribute writes report success. Also the `r.alive` guard runs after
the ccall has already executed at call sites like `step!` (arguments evaluate first), so a
closed Renderer passes `C_NULL` into ovrtx.

**Fix design (locked).**
- Change `enqueue_wait` to take the enqueue as a THUNK: `enqueue_wait(f, r, op; timeout_ns)`
  runs the alive check first, then `f()`. Mechanically convert all ~12 call sites
  (`open_usd!`, `open_usd_string!`, `step!`, `reset!`, `remove_usd!`, `add_usd_reference!`,
  binding create/unmap/destroy/write, pick enqueue) to `do`-blocks.
- After a successful wait, if `num_error_ops != 0`, resolve messages via
  `ovrtx_get_last_op_error` (copy the transient thread-local strings IMMEDIATELY, same
  discipline as `LibOVRTX.check`) and throw `OVRTXError(op, msg)`.
- Check the return of `ovrtx_unmap_render_var_output` on the two `map_cpu*` success paths
  (OV.jl:229, 269). Leave close/finalizer-path destroys unchecked (defensible).
- SPIKE (first step, subprocess): confirm `open_usd!` on a missing path populates
  `error_op_ids`; record what an attribute write to a nonexistent attr reports (both
  outcomes documented in the task ledger — the second feeds B4's design note).

**Tests.** Subprocess: `@test_throws OVRTXError open_usd!(r, "/nonexistent.usda")`;
`step!`/`reset!`/`colorbuffer` on a closed-Renderer errors cleanly (alive check, no ccall);
existing m0/m1 render testsets stay green.

### Task A2: leak-proof readback + `with_mapped_hdr` [M]
- [ ] Done

**Problem.** `map_cpu`/`map_cpu_f32` (OV.jl:207-272) leak the mapping if the copy throws
(no try/finally, unlike `read_pick_hit`); `step!` (OV.jl:146-157) leaks its results handle
when the wait times out/fails (h[] filled by the enqueue, `StepResult` never constructed);
offscreen readback then does a SECOND full pass (`cwh_to_matrix`, dlpack.jl byte-by-byte
repack) even though `RGBA{N0f8}` is layout-identical to the 4-byte texel.

**Fix design (locked).**
- Wrap both map bodies' copy in `try/finally unmap`.
- `step!`: on `enqueue_wait` failure with `h[] != 0`, best-effort
  `ovrtx_destroy_results` before rethrow.
- Single-pass readback: build the `Matrix{RGBA{N0f8}}` directly from the mapped `[C,W,H]`
  memory in one pass (reinterpret/permute — implementer's choice), preserving the verified
  top-left-origin/no-flip orientation (dlpack.jl header, m1_orientation_test). Retire or
  slim `cwh_to_matrix` accordingly.
- Add `with_mapped_hdr(f, sr::StepResult, name="HdrColor")` — calls `f(raw16, W, H)` with
  the still-mapped `[C,W,H]` `Float16` view, unmaps in `finally`, returns `f`'s result.
  **Produces (for INT-2):** exactly that signature. `map_cpu_f32` becomes a thin wrapper.
- While in these bodies: inspect `ro[].status`/`error_message` after a successful map call
  (review: never checked in `map_cpu`/`map_cuda`) and throw `OVRTXError` on failure.

**Tests.** m1 orientation + m0 render testsets green (proves single-pass readback is
pixel-identical); a pure test for `with_mapped_hdr` result passthrough + unmap-on-throw can
run in the CUDA/GL-free subprocess using an ovrtx step (env-gated).

### Task A3: `PathResolver` lifetime + cache invalidation [M]
- [ ] Done

**Problem.** OV.jl:472 says the resolver is "valid while the stage composition is
unchanged"; screen.jl:417-419 caches one per Screen claiming renderer-lifetime validity and
reuses it across insert/delete. One claim is wrong; the struct also holds no `Renderer`
reference (nothing roots it during `@ccall` through raw vtable pointers, no alive guard).

**Fix design (locked).** Treat composition-scoped as ground truth (the conservative
reading): store `r::Renderer` in `PathResolver`, alive-guard + `GC.@preserve r` around the
vtable ccalls in `resolve_prim_path`; invalidate the Screen cache
(`screen.path_resolver = nothing`) wherever the stage composition changes —
`add_usd_reference!`/`remove_usd!` call sites in screen lifecycle (`insertplots!`, plot
delete, `empty!`). Resolver is already lazily rebuilt (screen.jl:420-426), so invalidation
is one assignment. Fix both docstrings to say composition-scoped.

**Tests.** Subprocess: pick → `delete!` a plot → add a new plot → pick again resolves the
NEW plot (this is the stale-resolver scenario; today it may silently mis-resolve). Existing
M6.B pick testsets stay green.

### Task A4: `Binding` finalizer must not block/throw; `ovx_string` SubString fix [M/L]
- [ ] Done

**Problem.** `finalizer(destroy!, b)` (OV.jl:971) reaches an infinite-timeout
`enqueue_wait` + throwing `check` inside GC finalization (can stall GC on a wedged queue;
a throw aborts before `alive=false`). Separately, `ovx_string`'s `SubString{String}` branch
(LibOVRTX.jl:31-32) is a guaranteed MethodError — `unsafe_convert(Cstring, ::SubString)`
has no method — and the comment asserts support Base doesn't provide.

**Fix design (locked).** `destroy!(b; from_finalizer::Bool=false)`: finalizer path uses a
finite timeout and swallows errors after marking `alive=false`/`map_handle=0` (leaking one
binding beats wedging GC; renderer close frees the pool anyway). Explicit `destroy!`
unchanged (throws). For strings: delete the SubString branch; add
`ovx_string(s::AbstractString) = ovx_string(String(s))` (ovx_string_t is ptr+len so the
String conversion is the only safe general path); fix the comment.

**Tests.** Pure: `ovx_string(SubString("abc/def", 1, 3))` round-trips via
`String(::ovx_string_t)`. Binding teardown covered by existing m2_binding/m2_delete
testsets staying green.

### Task A5: index-config synthesis robustness [L]
- [ ] Done

**Problem.** `_synth_index_config` (index_config.jl:32-41) interpolates the env-supplied
libs path into JSON unescaped (a `"` or `\` yields invalid JSON carb rejects at init) and
merges at the FIRST `"app": {` occurrence with exact-spacing sensitivity — a nested match
would plant the token where carb ignores it, silently.

**Fix design (locked).** Add a tiny `_json_escape(s)` (backslash, quote, control chars)
for the interpolated value; anchor the merge to a line-start `"app": {` occurrence
(regex with `m` flag) and error clearly when no top-level match is found (the existing
no-block error message stays). Everything still lands in `_ensure_index`'s catch→warn
(never throws to the user) — that contract is unchanged.

**Tests.** Pure: a libs path containing `"` produces valid JSON (parseable by a strict
parser in the test); a fixture config with a NESTED `"app": {` before the top-level one
gets the token in the top-level block.

---

## Track B — Translation correctness (`src/compute.jl` + translation/)

### Task B1: one colormap mapper + one colorrange resolver, NaN-safe [H]
- [ ] Done

**Problem.** The `to_colormap` + `interpolated_getindex` comprehension exists 3× (
materials.jl:37-41, primitives.jl:303-313, compute.jl:113-120) with two divergent
colorrange resolvers (`_colorrange` materials.jl:49-56 vs `_volume_colorrange`
compute.jl:303-307). None is NaN-safe: automatic colorrange falls back to raw `extrema`
→ `(NaN, NaN)`, and Makie's `interpolated_getindex` hard-errors on non-finite lookups —
so a NaN-masked `surface!` (whose mesh path DELIBERATELY skips non-finite cells,
primitives.jl:294) throws at first colorbuffer.

**Fix design (locked).** In materials.jl: `_resolve_colorrange(plot, values)` (explicit
2-tuple honored verbatim; automatic → extrema over FINITE values only; no finite values →
`(0f0, 1f0)`) and `_map_through_colormap(plot, values) -> Vector{NTuple{3,Float32}}`
(finite → interpolated lookup; non-finite → `_rgb(to_color(plot.nan_color[]))`, guarded by
`haskey`; RGB-only is fine — non-finite surface vertices belong to no emitted face).
Replace all 3 sites; `_volume_colorrange` delegates (keeps its Float64 return + Automatic
handling). **Produces:** those two signatures (B2 does not depend on them; nothing else
does either — self-contained).

**Tests.** Pure: surface-colors path with NaN z produces finite RGB for every vertex and
no throw; explicit `colorrange=(a,b)` byte-identical to today for finite data (golden USDA
string comparison on a small mesh, captured before the change); numeric-scatter and volume
colorrange behavior unchanged.

### Task B2: NaN-separated lines [H]
- [ ] Done

**Problem.** Makie's standard broken-line idiom (NaN separators, e.g. contour output)
flows NaN into USDA: `_bbox_diag` (primitives.jl:73-85) propagates NaN through min/max and
its `d < 1e-8` guard (false for NaN) → `_curve_width` (primitives.jl:214-219) emits
`float[] widths = [NaN]`; the Lines author (compute.jl:256-276) always emits ONE polyline
`curveVertexCounts=[n]` including the NaN points.

**Fix design (locked).**
- `_split_nan_runs(pts) -> (finite_pts, counts)`: contiguous finite runs of length ≥ 2
  become curve entries (BasisCurves is natively multi-curve); runs of 1 dropped; no runs →
  the author returns `nothing` (existing empty behavior).
- Per-vertex color arrays are filtered with the SAME mask so colors stay index-aligned.
- `_bbox_diag` skips non-finite points; `_curve_width` therefore stays finite.
- LineSegments: drop any segment with a non-finite endpoint.
- Live positions push (`push_to_ovrtx!` `:positions_transformed_f32c` route): when the new
  positions contain non-finite values, re-split and write `points` AND `curveVertexCounts`
  (counts via `OV.write_array_attribute!`, same shape as `_push_faces!`). SPIKE first:
  verify a live `curveVertexCounts` write is honored (subprocess render, pixel-count
  change). If not honored → fall back to remove+re-reference (the volume-reload pattern,
  translation/volume.jl:155-205) for NaN-topology edits, and say so in a comment.
- NOTE: the persistent `points` binding is sized at author time; a split changes point
  count. When lengths differ from the bound size, take the one-shot write path (binding
  route only when `length == bound length`) — implementer verifies which path handles
  resized arrays and documents it.

**Tests.** Pure: `_split_nan_runs` unit cases (leading/trailing/consecutive NaN, all-NaN,
no-NaN — no-NaN must be byte-identical USDA to today, golden-captured first). Subprocess:
`lines!` of a NaN-separated path renders (>N lit px) with a visible GAP (two clusters'
centroids), and a live edit that moves the NaN renders without error.

### Task B3: empty→fill rebuild is universal [M]
- [ ] Done

**Problem.** The diff-node callback (compute.jl:687-710) rebuilds only on first resolve or
new screen; when `author_usd_prim!` returned `nothing` (empty scatter/lines/mesh/volume
guards), `robj` stays `nothing` forever on that screen and later data edits are silently
dropped — "start empty, fill in the animation loop" never renders. Documented for Volume
only (compute.jl:329-333); actually universal. Also the pick maps (`plot2robj`/`path2plot`)
are only populated at register time (compute.jl:713-718), so a later in-callback rebuild
wouldn't be pickable.

**Fix design (locked).** In the callback's else-branch: if `robj === nothing` and any
tracked input changed → attempt the build path (`author_usd_prim!` + `bind_hot_attributes!`,
same as first-resolve). Extract `_register_robj_maps!(screen, plot, robj)` (sets `.plot`,
`plot2robj`, `path2plot`) and call it from register_ovrtx_robj! AND from the callback after
a successful late build. Delete the Volume-only limitation comment (compute.jl:329-333) —
the guard `all(iszero, scalars) && return nothing` stays; it now self-heals.

**Tests.** Subprocess: `scatter!(scene, Point3f[])` → colorbuffer (blank) → fill positions
→ colorbuffer renders (>N lit px) and the plot is pickable. UPDATE
`test/volumes_live_test.jl`'s all-zero testset: it currently pins the no-op; it must now
assert the FILL renders (graded data — uniform is invisible).

### Task B4: scatter/meshscatter live positions [H]
- [ ] Done

**Problem.** Scatter/MeshScatter track `:positions_transformed_f32c` (compute.jl:82-83)
but `_points_binding_attr(::Any) = nothing` (compute.jl:573) drops the push to
`write_array_attribute!(r, prim, "points", value)` (compute.jl:509-515) — on a
PointInstancer the attribute is `positions`, so a live position edit is a visual no-op that
still sets `routed=true` → burns an accumulation reset. On a MATERIALIZED scatter the prim
is the merged tessellated mesh: the same route writes n instance positions over an
n×~150-vertex `points` array. Known M3 carry (bench/RESULTS.md), never routed.

**Fix design (locked).**
- SPIKE first (subprocess): write `positions` on an authored instancer via
  `write_array_attribute!` and confirm the render moves (pixel centroid). Then try a
  persistent binding on `positions` (`bind_array_attribute`-equivalent path used for
  `points`); adopt it if it works, else one-shot writes.
- Route: `push_to_ovrtx!` writes `positions` for non-materialized Scatter/MeshScatter
  (count-change caveat: same one-shot-vs-binding size rule as B2).
- Materialized (merged mesh): `@warn maxlog=1` "live position edits on a materialized
  scatter need a re-author — skipped" + `routed = false` (no reset burn, no corrupting
  write). Full merged-mesh re-tessellation is a NON-GOAL (below).
- Update the `push_to_ovrtx!` docstring routing table (compute.jl:484-494) and strike the
  carry note in bench/RESULTS.md.

**Tests.** Subprocess: non-materialized `scatter!` live position edit moves the lit
centroid ≥ threshold; materialized scatter edit: image UNCHANGED, exactly one `@warn`, and
`_PUSH_OBSERVER` shows no routed write (existing observer hook, compute.jl:417).

### Task B5: dedupe the materialized-reference epilogue [M]
- [ ] Done

**Problem.** `h = add_usd_reference!; bind_material!; robj = OvrtxRObj(path, h);
robj.material_shader = material_prim_path(plot) * "/Shader"` is copy-pasted 6× (
compute.jl:171-175, 206-209, 240-243, 266-269, 287-290, 669-670); the `"/Shader"` suffix
must stay in sync with `def Shader "Shader"` (materials.jl:130).

**Fix design (locked).** One `_add_materialized_reference!(screen, path, usda_or_handle,
plot) -> OvrtxRObj` used at all 6 sites (the register_ovrtx_robj! site at :669 already has
a handle — accept either, implementer's call on one function vs two thin methods). Pure
refactor; no behavior change.

**Tests.** m3_material + m3_primitives_material + m1_mesh_render testsets green (these
exercise all six shapes).

### Task B6: USD string hygiene + texture-name collision [L]
- [ ] Done

**Problem.** User-supplied strings enter USDA unescaped: `@$(path)@` asset refs break on
`@` in the path (materials.jl:~121 emit, volume.jl:84 filePath); a VDB `field` name becomes
a prim identifier (volume.jl:80 `def OpenVDBAsset "$(field)"`); `_validate_camera_path`
(camera.jl:76-85) passes `"/World/My Camera"`. Separately, `_merge_material_input!` passes
`plot = nothing` to `_texture_asset_for` (materials.jl:318-321), so two plots passing an
image as `base_color_texture` collide on `tex_$(objectid(nothing)).png` and overwrite each
other.

**Fix design (locked).** In usd.jl: `_usd_identifier(name)` (assert `[A-Za-z_][A-Za-z0-9_]*`,
clear error naming the offender) and `_usd_asset_path(path)` (contains `@` → wrap `@@@…@@@`;
contains `@@@` → error). Apply at the three emit sites + camera path segments. Thread the
real `plot` from `material_inputs_from` (materials.jl:297-303) into
`_merge_material_input!` → `_texture_asset_for(value, plot)`, and suffix the temp-PNG name
with the input key (`tex_<objectid>_<key>.png`) so two image inputs on ONE plot don't
collide either.

**Tests.** Pure: identifier rejection message; `@`-path wraps to `@@@`; camera path with a
space errors; two plots + image textures → distinct paths (filenames differ).

### Task B7: authoring-path allocations [L]
- [ ] Done

**Problem.** Mesh conversion allocates one `Int[]` per face, in three identical
comprehensions (compute.jl:158, 203, 229), only for `usda_mesh` (usd.jl:64-67) to
re-flatten; `usda_mesh`/`_point3f_list` build a temporary String per vertex before `join`
(usd.jl:60-61, 70-71); `_merged_instances_mesh` (primitives.jl:150-171) push!es without
sizehint. Semi-hot for large meshes.

**Fix design (locked).** Emitters accept flat `(counts::Vector{Int}, indices::Vector{Int})`
built once by a shared `_flat_faces(faces)` helper (replaces the 3 comprehensions), and
stream number lists through one `IOBuffer` with `print` loops instead of per-element
Strings + join. Output must be BYTE-IDENTICAL — capture golden USDA strings for a small
mesh/instancer fixture before refactoring and assert equality after (Julia `print` of
Float32 has no `f0` suffix — already relied on). Also on the hot write path: type
`OvrtxRObj.bindings` as `Dict{Symbol,OV.Binding}` (compute.jl:30 — values are always
`OV.Binding`; kills a dynamic dispatch per live write) and remove the full-buffer
`collect(reinterpret(Float32, src))` copy in `_push_points_binding!` (compute.jl:408-413)
— write through the reinterpreted view under `GC.@preserve` (verify `write_binding!`'s
`pointer(data)` works on a `ReinterpretArray`; if not, reuse a per-binding scratch buffer).

**Tests.** Pure golden-equality tests; `@allocated` on `usda_mesh` for a 10k-vertex fixture
reduced ≥ 5× (loose bound, sanity not benchmark); m1 render testsets green.

---

## Track L — Lights (`src/translation/lights.jl`)

### Task L1: structural light change: fail loud, not corrupt [M]
- [ ] Done

**Problem.** On a light-count change `sync_lights!` (lights.jl:355-365) refreshes
`screen.last_lights` and returns false; nothing ever re-authors (docstring admits "needs a
stage re-open" that no code performs). The added light silently never renders — and because
the snapshot now has the new length, the NEXT edit diffs against never-authored prims.
Also RectLight re-implements `_direction_to_xform_matrix` inline (lights.jl:172-181 vs
:65-74) — drift risk.

**Fix design (locked).** On count mismatch: DO NOT advance the snapshot; `@warn maxlog=1`
("adding/removing lights on a live Screen is not supported — create a new Screen");
return false. (The mismatch then stays detectable every frame; the warn is once.) First
call (`old === nothing`) keeps today's seed behavior. RectLight: build via
`M = copy(_direction_to_xform_matrix(dir)); M[4,1:3] .= p` — delete the inline copy.

**Tests.** Pure: snapshot preserved + warn fired on a count change (`_lights_snapshot` and
the mismatch branch are testable without a renderer by faking the screen field); RectLight
matrix equals DistantLight matrix + translation row (golden numeric check).

### Task L2: sync_lights! allocation diet + typed Screen snapshots [M]
- [ ] Done

**Problem.** `_lights_snapshot` runs EVERY frame (screen.jl:321-323 → lights.jl:357) even
when nothing changed: `Dict{DataType,Int}` in `_enumerate_lights` (lights.jl:242-252),
fresh path strings (`light_prim_name` interpolation, :31), 4×4 `Matrix{Float64}` through
~6 temporaries per directional light (:65-74), all stored in `Any[]` (:326) compared by
dynamic dispatch. `Screen.last_camera`/`last_lights` are `::Any` (screen.jl:20-21).

**Fix design (locked).** Introduce `struct LightState` (intensity::Float32,
color::NTuple{3,Float32}, xform::Union{Nothing,NTuple{16,Float64}}) — xform as a stack
tuple, converted to the Matrix form only when actually writing. Snapshot type:
`Vector{Tuple{String,Union{LightState,Nothing}}}`; reuse the PATH strings from the old
snapshot when the count is unchanged (paths are invariant between structural changes,
which L1 now makes terminal). Type the two Screen fields concretely
(`Union{Nothing,<snapshot type>}` / the camera NamedTuple's concrete type). Keep
`_lights_snapshot`'s public shape (test/m2 uses it? — grep and preserve callers).

**Tests.** Pure: `@allocated` building a second snapshot for an unchanged 3-light scene
is ≤ a small constant bound (no Dict, no fresh strings — assert e.g. < 1KiB, tune to
measured); state comparison still detects each of intensity/color/direction changes
(existing m1_lights + m2 testsets green).

---

## Track C — Present paths (`ext/`, `src/tonemap.jl`)

### Task C1: hoist `exp2` out of the per-pixel tonemap [L]
- [ ] Done

**Problem.** `scale = exp2(exposure)` sits inside `@inline tonemap` (tonemap.jl:16-19),
executed W×H times per frame on CPU (tonemap_frame :23-30) and per CUDA thread
(CUDAExt:51-60).

**Fix design (locked).** `tonemap(rgb, scale)` takes the PRE-COMPUTED scale;
`tonemap_frame` and the CUDA kernel compute `exp2(exposure)` once per frame/launch and
pass it. Keep a deprecated-free story: just change the two call sites + kernel signature
(internal API, no external users). Docstring: parameter is now a linear scale, callers
convert from EV.

**Tests.** Existing M6 "host vs CUDA-kernel tonemap agreement" testset green (it pins
byte-equality of both paths — the refactor must not change a single pixel).

### Task C2: CPU present — one pass, zero steady-state garbage [H]
- [ ] Done

**Problem.** `present!(::Val{:cpu})` (GLMakieExt:346-353) chains 4 full-frame allocations
per tick: `Float32.(raw16)` (OV.jl:267) → `tonemap_frame` out (tonemap.jl:25) →
`permutedims` → `reverse` (`_orient_for_display`, GLMakieExt:24) → sets the image
Observable. ~13MB/tick at 800×600, ~0.8GB/s garbage at 60Hz on GLMakie's render task.

**Fix design (locked).** Fuse tonemap+orientation into ONE loop writing a session-owned
pre-oriented `[W,H]` `Matrix{RGBA{N0f8}}` cached on `ViewportSession` (recreate on size
change — resize already rebuilds the Screen); then mutate the image Observable's existing
array in place + `notify`. Orientation must reproduce `reverse(permutedims(frame), dims=2)`
exactly (`out[j, H+1-i] = tonemap(...)`) — the same fused indexing C3 uses, keep them
visibly parallel. The `map_cpu_f32` Float32 conversion remains this task's ONE transient
(removed later by INT-2). `resize_viewport!`/`interactive_display` orientation contract
(GLMakieExt:22-24 comment) updated to name the fused loop as the one implementation.

**Tests.** GPU/GL subprocess: existing m5 viewport testsets green; new assertion — second
tick's `@allocated` below a threshold that fails against today's 4-buffer chain (e.g.
< 2 × frame bytes; tune to measured); pixel-equality of one presented frame vs the old
chain (compute both in the test program).

### Task C3: GPU present — fused oriented kernel + cached buffers [M]
- [ ] Done

**Problem.** `_tonemap_oriented` (CUDAExt:74-76) = kernel + `permutedims` + `reverse` = 3
kernels + 3 device allocations per frame; `_image_texture(session)` re-resolves the GL
texture through `plot2robjs` + a uniforms Dict every `_gpu_present!` (:93-97, :188).

**Fix design (locked).** New kernel writing oriented output directly
(`out[j, H+1-i] = tonemap(rgb, scale)`) into a `CuArray{RGBA{N0f8}} [W,H]` cached on
`GPUBlitState` (allocate lazily/on size change; `gpu_unregister!` frees). Cache the
resolved `Texture` on `GPUBlitState` too, invalidated in `_unregister!`; the existing
`st.tex_id != tex.id` re-register guard keeps working against the cached object.
KEEP `cuStreamSynchronize` (:239) — dropping it is a NON-GOAL (unverified safety).
KEEP `tonemap_kernel_to_matrix` (test hook) and `_tonemap_dev` if the agreement test needs
the unoriented form; the agreement test may instead be extended to cover the oriented
kernel against `_orient_for_display(tonemap_frame(...))` on the host.

**Tests.** CUDA subprocess: M6 agreement + GPU-direct blit + benchmark testsets green;
new: oriented-kernel output == host `_orient_for_display(tonemap_frame(hdr, ...))`
byte-equal; steady-state device allocations per tick == 0 (CUDA.@allocated or pool stats,
implementer's choice).

---

## Track D — NanoVDBWriter (`lib/NanoVDBWriter/`)

### Task D1: golden anchor + input guards + self-documenting constants [M]
- [ ] Done

**Problem.** No guards: a NaN voxel counts as active (`v != background`, :256-262) and
poisons vmin/vmax; zero `extent` writes `Inf` voxel size silently (:423-426); zero-dim
arrays fall into a misleading error. The `+ 63` node-offset bias (:510-512) is the file's
one unexplained magic number; the little-endian assumption is unstated.

**Fix design (locked).** FIRST capture golden SHA256s of `save_nanovdb` output for two
fixtures (single-leaf 8³, multi-lower-node ~64³ graded) into the sub-package test — the
regression anchor for D2. Then: up-front validation with clear errors (`all(isfinite,
data)`, `all(>(0), size(data))`, `all(>(0), extent)`); a named constant for the offset bias
derived from the existing constants block with a one-line derivation comment; a top-level
`@assert ENDIAN_BOM == 0x04030201` (little-endian hosts only) with a module-note.
Attribution header untouched.

**Tests.** (all pure, no GPU) `@test_throws` for NaN input / zero extent / zero dims /
all-background; golden hashes recorded and asserted; multi-node fixture's
`parse_nanovdb_header` invariants (magic, version 32, voxel_count) + node-offset sanity
via the named constant.

### Task D2: memory refactor — no leaf copies, no header concat [H]
- [ ] Done

**Problem.** Phase 1 stores `copy(scratch)` per leaf (:268) — 262k × 2KiB held until
Phase 5 for a 512³ volume; `save_nanovdb` duplicates the entire node buffer just to
prepend a 736-byte header (:463-464); Phase 5 writes 512 scalars per leaf via `write_buf!`
(:345-347). ~1.6GiB transient per write — and this runs on EVERY live volume edit.

**Fix design (locked).** Extract the leaf-scratch fill into a helper used by Phase 1
(min/max/mask) and RE-USED by Phase 5 to re-read values straight from `data` via
`leaf_coords` (drop `leaf_values` entirely); write the file as three sequential writes
(io header, grid header, node buffer) with `grid_size` computed arithmetically; bulk-copy
each leaf's 512 Float32 payload (reinterpret/unsafe_copyto!, little-endian guaranteed by
D1's assert). BYTE-IDENTICAL to D1's golden hashes — that is the acceptance bar.

**Tests.** Golden hashes from D1 unchanged; `@allocated save_nanovdb` for the 64³ fixture
< 3× payload bytes (vs ~10×+ today; tune to measured); the GPU volumes testsets
(volumes_writer/plot/live) green at track end.

### Task D3: drop the dead Zlib path [L]
- [ ] Done

**Problem.** `Zlib_jll` is a hard dep used only by `compress_zlib` (:590-604), which
nothing calls; the kept code has a latent bug (deflate return never checked, hardcoded
"1.2.11" version string).

**Fix design (locked).** Delete `compress_zlib` + the import; `Pkg.rm("Zlib_jll")` in the
sub-package project (Pkg op, not TOML editing). Module docstring notes the ZIP-codec path
was removed and lives in git history. GeometryBasics STAYS (used in signatures).

**Tests.** Sub-package `Pkg.test` green; `Pkg.status` shows Zlib_jll gone; main-package
volumes pure tests green.

---

## Track E — Test harness (`test/`)

### Task E1: truthful watchdog [H]
- [ ] Done

**Problem.** helpers.jl:82-85: after `kill(p)` the exitcode is read WITHOUT `wait(p)` —
live-verified `typemin(Int64)` while dying, `0` once reaped (SIGTERM sets termsignal=15,
exitcode=0) — so `@test exitcode == 0` can PASS on a timed-out child; `take!(out)` races a
still-writing child; SIGTERM-only (a GPU-wedged child can ignore it).

**Fix design (locked).** On timeout: `kill(p)` (TERM) → `timedwait` ~10s → still running →
`kill(p, Base.SIGKILL)` → ALWAYS `wait(p)` before reading output. Return contract stays a
2-tuple `(exitcode, output)` for the ~50 call sites, but exitcode is now truthful:
`success(p) ? 0 : (timed_out ? -9 : (p.termsignal != 0 ? -p.termsignal : p.exitcode))` —
any non-zero fails the existing `@test exitcode == 0` sites, which is the point. Docstring
documents the encoding.

**Tests.** Pure (no ovrtx): run the helper against (a) `exit(0)` child → 0, (b) `exit(3)`
child → 3, (c) `sleep(60)` child with timeout=2 → negative + output captured, (d) a child
that traps SIGTERM and keeps sleeping → SIGKILL path returns negative in bounded time.

### Task E2: shared harness capabilities + m0 migration [M]
- [ ] Done

**Problem.** The 4 pre-helpers m0 tests duplicate the runner inline with an UNBOUNDED
`wait(p)` under a comment claiming a 300s timeout (m0_render_test.jl:61-64) — a hung m0
child blocks `Pkg.test` forever; helpers.jl:3 even cites m0_render as its source. The
required ovrtx-startup retry loop + `_LIBS` default are copy-pasted 4× in volumes tests;
~23 subprocess progs re-implement the lit-pixel helper and re-derive `>300`/`0.04`
thresholds.

**Fix design (locked).** Extend `run_ovrtx_subprocess` with `retries::Int=1` +
`ready_marker::AbstractString=""` (retry while marker absent, preserving each attempt's
output on failure); add `_HELPER_INDEX_LIBS` (the volumes default, env-overridable) and a
`PROG_PIXEL_HELPERS::String` prelude (defines `nonblack(img)`, `lit_centroid(img)`,
`LIT_PX_MIN=300`, `LUM_MIN=0.04f0`) that progs splice in. Move `include("helpers.jl")`
above the m0 includes in runtests.jl and port the 4 m0 files to the helper (delete the
inline runners + wrong comment). **Produces:** the kwargs + const names above (E3 adopts).

**Tests.** m0 testsets green; a pure test that `ready_marker` retries (child that fails
first via a flag file) and that the prelude parses in a child.

### Task E3: adopt across the suite [M]
- [ ] Done

**Problem.** Volumes tests carry 4 copies of the retry loop + `_LIBS`; the equally
crash-prone m1–m6 render tests have NO retry at all (the startup crash is not
volumes-specific).

**Fix design (locked).** Mechanical adoption, file-by-file: volumes_{writer,plot,live,
color}_test.jl use `retries=4, ready_marker=...` + `_HELPER_INDEX_LIBS` (delete local
loops/consts — but DO NOT touch the all-zero testset body in volumes_live_test.jl, Track B
owns it); add `retries=2` + the appropriate ready markers to m1_mesh_render, m1_lights,
m1_camera, m1_orientation, m1_primitives, m1_save_record, m2_*render-ish, m3_texture,
m3_material*, m5_*, m6b_* test files (each file's marker = its existing sentinel). Adopt
`PROG_PIXEL_HELPERS` only where the local helper is a drop-in match — no semantic changes.

**Tests.** Full suite is the test (run at integration); in-task: spot-run 3 representative
converted testsets (one m1, one m3, one volumes).

---

## Track F — Docs & hygiene

### Task F1: root README + gitignore intent [M]
- [ ] Done

**Problem.** No root README.md (examples/ and lib/NanoVDBWriter/ both have one);
`examples/Manifest.toml` is tracked yet matched by `.gitignore`'s bare `Manifest.toml` —
tools/clones disagree about its status.

**Fix design (locked).** README: what it is (Makie backend → ovrtx RTX path tracer),
status table (M0–M6 + volumes, grayscale-volumes caveat + pointer to the
colors-need-Kit-runtime constraint), requirements (GPU, ovrtx install, env vars
`OVRTX_LIBRARY_PATH` / `OMNIVERSEMAKIE_INDEX_LIBS` / `OMNIVERSEMAKIE_OVRTX_CONFIG`),
quickstart (offscreen colorbuffer + `interactive_display`), pointers to
ARCHITECTURE.md/examples/bench. Add `!examples/Manifest.toml` to .gitignore with a
one-line comment (reproducible example gallery).

**Tests.** None (docs); verification = links/paths named in README exist.

### Task F2: ARCHITECTURE.md truth + plan-file consolidation [M]
- [ ] Done

**Problem.** ARCHITECTURE.md:386-387 presents M6 as FUTURE work (and describes an AOV pick
design; shipped was native ray-query); :213 says volume live updates use "field attribute
writes" — the opposite of the shipped remove+re-reference (a filePath write is a no-op).
Root carries M1/M2/M3/M5_PLAN.md while newer plans live in docs/superpowers/plans/.

**Fix design (locked).** Update the M6 section to shipped reality (ray-query pick,
selection outline constraint), fix the Volume row (fresh-temp reload, grayscale-only note),
stamp the doc header "accurate as of 2026-07 (post-M6/volumes-M2)". `git mv` the four root
plan files to `docs/plans-history/`; leave a one-line pointer in ARCHITECTURE.md. Do NOT
rewrite history sections that are still accurate.

**Tests.** None; verification = grep ARCHITECTURE.md for "filePath" / "AOV" claims removed,
files moved, no dangling references (grep repo for `M2_PLAN.md` etc.).

---

## Integration (controller, sequential)

### INT-1: merge gates
- [ ] Done

Merge `B` → full targeted re-run of m1/m3/volumes → merge `A` → **full `Pkg.test()`** —
A's op-error throwing is the honesty gate: any newly-surfaced `OVRTXError` is a REAL
latent failure; triage each (fix-forward in a follow-up commit or document + skip with an
issue note in the ledger). Then merge `L`, `C`, `D`, `E`, `F` (each: rebase, resolve the
known screen.jl / runtests.jl hunks, targeted tests) → final **full `Pkg.test()`** +
`bench/hot_path.jl` re-run; update bench/RESULTS.md numbers (esp. Path A after B4/B7 and
the present-path notes after C2/C3).

### INT-2 (optional capstone): zero-copy CPU present
- [ ] Done

Adopt A2's `with_mapped_hdr` inside C2's fused CPU present (tonemap straight from the
still-mapped Float16 buffer → cached oriented matrix): steady-state ZERO full-frame
allocations. Only after both tracks merged; gated on the C2 pixel-equality test staying
green. Update RESULTS.md.

## Non-goals (explicitly out of scope, with reasons)

- `pull_ovrtx_nodes!` dirty-set redesign — deliberate pull architecture; O(n plots)/frame
  is fine at current scale (review: "worth a note before figure sizes grow").
- Dropping `cuStreamSynchronize` in `_gpu_present!` — safety unverified; one sync/frame.
- Live material SWAP on non-materialized plots; merged-mesh live position re-tessellation
  (B4 warns instead) — both need re-author machinery, separate design.
- Volume colormap COLORS — impossible in standalone ovrtx (needs a Kit runtime; see
  memory/volume-colors-need-composite-runtime).
- Julia 1.12 `public` declarations — nice-to-have flagged in review, not a fix.

## Self-review notes

- Every review finding maps to a task: A1 (error swallowing, unmap returns), A2 (map
  leaks, step! leak, double-copy readback, map-status checks), A3 (resolver), A4
  (finalizer, SubString), A5 (index-config JSON escaping + merge anchor),
  B1 (NaN colors + 3× dup + 2 resolvers), B2 (NaN lines), B3 (empty→fill), B4 (scatter
  positions), B5 (epilogue ×6), B6 (escaping, camera path, texture collision), B7 (mesh
  allocs + typed bindings Dict + push-copy), L1 (light count + RectLight dup), L2
  (per-frame snapshot churn + Any fields),
  C1 (exp2), C2 (CPU 4-buffer), C3 (GPU 3-kernel + texture lookup), D1-D3 (guards, memory,
  Zlib), E1-E3 (watchdog, m0, retry/dedup), F1-F2 (README, gitignore, ARCHITECTURE,
  plan files). Deferred items are in Non-goals.
- Cross-track interfaces: A2→INT-2 (`with_mapped_hdr(f, sr, name)`), E2→E3 (kwargs +
  consts), B1 internal only. No task depends on another TRACK's unmerged work.
- Shared-file hunks declared: screen.jl (A resolver-cache vs L field types),
  volumes_live_test.jl (B all-zero testset vs E preamble), runtests.jl (E include order vs
  per-track new-test includes — controller resolves at merge).
