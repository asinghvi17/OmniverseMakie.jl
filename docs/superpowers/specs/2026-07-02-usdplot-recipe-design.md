# `usdplot` recipe + `bind_usd!` observable bindings — design

**Date:** 2026-07-02 · **Status:** spike-validated design, awaiting review
**Depends on:** open-stage diff architecture (M2), accumulate-across-frames (09ae8c2)

## Goal

Place an **external USD file** (a DCC export, a vendor asset, a Kit-authored scene) into a
Makie scene as a first-class plot, rendered through ovrtx alongside ordinary Makie plots —
and **tie Julia `Observable`s to prims/attributes inside that file** so updating an
observable live-updates the render (a robot arm driven by a joint-angle observable, a
material color driven by a slider).

```julia
p = usdplot!(ax, "assets/robot.usdc"; bbox = Rect3f(Point3f(-1), Vec3f(2)), up = :y)

translate!(p, 0, 0, 1)          # ordinary Makie transforms → the asset's root transform
p.visible[] = false             # ordinary visibility

arm = Observable(Makie.translationmatrix(Vec3f(0)))
bind_usd!(p, "/Arm", arm)                                        # prim binding → transform
bind_usd!(p, "/Arm/Geo.primvars:displayColor", color_obs)        # attribute binding
arm[] = Makie.rotationmatrix_z(0.4f0)                            # live update, no re-author
```

`usdplot` works offscreen (`save`/`colorbuffer`), in `interactive_display`, and inside a
`replace_scene!` panel. In a plain GLMakie window it renders nothing (documented).

## Spike evidence (2026-07-02, two GPU spikes; findings files in the session scratchpad)

Every claim below is pixel-verified (18496-px quad oracle, centroid oracle for transforms).

**Spike A — composition.** ALL WORK: `ovrtx_add_usd_reference_from_file` composes `.usda`
AND `.usdc` mid-session; nested references and relative sub-assets resolve (anchored at the
referenced file's directory); **payloads auto-load**; a **self-contained material inside the
referenced layer binds and applies** — both OmniPBR/MDL (bare `@OmniPBR.mdl@` resolves) and
**UsdPreviewSurface** (what DCC exports use). The M3 "materials must be pre-authored at
open" limitation is cross-arc-only; the self-contained (mesh+material+binding in one file)
case — the normal DCC-export shape — works when added mid-session.

**Spike B — live writes.**
- `write_xform!` (`omni:xform`) **works on subprims nested inside the referenced hierarchy**
  (181-px centroid move, exact restore), not just the reference root. `omni:xform` is
  intrinsic to every Xformable prim in Fabric — no declaration needed.
- File-declared `xformOp:*` attributes are **baked at load and NOT live-writable** (silent
  no-op under EXISTING_ONLY; MUST_EXIST write throws "not found in stage").
- Ordinary attributes declared in the referenced layer (e.g. `primvars:displayColor`) ARE
  live-writable on composed subprims (full red→green flip).
- **Bind-time validation:** `create_binding` validates nothing in either prim mode (creation
  is lazy). The only proven validator is a **MUST_EXIST one-shot write**, which throws a
  precise `OVRTXError` naming the missing `path.attr`. EXISTING_ONLY writes to bogus targets
  stay silent no-ops (the known ovrtx hazard).

**Four load-bearing composition rules** the design encodes:
1. A reference pulls in the file's **defaultPrim subtree only**, remapped onto the reference
   prim → user-facing bind paths are **relative to the defaultPrim**.
2. Files with relative sub-assets/payloads/textures must compose **from file** (directory
   anchor), not from an anonymous string.
3. Targets inside a layer must be layer-relative (existing repo rule; matters only if we
   ever author wrapper layers with internal targets).
4. Transforms route through `omni:xform` only; `xformOp:*` targets are refused up front.

## Design

### The recipe (Makie 0.24 form)

```julia
@recipe USDPlot (path,) begin
    "Axis-aligned bounds used for camera framing (the USD file is not parsed)."
    bbox = Rect3f(Point3f(-1), Vec3f(2))
    "Source up-axis: `:z` (USD default here) or `:y` (typical DCC export) — `:y` folds a +90° X rotation into the model."
    up = :z
    Makie.mixin_generic_plot_attributes()...
end
Makie.plot!(p::USDPlot) = p          # NO child plots → our backend treats it as atomic
Makie.convert_arguments(::Type{<:USDPlot}, path::AbstractString) = (abspath(String(path)),)
Makie.data_limits(p::USDPlot) = p.bbox[]
```

The backend's plot walker already dispatches on `isempty(plot.plots)` (screen.jl:162,
compute.jl:912), so a childless recipe flows into `register_ovrtx_robj!` →
`author_usd_prim!(::USDPlot)` with zero walker changes. `convert_arguments` absolutizes the
path immediately (cwd may change before display).

### Authoring (one prim, one handle — the volume pattern)

`author_usd_prim!(screen, scene, plot::USDPlot, args)`:
1. Validate the file exists (clear `ArgumentError` otherwise; ovrtx also throws on a missing
   file — we pre-empt with a friendlier message).
2. `robj.usd_handle = OV.add_usd_reference_from_file!(r, path, prim_path)` — a NEW `OV`
   wrapper mirroring `add_usd_reference!` around `ovrtx_add_usd_reference_from_file`
   (libovrtx_api.jl:1395). The file's defaultPrim composes onto
   `/World[/Scene_<sid>]/plot_<oid>`.
3. Apply any bindings stashed by pre-display `bind_usd!` calls (below).

`consumed_inputs(::USDPlot) = [:model_f32c, :visible]`. The existing `:model_f32c` push
writes `omni:xform` on the plot prim — for `up = :y` the push folds the correction in
(`_model_to_usd_xform(model * ROT_X_90)`), via a small `USDPlot`-aware hook in the
`:model_f32c` branch. Consequence (documented): **Makie owns the asset's root transform** —
the defaultPrim's own root-level transform is replaced by the plot's
`translate!`/`rotate!`/`scale!` (omni:xform is a wholesale REPLACE; M1-proven). Interior
prim transforms are untouched. `metersPerUnit` mismatches are the user's `Makie.scale!`.

Teardown is the existing path unchanged: `_teardown_usd_reference!` (screen.jl:198) removes
the handle; `delete!`/`empty!`/screen-close all work; structural add/remove flows through
`_note_composition_change!` (accumulate-correct) for free because usdplot rides the normal
robj lifecycle.

### `bind_usd!(p::USDPlot, target::String, obs) -> obs` / `unbind_usd!(p, target)`

**Target grammar** — split at the FIRST `.` (USD prim names cannot contain `.`):
- `"/Arm"` (no dot) → **prim binding**: obs values are Makie-convention 4×4 matrices →
  `write_xform!(prim * target, _model_to_usd_xform(value))`.
- `"/Arm/Geo.primvars:displayColor"` → **attribute binding**: value dispatched by type
  (table below).
- Paths are relative to the file's **defaultPrim** and prefixed with the plot's prim path.
- **Refused with a clear error:** `xformOp:*` attributes ("baked by ovrtx at load — bind
  the prim itself (`\"/Arm\"`) with a matrix observable instead"), and malformed paths
  (each segment validated with the existing `_usd_identifier` rules).

**Registry & lifecycle.** Bindings live in a module-level
`WeakKeyDict{AbstractPlot, Vector{USDBinding}}` (a plot may be bound before display and may
be shown on several screens). Wiring to a live screen registers one observable listener per
binding; the `ObserverFunction`s are stored in `robj.meta[:usd_binding_obsfuncs]` and
detached in `destroy_bindings!` (compute.jl:769) — same lifecycle as hot-path bindings.
Re-binding an already-bound target replaces it; `unbind_usd!` detaches and leaves the last
written value in place.

**Fail-fast validation (the answer to ovrtx's silent-ignore hazard).** At wire-time the
binding is applied once as a **MUST_EXIST write of the observable's current value** — one
call that validates AND establishes consistent initial state. A typo'd prim or attr throws
ovrtx's precise `OVRTXError` ("path or attribute not found in stage: <prim>.<attr>") at
`bind_usd!` time; a dtype mismatch throws the (also precise) size-mismatch error. This
requires threading a `prim_mode` kwarg through `OV._write_attribute!` (default
`EXISTING_ONLY`, byte-identical behavior for all existing callers; probes pass
`MUST_EXIST`). When `bind_usd!` runs before display, bindings are stashed and applied at
author time — a probe failure there degrades to a loud `@warn` (naming the target and the
ovrtx error) + skip, so one bad binding can't kill the whole figure. Implementation
checkpoint: confirm a MUST_EXIST write accepts the intrinsic `omni:xform` on a real prim
(spike B confirmed MUST_EXIST writes to real live attrs don't throw, but did not test the
intrinsic-attr case specifically; if it surprises, prim-binding probes fall back to an
EXISTING_ONLY write, documented as "prim bindings validate the write, not prim existence" —
attribute bindings keep full MUST_EXIST validation either way).

**Update flow (coalescing, accumulate-aware).** Listeners do not write immediately; they
enqueue `(prim_path, attr, value)` into a per-screen `Dict` keyed by target (latest value
wins) and set `requires_update`. The queue is flushed inside `_sync_and_needs_reset!`,
OR-ing "any write happened" into `need_reset` BEFORE the accumulate gate — so default mode
reconverges (like camera/light edits) and `accumulate_across_frames = true` keeps
accumulating (bound writes are non-structural; RT2 reprojection absorbs them). This is
exactly the semantics the accumulate feature defined for attribute edits. Flush uses
one-shot writes (bound updates are user-rate, not per-frame×10⁵; a `create_binding` hot
path is a documented follow-up if someone drives 60 Hz animation through a binding).

**Value → write dispatch** (attribute bindings; `ArgumentError` naming supported types
otherwise):

| Julia value | USD write |
|---|---|
| `Mat4`-like (prim binding only) | `write_xform!` via `_model_to_usd_xform` |
| `Real` | float, `{kDLFloat,32,1}` scalar |
| `VecTypes{3}` / `NTuple{3,Real}` / `Colorant` (→`RGBf`) | `{kDLFloat,32,3}`, is_array=false, shape `[1]` |
| `AbstractVector{<:VecTypes{3}}` / `AbstractVector{<:Colorant}` | `{kDLFloat,32,3}`, is_array=true, shape `[n]` (the proven displayColor recipe) |

### Picking integration (small, in scope)

`_path_to_oid`'s ancestor walk strips a single trailing segment — a pick landing on
`/World/plot_<oid>/Robot/Arm/Bolt` currently resolves to `nothing`. Extend it to a full
longest-prefix walk over `path2plot` so picking a usdplot returns the plot from any depth.
(Exposing the picked *subprim path* as a public accessor is a follow-up.)

## Files

- `src/translation/usdplot.jl` (NEW): recipe + `plot!`/`convert_arguments`/`data_limits`,
  `author_usd_prim!(::USDPlot)`, `USDBinding` + registry, `bind_usd!`/`unbind_usd!`, target
  parsing/validation, value dispatch, queue flush helper.
- `src/binding/OV.jl`: `add_usd_reference_from_file!`; `prim_mode` kwarg on
  `_write_attribute!` (default preserves current behavior).
- `src/compute.jl`: `consumed_inputs(::USDPlot)`; `up`-fold hook in the `:model_f32c`
  branch; listener detach in `destroy_bindings!`; full ancestor walk in `_path_to_oid`.
- `src/screen.jl`: `pending_usd_writes` field on `Screen` + flush call in
  `_sync_and_needs_reset!`.
- `src/OmniverseMakie.jl`: include; export `usdplot`, `usdplot!`, `bind_usd!`,
  `unbind_usd!`.
- `README.md`: a "Placing USD assets" section (incl. the defaultPrim path rule, the
  root-transform-ownership rule, and the GLMakie-renders-nothing note).
- `test/usdplot_test.jl` + runtests wiring.

## Testing

- **Pure:** target parsing (prim vs attr split, `xformOp:*` refusal, segment validation),
  value-dispatch table, registry add/replace/unbind, `convert_arguments` absolutization,
  `data_limits == bbox`.
- **Subprocess GPU** (the spike assets, re-authored small): compose an `arm.usda`-style
  file next to a Makie anchor → red quad renders alongside; `bind_usd!` prim binding →
  update → centroid moves ≫20 px; attribute binding displayColor → red→green flip; a typo'd
  target → `OVRTXError` thrown AT BIND TIME; `delete!(ax, p)` → pixels gone (handle
  removed, no leak); with `accumulate_across_frames = true`, a bound update triggers NO
  reset (`OV._RESET_OBSERVER` count) while a plot insert still does; `up = :y` renders a
  Y-up-authored quad upright. One `.usdc` + one payload-file smoke composes and renders.

## Non-goals (v1)

- Construction-time `bindings = ...` kwarg (post-hoc `bind_usd!` covers it; compute-graph
  plumbing for dynamic user observables is the sharp edge we're avoiding).
- Auto-`bbox` / defaultPrim auto-detection (requires parsing USD; `.usdc` can't be parsed
  without USD libs).
- Texture/asset-input bindings (proven impossible today — no writable asset-input path).
- Raw `xformOp:*` bindings (proven baked).
- `.usdz` archives (untested; likely needs its own spike).
- `create_binding`-based per-frame hot path for bound attrs; picked-subprim-path accessor;
  generalizing `bind_usd!` to non-USDPlot plots.

## Open questions

1. MUST_EXIST write against intrinsic `omni:xform` (implementation checkpoint above).
2. Should `bind_usd!` on a displayed-on-zero-screens plot warn (deferred probe) or stay
   silent? Proposed: silent stash, loud on author failure.
