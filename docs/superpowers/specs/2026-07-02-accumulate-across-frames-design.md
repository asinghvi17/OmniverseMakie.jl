# Accumulate-Across-Frames Rendering Mode — Design

**Date:** 2026-07-02 · **Status:** approved (user) · **Origin:** user experiment — overriding
`OmniverseMakie.OV.reset!(...) = nothing` recorded 1351 frames in 5.3 min vs 54 min (10.2×,
~4 fps vs 0.37) with A/B frames at peak motion visually indistinguishable. RT2's temporal
reprojection + denoiser handle object motion exactly as in the interactive viewport; per-frame
full reconvergence bought nothing for that scene.

## Goal

Make the hack a first-class, opt-in `ScreenConfig` mode: RT2 accumulation carries across
frames while the per-frame USD diff writes continue as normal. Recording (`Makie.record`)
becomes realtime-style; the same flag governs every consumer of the Screen.

## Configuration

Two new **trailing** `ScreenConfig` fields (positional-ctor discipline — Makie's
`merge_screen_config` constructs positionally; defaults live in `__init__` theme
registration, NOT in the struct — the `selection_outline` precedent):

| Field | Default | Meaning |
|---|---|---|
| `accumulate_across_frames::Bool` | `false` | Never reset RT2 accumulation for camera/light/attribute changes; diffs still written every frame. |
| `accumulation_preroll::Int` | `40` | Extra RTX steps before the FIRST readback so frame 1 is converged (≈ the experiment's ~10 warm-up colorbuffers × `warmup = 4`). |

`warmup` keeps its existing meaning — RTX steps per rendered frame (with no reset, this is
the per-frame refinement budget; the experiment used 4, the interactive viewport uses 2).
No default changes. Recipe documented in README + `activate!` docstring:

```julia
OmniverseMakie.activate!(accumulate_across_frames = true, warmup = 4)
Makie.record(fig, "out.mp4", frames) do f ... end
```

## Reset gating — one choke point

Both reset consumers (offscreen `Makie.colorbuffer` at `src/screen.jl` and the interactive
tick at `ext/OmniverseMakieGLMakieExt.jl`) funnel through `_sync_and_needs_reset!`. The gate
lives INSIDE that helper:

- Flag off: behavior byte-identical to today (`need_reset` as computed).
- Flag on: return `need_reset && screen.structural_dirty`; clear `structural_dirty` with the
  same two-write discipline the helper already uses for `requires_update`.

**Scope (user decision):** Screen-level — the flag governs offscreen AND
`interactive_display`. An accumulating viewport smears briefly on fast orbits instead of
flashing noise (game-engine behavior); default-off means nothing changes unless opted in.

## Structural signal (user-approved assumption: auto-reset on structural)

Attribute/camera/light writes never reset in accumulate mode (the validated fast path), but
anything that ADDS or REMOVES a USD reference fires ONE reset — RT2's history has no
reprojection for a prim that didn't exist, and structural edits are rare and already
expensive. New field `Screen.structural_dirty::Bool`, set at exactly the composition-change
funnel the review-fixes established for PathResolver invalidation:

- `_register_robj_maps!` (src/compute.jl) — covers plot insert, the Surface no-diff-node
  branch, and the B3 empty→fill late build. (Also fires on a harmless re-register of an
  existing robj — one spurious reset per re-display event, accepted.)
- `_teardown_usd_reference!` (src/screen.jl) — covers `delete!` / `empty!`.
- `reload_volume_data!` (src/translation/volume.jl) — a volume data reload swaps references.

Escape hatch: `reset_accumulation!(screen)` = manual `OV.reset!` (unexported, documented) for
anything the funnel misses or the user wants to force.

## Pre-roll

In `colorbuffer` only (the viewport converges live as its normal UX): when
`accumulate_across_frames && !screen.preroll_done`, run `accumulation_preroll` extra RTX
steps before the first readback (fold into the first render call's step count), then set
`preroll_done = true` (new `Screen` field, never cleared for the Screen's lifetime).

## Observability & testing (house subprocess pattern)

New diagnostic hook `OV._RESET_OBSERVER` mirroring `_PUSH_OBSERVER` (a `Ref{Any}(nothing)`;
`OV.reset!` fires it when set; `nothing` = zero overhead) so tests count resets exactly.

1. **Pure:** ScreenConfig field count/order/defaults via the theme; positional ctor intact.
2. **Motion (GPU):** `accumulate_across_frames=true, warmup=4`; author a mesh; N frames of
   per-frame xform edits via `colorbuffer`; assert lit-centroid MOVES frame 1→N (diffs
   applied), reset count == 0 (observer), frame 1 non-black (preroll landed).
3. **Structural (GPU):** mid-sequence `insert!` of a new plot → exactly 1 reset + the new
   plot renders; mid-sequence volume data reload → exactly 1 reset.
4. **Default-mode regression:** flag off → reset fires on a camera/attribute change
   (observer ≥ 1), pinning that the gate changed nothing by default.

Volume/graded-data and GPU-flock rules per the house test conventions.

## Non-goals

- Changing `warmup`/quality defaults, or any behavior when the flag is off (byte-identical).
- A `record`-specific kwarg (Makie owns `record`'s loop; ScreenConfig flows through it).
- Pure never-reset semantics (rejected in favor of auto-reset-on-structural; flipping later
  is a two-line change given the single gate).
- Reprojection-quality guarantees for fast camera fly-throughs (documented caveat: same
  trade-off the interactive viewport makes all day).

## Files

`src/settings.jl` (+2 fields, docstring), `src/OmniverseMakie.jl` (`__init__` theme
defaults), `src/screen.jl` (2 Screen fields, gate in `_sync_and_needs_reset!`, preroll in
`colorbuffer`, `structural_dirty` in `_teardown_usd_reference!`, `reset_accumulation!`),
`src/compute.jl` (`structural_dirty` in `_register_robj_maps!`), `src/translation/volume.jl`
(`structural_dirty` in `reload_volume_data!`), `src/binding/OV.jl` (`_RESET_OBSERVER`),
`test/` (1 pure + 1 GPU subprocess file), `README.md` (recording section), `test/m3_material_prog.jl`
(the one positional-ScreenConfig-ctor site, if its arity assert needs the 2 new fields).
