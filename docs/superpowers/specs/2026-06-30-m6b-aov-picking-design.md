# M6.B — AOV Picking in the OmniverseMakie RTX Viewport (Design)

**Status:** approved design → ready for implementation plan
**Date:** 2026-06-30
**Milestone:** M6.B (follows M6.A GPU-direct HDR viewport, on origin/main @ `1f622a1`)

## Goal

Implement Makie-style picking for the OmniverseMakie backend: given a screen pixel, return the originating Makie `(plot, index)` (and the 3D world position/normal), and visually highlight the picked object with ovrtx's selection outline. Picking is driven by ovrtx's **native ray-query pick API**, surfaced through Makie's standard `pick` interface, with an **opt-in attachable interaction** for the live viewport.

## Summary of the mechanism (validated by research)

ovrtx exposes a native pick path — there is **no per-pixel id raster AOV** to read; instead:

1. `ovrtx_enqueue_pick_query(desc)` is queued **before** a `step!`; the next step produces a synthetic `ovrtx_pick_hit` render var.
2. `ovrtx_pick_hit` (mapped **CPU-only**) carries `params` (`magic`, `version`, `hitCount`) and `tensors` (`primPath`, `objectType`, `geometryInstanceId`, `worldPositionM` float64`[h,3]`, `worldNormal` float32`[h,3]`).
3. `primPath` is an opaque `ovx_primpath_t` (UInt64) resolved to the exact authored prim-path string (`/World/plot_<objectid>`) via `ovrtx_get_path_dictionary` + the path-dictionary vtable resolvers.
4. We already author `plot_prim_path = /World[/Scene_<sid>]/plot_<objectid(plot)>` and keep the forward map `Screen.plot2robj`; a small `path2plot` reverse map closes the loop back to the Makie `Plot`.

The pick is a **renderer query**, not a window/GL operation — so the core needs **no GLMakie/CUDA** and works on any `Screen` (offscreen and interactive).

## Architecture

| Layer | Where | Responsibility |
|-------|-------|----------------|
| Pick FFI | `src/binding/OV.jl` (main module) | enqueue pick query; map + decode `ovrtx_pick_hit` (CPU); resolve `primPath`→string via the path-dictionary vtable; selection-outline + renderer-config helpers |
| Pick core | `src/screen.jl` (main module) | `path2plot` reverse map; `Makie.pick`/`pick_closest`/`pick_sorted` overrides; element-index extraction; `select!`/`clear_selection!` outline API |
| Renderer config | `src/binding/OV.jl` + `src/settings.jl` | `ScreenConfig.selection_outline` (default off) → create the renderer with the creation-time outline flag |
| Attachable interaction | `ext/OmniverseMakieGLMakieExt.jl` | opt-in `attach_picking!(session; …)` wiring click → pick → outline + `on_hit` callback/Observable |

This mirrors the M6.A split: backend-universal capability in the main module; GLMakie-specific wiring in the extension.

## FFI additions (`src/binding/OV.jl`)

Already bound in `lib/LibOVRTX` (no binding work): `ovrtx_enqueue_pick_query`, `ovrtx_get_path_dictionary`, `ovrtx_set_selection_group_styles`, `path_dictionary_instance_t`/`vtable`, `ovrtx_write_attribute`.

To add as Julia helpers over already-wrapped primitives:

- **`OV.enqueue_pick_query(r, product_path, (left, top, right, bottom); flags=0)`** — fills `ovrtx_pick_query_desc_t` (RenderProduct **pixel** space; `left`/`top` inclusive, `right`/`bottom` exclusive; a single-pixel click = `(x, y, x+1, y+1)`).
- **`OV.read_pick_hit(sr)`** — map the `ovrtx_pick_hit` var with `OVRTX_MAP_DEVICE_TYPE_CPU`; validate `magic == OVRTX_PICK_HIT_MAGIC` and `version == OVRTX_PICK_HIT_VERSION` (mismatch → treat as no hit); read `hitCount`, then the `primPath`/`geometryInstanceId`/`worldPositionM`/`worldNormal` tensors **by name** (params and tensors are separate arrays on `ovrtx_render_var_output_t`). Returns a vector of raw hit records.
  - **Why CPU (and what the alternatives are):** `OVRTX_MAP_DEVICE_TYPE_CPU` is **required by ovrtx**, not a choice — the synthetic `ovrtx_pick_hit` var can *only* be mapped on CPU/default (documented in the LibOVRTX `ovrtx_pick_hit` docstring + ovrtx's picking SKILL). The two CUDA modes M6.A uses for `HdrColor` — `OVRTX_MAP_DEVICE_TYPE_CUDA` (linear device ptr) and `OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY` (opaque array) — are **not available** for the pick-hit var. It is also irrelevant to performance: pick-hit is a *tiny* result (a few hit records — `primPath` u64, `worldPositionM` 3×f64, `worldNormal` 3×f32, plus the `magic`/`version`/`hitCount` u32 params — one row per hit, not a per-pixel image), so there is no host-roundtrip cost worth a device path (unlike the per-frame megapixel `HdrColor` blit). CPU is both mandatory and the right fit.
- **`OV.resolve_prim_path(path_dict, primpath_id) -> String`** — ★ the one genuinely novel FFI: the resolvers `path_dictionary_get_tokens_from_paths` / `get_strings_from_tokens` are C `static inline` **vtable dispatches**, not exported symbols. Call them through the `path_dictionary_instance_t.vtable` function pointers from Julia (`@ccall $(fnptr)(...)`). VERIFY the exact pointer signatures in a REPL before relying on them (a new-dep unknown, handled like M6.A's CUDA calls). Sequence per the C `docs_resolve_primpath` helper: `get_tokens_from_paths(pd, &id, 1, buf, …)` → for each token `get_strings_from_tokens(pd, &tok, 1, &str)`, join with `/`.
- **`OV.set_selection_outline_group(r, prim_paths, group_ids)`** — implement via the already-wrapped `ovrtx_write_attribute` writing the `omni:selectionOutlineGroup` uint8 attribute (`OVRTX_ATTR_NAME_SELECTION_OUTLINE_GROUP`); group `1` = selected, `0` = cleared.
- **`OV.set_selection_group_styles(r, group_ids, styles)`** — thin wrapper over the exported `ovrtx_set_selection_group_styles` (`ovrtx_selection_group_style_t{outline_color::NTuple{4,Cfloat}, fill_color::NTuple{4,Cfloat}}`).
- **Renderer config builders** — build `ovrtx_config_entry_t` in Julia for `OVRTX_CONFIG_SELECTION_OUTLINE_ENABLED` (bool), `OVRTX_CONFIG_SELECTION_OUTLINE_WIDTH` (int), `OVRTX_CONFIG_SELECTION_FILL_MODE` (int), passed to `ovrtx_create_renderer` when `ScreenConfig.selection_outline` is on (creation-time only).

## Data flow (one pick)

```
Makie.pick(scene, screen, xy)                       # xy: Makie pixel coords (bottom-left origin, Float64)
  → (px, py) = to_ovrtx_pixels(xy, screen.fb_size)  # ★ y-flip to top-left RenderProduct pixels (VERIFY in REPL)
  → OV.enqueue_pick_query(r, screen.product, (px, py, px+1, py+1))
  → sr = OV.step!(r, screen.product; …)             # the pick consumes a step
  → hits = OV.read_pick_hit(sr)                      # CPU map + magic/version validate
  → hitCount == 0 ? return (nothing, 0)
  → pathstr = OV.resolve_prim_path(path_dict, hits[1].primPath)   # → "/World/plot_<id>"
  → plot = screen.path2plot[pathstr]                 # reverse map → Makie Plot
  → index = element_index(plot, hits[1].geometryInstanceId)       # per-kind (below)
  → return (plot, index)        # world_position/normal available via the richer pick_hit path
```

After a pick step the RT2 accumulation is reset and resumes (one-frame hiccup; acceptable for on-demand picking).

## Makie interface

- **`Makie.pick(scene, screen::Screen, xy::Vec{2,Float64}) -> (AbstractPlot, Int)`** — single-pixel ray query (above). Miss → `(nothing, 0)`.
- **Override `Makie.pick_closest(scene, screen, xy, range)`** and **`Makie.pick_sorted(scene, screen, xy, range)`** — both do a single-pixel ray query at `xy` (a ray tracer returns the closest front hit natively, so `range` is largely irrelevant); `pick_sorted` returns the front hit as a 1-element vector. Overriding these means Makie's generic `pick(rect)` fallback is never invoked, so we need not implement the per-pixel matrix.
- **Non-goal:** `Makie.pick(scene, screen, rect::Rect2i)` (the per-pixel `Matrix{(plot,index)}`). It is a poor fit for a ray-query backend (ovrtx marquee returns a deduplicated hit-list, not a per-pixel matrix). Deferred; can be marquee-backed later if needed.
- **Richer hit (beyond Makie's contract):** an internal `OmniverseMakie.pick_hit(screen, xy) -> Union{Nothing, NamedTuple{(:plot,:index,:world_position,:normal)}}` that `Makie.pick` wraps (dropping the extras) and the attachable interaction consumes (keeping the 3D position/normal).

Because we implement the standard `pick`/`pick_closest`/`pick_sorted`, Makie's own pick-based tools (`DataInspector`, `onpick`) compose with the viewport for free when the user attaches them.

## Element-index fidelity (confirmed scope)

Plot is **always** exact (via `path2plot`). Element index:

- **Scatter / MeshScatter** (non-materialized `UsdGeomPointInstancer`): index = `geometryInstanceId` (instance 0 = scatter point 0). **Exact** — `geometryInstanceId` is well-defined for instancers.
- **Surface** (`UsdGeomMesh`, i-major grid): map the hit's face/primitive index to the linear `(i-1)*ny + j` index Makie expects. **Exact *iff* a mesh pick hit exposes a per-face/primitive index** (in `geometryInstanceId` or another hit field) — ★ VERIFY in REPL whether ovrtx provides a sub-prim index for a plain `UsdGeomMesh` hit. If it does **not** (the hit only identifies the prim), surface **degrades to plot-level** (index `0`) like Mesh, and the cell-index mapping moves to the follow-up. The plan must include this verification and branch accordingly.
- **Mesh, Lines, LineSegments, materialized (merged-mesh) Scatter/MeshScatter:** return index `0` (plot-level). The per-kind face→vertex / merged-marker-division mapping (and the metadata it needs, e.g. marker face counts) is a documented **follow-up**, not in M6.B.

## Selection outline

- **Highlight is OFF by default at every level** — opt-in, never automatic:
  - `ScreenConfig.selection_outline` defaults to `false` (no outline pass on the renderer). Offscreen `colorbuffer`/`save` Screens stay off.
  - `interactive_display(...; selection_outline=false)` — the interactive viewport also defaults the flag **off**; pass `selection_outline=true` to create the viewport's `Screen` with the (creation-time-only) outline capability.
  - `attach_picking!(...; outline=false)` — even with picking attached, no highlight is drawn unless you ask for it.
- Pick **data** (`Makie.pick`, `pick_hit`, `on_hit`) works **regardless** of the outline flag — only the visible highlight depends on it. Because the flag lives in `ScreenConfig`, `resize_viewport!` (which rebuilds the Screen) preserves it.
- The outline flag is **creation-time-only**, so `attach_picking!(...; outline=true)` requires a Screen built with `selection_outline=true`. If it isn't, `attach_picking!` `@warn`s once and falls back to no-highlight (hinting `interactive_display(...; selection_outline=true)`) rather than rebuilding the Screen.
- **`OmniverseMakie.select!(screen, plot; group=1)`** writes `omni:selectionOutlineGroup = group` on the plot's prim (`OvrtxRObj.prim_path`); **`clear_selection!(screen[, plot])`** writes group `0`. Outline color via `OV.set_selection_group_styles` (default a high-contrast outline). Width/fill-mode are creation-time config defaults.

## Reverse map (`path2plot`)

Add `Screen.path2plot::Dict{String, UInt64}` (prim-path → `objectid(plot)`), populated wherever `plot2robj[objectid(plot)] = OvrtxRObj(...)` is set (the diff-node and direct paths in `compute.jl`), and cleared in the typed `delete!`/`empty!` teardown — so it stays churn-safe and leak-free alongside `plot2robj`. `select!`/`pick` look up `plot2robj` (forward, for the prim path) and `path2plot` (reverse, for the plot).

## Attachable interaction (GLMakie ext)

- **`OmniverseMakie.attach_picking!(session; on_hit=nothing, outline=false, button=Mouse.left) -> handle`** — opt-in (**off by default**); the highlight is **also off by default** (`outline=false`). Wires a discrete click (press+release without drag, so it does not fight the existing left-drag orbit) on the viewport: compute the RenderProduct pixel under the cursor → `pick_hit(session.screen, px)` → if `outline` (and the Screen has the capability — see Selection outline), `select!` the hit plot (and `clear_selection!` the previous) → call `on_hit(hit)` and push the hit to a `selected` Observable on the handle. `detach_picking!(handle)` / closing the session tears down the listener.
- The handle exposes `selected::Observable{Union{Nothing, hit}}` for the user to react to. Nothing about picking runs unless the user calls `attach_picking!`.

## Error handling & edge cases

- **Miss / empty space:** `hitCount == 0` → `(nothing, 0)`; the interaction `clear_selection!`s.
- **Magic/version mismatch:** validate before reading tensors → treat as a miss (don't read garbage).
- **Coordinate convention:** Makie xy is bottom-left-origin Float64 pixels; the pick desc is top-left-inclusive RenderProduct pixels → a y-flip + round to Int. **VERIFY in REPL** (the M6.A-style orientation unknown).
- **CPU-only:** `ovrtx_pick_hit` must be mapped with `OVRTX_MAP_DEVICE_TYPE_CPU` — never `map_cuda`.
- **Accumulation:** a pick consumes a step → reset + resume RT2 (one-frame hiccup); the interaction re-seeds the viewport frame after a pick.
- **Path resolved to a non-plot prim** (camera/light/looks): `path2plot` miss → `(nothing, 0)` (we only register plot prims).

## Testing

Core tests are **offscreen subprocess** (no GLMakie needed):

- **Pick correctness:** author a known scene (a cube at a known location + a scatter), build a `Screen`, `Makie.pick` over a pixel known to be on the cube → assert the returned plot **is** the cube plot and `world_position ≈` the expected point; over empty space → `(nothing, 0)`; over a specific scatter point → correct plot **and** the right `geometryInstanceId` index.
- **Path-dictionary round-trip:** a `primPath` id resolves to the exact `/World/plot_<id>` string (the vtable FFI).
- **Reverse map:** `path2plot` populated at insert, cleared at delete; 50× churn → baseline (no leak), mirrors the `plot2robj` lifecycle test.
- **Element index:** scatter point + surface cell exact; mesh/lines = `0` (asserted + documented).
- **Outline:** a `Screen` with `selection_outline=true`; `select!(screen, plot)` then render → assert outline-colored pixels appear around the prim (and `clear_selection!` removes them).
- **Attachable interaction (subprocess, GLMakie):** `attach_picking!(session)`; drive a synthetic click at a known pixel → assert `on_hit`/`selected` fires with the right plot and the outline is applied. (Account for the M5 GLMakie background-thread event race — test the pick path directly + the wiring synchronously, as M5's orbit test did.)

## Non-goals (deferred / out of M6.B)

- Per-pixel `pick(rect)` matrix and full marquee/region picking.
- Full element-index fidelity for Mesh/Lines/merged-marker scatter.
- Hover/DataInspector-by-default (picking is opt-in; DataInspector still works if the user attaches it).
- Semantic-segmentation AOV path (class-level, needs per-prim labels — not used).

## Implementation plan shape (~5 tasks)

1. **Pick FFI** (`OV.jl`): `enqueue_pick_query`, `read_pick_hit` (CPU decode + magic/version), `resolve_prim_path` (path-dictionary vtable — REPL-verify-heavy), outline/style/config helpers.
2. **Renderer config + reverse map**: `ScreenConfig.selection_outline` + renderer creation with the outline flag; `Screen.path2plot` populated/cleared with `plot2robj`.
3. **Makie.pick**: `pick`/`pick_closest`/`pick_sorted` + `pick_hit`; coordinate y-flip; element-index extraction (scatter/surface exact).
4. **Selection outline API**: `select!`/`clear_selection!` + group styles + the outline render test.
5. **Attachable interaction** (GLMakie ext): `attach_picking!`/`detach_picking!` + viewport wiring + smoke test.
