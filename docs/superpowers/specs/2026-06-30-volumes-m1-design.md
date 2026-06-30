# Volumes Milestone 1 — Render-Path Proof (low-level USD/Screen entry) — Design

**Status:** approved direction (user: staged scope, `OncePerProcess` config, low-level entry, "just do it").

## Goal

Prove the (now-cracked) ovrtx volume render path end-to-end from Julia at the **USD/Screen level** — no Makie `volume!` recipe, no dense-array writer. Deliverable: enable NVIDIA IndeX once per process, author a `UsdVolVolume` from an on-disk `.vdb`/`.nvdb` file into a Screen's stage, render it through the existing RT2 path, and get a non-black, **colormap-shaded** volume.

## Background (verified — see `.superpowers/sdd/m6b/volume-spike-report.md`)

- ovrtx renders a volume DECLARATIVELY: a `UsdVolVolume` prim with `rel field:density` → `UsdVolOpenVDBAsset{ fieldName, fieldDataType, filePath=@….vdb@ }`. RT2 has no native VDB integrator — it auto-routes `UsdVol` prims to **NVIDIA IndeX Direct** (integrated into the RTX path; not a separate render product).
- **The IndeX-init blocker is cracked.** `getIndexLibDir()` resolves the carb **token** `${omni.index.libs}` (a Kit-only token, unregistered in standalone ovrtx) to find `<dir>/bin/nvindex-libs`. Fix: register that token via carb settings. ovrtx loads `…/ovrtx/bin/ovrtx.config.json` as carb settings at startup, and carb registers tokens from the `/app/tokens/*` settings subtree. Proven recipe (exp6, no-install-edit): a scratch config = `ovrtx.config.json` + `{"app":{"tokens":{"omni.index.libs":"<KIT_INDEX_LIBS_DIR>"}}}`, delivered via env `CARB_FRAMEWORK_CONFIG_NAME` set to a path **relative** to `CARB_APP_PATH` (which `libovrtx-dynamic.so` force-sets to `…/ovrtx/bin`; carb mangles absolute values, so a relative `../…` path is required). Result: `IndeX Direct: initialization successful`, the `torus_fog` VDB renders (0 → 9179 px @ 512²).
- **The complete IndeX libs are the Kit ext, not ovrtx's bundle.** ovrtx's own `…/bin/libs/iray/` has only 4 of 13 libs + a mismatched `libnvindex.so`. The complete `lx64.r` release set is `~/.local/share/ov/data/exts/v2/omni.index.libs-1287db94366cf6fe` (the token must point here; loader appends `/bin/nvindex-libs`).
- The carb config is consumed ONCE at carb init (first `ovrtx_create_renderer`), so enablement is **process-global, set-once** — hence a `Base.OncePerProcess`.

## Scope

**In (M1):** IndeX enablement; low-level UsdVol+material authoring into a Screen's stage; a render test. **Out (→ M2):** Makie `volume!`/`vdb_volume!` plot recipe; the dense `Array{Float32,3}` → `.nvdb` writer; live volume edits.

## Architecture — three independently-testable units

### Unit 1 — IndeX enablement (`Base.OncePerProcess`, process-global)

A module-level `const _ensure_index = Base.OncePerProcess{Bool}() do … end` (Julia 1.12) that runs **once**, **before the first `ovrtx_create_renderer`** (called at the top of `OV.Renderer()` so every renderer-creating path triggers it). It reads env vars and, if volume rendering is requested, sets up the carb config + `CARB_FRAMEWORK_CONFIG_NAME` so IndeX can find its libs. Returns whether IndeX was enabled.

Env-var contract (both optional; absent → IndeX not enabled = today's behavior, zero overhead):
- `OMNIVERSEMAKIE_OVRTX_CONFIG` = absolute path to a ready ovrtx carb config (the user manages it; it already registers `/app/tokens/omni.index.libs`). Highest precedence.
- `OMNIVERSEMAKIE_INDEX_LIBS` = path to the `omni.index.libs` ext dir. OmniverseMakie SYNTHESIZES a config (copy of the install's `ovrtx.config.json` + the `app.tokens."omni.index.libs"` key) and wires `CARB_FRAMEWORK_CONFIG_NAME`.

**Mechanism note (carb-config injection — VERIFY-in-REPL during the plan, the one genuinely tricky bit):** `CARB_FRAMEWORK_CONFIG_NAME` is force-joined onto `CARB_APP_PATH=…/ovrtx/bin` and absolute values are mangled (proven in the spike). The implementation must deliver the synthesized config via a path that survives that join — re-derive the robust form in a REPL against the spike's exp6 recipe (the spike used a relative `../`×N path to a scratch file; harden it: write the synthesized config to a deterministic location and compute the relative path, OR set `CARB_APP_PATH` ourselves to a controlled dir). This is the M6.B-style "VERIFY the FFI/runtime ABI before relying on it" step. The OncePerProcess must NOT throw if disabled/misconfigured — it `@warn`s once and returns `false` (volumes stay unavailable; non-volume rendering is unaffected).

### Unit 2 — UsdVol authoring + colormap volume material

`OmniverseMakie.author_vdb_volume!(screen, scene, vdb_path; prim_path="/World/Volume", field="density", field_dtype="float", colormap=:viridis, colorrange=nothing)` authors into the Screen's already-open stage (after `author_root_from_scene!` set up camera/lights/render-product from the Makie `scene`):
- a `Volume` prim at `prim_path` with `rel field:<field>` → a child `OpenVDBAsset{ fieldName=field, fieldDataType=field_dtype, filePath=@vdb_path@ }`;
- an IndeX **`nvindex:volume` material with a `Colormap`** transfer function built from `colormap`+`colorrange` (density → RGBA), bound to the Volume (form per the on-disk reference `…/omni.rtx.index_composite-*/data/tests/usd/torus-volume-with-geometry.usda`). Mirrors existing `src/translation/usd.jl` / `add_usd_reference!` authoring (newline-separated metadata, multi-line prim bodies — the spike's "Invalid RenderProduct Prim" gotcha).

A bare default-material render is the fallback if the colormap material proves finicky (the spike confirmed the bare volume already renders), but the target is the colormap-shaded render.

### Unit 3 — Render + test

Render the Screen's stage through the existing `OV.render_to_matrix` / `Makie.colorbuffer` path (the volume is in the open stage; IndeX renders it inline with RT2). Subprocess test: set the enable env → build a `Scene` with a `cam3d!` framing the origin → `Screen` → `author_root_from_scene!` → `author_vdb_volume!(screen, scene, <sample.vdb>)` → render → assert a meaningful count of non-black, non-background (colormap-colored) pixels in the volume region; assert the IndeX-init log line succeeded.

## Error handling

- IndeX disabled / libs not found / config malformed → `_ensure_index` `@warn maxlog=1` + returns `false`. `author_vdb_volume!` then **errors clearly** ("volume rendering requires IndeX — set `OMNIVERSEMAKIE_INDEX_LIBS` or `OMNIVERSEMAKIE_OVRTX_CONFIG` before creating a Screen") rather than silently authoring a prim that won't render — a clear failure beats a black no-op.
- A failed `open_usd`/author must not poison later renders — follow the spike's "fresh `OV.Renderer()` per stage" discipline in tests.

## Testing

Subprocess (env-gated, **skips gracefully** when the IndeX libs / a sample VDB are absent, so CI without them is green): the Unit-3 render assertion. Sample asset: the on-disk `…/omni.rtx.index_composite-*/data/tests/volumes/torus.vdb` (grid `torus_fog`) — referenced by path, not vendored (large). A second test asserts the disabled path (no env) is a clean no-op/clear-error and does not regress non-volume rendering.

## Portability / dependencies (documented, not solved in M1)

Volume rendering requires (a) the complete Kit `omni.index.libs` ext present on the machine (ovrtx's bundle is incomplete) and (b) a VDB/NVDB file. M1 surfaces both via the env var + the `vdb_path` arg and documents the dependency; **vendoring/bundling the 13-lib `nvindex-libs` for deployment is a later concern**, not M1.

## Non-goals (M1)

Makie `volume!`/`vdb_volume!` recipe; dense `Array→.nvdb` writer (M2 — verified path: a ~40-line C++ shim over vendored header-only NanoVDB, MPL-2.0, pinned `NANOVDB_MAJOR_VERSION==32`; `NanoVDB_jll` does not exist and Hikari.jl is unlicensed/un-vendorable — see `.superpowers/sdd/volume-nanovdb-research.md`); live volume edits; multi-field/vector grids; homogeneous MDL media.

## Forward (M2, recorded so M1 doesn't paint into a corner)

`author_vdb_volume!` is the authoring primitive M2's `volume!(x,y,z,array)` reuses: M2 writes the dense array to a temp `.nvdb` (the C++ NanoVDB shim) and calls `author_vdb_volume!` with that path + the plot's colormap/colorrange. Keep Unit-2's signature array-agnostic (a file path + render attrs) so M2 layers on top without changing it.
