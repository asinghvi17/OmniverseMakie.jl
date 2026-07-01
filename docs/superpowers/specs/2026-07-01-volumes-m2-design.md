# Volumes Milestone 2 — Makie `volume!` with real colors + live edits — Design

**Status:** approved direction (user, 2026-07-01: full scope — writer + `volume!` recipe + composite colormap COLORS + full live edits incl. data; NanoVDB writer as a `lib/` sub-package; spike-gated colors + reload, "sounds good").

## Goal

`volume!(x, y, z, scalars::Array{Float32,3})` in a Makie scene renders a **colormap-shaded** volume through OmniverseMakie → ovrtx → NVIDIA IndeX, and updates **live** when the data, colormap, or colorrange changes. Deliverable: a dense Julia 3-D array becomes a first-class Makie volume plot, rendered and interactively editable.

## Background (what M1 established — see `docs/superpowers/specs/2026-06-30-volumes-m1-design.md`)

- **IndeX is enabled** once per process (`OV._ensure_index`, carb-token injection) from `OMNIVERSEMAKIE_INDEX_LIBS` / `OMNIVERSEMAKIE_OVRTX_CONFIG`. Volumes render through the existing RT2 → IndeX Direct path.
- **`author_vdb_volume!(screen, scene, vdb_path; prim_path, field, field_dtype, colormap, colorrange)`** (`src/translation/volume.jl`) authors a `Volume` + `OpenVDBAsset` + a `nvindex:volume` `Colormap` material into the open stage from an on-disk `.vdb`/`.nvdb`, erroring clearly when IndeX is off. M2's `volume!` recipe **reuses this authoring primitive unchanged** (it was designed array-agnostic: a file path + render attrs).
- **★ M1 verified constraint (the reason Unit 3 exists):** the Colormap transfer-function COLOURS do NOT apply via IndeX **Direct** — a `nvindex:volume` material added by reference renders with Direct's default grayscale-density TF (a colourful viridis TF produced byte-identical gray output). The authored colours require the IndeX **composite** path (verified renderable by NVIDIA's reference `…/omni.rtx.index_composite-*/data/tests/usd/torus-volume-with-geometry.usda`).
- **★ Writer research (see `.superpowers/sdd/volume-nanovdb-research.md`):** `NanoVDB_jll` does not exist; `OpenVDB_jll` is `.vdb`-only + C++-ABI + unregistered; **Hikari.jl** has exactly the pure-Julia writer we want but was unlicensed. **The user has the author's (Simon Danisch / Anton Smirnov) verbal OK to lift it.** On the `sd/vk-hw-accel` branch, Hikari's `save_nanovdb` already emits a **standard major-32 NanoVDB file** (magic `NanoVDB0`, gridType Float, gridClass FogVolume) — a *lift, not a rewrite*.

## Scope

**In (M2):** (1) a `lib/NanoVDBWriter` sub-package (dense `Array{Float32,3}` → `.nvdb`); (2) a Makie `volume!` recipe (array → temp `.nvdb` → `author_vdb_volume!`); (3) the composite-colormap path (real COLOURS); (4) live edits — colormap/colorrange (cheap) and volume data (re-write + reload). **Out (→ M3+):** multi-field / vector / non-`Float32` grids; `.vdb` (OpenVDB) output; homogeneous MDL-media volumes; a `vdb_volume!` file-plot convenience (the low-level `author_vdb_volume!` already covers on-disk files); GPU-resident volume writes.

## Architecture — four sequentially-dependent, independently-testable units

### Unit 1 — `lib/NanoVDBWriter` sub-package (dense array → `.nvdb`)

A new **in-repo sub-package** at `lib/NanoVDBWriter/` (own `Project.toml`; wired into the main package via `[workspace].projects` + `[sources]` + `[deps]` + `[compat]`, mirroring `lib/LibOVRTX` — managed via Pkg where possible, not hand-edited TOML). Kept separate so it can be **extracted into a standalone package later** with zero coupling to OmniverseMakie.

- **Contents:** lift Hikari's pure-Julia NanoVDB writer from `JuliaGraphics/Hikari.jl` @ `sd/vk-hw-accel` (`src/integrators/volpath/nanovdb.jl`): `build_nanovdb_from_dense` + both `save_nanovdb` methods + the coord hashers / `write_buf!` / `bitmask_set!` helpers + `ZStream`/`compress_zlib` + a minimal reader for tests (`parse_nanovdb_buffer`/`extract_nanovdb_metadata`). ~450 lines. Re-clone Hikari to lift from (the prior clone is gone).
- **Public API (mirrors the source, the only surface M2 depends on):** `save_nanovdb(path, data::Array{Float32,3}, origin, extent)` — `origin`/`extent` as `GeometryBasics.Point3f`/`Vec3f` (voxel size = `extent ./ size(data)`); background/name default to `0.0f0`/`"density"`.
- **★ Two deliberate changes for ovrtx safety (the research's caveats):** emit **`Codec::NONE`** (uncompressed grid payload, codec field 0 — warp's sample `.nvdb`s are uncompressed; ZIP is unverified in IndeX), and confirm the **checksum** field (port the GridChecksum, or write the disabled sentinel `typemax(UInt64)` if IndeX skips it).
- **Deps:** `Zlib_jll` (retained for the reader / optional ZIP) + `GeometryBasics` (the `Point3f`/`Vec3f` sig; replaceable with `NTuple{3,Float32}`). Nothing else — no C++, no OpenVDB/TBB.
- **License / attribution:** header on the lifted file crediting Hikari.jl (Simon Danisch, Anton Smirnov) and the NanoVDB format (MPL-2.0); the upstream license to be formalized. A `lib/NanoVDBWriter/README` records provenance + the author's grant.
- **Validated FIRST, before Unit 2 builds on it (M1 verify-first discipline):** (a) a pure round-trip — write → parse the header → assert magic `NanoVDB0`, major 32, gridType Float, expected voxel count; (b) an **ovrtx-render spike** — a written `.nvdb` loads and renders non-black via `author_vdb_volume!` (this is the go/no-go for the whole codec/checksum/version choice).

### Unit 2 — the Makie `volume!` recipe (translation; static author-once)

A `to_ovrtx_object` / `author_usd_prim!` branch for Makie's built-in `Volume` plot, so standard `volume!(x, y, z, array)` renders through the backend like `mesh!`/`scatter!`.

- **Data flow:** `Volume` plot → derive world `origin`/`extent` from the `x`/`y`/`z` ranges (the plot's sampled grid bounds) → `NanoVDBWriter.save_nanovdb` to a temp `.nvdb` (a screen-owned temp path) → `author_vdb_volume!(screen, scene, tmp; field="density", colormap, colorrange)`.
- **Attributes honoured:** `colormap`, `colorrange` (→ the `Colormap` TF, per M1). Makie `algorithm`/`isovalue`/`absorption` are **best-effort/ignored** — IndeX has its own volume model (documented; not silently dropped).
- **★ Orientation:** Julia column-major `(i,j,k)` → NanoVDB `Coord` + world placement so the rendered volume matches the `volume!(x,y,z)` axes. **Verified against a deliberately asymmetric array** (a value at a known corner lands at the correct world corner), reusing the orientation discipline that fixed the M4 surface texcoords.
- **Lifecycle:** static author-once — the temp `.nvdb` is written at author time; the screen owns the temp path for cleanup (Unit 4 replaces it on a live edit; `close` deletes it).

### Unit 3 — the composite colormap path (real COLOURS; spike-gated, verify-or-degrade)

Make the authored `Colormap` TF actually colour the volume, via the IndeX **composite** path. From NVIDIA's reference asset, composite needs: **volume-prim flags** (`nvindex:composite=1`, `omni:rtx:skip=1`, `apiSchemas=["MaterialBindingAPI"]`, `material:binding` → the `nvindex:volume` Colormap material — all of which compose fine through `add_usd_reference!`), **plus** a render setting `rtx:index:compositeEnabled=1` (+ `compositeDepthMode`).

- **★ The open question the spike resolves:** how to enable `compositeEnabled` — as a **carb setting** (cleanest: extend `index_config.jl`'s synthesized config with `/rtx/index/compositeEnabled`, no shared-authoring change) **vs.** root-layer `customLayerData.renderSettings.rtx:index:compositeEnabled` (modifies `author_render_root!` in `usd.jl`). The spike authors the composite volume both ways and finds the minimal form that renders the TF colours.
- **★ Matched set (trap):** the composite flags and `compositeEnabled` are all-or-nothing — `omni:rtx:skip=1` tells RT2 to skip the prim, so WITHOUT `compositeEnabled` the volume renders BLACK (RT2 skips it, composite off, Direct never sees it). So Unit 3 adds the composite flags **only** in the composite path; the degrade path keeps M1's bare-Direct `_vdb_volume_usda` form verbatim (no `skip`, no `composite`).
- **Verify-or-degrade (M1 pattern):** if colours render, wire the minimal form into `_vdb_volume_usda` (composite flags, gated) + wherever `compositeEnabled` must live; a `volume!` render then shows colormap colours. **If composite proves to need more than M2 can carry, degrade to grayscale-Direct** (M1's shipped behavior) and defer colours — surfaced to the user before shipping grayscale, not silently.

### Unit 4 — live edits (diff-path integration)

- **Colormap / colorrange edits (cheap):** an in-place USD attribute write to the `Colormap` prim (`rgbaPoints` from the new colormap, `domain` from the new colorrange) + `reset!`, routed through the existing `push_to_ovrtx!` diff path. Only meaningful once Unit 3 lands.
- **Volume data edits (the expensive one; reload-spike-gated):** a `push_to_ovrtx!` branch on the `Volume` plot's scalar input → re-run `save_nanovdb` to a **fresh** temp `.nvdb` → reload in IndeX → `reset!`. **Starts with a reload-behavior spike:** does writing a new `filePath` on the `OpenVDBAsset` reliably reload past carb's asset cache (the spike logs showed asset eviction), or is `remove_usd!`+`add_usd_reference!` needed? The spike picks the mechanism. **Temp-file GC:** each data edit writes a new temp path (a reused path risks stale cache); the screen deletes the prior temp on each edit and on `close`. **If neither reload mechanism is reliable, stop and surface** — live *data* edits defer to M3; colormap edits + static `volume!` still ship.
- **Cost is honest, not hidden:** data edits re-run the writer (O(voxels), not a zero-copy hot path — UsdVol is strictly file-based); fine for interactive grids, documented as slow for very large ones.

## Error handling

- Writer: assert `eltype(data)==Float32` / `ndim==3` / finite `extent`; a zero/degenerate grid writes an empty-but-valid file (no crash).
- `volume!`: reuses `author_vdb_volume!`'s clear IndeX-disabled error; a data edit that fails to reload `@warn`s once and keeps the last good frame (never a black flash mid-interaction).
- Temp files: always cleaned on `close`; a failed write leaves the prior good file in place.

## Testing

- **Unit 1:** pure round-trip (header/magic/version/voxel-count) — always runs; ovrtx-render spike of a written `.nvdb` (subprocess, env-gated skip-if-absent).
- **Unit 2:** `volume!(x,y,z,array)` end-to-end render (subprocess) — non-black; orientation asserted on an asymmetric array.
- **Unit 3:** colored render (subprocess) — the volume region shows colormap colours (mean R≠G≠B distinguishing it from grayscale), **or** a documented-degrade test asserting non-black grayscale.
- **Unit 4:** live-update renders (subprocess) — after a colormap edit the frame's colours change; after a data edit the volume's occupied region changes; temp-file count stays bounded across edits.
- Full `Pkg.test()` — no regression across M0–Volumes-M1 (the writer sub-package + `volume!` add no GLMakie/CUDA to the offscreen path; `using OmniverseMakie` stays offscreen-pure).

## Portability / dependencies

`Zlib_jll` + `GeometryBasics` are the only new deps, isolated in the `lib/NanoVDBWriter` sub-package (pure Julia, no native toolchain). Rendering still requires the Kit `omni.index.libs` ext present at runtime (M1's env contract), surfaced by the existing skip-if-absent test gate. The sub-package carries a pinned NanoVDB **major version 32** in its format constants (matches the box's v32.8; IndeX's `Version::isCompatible` requires equal major).

## Non-goals (M2)

`.vdb` (OpenVDB) output; multi-field / vector / non-`Float32` grids; homogeneous MDL-media (`outputs:mdl:volume`) volumes; a `vdb_volume!` file convenience plot; GPU-resident / incremental NanoVDB updates; sub-package registration/publishing (kept separable, but shipped in-repo).

## Forward (M3+, recorded so M2 doesn't paint into a corner)

The `lib/NanoVDBWriter` boundary + the `save_nanovdb(path, data, origin, extent)` API are the extraction seam for a future standalone package. If Unit 3 degrades (grayscale) or Unit 4-data defers (reload unreliable), each is an isolated M3 slice built on the same units. `.vdb` output and multi-field grids extend Unit 1's writer without touching Units 2–4.
