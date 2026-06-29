# M4 — Examples Gallery — Design

**Date:** 2026-06-29
**Milestone:** M4 (Examples gallery) — the first post-M3 milestone in the reordered roadmap (examples land before the interactive viewport so the gallery has full materials).
**Status:** design approved in brainstorming; pending spec review → `writing-plans` (the bite-sized `M4_PLAN.md`).

## Goal

Port a **broad set of real, recognizable path-traced scenes** from the two existing Makie ray-tracing galleries — `references/RPRMakieNotes/` and `references/raydemo/` — into `OmniverseMakie.jl/examples/`, rendered end-to-end through OmniverseMakie. This is both the project's **showcase** and the **end-to-end validation** that the M0–M3 backend handles real-world scenes (mesh / scatter / meshscatter / lines / surface + OmniPBR materials + image textures + lights + camera).

## Non-goals (M4)

- **No new `src/` backend features (STRICT PORT-ONLY).** M4 adds zero translation code. A scene that needs an unimplemented *render path* is marked **⛔ blocked** in the port-status table, not enabled. (Loading a mesh from a file via MeshIO/FileIO is NOT a backend feature — our M1 Mesh path already renders any `GeometryBasics` mesh — so file-loaded meshes are in scope.)
- **No committed reference-image regression.** Verification is property-asserts (non-black / expected regions), the M1–M3 test style — not pixel-compared committed reference PNGs.
- **No Documenter docs site.** The gallery is `examples/renders/*.png` + `examples/README.md`.
- **No edits to the originals.** `references/` is read-only; we adapt out of it.

## User-facing shape

`examples/` is a self-contained, separately-environment'd gallery:

```
examples/
  Project.toml            # OWN env: OmniverseMakie + MeshIO/FileIO/GeometryBasics/Colors/ImageIO/… (via Pkg)
  common/
    harness.jl            # activate!, run_example, asset resolver, property-assert helpers
  assets/                 # GITIGNORED — populated by fetch_assets.jl (copy from references/ + downloads)
  renders/                # COMMITTED — examples/renders/<scene>.png — the viewable gallery
  rpr/<scene>.jl          # ported RPRMakieNotes scenes (one file each)
  raydemo/<scene>.jl      # ported raydemo scenes (one file each)
  fetch_assets.jl         # one-time setup: populate assets/ from references/ + URLs (NOT at render time)
  run_all.jl              # the M4 gate: render every ported scene → renders/ + property-assert
  README.md               # gallery index + port-status table + embedded hero renders
```

### The harness (`examples/common/harness.jl`)

- Each scene file defines `scene_<name>(; kw...) -> Makie.Figure` (or `Scene`) — pure scene construction, no `save`/`display`.
- `run_example(name::AbstractString, scene_fn; size, out) -> Matrix{RGBA{N0f8}}` — activates OmniverseMakie (`OmniverseMakie.activate!()`), builds the figure, `Makie.save("examples/renders/<name>.png", fig)` (which routes through our `colorbuffer`), and returns the image for asserts.
- `asset(scene, relpath) -> String` — resolves an asset to its on-disk path under `examples/assets/<scene>/…`, erroring with a "run fetch_assets.jl first" message if missing.
- Property-assert helpers reused from the M1–M3 test idiom: `nonblack_count`, `assert_nonblack`, region/luminance checks. Renderer runs are **subprocess-isolated** (carb signal handlers) using the same harness pattern as `test/helpers.jl`.

### Environment (`examples/Project.toml`)

Its own Pkg environment (so example deps never touch the package's test env), declaring `OmniverseMakie` (dev path) + the example dependencies (`MeshIO`, `FileIO`, `ImageIO`, `GeometryBasics`, `Colors`, `ColorTypes`, plus per-scene needs). Managed via Pkg, never hand-edited.

### Assets (`examples/fetch_assets.jl` + gitignored `examples/assets/`)

Several scenes carry large data (Crown 112 MB, Volumes 114 MB, ProtPlot 25 MB, Trixi 13 MB; ~250 MB total). Committing that would bloat the repo. Instead:

- `examples/assets/` is **gitignored**.
- `examples/fetch_assets.jl` is a **one-time setup step** (NOT run at render time): for each scene it copies the scene's assets from the read-only `references/raydemo/<scene>/…` / `references/RPRMakieNotes/…` into `examples/assets/<scene>/…`, and `Downloads.download`s the few network textures (e.g. RPRMakie's Earth jpg) into the same place.
- Scenes load assets only via `asset(scene, relpath)` (never a raw `Downloads.download` at render time → no network during rendering).
- Truly tiny shared assets MAY be committed directly; the default is fetch-into-gitignored.

### Verification & gate (`examples/run_all.jl`)

- `run_all.jl` (in the examples env) renders every **ported** scene through the harness, writes `examples/renders/<scene>.png`, and property-asserts each (non-black + scene-appropriate region/feature checks). It prints a per-scene pass/fail summary and exits non-zero on any failure.
- **M4 GATE:** `run_all.jl` green on the full ported set **AND** the port-status table in `examples/README.md` complete (every candidate scene classified) **AND** the hero renders embedded in the README.
- **Not wired into `Pkg.test`** — env separation keeps the main suite fast; the gallery is its own runner / CI job.

## Scenes

**Ported set (~27 — every clearly-portable scene):**

- **RPRMakieNotes (15):** sphere_plane_greysky, sphere_source_light, reflections_glass_material, transparentM, transparentMaterial, materials_julia_room, uberMExample, helix, rrg, earthquakes, earthquakesLight, submarineCables, twoEarths, earth_ina_julia_box, pointsfont (font glyph → `normal_mesh`, NOT `text!`).
- **raydemo (12):** Ark, BlackHole, Crown, Geant4, KillerooGold, koeln_flooding, Materials, Plants, ProtPlot, SandCat, Trixi, Volumes (uses mesh/surface despite the name — no `volume!`).

**⛔ Blocked (6 — need an unimplemented render path; documented in the table, deferred):**

| Scene | Source | Missing render path |
|---|---|---|
| betterview | RPRMakieNotes | `text!` (axis labels) + `arrows!` |
| freetype_text | RPRMakieNotes | `text!` |
| saveRPR | RPRMakieNotes | `image!` (2-D image plot) |
| volumeM | RPRMakieNotes | `volume!` |
| GLTF | raydemo | GLTF scene-graph import |
| Waterlily | raydemo | `volume!` |

### Porting model (per scene)

Each ported scene adapts the original: swap `GLMakie`/`RPRMakie` → `OmniverseMakie`; translate RPR/`matsy` materials (`RPR.Matte`/`UberMaterial`/MaterialX) → our `material=` (metallic/roughness/emissive/opacity) and image `color=`; replace runtime `Downloads.download` with `asset(scene, …)`; drop any unsupported sub-plot (and note the omission in the scene's docstring + the table if it changes fidelity). The scene file stays small by leaning on `common/harness.jl`.

## Testing

`run_all.jl` is the gate (above). Each scene's property-assert is scene-appropriate (a textured globe asserts the texture's two dominant colors appear; a metallic scene asserts a bright specular fraction; a multi-primitive scene asserts non-black + expected lit regions). Asserts target *robustness under RTX nondeterminism*, never bit-exactness.

## Risks / open items (resolved in the plan)

1. **RPR material API → OmniPBR mapping fidelity.** RPR materials (`RPR.Matte`, `UberMaterialv3`, MaterialX `.mtlx`) don't map 1:1 to OmniPBR. Mitigation: map to the closest OmniPBR `material=` (metallic/roughness/emissive/opacity); a scene whose look depends on an unmapped RPR feature is ported best-effort with a docstring note. No MaterialX-primary support (M3 non-goal).
2. **Asset availability.** `fetch_assets.jl` assumes `references/` is present (it is, in this repo). A scene whose asset is missing fails fast with a clear message.
3. **Per-scene render cost / total runtime.** ~27 RTX renders × tens of seconds → `run_all.jl` is minutes-long; acceptable for a separate runner (not `Pkg.test`). `run_all.jl` supports a single-scene filter for iteration.
4. **Heavy/niche domain scenes** (Geant4, Trixi, ProtPlot, koeln_flooding) are portable (mesh-based) but large; included in the ported set, flagged in the table as heavy-asset.
