# M4 — Examples Gallery — Design

**Date:** 2026-06-29
**Milestone:** M4 (Examples gallery) — the first post-M3 milestone in the reordered roadmap (examples land before the interactive viewport so the gallery has full materials).
**Status:** design approved; scope corrected 2026-06-29 after a per-scene catalog pass (see "Scope correction" below). Proceeding to `writing-plans`.

## Goal

Port a **broad set of real, recognizable path-traced scenes** from the `references/RPRMakieNotes/` gallery into `OmniverseMakie.jl/examples/`, rendered end-to-end through OmniverseMakie. This is both the project's **showcase** and the **end-to-end validation** that the M0–M3 backend handles real-world scenes (mesh / scatter / meshscatter / lines / surface + OmniPBR materials + image textures + lights + camera + per-plot transforms).

## Scope correction (2026-06-29)

A scene-by-scene catalog of both reference galleries overturned the original spec's premise:

- **`references/raydemo/` is NOT an RPRMakie gallery** — it is built on the **Hikari / RayMakie / Raycore** stack (a different path tracer). Its scenes overwhelmingly require render paths outside the M0–M3 backend (NanoVDB participating-media volumes, PBRT-v4 / Geant4-GDML scene-graph import, custom `Hikari.Medium`) or heavy domain libraries (WaterLily, ProteinChains, PlantGeom/XPalm, Geant4.jl) with non-mesh plots (`ribbon!`, `plantviz!`, `streamplot!`, `volume!`). **Essentially zero raydemo scenes are clean ports.** raydemo is therefore **deferred** out of M4 (revisit in a later milestone if/when those paths land).
- **`references/RPRMakieNotes/` is the real, mappable gallery** — genuine RPRMakie ray-tracing scenes whose RPR materials map onto our OmniPBR `material=`, using exactly the plot types and lights M0–M3 implements. M4 ports **all 14 portable RPRMakieNotes scenes**.

## Non-goals (M4)

- **No new `src/` backend features (STRICT PORT-ONLY).** M4 adds zero translation code. A scene that needs an unimplemented *render path* is marked **⛔ blocked** in the port-status table, not enabled.
- **raydemo is out of M4** (deferred — different renderer / out-of-scope render paths).
- **No committed reference-image regression.** Verification is property-asserts (non-black / expected colours / expected lit regions), the M1–M3 test style — not pixel-compared committed reference PNGs.
- **No Documenter docs site.** The gallery is `examples/renders/*.png` + `examples/README.md`.
- **No edits to the originals.** `references/` is read-only; we adapt out of it.

## User-facing shape

`examples/` is a self-contained, separately-environment'd gallery:

```
examples/
  Project.toml            # OWN env: OmniverseMakie (dev) + all example deps (via Pkg)
  common/
    harness.jl            # run_example, asset() resolver, property-assert helpers
    run_one.jl            # child entry: render ONE scene in a subprocess + assert
  assets/                 # GITIGNORED — populated by fetch_assets.jl (copy from references/ + downloads)
  renders/                # COMMITTED — examples/renders/<scene>.png — the viewable gallery
  rpr/<scene>.jl          # ported RPRMakieNotes scenes (one file each; defines scene_<name>())
  fetch_assets.jl         # one-time setup: populate assets/ from references/ + URLs (NOT at render time)
  run_all.jl              # the M4 gate: for each scene, subprocess-render → renders/ + property-assert
  README.md               # gallery index + port-status table + embedded hero renders
```

### `references/` location

`references/` is a **sibling of the repo** (`<repo>/../references/`), not inside it. `fetch_assets.jl` resolves it via the `OM_REFERENCES_DIR` environment variable, defaulting to `joinpath(repo_root, "..", "references")`.

### The harness (`examples/common/harness.jl`)

- Each scene file defines `scene_<name>() -> Makie.Figure` — pure scene construction (lights via `scenekw=(;lights=…)`, camera via `update_cam!`/`cameracontrols`, plots, materials), no `save`/`display`.
- `run_example(name::AbstractString, scene_fn; size=(900,900), out=<renders>/<name>.png) -> Matrix{RGBA{N0f8}}` — calls `OmniverseMakie.activate!()`, builds the figure, `Makie.save(out, fig)` (which routes through our `colorbuffer`), and returns the image for asserts.
- `asset(scene, relpath) -> String` — resolves `examples/assets/<scene>/<relpath>` to an absolute path, erroring with a "run fetch_assets.jl first" message if missing.
- Property-assert helpers (the M1–M3 idiom): `nonblack_count`, `assert_nonblack`, `dominant_color_fraction` / region-luminance checks. Asserts target **robustness under RTX nondeterminism**, never bit-exactness.

### Rendering is subprocess-isolated

ovrtx installs carb signal handlers, so each render runs in a **child `julia` process** (the `test/helpers.jl` pattern). `run_all.jl` (the parent) spawns `examples/common/run_one.jl <scene>` per scene; `run_one.jl` (the child) `include`s the harness + the scene file, calls `run_example`, runs the scene's property-asserts, prints `PASS:<scene>` / `FAIL:<scene>` markers, and exits 0/1. The child sets `OVRTX_LIBRARY_PATH` (default mirrors `test/helpers.jl`). The **GPU is single (one A5000)** — renders are serialized at execution time even though scene *files* are independent.

### Environment (`examples/Project.toml`)

Its own Pkg environment (so example deps never touch the package's test env), declaring `OmniverseMakie` (dev path) + the example dependencies. The **full** dep set is added up front (so per-scene port tasks never edit `Project.toml`): `GeometryBasics`, `Colors`, `ColorTypes`, `FileIO`, `ImageIO`, `Downloads`, `Luxor` (helix glyph meshes), `CSV` + `DataFrames` (earthquakes), `GeoMakie` + `GeoInterface` + `GeoJSON` (submarineCables). Managed via Pkg, never hand-edited. (Dep policy was relaxed 2026-06-28: new deps OK when justified.)

### Assets (`examples/fetch_assets.jl` + gitignored `examples/assets/`)

`fetch_assets.jl` carries a **complete declarative manifest** for all 14 scenes (built up front from the catalog, so scene port tasks never edit it). For each entry it either **copies** a file from `OM_REFERENCES_DIR/RPRMakieNotes/…` into `examples/assets/<scene>/…`, or **`Downloads.download`s** a URL into the same place. It is a **one-time setup step, NOT run at render time** (no network during rendering). `examples/assets/` is **gitignored**. A scene whose asset is missing fails fast via `asset()` with a clear "run fetch_assets.jl first" message.

Asset inventory:

| Asset | Source | Used by |
|---|---|---|
| `lights/envLightImage.exr` (412K) | copy from references | reflections_glass_material *(see env-light caveat — may be skipped)* |
| `imgs/makie_logo_transparent.png` (76K) | copy from references | materials_julia_room, helix |
| `imgs/lazaro2.png` (23K) | copy from references | materials_julia_room, helix |
| `8k_earth_daymap.jpg` | download (solarsystemscope.com) | transparentMaterial, uberMExample, earth_ina_julia_box, earthquakesLight, submarineCables |
| `2k_earth_daymap.jpg` | download (wikimedia) | twoEarths |
| `2021_01_2021_05.csv`, `2021_06_2022_01.csv` | download (BeautifulMakie) | earthquakes, earthquakesLight |
| `landing-point-geo.json`, `cable-geo.json` | download (telegeography) | submarineCables |

### Verification & gate (`examples/run_all.jl`)

- `run_all.jl` discovers `examples/rpr/*.jl`, subprocess-renders each through the harness, writes `examples/renders/<scene>.png`, and property-asserts each. Supports a **single-scene filter** (`julia --project=examples examples/run_all.jl <scene>`) for iteration. Prints a per-scene pass/fail summary; exits non-zero on any failure.
- **M4 GATE:** `run_all.jl` green on the full ported set **AND** the port-status table in `examples/README.md` complete (every candidate scene classified) **AND** the hero renders embedded in the README.
- **Not wired into `Pkg.test`** — env separation keeps the main suite fast; the gallery is its own runner / CI job.

## Scenes (RPRMakieNotes)

**Ported set (14):**

| # | Scene | Plots | Material mapping | Assets | Extra deps |
|---|---|---|---|---|---|
| 1 | sphere_plane_greysky | mesh! | Diffuse→displayColor | — | — |
| 2 | sphere_source_light | mesh! | Diffuse + Emissive→`emissive=` | — | — |
| 3 | reflections_glass_material | mesh! | Glass→`opacity=`/low-rough | (env: neutral dome) | — |
| 4 | transparentM | mesh! | Uber transparency→`opacity=` | — | — |
| 5 | transparentMaterial | mesh! | Diffuse + Uber transparency; earth texture `color=img` | earth jpg | — |
| 6 | uberMExample | mesh! | Uber metallic/roughness→`material=(; metallic, roughness)`; earth texture | earth jpg | — |
| 7 | materials_julia_room | mesh! (+transforms) | Diffuse/Emissive/Glass; 2 PNG textures | 2 PNGs | — |
| 8 | earth_ina_julia_box | mesh! (+transforms) | Emissive corners; earth texture | earth jpg | — |
| 9 | twoEarths | mesh! | plain color/texture (GLMakie original) | 2k earth jpg | — |
| 10 | rrg | linesegments! + meshscatter! + mesh! | Diffuse + colormap | — | — |
| 11 | helix | mesh! + meshscatter! + lines! (+transforms) | Uber/Diffuse/Emissive; glyph meshes | 2 PNGs | Luxor |
| 12 | earthquakes | meshscatter! + mesh! | colormap scatter + Glass box | 2 CSVs | CSV, DataFrames |
| 13 | earthquakesLight | meshscatter! + mesh! | Diffuse + Emissive; earth texture | earth jpg, 2 CSVs | CSV, DataFrames |
| 14 | submarineCables | surface! + meshscatter! + lines! + mesh! | Diffuse + colormap; earth texture | earth jpg, 2 GeoJSON | GeoMakie, GeoInterface, GeoJSON |

**⛔ Blocked (4 — need an unimplemented render path; documented in the table, deferred):**

| Scene | Missing render path |
|---|---|
| betterview | `text!` (axis labels) + `arrows!` |
| freetype_text | `text!` (custom glyph render) |
| saveRPR | `image!` (2-D image plot) — also a post-render compositor, not a scene |
| volumeM | `volume!` |

*Note:* `pointsfont.jl` is a **helper** (font-glyph → `normal_mesh`), not a standalone scene — it is `include`d by helix and ported as part of scene #11.

### RPR → OmniPBR material-mapping protocol

| RPR / original | OmniPBR `material=` |
|---|---|
| `RPR.DiffuseMaterial(matsys)` (solid colour) | drop `material=`; keep `color=` (USD `displayColor` is the faithful matte) |
| `RPR.EmissiveMaterial(c)` (× mult) | `material=(; emissive=c)` (authors emissive_color + enable_emission + intensity 5000) |
| `RPR.Glass(matsys)` | `material=(; opacity=0.15f0, roughness=0.0f0, metallic=0.0f0)` — **best-effort** transparent (no true refraction/IOR input; docstring note) |
| `RPR.UberMaterial(reflection_metalness=m, reflection_roughness=r)` | `material=(; metallic=m, roughness=r)` |
| `RPR.UberMaterial(transparency=t)` | `material=(; opacity=1−t)` |
| image `color = img'` (texture) | keep `color = img'` → backend auto-emits `diffuse_texture` + `st` UV primvar (M3.3) |

### Env-light caveat

The backend maps `EnvironmentLight` → `UsdLuxDomeLight` but **image-based env textures are not yet honored** (deferred). Scenes that pass a 1×1 colorant "image" (most RPRMakieNotes scenes — `grey90`, `white`, `black`) map cleanly to a solid-colour dome. Scenes passing a real EXR/JPEG env map (reflections_glass_material) drop to a **neutral dome** (the EXR is not loaded), noted in the scene docstring. The key/fill `PointLight`s — which carry these scenes' actual lighting — translate faithfully.

### Porting model (per scene)

Each ported scene adapts the original: swap `using GLMakie, RPRMakie, RadeonProRender` → `using OmniverseMakie`; drop `RPRMakie.RPRScreen` / `matsys` / `replace_scene_rpr!` / `saveRPR` / `GLMakie.activate!()`; translate RPR materials per the protocol; replace runtime `Downloads.download(url)` / `load("./…")` with `asset(scene, relpath)`; keep the camera (`update_cam!` / `zoom!` / `cameracontrols`) and lights (`scenekw=(;lights=…)`); wrap the body in `scene_<name>() -> Figure`. Any unsupported sub-plot is dropped with a docstring note. The scene file stays small by leaning on `common/harness.jl`.

## Testing

`run_all.jl` is the gate. Each scene's property-assert is scene-appropriate (a textured globe asserts the texture's dominant colours appear; a metallic scene asserts a bright specular fraction; a multi-primitive scene asserts non-black + expected lit regions). Asserts target robustness under RTX nondeterminism, never bit-exactness.

## Risks / open items

1. **RPR material API → OmniPBR fidelity.** RPR materials don't map 1:1 to OmniPBR; map to the closest `material=` (metallic/roughness/emissive/opacity). A look that depends on an unmapped RPR feature (true glass refraction, reflection_weight) is ported best-effort with a docstring note.
2. **Heavy-dep scenes** (helix/Luxor, earthquakes(Light)/CSV+DataFrames, submarineCables/GeoMakie+GeoInterface+GeoJSON) add packages to the examples env (acceptable under the relaxed dep policy) and need network downloads at setup time.
3. **Per-scene render cost / single GPU.** ~14 RTX renders × tens of seconds → `run_all.jl` is minutes-long and serialized on one GPU; acceptable for a separate runner (not `Pkg.test`).
4. **Network at setup.** `fetch_assets.jl` downloads several textures/data files; a missing/blocked URL fails fast with a clear message. Renders never touch the network (assets are local by then).
