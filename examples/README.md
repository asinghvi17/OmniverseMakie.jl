# OmniverseMakie Examples Gallery

Real, recognizable path-traced scenes, ported from the
[`RPRMakieNotes`](https://github.com/lazarusA/RPRMakieNotes) gallery and rendered
**end-to-end through OmniverseMakie** — Makie's scene graph translated to an OpenUSD stage
and rendered by NVIDIA's `ovrtx` RTX path tracer. This is both the project's showcase and an
end-to-end check that the M0–M3 backend handles real-world scenes: mesh / scatter /
meshscatter / lines / surface, OmniPBR materials, image textures, lights, cameras, and
per-plot transforms.

## Running the gallery

The gallery is a **self-contained Pkg environment** (it dev-depends on the parent package).

```bash
# 1. instantiate the examples environment
julia --project=examples -e 'using Pkg; Pkg.instantiate()'

# 2. one-time: populate examples/assets/ (copies from ../references + a few downloads)
julia --project=examples examples/fetch_assets.jl

# 3. render every scene → examples/renders/<scene>.png, with per-scene property asserts
julia --project=examples examples/run_all.jl

# (render a single scene)
julia --project=examples examples/run_all.jl uberMExample
```

`run_all.jl` renders each scene in an isolated subprocess (ovrtx installs `carb` signal
handlers), writes the PNG to `examples/renders/`, and property-asserts the result (non-black
+ a scene-appropriate signal). It is **not** part of `Pkg.test` — it is its own runner.
By default the in-repo `OVRTX_jll` artifact downloads the official ovrtx C archive on first
use; set `OVRTX_LIBRARY_PATH` only to override that runtime with a manual install.

## Gallery

| Scene | Plots exercised | Render |
|---|---|---|
| **uberMExample** | `mesh!` + OmniPBR metallic/roughness + image texture | ![](renders/uberMExample.png) |
| **materials_julia_room** | `mesh!` ×many, emissive + glass + textures + transforms | ![](renders/materials_julia_room.png) |
| **helix** | `mesh!` + `meshscatter!` + `lines!`, Luxor glyph meshes | ![](renders/helix.png) |
| **earthquakes** | `meshscatter!` (colormap) + glass box | ![](renders/earthquakes.png) |
| **earthquakesLight** | `meshscatter!` + emissive boxes + earth texture | ![](renders/earthquakesLight.png) |
| **submarineCables** | `surface!` + `meshscatter!` + `lines!`, GeoJSON | ![](renders/submarineCables.png) |
| **rrg** | `linesegments!` + `meshscatter!` (colormap) | ![](renders/rrg.png) |
| **earth_ina_julia_box** | `mesh!`, emissive corners + earth texture | ![](renders/earth_ina_julia_box.png) |
| **twoEarths** | `mesh!` ×2, earth texture | ![](renders/twoEarths.png) |
| **transparentMaterial** | `mesh!`, opacity + earth texture | ![](renders/transparentMaterial.png) |
| **transparentM** | `mesh!`, nested opacity | ![](renders/transparentM.png) |
| **reflections_glass_material** | `mesh!`, glass in a room box | ![](renders/reflections_glass_material.png) |
| **sphere_source_light** | `mesh!`, emissive source spheres | ![](renders/sphere_source_light.png) |
| **sphere_plane_greysky** | `mesh!`, diffuse sphere + plane | ![](renders/sphere_plane_greysky.png) |

## Port status

All 14 portable RPRMakieNotes scenes are ported (✅). Three need plot types OmniverseMakie
does not implement yet (`text!` / `arrows!` / `image!`); `volumeM` is unported because its
colored volume needs a Kit composite runtime (grayscale `volume!` has shipped). The
`raydemo` gallery is **deferred** (see below).

| Scene | Status | Note |
|---|---|---|
| sphere_plane_greysky, sphere_source_light, reflections_glass_material, transparentM, transparentMaterial, uberMExample, materials_julia_room, earth_ina_julia_box, twoEarths, rrg, helix, earthquakes, earthquakesLight, submarineCables | ✅ ported | rendered + property-asserted by `run_all.jl` |
| betterview | ⛔ blocked | needs `text!` + `arrows!` |
| freetype_text | ⛔ blocked | needs `text!` |
| saveRPR | ⛔ blocked | needs `image!` (also a post-render compositor, not a scene) |
| volumeM | ⚠ unported | `volume!` ships **grayscale** (small UsdVol + dense-Array→.nvdb); volume COLORS need a Kit composite runtime, so this colormapped scene has no faithful standalone port |
| **`raydemo/` gallery** | ⏸ deferred | built on a different renderer (Hikari / RayMakie) needing participating-media volumes, PBRT/GDML scene-graph import, and heavy domain libraries — out of scope for this milestone |

## Porting notes

These scenes were authored for RPRMakie; porting to OmniverseMakie applies a few consistent
adaptations:

- **Materials** map RPR materials to our OmniPBR `material=` escape hatch:
  `DiffuseMaterial` → plain `color=` (USD displayColor matte); `EmissiveMaterial(c)` →
  `material=(; emissive=c)`; `Glass` → `material=(; glass=true, ior=…)` (TRUE refractive
  OmniGlass — a real `UsdShade` glass shader with `glass_ior`, not an alpha cut-out);
  `UberMaterial` metalness/roughness/transparency → `material=(; metallic, roughness, opacity)`.
- **Image textures** are passed in **natural orientation** (`color = img`). RPRMakie's UV
  convention is transposed relative to ours, so the originals' `img'` transpose is dropped —
  with it, the texture renders rotated 90°. Image textures are carried through both `mesh!`
  and `surface!` (the surface samples a `diffuse_texture` over its grid's parametric `st` UVs).
- **Lights** keep `PointLight`/`EnvironmentLight`; note `PointLight` is `PointLight(color,
  position)` here. Image-based environment maps ARE honored: a scene `EnvironmentLight`
  image is authored into a DomeLight at author time (`src/screen.jl` `_author_env_light!`;
  a matrix goes through a temp PNG, an `.exr`/`.hdr` path is used directly for true HDR).
  These ports pass a neutral 1×1 grey dome for simplicity and let the key/fill point lights
  carry the lighting — swap in a real EXR/HDR to drive image-based lighting.
- **Assets** are fetched once into the gitignored `examples/assets/` by `fetch_assets.jl`
  (never downloaded at render time); scenes resolve them via `asset(scene, relpath)`.

## Layout

```
examples/
  Project.toml          # own env (dev-deps OmniverseMakie + example deps)
  common/harness.jl     # run_example, asset(), property-assert helpers
  common/run_one.jl     # renders ONE scene in a subprocess + asserts
  fetch_assets.jl       # one-time asset setup (copies + downloads)
  run_all.jl            # the gallery runner / gate
  rpr/<scene>.jl        # one file per scene: scene_<name>() + assert_<name>(img)
  assets/               # gitignored, populated by fetch_assets.jl
  renders/              # committed PNG gallery
```
