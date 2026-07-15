# OmniverseKitMakie.jl

Full-color NVIDIA IndeX volume rendering for Makie scenes, through a
persistent headless **Kit** render server.

Standalone ovrtx — what `OmniverseMakie.Screen` drives over C FFI — renders
volume transfer functions **grayscale-only** (IndeX Direct ignores authored
colormaps). The colored *composite* path needs a Kit runtime carrying the
`omni.rtx.index_composite` extension chain; the proof, recipe, and the
binary-level analysis of exactly what ovrtx is missing live in
[`examples/kit_index_composite/`](../../examples/kit_index_composite/README.md).

This subpackage contains **all** Kit-specific machinery, so OmniverseMakie
itself stays untouched — if a future ovrtx build ships the composite marker
path, the core backend absorbs it and this package remains an optional
transport.

```julia
using Makie, OmniverseKitMakie

scene = Scene(size = (512, 512); lights)
cam3d!(scene)
volume!(scene, 0..1, 0..1, 0..1, density; colormap = :viridis)
update_cam!(scene, Vec3f(2.4, 2.4, 1.6), Vec3f(0.5, 0.5, 0.4), Vec3f(0, 0, 1))

screen = KitScreen(scene)          # starts the server, authors + opens the stage
img = Makie.colorbuffer(screen)    # camera sync + converge + capture — IN COLOR
close(screen)
```

## What it does

- `start_kit_server()` — launches the bare `kit` kernel headless with the
  six-extension composite chain (GPU-lock + timeout hygiene, libGLU shim),
  running a line-JSON RPC loop (`kit_server.py`) for its lifetime; one
  ~seconds startup amortizes over many renders.
- `stage_usda(scene)` — authors the scene's `volume!` plots (via
  OmniverseMakie's own `_vdb_volume_usda` emitter), lights (`lights_usda`),
  and camera into a self-contained stage with the composite enablement
  (`rtx:index:compositeEnabled`, per-prim `nvindex:composite` +
  `omni:rtx:skip`). Live screens ship volume payloads as classic OpenVDB
  `.vdb`, converted server-side via `omni.volume`'s pyopenvdb — Kit's
  composite importer cannot fetch NanoVDB data (found the hard way; the
  design doc has the evidence).
- `KitScreen` — scene-backed (`Makie.colorbuffer` = camera sync + render) or
  stage-backed (`render_stage!` for externally-authored USD).

**v1 scope:** volume plots + lights + camera. Other plot types are skipped
with a warning — they are the standalone backend's job. Live volume-data
updates and `Base.display` integration are future work; the roadmap
(including the in-process `libkitjl` C-shim phase over Carbonite's
function-pointer ABI and the GPU data planes — Fabric in,
`omni.syntheticdata` CUDA tensors out) is in
[`docs/superpowers/specs/2026-07-15-omniverse-kit-makie-design.md`](../../docs/superpowers/specs/2026-07-15-omniverse-kit-makie-design.md).

## Requirements

A built Kit runtime with the IndeX composite extensions resolved (default:
the DSX blueprint's kit-cae build; override with `KIT_RELEASE_DIR`). See the
`examples/kit_index_composite` README for how the extension cache gets
materialized and the system-library gotchas (`libGLU`) the launcher handles.

## Tests

`julia --project=lib/OmniverseKitMakie -e 'using Pkg; Pkg.test()'` — pure
authoring/codec tests always run; the end-to-end GPU testset self-skips
unless a Kit runtime is present (`OMK_SKIP_GPU=1` forces the skip).
