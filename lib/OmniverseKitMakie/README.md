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
updates and `Base.display` integration are future work; the roadmap (the GPU
data planes — CUDA IPC in, `omni.syntheticdata` CUDA tensors out) is in
[`docs/superpowers/specs/2026-07-15-omniverse-kit-makie-design.md`](../../docs/superpowers/specs/2026-07-15-omniverse-kit-makie-design.md).

**Why a subprocess (and not in-process)?** An in-process transport
(`libkitjl`, a C shim over Carbonite's ABI) was built and then **removed**:
controlled experiments proved Kit *cannot* be hosted in-process by a non-Kit
executable — `OMNI_APP_GLOBALS` (Kit's omni-core client context) must live in
the process **main executable**, which `julia`'s cannot carry; startup hangs
in `omni.usd_resolver`'s static initializer otherwise (a Julia-free C harness
reproduces it; a C++ main-exe with identical init works). Full evidence:
[`docs/superpowers/specs/2026-07-15-libkitjl-design.md`](../../docs/superpowers/specs/2026-07-15-libkitjl-design.md).
The only in-process route would be inverting the host (a Kit-globals
executable embedding `libjulia`).

## GPU data plane

Frames out of the server, three ways (`render!(screen; device = …)`;
[design](../../docs/superpowers/specs/2026-07-16-kit-gpu-data-plane-design.md)):

- **`:cpu`** (and `:auto`, which `Makie.colorbuffer` rides) — zero-disk
  capture: the server memmoves the RGBA8 frame into POSIX shared memory and
  Julia mmaps it. No PNG encode, no disk. Available whenever the server
  reports `shm_out` in [`gpu_caps`](@ref).
- **`:cuda`** — device-resident frames: `omni.syntheticdata`'s `LdrColorSDPtr`
  node exposes the render var as a CUDA device pointer; the server copies it
  device→device into a `cudaMalloc`'d buffer exported once over **CUDA IPC**,
  and Julia (with CUDA.jl loaded — the `OmniverseKitMakieCUDAExt` extension)
  wraps it as a `CuMatrix{RGBA{N0f8}}`. Zero host copies. Requires the same
  GPU on both sides and the `omni.syntheticdata` extension (fetched once from
  NVIDIA's registry; cached thereafter — the server launch enables it by
  default, `syntheticdata = false` opts out).
- **`:png`** — the original file round-trip, kept as the compatibility floor.

Volumes in, live: `gpu_update_volume!(screen, plot; data::CuArray{Float32,3})`
copies a GPU sim field device→device into a server-owned IPC staging buffer;
the server writes a **fresh** `.vdb` (IndeX's importer is file-based) and
swaps the prim's `filePath`. No Julia-side host copy — the Kit-backend
symmetric of the standalone backend's `gpu_update_mesh!`. The grid size is
frozen at author time.

## Requirements

A built Kit runtime with the IndeX composite extensions resolved (default:
the DSX blueprint's kit-cae build; override with `KIT_RELEASE_DIR`). See the
`examples/kit_index_composite` README for how the extension cache gets
materialized and the system-library gotchas (`libGLU`) the launcher handles.

## Tests

`julia --project=lib/OmniverseKitMakie -e 'using Pkg; Pkg.test()'` — pure
authoring/codec tests always run; the end-to-end GPU testset self-skips
unless a Kit runtime is present (`OMK_SKIP_GPU=1` forces the skip).
