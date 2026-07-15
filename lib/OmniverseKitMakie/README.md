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

screen = KitScreen(scene)          # starts the backend, authors + opens the stage
img = Makie.colorbuffer(screen)    # camera sync + converge + capture — IN COLOR
close(screen)
```

## Transports

`KitScreen` reaches a Kit runtime through a **`KitTransport`**; the same
Julia API (`KitScreen` / `colorbuffer` / `open_stage!` / `render!` /
`set_attr!` / `close`) drives both:

- **`:subprocess`** (default) — a persistent headless `kit` subprocess talked
  to over a line-JSON FIFO RPC (`server.jl` + `kit_server.py`). Proven, and
  the **only** path that coexists with in-process standalone ovrtx (two
  processes = two carb frameworks).
- **`:inprocess`** (opt-in, **implemented but not yet functional** — see the
  limitation below) — Kit hosted **in this Julia process** via the `LibKitJL`
  C shim (`libkitjl.so` over Carbonite's ABI): no subprocess, no FIFO.
  Lifecycle + settings are native `kitjl_*` calls; stage open / `set_attr` /
  capture / `write_vdb` reuse `kit_server.py`'s proven handler bodies through
  Kit's Python scripting hatch, driven by Julia pumping `kitjl_update()`.
  Intended for faster repeated renders and, later, zero-copy GPU data planes.

```julia
screen = KitScreen(scene; transport = :inprocess)   # or OMK_KIT_TRANSPORT=inprocess
```

> **⚠ KNOWN LIMITATION (v1): in-process startup deadlocks.**
> The native shim, lifecycle, settings, argv construction, the embedded
> Python helper, and the whole transport op surface are implemented, the
> `libkitjl.so` build + pure symbol tier is green, and in-process framework
> startup *initiates* (carb framework acquired, ~36 Kit extensions load, the
> RTX GPU is detected). But `IApp::startup` then **deadlocks** during the
> `omni.usd_resolver` Python-extension `dlopen` when Kit is co-hosted in the
> Julia process — the calling thread spins (~100% CPU) in a carb loader lock
> and never returns. Reproduced identically with/without the signal guard,
> with `--handle-signals=no`, with a full `carb::startupFramework` init, and
> with a system-`libstdc++` `LD_PRELOAD`. **No in-process frame renders yet;
> use the subprocess transport (the default) for working colored volumes.**
> Details + the still-open seam are in
> [`docs/superpowers/specs/2026-07-15-libkitjl-design.md`](../../docs/superpowers/specs/2026-07-15-libkitjl-design.md).

**Hazard (b):** the in-process Kit app **cannot coexist with in-process
standalone ovrtx** — two carb frameworks would fight over `g_carbFramework`,
`CARB_APP_PATH`, and the crashreporter singleton. A session picks **one**
in-process backend; there is only ever **one in-process Kit app per process**
(carb cannot cleanly restart it — the transport enforces this). If you also
drive ovrtx in-process, use the subprocess transport (the default). Kit is
started inside OmniverseMakie's crashreporter/GC signal guard (breakpad vs
Julia's GC-safepoint SIGSEGV — the same fix ovrtx needs), belt-and-braces with
`--/crashreporter/enabled=false`.

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
updates and `Base.display` integration are future work.

The in-process `libkitjl` C-shim transport (over Carbonite's function-pointer
ABI) is **implemented and opt-in but not yet functional** — its startup
deadlocks when co-hosted with Julia (see the limitation above). Its design,
the deadlock analysis, and the still-open v2 seams (Fabric GPU-pointer
geometry in; `omni.syntheticdata` CUDA AOV readback out to replace the PNG
capture; a thread-pinned background pump for interactive use) are in
[`docs/superpowers/specs/2026-07-15-libkitjl-design.md`](../../docs/superpowers/specs/2026-07-15-libkitjl-design.md).
The broader roadmap is in
[`docs/superpowers/specs/2026-07-15-omniverse-kit-makie-design.md`](../../docs/superpowers/specs/2026-07-15-omniverse-kit-makie-design.md).

## Requirements

A built Kit runtime with the IndeX composite extensions resolved (default:
the DSX blueprint's kit-cae build; override with `KIT_RELEASE_DIR`). See the
`examples/kit_index_composite` README for how the extension cache gets
materialized and the system-library gotchas (`libGLU`) the launcher handles.

The in-process transport additionally needs the Kit **dev headers**
(`$KIT_RELEASE_DIR/kit/dev/include` + `.../fabric/include`) and a C++17
compiler (g++), used once by `lib/LibKitJL/deps/build.jl` to compile
`libkitjl.so` (linking only `-lcarb`; the extension chain loads at runtime).
Kit is NVIDIA-proprietary and **never vendored** — it is located via
`KIT_RELEASE_DIR` exactly like ovrtx, so `libkitjl.so` and the generated
`deps/deps.jl` are gitignored. When Kit/headers are absent the build writes an
"unavailable" marker instead of failing precompile, and the in-process
transport simply reports unavailable (the subprocess transport is unaffected).

## Tests

`julia --project=lib/OmniverseKitMakie -e 'using Pkg; Pkg.test()'` — pure
authoring/codec/transport-dispatch tests plus the LibKitJL build/symbol smoke
(`kitjl_sdk_version`, no GPU) always run; the end-to-end GPU testsets
(subprocess A/B and the spawned, process-isolated in-process A/B) self-skip
unless a Kit runtime is present (`OMK_SKIP_GPU=1` forces the skip).

GPU recipe (serialize on the shared lock; cap with a timeout). The subprocess
test flocks internally, so run `Pkg.test` **without** an outer flock; the
in-process test is spawned as its own process (hazard b) with its own flock.
The standalone in-process parity check:

```bash
DISPLAY=:0 XDG_RUNTIME_DIR=/run/user/1000 \
XAUTHORITY=$(ls /run/user/1000/.mutter-Xwaylandauth.* | head -1) \
KIT_RELEASE_DIR=<release> OMNI_KIT_ACCEPT_EULA=YES \
flock -w 3600 /tmp/omniversemakie-gpu.lock timeout 900 \
  julia --project=lib/OmniverseKitMakie lib/OmniverseKitMakie/test/inprocess_gpu.jl
```
