# Full-color NVIDIA IndeX volume rendering (Kit composite runtime)

**Proven 2026-07-15 on this box:** volume transfer-function *colors* render
correctly through NVIDIA IndeX's **RTX compositing** mode when the scene is
rendered by a Kit runtime carrying `omni.rtx.index_composite`. This is the
capability that is architecturally absent from standalone ovrtx
(`libovrtx-dynamic.so`), where volumes always go through **IndeX Direct** —
scalar-only, default grayscale ramp, authored `Colormap`s ignored (proven
twice during Volumes M2; `.superpowers/sdd/libovrtx-volume-deps-research.md`
has the full dependency research).

Evidence (`renders/`, produced by `launch.sh` itself): the same `torus.vdb`,
same camera, rendered headlessly at 1280×720 — NVIDIA's gray transfer
function vs a blue→green→yellow→red one:

| | gray TF | colored TF |
|---|---|---|
| non-black px | 351,387 | 335,298 |
| px with chroma > 0.15 | **0** | **322,046** |
| mean chroma in volume footprint | 0.0 | **0.60** |

Run it yourself: `./launch.sh` (serializes on `/tmp/omniversemakie-gpu.lock`,
~2 min). It generates both stage variants, launches Kit headless, captures
both frames, and chroma-checks them.

The launch is deliberately *from parts* — the bare `kit` kernel with only the
needed extensions enabled (dependencies auto-resolve), no full editor app:

```
kit --empty --no-window --ext-folder exts --ext-folder extscache \
    --enable omni.kit.mainwindow --enable omni.kit.viewport.window \
    --enable omni.kit.viewport.utility --enable omni.hydra.rtx \
    --enable omni.rtx.index_composite --enable omni.kit.exec.core \
    --/rtx/index/compositeEnabled=true \
    --/rtx/index/overrideSubdivisionMode="kd_tree" \
    --/rtx/index/overrideSubdivisionPartCount=1 \
    --exec probe.py
```

The full CAE editor app works identically (`./omni.cae.kit.sh --no-window …`,
first proof ran that way) but starts slower and its teardown stalls headless
(`probe.py` hard-exits as a fallback). Without the `overrideSubdivision*`
pair the volume still renders in color but with heavy banding.

## The two tiers

| | standalone ovrtx (what OmniverseMakie drives) | Kit composite runtime (this probe) |
|---|---|---|
| IndeX mode | Direct (auto-engaged inside RT2) | RTX compositing (IndeX as secondary renderer, depth-correct composite) |
| colors | **grayscale only** | **full transfer-function colors** |
| enable | carb token `/app/tokens/omni.index.libs` (`OMNIVERSEMAKIE_INDEX_LIBS`, automated in `src/binding/index_config.jl`) | extension `omni.rtx.index_composite` + settings below |
| runtime | `libovrtx-dynamic.so` via C FFI | a Kit application (here: the DSX blueprint's `omni.cae.kit`) |

## The correct set of libraries

Everything is NVIDIA-proprietary Kit extensions (ext-registry, **not** pip;
do not vendor). Dependency chain, all resolved in
`~/.local/share/ov/data/exts/v2/` by building/running any Kit app that
declares `omni.rtx.index_composite` (the DSX blueprint's kit-cae does, via
`omni.cae.index`):

```
omni.rtx.index_composite  (pure-Python marker: settings + deps only)
├── omni.hydra.rtx            (Kit's RTX Hydra delegate — has the compositor ovrtx lacks)
├── omni.index.renderer ─── omni.index ─── omni.index.libs   (the 13 .so IndeX payload,
├── omni.index.usd                                            libdice.so + libnvindex.so + …)
└── omni.index.settings.core
```

System libraries Kit needs beyond the ext set (found missing on a minimal
Linux install):

- **`libGLU.so.1` (+ GLVND `libOpenGL.so.0`/`libGLdispatch.so.0`)** — required
  by the MDL SDK (`libneuray.so` in `omni.iray.libs`), which the RTX scene
  renderer hard-requires. **Without it the failure is misleading:**
  `UsdManager::addHydraEngineWithConfiguration - Invalid sync scope created.
  Failed to add Hydra engine` (the "scope" is the *neuray database* scope, not
  a Vulkan sync scope). `launch.sh` extracts the packages locally — no root.
- `libxml2.so.2` — only for asset-*importer* exts; its startup errors are
  benign for rendering.

## The correct config

**App/carb settings** (the CAE app sets the first three; `launch.sh` passes
the rest):

```
rtx.index.compositeEnabled = 1          # master switch (also queryable live at /rtx/index/*)
rtx.index.overrideSubdivisionMode = "kd_tree"
rtx.index.overrideSubdivisionPartCount = 1
--/app/asyncRendering=false             # deterministic headless capture
--/omni.kit.plugin/syncUsdLoads=true
```

Live tuning knobs (NVIDIA's own render test drives these):
`/rtx/index/compositeDepthMode` (0–3), `/rtx/index/resolutionScale`.

**Stage authoring** (see `torus_colormap.usda.in`; this is the shape
NVIDIA's own golden-image test uses):

- root layer `customLayerData.renderSettings`:
  `bool "rtx:index:compositeEnabled" = 1`, `int "rtx:index:compositeDepthMode" = 3`
- on the `Volume` prim: `custom bool nvindex:composite = 1` and
  `custom bool omni:rtx:skip = 1` (IndeX draws it; RTX skips it)
- material: `token outputs:nvindex:volume.connect` → a `Shader` whose
  `inputs:colormap.connect` → a `Colormap` prim with
  `float4[] rgbaPoints` / `float[] xPoints` / `float2 domain` — **this is the
  transfer function, and in composite mode it is honored.**

**Capture gotcha:** `omni.kit.viewport.utility.capture_viewport_to_file`
must be awaited — keep the returned object and
`await cap.wait_for_result(completion_frames=60)`; fire-and-forget writes
nothing (`probe.py` does it right).

## How close standalone ovrtx is (in-process probe, 2026-07-15)

Follow-up binary + runtime investigation of whether the same composite path
can run **inside Julia-hosted ovrtx** (no Kit process). Result: *almost* —
every component but one is present and responsive:

- ovrtx's `libcarb.scenerenderer-rtx.plugin.so` carries the **full**
  `/rtx/index/*` composite settings family and the composite machinery
  (`rtx::index::IndexCompositeRendererContext`, composite-before-AA passes).
  An earlier claim that these strings exist in zero ovrtx binaries was wrong.
- ovrtx bundles `carb.scenerenderer-index.plugin` (the IndeX scene renderer,
  self-contained, colormap-capable) and its dir is in the engine's plugin
  search paths; `rtx.indexlib` (IndexInstance) loads.
- Setting `/rtx/index/compositeEnabled=true` via the carb config **is
  honored as a composite request**: with `/nvindex/compositeRenderingAvailable`
  deliberately absent, ovrtx logs the same diagnostic Kit would
  ("NVIDIA IndeX Compositing was requested but … is not set").
- **The one missing link:** nothing in ovrtx reads the per-prim marker
  `nvindex:composite` (or the layer `rtx:index:*` renderSettings). In Kit
  that reader is `omni.index.usd`'s `libomni.index.usd.plugin.so` — the
  "Use IndeX compositing" property toggle. It cannot be transplanted: it
  links Kit's split USD (`libusd_*.so`) + `libpython3.12`, while ovrtx embeds
  a monolithic `libov_25.11usd_ms.so` (different pxr inline namespace).
- Net behavior in-process: composite requested + volume skipped → black
  (empty composite); volume not skipped → IndeX **Direct** grayscale as
  always. Also tried and refuted: `/renderer/enabled = "rtx,index"` (no
  second engine), settings-only volume marking (no reader exists).

So standalone full-color volumes are **one NVIDIA-shipped component away**
(an ov-USD-built `nvindex:composite` reader, or a carb-settings fallback to
mark all volumes) — worth an upstream request against ovrtx; until then,
colors require this Kit runtime.

## What this means for OmniverseMakie

`volume!` colors stay a documented grayscale degrade on the standalone-ovrtx
backend (the `COLORED=false` tripwire in the volume tests flips loudly if an
ovrtx build ever changes this; PyPI still ships only `0.3.0.312915`,
re-checked 2026-07-15). Getting Makie volume colormaps to actually show
means rendering through a Kit runtime like this probe's — a deployment
change (Kit hosts the render loop; the current C FFI drives libovrtx
directly), not a library fetch into ovrtx.

## Files

- `launch.sh` — end-to-end: discovers paths, generates stages, launches Kit
  headless under the GPU lock, chroma-verifies. `KIT_RELEASE_DIR` env
  overrides the Kit build (default: DSX kit-cae).
- `probe.py` — Kit-side `--exec` script: opens each stage, converges,
  captures the viewport.
- `torus_colormap.usda.in` — colored-TF stage template (`@VDB_PATH@`
  placeholder; the gray variant is generated by `launch.sh`).
- `analyze.jl` — chroma oracle (gray frame ≈ chroma-free, colored frame
  strongly chromatic, colormap swap must move the volume's pixels).
- `renders/` — the two captured frames backing the numbers above.
