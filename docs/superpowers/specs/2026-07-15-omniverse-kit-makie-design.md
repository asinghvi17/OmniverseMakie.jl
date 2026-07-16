# OmniverseKitMakie.jl — Kit-backend subpackage (design)

2026-07-15. Short-form spec (project practice for well-scoped work).

## Why

Standalone ovrtx renders volume transfer functions grayscale-only (IndeX
Direct); full colors need a Kit runtime with the `omni.rtx.index_composite`
chain — proven and documented in `examples/kit_index_composite/`. NVIDIA may
eventually ship the missing piece in ovrtx (the composite machinery is
already compiled in; only the `nvindex:composite` marker path is absent — see
the README's seam analysis). **OmniverseMakie's implementation must stay
untouched** so it can simply absorb that fix. Everything Kit-related lives in
a subpackage.

## Shape

`lib/OmniverseKitMakie.jl` — a workspace member (same pattern as
`lib/LibOVRTX`) that depends on OmniverseMakie via a path source and reuses
its internals (`_vdb_volume_usda`, `_volume_colorrange`, `usda_light`,
`NanoVDBWriter`). Nothing in OmniverseMakie references it.

### v1 (this iteration): volume colors behind the normal interface, RPC transport

- **Transport** (migrated from the passing spike, was `src/kit/`):
  `KitServer` = persistent headless `kit --empty` subprocess (six-extension
  composite launch, libGLU shim, GPU flock + timeout) driven by line-JSON
  RPC over a FIFO; ops `ping/open_stage/render/set_attr/quit`. `set_attr`
  gains a `usd_type="matrix4d"` branch for camera updates.
- **Authoring** (`stage_usda`): walk a Makie `Scene`; for each `Makie.Volume`
  plot write a volume payload (pluggable `volume_writer`; exactly the
  conversions `author_usd_prim!(::Volume)` does) + a `_vdb_volume_usda`
  fragment file, referenced into the stage with `nvindex:composite=1` +
  `omni:rtx:skip=1` (+ `xformOp:transform` when the plot model is
  non-identity). Lights via `lights_usda`; camera prim from `inv(view)` (GL
  camera convention == USD camera convention; row-vector layout like
  `_model_to_usd_xform`), focal length from the controls' vertical fov. Root
  `customLayerData`: `rtx:index:compositeEnabled=1`, `compositeDepthMode=3`,
  `boundCamera`.
- **Volume payload format (discovered during v1):** Kit's IndeX composite
  importer (`OpenVDB_importer_NanoVDB`) **fails to fetch NanoVDB data** —
  from `NanoVDBWriter` output (codec NONE *and* a ZIP-recoded variant) *and*
  from Warp's own sample `.nvdb`s — while classic OpenVDB `.vdb` (the torus
  sample) renders. Standalone ovrtx reads our `.nvdb` fine because IndeX
  Direct ingests through ovrtx's `rtx.scenedb`, a different reader. So a
  live `KitScreen` writes payloads as **`.vdb` server-side**: a `write_vdb`
  RPC ships the dense Float32 array (raw column-major file) and the server
  converts via the `omni.volume` extension's bundled pyopenvdb bindings
  (`--enable omni.volume` added to the launch). The pure-Julia `.nvdb`
  writer remains the default for offline/text-level authoring.
- **Screen**: `KitScreen(scene; ...)` authors + opens the stage;
  `Makie.colorbuffer(screen)` re-syncs the camera (matrix4d `set_attr`) and
  renders. Non-volume atomic plots are skipped with a one-time warning —
  meshes etc. are the standalone backend's job (their usda-text emission
  doesn't exist and is out of scope for v1).
- **Scope guards**: one stage per screen; live volume-data updates deferred
  (fresh-`.nvdb` + `filePath` set_attr is the known shape); no
  `Base.display` integration — explicit `colorbuffer` only.

### Tests

- Pure: JSON codec round-trip; `stage_usda` contains composite layer keys,
  per-prim markers, colormap `rgbaPoints` (viridis vs `:grays` differ),
  `boundCamera`, a parseable camera matrix.
- GPU (skipped unless a Kit release dir exists): one server, A/B —
  `volume!(...; colormap=:viridis)` renders with high chroma;
  `colormap=Makie.to_colormap([:black, :white])` twin ≈ zero chroma.
  Server serializes on the shared GPU lock itself.

### Phase 2 (BUILT, ROOT-CAUSED AS IMPOSSIBLE, REMOVED): `libkitjl` C shim, in-process

**Status: the in-process transport was fully implemented (native shim + pure
tier green + transport abstraction, 2026-07-15), then root-caused as
architecturally impossible with Julia as the host and REMOVED (2026-07-16).**
Kit's `IApp::startup` hangs in `omni.usd_resolver`'s static initializer
because **`OMNI_APP_GLOBALS` (the omni-core client context) must live in the
process main executable** — Kit resolves it via the main-program handle,
which never searches a dlopened/preloaded library, and `julia`'s executable
cannot carry it. Proven by controlled bisection: a Julia-free C harness hangs
identically (not Julia); a C++ main-executable with identical init renders
(not the init sequence); `LD_PRELOAD` doesn't help (not symbol-scope order).
Full design, evidence, and the decisive experiments are preserved in
[`docs/superpowers/specs/2026-07-15-libkitjl-design.md`](2026-07-15-libkitjl-design.md)
— kept as the record so this is never re-attempted on the same axis. The only
in-process route is inverting the host (a Kit-globals executable embedding
`libjulia`) — out of scope. **The subprocess transport is the design.**

### Phase 3: GPU data planes — see `2026-07-16-kit-gpu-data-plane-design.md`

Implemented as its own spec: frames out (`:cuda` via syntheticdata
LdrColorSDPtr + CUDA IPC; `:cpu` via shared-memory capture; `:png` floor)
and live volumes in (`gpu_update_volume!` — IPC staging + fresh-`.vdb`
server-side write). Volume payloads stay file-based (IndeX importer) — an
IndeX data-import SDK integration remains a separate project.
