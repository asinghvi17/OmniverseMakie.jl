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

### Phase 2 (specced, not built): `libkitjl` C shim, in-process

C ABI over Carbonite (headers ship in `kit/dev/include` + `kit/dev/fabric`):
`IApp` lifecycle pumped from a Julia task, `ISettings`, Fabric
`IStageReaderWriter` for CUDA-pointer geometry, `IAppScripting` escape hatch
for stage/capture ops until native AOV readback lands. Replaces the
subprocess; same `KitScreen` surface. Known hazards + mitigations: breakpad
vs Julia GC signals (ovrtx fix pattern applies), no coexistence with
in-process ovrtx (one backend per session).

### Phase 3 (later): GPU data planes

In: Fabric GPU attributes (in-process) / CUDA IPC (subprocess). Out:
`omni.syntheticdata` CUDA annotators → `CuArray` → existing CUDA-GL blit.
Volume payloads stay file-based (`.nvdb`) — an IndeX data-import SDK
integration is a separate project.
