# LiDAR + radar sensor simulation (`lidar!` / `radar!`) — design

**Date:** 2026-07-05 · **Status:** design approved in conversation; Phase 1 opens with a
de-risking spike (spike-dependent claims are marked ⚠ SPIKE below)
**Depends on:** usdplot childless-recipe pattern (9d881ad), open-stage diff architecture
(M2), accumulate-across-frames (09ae8c2), M6.B multi-tensor readback pattern

## Goal

Expose ovrtx's native RTX sensor simulation — LiDAR and radar point clouds traced against
the same USD stage the camera renders — as first-class Makie objects:

```julia
fig = Figure()
ls  = LScene(fig[1, 1])
meshscatter!(ls, positions)                       # ordinary scene content

sensor = lidar!(ls; frame_rate = 10.0,
                channels = [:coordinates, :intensity])
translate!(sensor, 0, 0, 1.5)                     # sensor pose = plot transform
rotate!(sensor, ...)                              # +X is the sensor's forward axis

OmniverseMakie.activate!(sensors = true)          # motion BVH on (creation-frozen)
screen = display(fig)

# live point-cloud panel next to the RTX render:
pts = lift(r -> r.points, sensor.returns)
scatter(fig[1, 2], pts; color = lift(r -> r.intensity, sensor.returns))

for t in timeline
    # ... move scene objects ...
    step_sensors!(screen, 1/10)                   # one scan; updates sensor.returns
end
```

Works offscreen (`colorbuffer`/`save`/`record`), in `interactive_display`, and inside a
`replace_scene!` hybrid panel. In a plain GLMakie window the sensor draws nothing
(no-op `draw_atomic`, the USDPlot precedent).

## Recon evidence (2026-07-05, verified against the local SDK checkout)

- **Sensor sim is standalone-ovrtx's primary use case** (SDK AGENTS.md: "OpenUSD-based
  sensor simulation and rendering workflows (camera, lidar, radar)"). NOT a Kit-only
  capability — unlike volume colors and the `"sky"` background.
- The bundle we already link ships the full **omni_sensors USD schema plugin**
  (`bin/usd_plugins/omni_sensors`: `OmniLidar`, `OmniRadar`, `OmniAcoustic` prim types +
  `OmniSensorGenericLidarCoreAPI`, `OmniSensorGenericRadarWpmDmatAPI`, …) and the
  `sensors-gmo` / `sensors-checker` libs.
- The **C API we already ccall is the sensor API**: `ovrtx_step` takes an N-product
  `ovrtx_render_product_set_t` (ours currently hardcodes 1); sensor products yield 0–n
  frames per step; readback is the same `ovrtx_fetch_results` →
  `ovrtx_map_render_var_output` named-tensor flow the M6.B pick path implements.
- NVIDIA's tested examples (`examples/{python,c}/sensors/{lidar,radar}`) establish the
  scene pattern: `OmniLidar` prim (`omni:sensor:Core:elementsCoordsType`,
  `omni:sensor:frameRate = (10, 1)`, xform) + `RenderProduct` whose `camera` rel targets
  the *sensor* + `PointCloud` RenderVar with `string[] channels`; step at
  `dt = 1/frameRate`; ~3 warmup steps before the first read; `Coordinates` maps as a
  `[3, N]` Float32 tensor sliced to `Counts[0]`; copy before unmap.
- **Motion BVH is required by the sensor pipeline** (example comment; `ovrtx_config.h`:
  "required … non-visual sensor render products (lidar, radar, acoustic)"). It is a
  renderer-**creation** config entry (`ovrtx_config_entry_enable_motion_bvh`); changing it
  requires recreating the renderer. Our `OV.Renderer` already passes config entries
  (REPL-verified 24-byte opaque structs) — this is one more entry.
- **History-discard rule** (`ovrtx.h` step docs): accumulated sensor rendering history for
  every render product NOT in a step's product set is discarded. Camera image-quality
  steps and sensor time-steps therefore need deliberate set composition (§ Stepping).
- Example materials carry `omni:simready:nonvisual:base = "asphalt"` etc. — nonvisual
  tokens that drive lidar intensity / radar RCS physics. ⚠ SPIKE: whether sensors return
  hits at all on materials WITHOUT these tokens (expected: yes, with default response).

## Design

### Layer 1 — OV foundation

- **`OV.step!(r, products::Vector{<:AbstractString}; dt, timeout_ns)`** — generalize the
  existing single-product method (same enqueue_wait/StepResult/finalizer machinery;
  N-element set, all `ovx_string_t` GC-preserved). The 1-product method delegates.
- **`OV.read_pointcloud(sr, product) -> Vector{<:NamedTuple}`** — fetch results, walk
  outputs **scoped to `product`** (today's `_find_var` scans all products of the set —
  correct only while sets have one product; sensors force product-scoped lookup), find the
  `PointCloud` var per frame, map CPU, copy each named channel tensor sliced to
  `Counts[0]`, unmap in `finally`. One NamedTuple per output frame (instant mode: 1;
  partial-scan mode: 0–n).
- **Motion BVH**: `OV.Renderer(...; enable_motion_bvh::Bool = false)` appends the config
  entry. Plumbed from a new `ScreenConfig.sensors::Bool = false` (off by default — BVH
  build is a cost every scene would otherwise pay).

### Layer 2 — the recipes

```julia
@recipe Lidar () begin
    "Scan rate in Hz → `omni:sensor:frameRate`; also the natural `step_sensors!` dt."
    frame_rate = 10.0
    "Requested PointCloud channels (Counts/Flags always auto-included by the model)."
    channels = [:coordinates, :intensity]
    "One full scan per sensor step (`instantLidar`). `false` = time-resolved partial scans (advanced)."
    instant = true
    "Output frame of reference: `:sensor` (default, NVIDIA-recommended) or `:world`."
    output_frame = :sensor
    "Raw `omni:sensor:*` schema attributes for tuned sensor models (`Dict{String,Any}`)."
    attributes = Dict{String,Any}()
    "OUTPUT (backend-written): NamedTuple of validity-sliced channel arrays + sensor pose."
    returns = (points = Point3f[], counts = 0)
    Makie.mixin_generic_plot_attributes()...
end
Makie.plot!(p::Lidar) = p          # childless → atomic (USDPlot pattern)
```

`radar!` is the parallel recipe (`Radar`): prim `OmniRadar` +
`OmniSensorGenericRadarWpmDmatAPI`, attr prefix `omni:sensor:WpmDmat:`, default
`channels = [:coordinates, :rcs, :radial_velocity]`. ⚠ SPIKE (Phase 2): the radar
equivalent of `instantLidar`.

**Authoring** (`author_sensor_prim!`, dispatched from the existing childless-plot walker):
1. The sensor prim under the scene's USD Scope like any plot prim (`plot_<oid>`), typed
   `OmniLidar`/`OmniRadar` with the API schema prepended, output-defining attributes from
   kwargs (these are **author-time-only** per NVIDIA docs — live mutation of e.g.
   `channels` throws).
2. A `RenderProduct` + `PointCloud` `RenderVar` mirroring the existing camera-product
   authoring path (per-screen `/Render` scope), `camera` rel → the sensor prim.
3. `consumed_inputs(::Lidar) = [:model_f32c, :visible]` — the standard `:model_f32c` push
   writes `omni:xform`, so `translate!`/`rotate!` move the sensor through the existing
   diff pipeline (non-structural → no accumulation reset in accumulate mode). Sensor
   forward = **+X** (ovrtx convention), documented on the recipe. Our stages are Z-up
   (matches NVIDIA's Z-up lidar example directly). `register_model_f32c!` in `plot!`
   (childless recipes don't auto-emit it — the usdplot lesson).
4. `data_limits`: a small fixed box at the origin (sensors shouldn't inflate axis limits).

⚠ SPIKE: whether a `RenderProduct` authored **live** (sensor added after `display`) is
picked up by subsequent steps. If not: sensors added post-display throw a clear
"author-before-display" `ArgumentError` in v1, and the limitation is documented (the
re-author path — new Screen — always works).

**Channel map** (Julia kwarg → USD channel → returned field/type; `ArgumentError` naming
valid channels on anything unknown, and on radar-only channels passed to `lidar!`):

| kwarg | USD channel | returns field | eltype |
|---|---|---|---|
| `:coordinates` | `Coordinates` | `points` | `Vector{Point3f}` (from `[3,N]`, validity-sliced) |
| `:intensity` | `Intensity` | `intensity` | `Vector{Float32}` |
| `:time_offset_ns` | `TimeOffsetNs` | `time_offset_ns` | `Vector` (tensor dtype) |
| `:flags` | `Flags` | `flags` | `Vector` (tensor dtype) |
| `:rcs` (radar) | `RCS` | `rcs` | `Vector{Float32}` |
| `:radial_velocity` (radar) | `RadialVelocityMs` | `radial_velocity` | `Vector{Float32}` |
| — always | `Counts` | `counts` | `Int` (`Counts[0]`) |

`returns` always additionally carries `pose::Mat4f` (the sensor's world transform at the
step) so `:sensor`-frame points can be placed in world/data space by the consumer.

### Layer 3 — stepping & time

Sensor time is **simulation time and only the user's loop knows it** — so advancing it is
explicit:

- **`step_sensors!(screen_or_session, dt) -> nothing`** (results land in each sensor's
  `returns` observable) — one `OV.step!` with the set
  `{camera product, all sensor products}` at physical `dt`,
  then `read_pointcloud` per sensor and update each `sensor.returns[]` via **assignment**
  (`plot.returns[] = nt`, never `Base.notify` — the frozen-texture lesson). The camera
  product is included so its accumulation history survives the history-discard rule
  (⚠ SPIKE: confirm a sensors-only set really would discard it; if not, drop the camera
  from the set and save the extra image step).
- **Image-quality steps are unchanged**: warmup/accumulate loops step `{camera}` only.
  Sensor scan history in `instant` mode doesn't depend on cross-step history, so the
  image steps' history-discard of sensor products is harmless (and is WHY
  `instant = true` is the default).
- ~~**Warmup**~~ — DROPPED (SPIKE RESULTS #3: the first step already returns a full scan;
  no warmup bookkeeping exists in the implementation).
- **Recording sugar**: `record_frame!(session; sensor_dt = nothing)` — when set, calls
  `step_sensors!` before the presenting tick. Offscreen `Makie.record` users call
  `step_sensors!(screen, dt)` inside their frame function (one line, documented in the
  README recipe).
- `instant = false` (partial scans) is authored and readable — `read_pointcloud` already
  returns per-frame NamedTuples and `step_sensors!` concatenates them — but its
  interaction with image-step history discard makes it an **advanced, lightly-tested
  mode** in v1; refinement is a follow-up milestone if a real use case appears.

### Error handling

- `lidar!`/`radar!` on a screen whose renderer lacks motion BVH → a one-time `@warn` at
  author time (NOT a throw — SPIKE RESULTS: static scenes render correctly without BVH)
  telling the user to pass `sensors = true` to `activate!`/`Screen`, plus constructor
  auto-enable when sensors are already in the scene.
- Unknown channels / radar channels on lidar / live writes to output-defining attrs →
  `ArgumentError` naming the valid set.
- A step whose sensor product yields zero frames + zero counts is a valid empty scan
  (empty arrays, `counts = 0`), NOT an error — but `step_sensors!` throws if the
  `PointCloud` var is entirely absent for a registered sensor (authoring bug; the
  silent-ignore hazard rule: evidence or error).
- `read_pointcloud` checks `ovrtx_render_var_output_t.status` (`_check_var_output`) and
  unmaps in `finally`, exactly like `map_cpu`.

## Files

- `src/binding/OV.jl`: multi-product `step!`; `read_pointcloud` + product-scoped var walk;
  `enable_motion_bvh` config entry on `Renderer`.
- `src/sensors.jl` (NEW): `Lidar`/`Radar` recipes, channel map + validation,
  `author_sensor_prim!`, `step_sensors!`, warmup bookkeeping.
- `src/settings.jl`: `ScreenConfig.sensors::Bool = false` (+ docstring: motion BVH,
  creation-frozen).
- `src/screen.jl`: sensor registry on `Screen` (plot ↔ product path); author-walker hookup;
  plumb `sensors` → `OV.Renderer`.
- `ext/OmniverseMakieGLMakieExt.jl`: no-op `draw_atomic` for `Lidar`/`Radar`;
  `record_frame!` `sensor_dt` kwarg; `step_sensors!(::EmbeddedSession)`.
- `src/OmniverseMakie.jl`: include; export `lidar`, `lidar!`, `radar`, `radar!`,
  `step_sensors!`.
- `README.md`: "Sensor simulation" section (pose convention, stepping contract, recording
  recipe, nonvisual-materials note).
- `test/sensors/` (NEW dir) + runtests wiring; goldens in `authoring/`.

## Testing

- **Spike (first task, one GPU program, ≤3 renderer creations):** author NVIDIA's minimal
  lidar scene shape through OUR emitters (ground plane + cube + OmniLidar + product/var),
  `enable_motion_bvh`, step at dt=0.1, assert `Counts[0] > 0` and that the cube produces
  returns at the expected range band (the tensor-evidence oracle). Probes: (a) returns
  without nonvisual material tokens; (b) camera-accumulation discard on sensors-only
  steps; (c) live-authored RenderProduct pickup; (d) `instantLidar = true` behavior +
  warmup need; (e) dt semantics across repeated steps (scan advances). Spike findings
  amend this spec before implementation proceeds.
- **Pure:** channel-map validation (unknown / radar-on-lidar rejections), sensor USDA
  emission goldens (prim + API schema + frameRate rational + product/var wiring;
  `authoring/` conventions), `ScreenConfig` field contract, `data_limits`.
- **Subprocess GPU** (`test/sensors/`, `retries=4`, early `ready_marker`, flock): lidar
  end-to-end — known cube at known range → `counts > 0`, range-band assert on `points`,
  `intensity` present iff requested; pose move (`translate!` + `step_sensors!`) shifts
  the point centroid; `returns` observable fires (listener count + value change);
  `delete!` removes prim+product and subsequent steps succeed; motion-BVH-off author →
  throws; accumulate mode: `step_sensors!` triggers no image reset
  (`OV._RESET_OBSERVER`). Radar phase adds: `rcs`/`radial_velocity` channels present, a
  moving target yields nonzero radial velocity (sign: approaching < 0).
- **Acceptance demo:** Zeus ZS300 (`usdplot`) with a roof-mounted `lidar!`, spun wheels,
  `record` of the RTX panel + live `scatter!` point-cloud panel side by side → `.mp4`
  for user review.

## Phasing

1. **Phase 1 — lidar end-to-end:** spike → OV foundation → `Lidar` recipe + authoring →
   `step_sensors!` + returns observable → GLMakie ext bits → tests + README.
2. **Phase 2 — radar:** `Radar` recipe on the same machinery; radar-channel readback;
   radar GPU tests; radar instant-mode spike.
3. **Phase 3 (optional) — nonvisual materials:** `material = (; nonvisual_base =
   "asphalt", …)` tokens on our material emitter for physically meaningful
   intensity/RCS; goldens + a measurable intensity-contrast GPU assert.

## Non-goals (v1)

- Acoustic sensors (`OmniAcoustic`) and Sensor Processing Graphs (experimental config).
- GPU-mapped (CUDA) point-cloud readback — CPU map first; the map flag is a follow-up.
- Custom emitter-state firing patterns (`…EmitterStateAPI`) beyond the `attributes`
  passthrough; GMO (`GenericModelOutput`) output.
- `ObjectId`/`MaterialId` → Makie-plot resolution (natural follow-up tying into the
  M6.B `path2plot` machinery).
- Refined partial-scan (`instant = false`) semantics; per-frame hot-path binding of
  sensor pose (the coalesced xform write is user-rate already).
- Mounting sugar (`follow!(sensor, plot)`) — parenting/updating transforms covers it.

## SPIKE RESULTS (2026-07-05, A5000, 3 renderer creations — all 5 questions answered)

1. **Live RenderProduct authoring WORKS** (a live-referenced layer carrying OmniLidar +
   RenderProduct + RenderVar produced full scans) — no author-before-display restriction.
   Even relative rel targets composed; we still author layer-absolute per repo convention.
2. **Camera accumulation IS discarded by sensors-only steps** (next camera frame 8× noisier:
   masked diff 0.0602 vs 0.0076 control) — the camera product goes in every sensor step set,
   as designed.
3. **NO warmup needed** — the FIRST step returns a full 203k-point scan in BOTH default and
   instant modes (NVIDIA's 3 warmup steps are defensive).  The warmup bookkeeping is DROPPED
   from the design.  Default (non-instant) mode at `dt=1/60` yields clean proportional
   partial scans (~203k/6 points per step, one frame per step); `instantLidar=true` yields a
   FULL scan per step at any dt — confirming `instant = true` as the default.
4. **Returns work with ZERO material bindings** (bare displayColor meshes) — nonvisual
   tokens are a fidelity upgrade (Phase 3), not a requirement.
5. **`Coordinates` = `[3,N]` row-major Float32** → lands as a Julia `(N,3)` `Matrix{Float32}`
   via the reversed-dims copy, exactly as `OV.read_pointcloud` implements.  Delivered
   channels = requested + auto `Flags`/`Counts`.  Output is SENSOR-frame by default
   (ground plane at z=−1 with the sensor 1 m up).  The ~15.4 m ground-return radius is the
   default elevation fan hitting the ground, NOT a range cap (`farRangeM` defaults to 200).

6. **★★ SENSOR STAGES MUST BE METER STAGES** (found during test bring-up, probe chain
   2026-07-05): ovrtx's sensor engine works in PHYSICAL METERS — range attributes
   (`nearRangeM = 0.3`, `farRangeM = 200`), the ~±4° elevation fan, and the output
   `Coordinates` (meters!) are all absolute physics.  On the M1 centimeter root stage
   (`metersPerUnit = 0.01`) a typical Makie scene sits inside the 30-data-unit near-range
   dead zone → ZERO returns; scaling the range attributes does NOT compensate (probe: scaled
   near/far/minReflection/accuracy/resolution → still 0), while the SAME scene ×100 returns
   the byte-identical 203k-point spike signature.  FIX: a screen with sensors (the same
   `config.sensors || auto-detect` bit that enables motion BVH) authors the root stage at
   `metersPerUnit = 1` — 1 data unit = 1 meter, sensor physics and returns coincide with data
   units; sensor-free screens keep the cm stage byte-identically.  Post-display sensors on a
   cm screen keep working only as "off the happy path" (warn covers both BVH and scale).
   Along the way, ALSO fixed: sensors need NVIDIA's mount rotation `rotateXYZ(90,0,−90)`
   (= `Rz(-90°)·Rx(90°)` innermost; the model spins about its native +Y axis — without it the
   scan is a vertical fan) — baked into the written `omni:xform` while the reported `pose`
   stays the clean mount (model ∘ origin) map, since the model's output frame aligns with the
   mount frame.

**Design amendments from the spike + Makie mechanics:**

- **Motion BVH is NOT required for static scenes** (a no-BVH renderer returned full scans —
  it's required for correct MOTION effects, per the SDK skill's fine print).  So: the
  `Screen` constructor AUTO-ENABLES motion BVH when the displayed scene already contains
  sensor plots; `ScreenConfig.sensors = true` FORCES it on (for post-display `lidar!`
  workflows); authoring a sensor on a renderer without motion BVH is a one-time `@warn`
  (moving-object returns may be wrong), NOT a throw.
- Every `OV.reset!` (image-accumulation restart) is `ovrtx_reset(time)`, which also rewinds
  sensor simulation time — harmless under `instant = true` (scan-phase-free), one more
  reason it is the default; documented for `instant = false` users.
- Makie rejects zero-positional-arg recipes, so the recipes take `(origin,)` — the sensor
  position in data space (`lidar!(ax, Point3f(0,0,1.5))`), composed into the `omni:xform`
  via the `_usdplot_model` hook; a hand-added zero-arg method defaults it to `Point3f(0)`.
  Live origin changes are not diffed (use `translate!` for live moves) — documented.
- `returns` is a plain `Observable` behind an accessor `sensor_returns(plot)` (WeakKeyDict,
  the `_USD_BINDINGS` pattern) rather than a recipe attribute — plot attributes are
  ComputePipeline `Computed`s, and a plain Observable keeps user-side `lift`/`on` free of
  framework risk.
