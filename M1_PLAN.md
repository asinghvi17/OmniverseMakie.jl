# Milestone M1 — Static `Scene → USD → image` (bite-sized plan)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes. This expands IMPLEMENTATION_PLAN.md's task-level M1 into bite-sized TDD steps, grounded in the completed M0 `OV` API and the ovrtx example stage.

**Goal:** `colorbuffer(scene)` / `save("out.png", fig)` render a real Makie `Scene` (mesh + scatter + lines + surface, with camera + lights) through ovrtx and produce a correct PNG — build-once authoring, no live diffing yet (that's M2).

**Approach:** A `Screen <: Makie.MakieScreen` owns an `OV.Renderer`. On `display`/`colorbuffer` it authors an **OpenUSD stage the way ovrtx expects** (a render-config root + per-plot prim references), renders RT2, reads back `LdrColor`, and returns the image. M0's `OV` layer is the proven substrate; M1 adds the Makie contract + the USD translation.

**Guiding principle (per user steer): author USD the ovrtx-native way.** RPRMakie is the reference ONLY for the Makie-facing contract (Screen methods, which plot attributes to read, `activate!`/re-export). The translation TARGET is USD — use USD schemas (`UsdGeomMesh`, `UsdGeomCamera`, `UsdLux*`), the render-config root structure from `references/ovrtx/examples/c/minimal/torus-plane.usda`, and `omni:`/`xformOp` conventions. Do **not** port RPR-specific object building or RPRMakie's framebuffer flip.

---

## Global Constraints (M1)

Inherit all M0 Global Constraints (Pkg.jl-managed deps, generated bindings verbatim, GC.@preserve on hot paths, carb signal guard, subprocess-isolated renderer tests, `Pkg.test()` runner). Plus M1-specific:

- **ovrtx/USD-native authoring.** The skeleton root stage mirrors `torus-plane.usda`: a `RenderProduct` (with the `OmniRtxSettings*` apiSchemas), `RenderSettings`, and a `RenderVar "LdrColor"`; the product path passed to `OV.step!` is the authored RenderProduct prim path. Per-plot geometry is added as USD references via `ovrtx_add_usd_reference_from_string` (M0 research confirmed incremental add/remove works without re-opening the stage).
- **`colorbuffer` returns the `Matrix{RGBA{N0f8}}` from `OV.render_to_matrix` directly — keep alpha, NO y-flip, no conversion.** Makie accepts `RGBA` (in `N0f8`/`Float32`/`Float64`), so there's nothing to drop or convert. ovrtx `LdrColor` is top-left origin and M0's readback already yields JuliaNative `(H,W)`. (If a render ever comes out upside-down, fix orientation here — but the M0 evidence says top-left origin.)
- **Coordinate system:** author the stage with USD metadata `upAxis = "Z"` so Makie's Z-up data maps directly (validate in M1.2's test; if ovrtx ignores `upAxis=Z`, fall back to a root `xformOp:transform` converting Z-up→Y-up). The model transform is authored as `matrix4d xformOp:transform` (USD is row-major / row-vector, so transpose Makie's column-major `model`).
- **Build-once / static:** no ComputePipeline diff node yet. `Base.empty!(::Screen)` and re-render author from scratch. `insert!` is the build-once recursion (`isempty(plot.plots)` ⇒ atomic).
- **Deps added this milestone (via Pkg, pinned):** `Makie = "=0.24.12"`, `ComputePipeline = "=0.1.8"`, `GeometryBasics`, `Colors`, `ColorTypes` (already), `FixedPointNumbers` (already), `LinearAlgebra`. Registry deps — NO external `[sources]`.

**Carried-over hardening from the M0 opus review (do these in M1.1):**
- Wire `import LibOVRTX` (and `include` of OV) into the `OmniverseMakie` module so `OV`'s `using ..LibOVRTX` resolves when nested (today it resolves to `Main.LibOVRTX` only in tests).
- Factor a **shared test helper** (`test/helpers.jl`): the subprocess runner + path defaults (centralize the 5-file duplication) **with a `timedwait`+`kill` watchdog** so a wedged renderer can't hang the suite forever.
- `Screen` must close its `StepResult`s before its `Renderer` (deterministic GPU teardown).

---

## File structure (M1 adds)

```
src/
  OmniverseMakie.jl      # MODIFY: module — import LibOVRTX, include OV + screen + translation, activate!, re-export Makie
  screen.jl              # NEW: Screen <: MakieScreen, ScreenConfig, constructors, display/colorbuffer/insert!/size/close/backend_showable
  settings.jl            # NEW: RT2/PathTracing/Minimal mode, samples, bounces → omni:rtx settings strings
  translation/
    usd.jl               # NEW: skeleton-root authoring, OV.add_usd_reference!/remove_usd! wrappers, USDA string builders
    camera.jl            # NEW: author_camera! (UsdGeomCamera from scene.camera)
    lights.jl            # NEW: author_lights! (UsdLux from scene.compute lights)
    meshes.jl            # NEW: to_ovrtx_object(::Mesh) → UsdGeomMesh USDA
    primitives.jl        # NEW (M1.7): scatter/meshscatter/lines/surface → USD
    materials.jl         # NEW: displayColor primvar from plot color (MaterialX/OmniPBR deferred to M4)
test/
  helpers.jl             # NEW: shared subprocess runner + watchdog + path defaults
  m1_*_test.jl           # NEW per task
```

---

## Task M1.1 — Screen, ScreenConfig, activate!, module wiring, test harness

**Files:** `src/OmniverseMakie.jl` (rewrite stub), `src/screen.jl`, `src/settings.jl`, `test/helpers.jl`, `test/m1_screen_test.jl`. Deps: `Pkg.add(["Makie","ComputePipeline","GeometryBasics","Colors","LinearAlgebra"])` then `Pkg.compat` the exact pins.

**Interfaces — Produces:**
```julia
struct ScreenConfig
    mode::Symbol        # :rt2 (default) | :pathtracing | :minimal
    samples::Int        # offline SPP for :pathtracing (default 512)
    warmup::Int         # RT2 warmup frames (default 64)
    max_bounces::Int    # default 4
end
mutable struct Screen <: Makie.MakieScreen
    renderer::OV.Renderer
    fb_size::Tuple{Int,Int}
    product::String                 # authored RenderProduct prim path
    config::ScreenConfig
    scene::Union{Nothing,Makie.Scene}
    plot2usd::Dict{UInt64,UInt64}   # objectid(plot) => ovrtx_usd_handle_t  (M1.5 fills)
    open_results::Vector{OV.StepResult}  # closed before renderer (teardown order)
    setup::Bool                     # lazy-author flag
end
activate!(; screen_config...)       # set_screen_config! + set_active_backend!
Base.size(s::Screen) = s.fb_size
Base.isopen(s::Screen) = s.renderer.alive
Base.close(s::Screen)               # close open_results THEN renderer
```

- [ ] **Step 1 (failing test):** `test/m1_screen_test.jl` — subprocess (via the new `test/helpers.jl` runner) that `using OmniverseMakie; OmniverseMakie.activate!()`, builds a `Screen` for an empty `Scene` of size (800,600), asserts `size(screen)==(800,600)` and `Makie.current_backend()===OmniverseMakie`, then `close(screen)`; process exits 0 + prints "OK".
- [ ] **Step 2:** run → FAIL (no module API).
- [ ] **Step 3a — `test/helpers.jl`:** extract the proven M0 subprocess pattern into `run_ovrtx_subprocess(prog::String; timeout=300) -> (exitcode, output)`: write `prog` to a temp `.jl`, `run(setenv(\`julia --project=<root> file\`, OVRTX_LIBRARY_PATH=..., OM_USDA=..., PATH=..., HOME=...); wait=false)`, **`timedwait(() -> !process_running(p), timeout)` then `kill(p)` on timeout**, capture stdout, cleanup in `finally`. Centralize the default `.so`/USDA paths here. Refactor M0 tests to use it (optional now; required by M1 tests).
- [ ] **Step 3b — `src/settings.jl`:** `ScreenConfig` + `rtx_settings_usda(::ScreenConfig) -> String` emitting the `omni:rtx:rendermode`/`omni:rtx:rtpt:maxBounces` lines for the RenderSettings prim (values per `mode`/`max_bounces`).
- [ ] **Step 3c — `src/screen.jl`:** the `Screen` struct + `ScreenConfig` defaults + the **single core constructor** `Screen(scene::Scene, config::ScreenConfig)` → builds `OV.Renderer()`, stores `size(scene)`, sets `product = "/Render/OVMakie/RenderProduct"`, `setup=false`. Pass-through forms `Screen(scene, config, ::IO, ::MIME)` and `Screen(scene, config, ::Makie.ImageStorageFormat)` = `Screen(scene, config)` (RPRMakie pattern — backend-agnostic). `Base.size/isopen/close` (close closes `open_results` then the renderer). `activate!`.
- [ ] **Step 3d — `src/OmniverseMakie.jl`:** `module OmniverseMakie; using Makie, GeometryBasics, Colors, ColorTypes, FixedPointNumbers, LinearAlgebra; import LibOVRTX; include("binding/OV.jl"); include("settings.jl"); include("translation/usd.jl"); include("screen.jl"); … ; activate!() in __init__; re-export all Makie names via the reflection loop (RPRMakie.jl:68-73). end`. **This makes `OV`'s `using ..LibOVRTX` resolve to `OmniverseMakie.LibOVRTX`** (the carried-over M0 fix).
- [ ] **Step 4:** run → PASS (exit 0, backend active, Screen builds/closes).
- [ ] **Step 5:** commit `feat(M1.1): Screen + ScreenConfig + activate! + module wiring + test harness`.

**Acceptance:** backend activates as `Makie.current_backend()`; a `Screen` builds and tears down cleanly (results before renderer); subprocess exits 0.

---

## Task M1.2 — USD authoring: skeleton root + reference add/remove wrappers + USDA builders

**Files:** `src/translation/usd.jl`, `src/binding/OV.jl` (add the two wrappers). Test: `test/m1_usd_test.jl`.

**Interfaces — Produces:**
```julia
# OV.jl (over the raw ccalls — M0 research §3):
OV.add_usd_reference!(r::Renderer, usda::AbstractString, prim_path::AbstractString) -> L.ovrtx_usd_handle_t
OV.remove_usd!(r::Renderer, handle::L.ovrtx_usd_handle_t) -> nothing
# usd.jl:
author_render_root!(screen; resolution, camera_path="/World/Camera") -> Nothing
    # OV.open_usd_string! a root: stage upAxis="Z", /World (Xform), /Render/OVMakie/RenderProduct
    # (resolution, rel camera, orderedVars [LdrColor], the OmniRtxSettings* apiSchemas), RenderSettings, RenderVar "LdrColor".
usda_mesh(points, faces, normals, displaycolor; model) -> String   # UsdGeomMesh (M1.5 uses)
usda_matrix4d(model::Mat4) -> String                               # "( (r00,..), ... )" row-major transpose of Makie model
```

- [ ] **Step 1 (failing test):** subprocess — `r=OV.Renderer()`; `author_render_root!` a 256×256 root; add a hand-written cube `UsdGeomMesh` reference at `/World/cube` via `OV.add_usd_reference!`; `OV.render_to_matrix(r, "/Render/OVMakie/RenderProduct"; warmup=32)`; assert size `(256,256)` and **non-black** pixel count > 1000; then `OV.remove_usd!(handle)`, `OV.reset!(r)`, re-render, assert the cube's pixels are gone (non-black count drops sharply). Exit 0.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3a — OV wrappers** (mirror the M0 `_write_attribute!` GC discipline): `add_usd_reference!` builds two `ovx_string_t` (usda, prim_path) under `GC.@preserve`, a `Ref{L.ovrtx_usd_handle_t}(0)`, calls `enqueue_wait(r, L.ovrtx_add_usd_reference_from_string(r.ptr, layer_ovx, path_ovx, handle_ref), "add_usd")`, returns `handle_ref[]`. `remove_usd!` = `enqueue_wait(r, L.ovrtx_remove_usd(r.ptr, handle), "remove_usd")`.
- [ ] **Step 3b — skeleton root** authored as a Julia triple-quoted USDA string mirroring `torus-plane.usda` lines 163-237 (RenderProduct with the `OmniRtxSettings*` apiSchemas + `int2 resolution` + `rel camera` + `rel orderedVars`; RenderSettings with `rtx_settings_usda(config)`; `RenderVar "LdrColor"`), plus `(upAxis = "Z")` stage metadata and a `/World` Xform. **This step validates the riskiest M1 assumption** — that a from-scratch authored stage renders; iterate the USDA against the example until the cube renders.
- [ ] **Step 3c — `usda_matrix4d`** (transpose Makie `model` to USD row-vector matrix4d) and a minimal `usda_mesh` for the test cube.
- [ ] **Step 4:** run → PASS (cube renders non-black; remove makes it disappear).
- [ ] **Step 5:** commit `feat(M1.2): USD render-root authoring + reference add/remove wrappers`.

**Acceptance:** an authored-from-scratch stage renders; per-plot reference add/remove round-trips through a render. Confirms `upAxis="Z"` (or records the fallback).

---

## Task M1.3 — Camera translation (`UsdGeomCamera`, ovrtx attributes)

**Files:** `src/translation/camera.jl`. Test: `test/m1_camera_test.jl`.

**Interfaces — Produces:** `author_camera!(screen, scene) -> Nothing` — emit/replace a `UsdGeomCamera` at `/World/Camera` from `scene.camera` using the **USD camera attributes ovrtx uses** (`projection="perspective"`, `focalLength`, `horizontalAperture`, `verticalAperture`, `clippingRange`, + `matrix4d xformOp:transform` look-at). Reads `scene.camera.eyeposition[]`, `.view_direction[]`/lookat, `.upvector[]`, `.projection[]`→fov (`2*atan(1/proj[2,2])`), `.resolution[]`.

- **Key code:** map Makie eye/lookat/up → a camera-to-world matrix (the `xformOp:transform`); `focalLength = horizontalAperture / (2*tan(fov_h/2))` (or the vertical-FOV form `res[2]/(2*tand(fov/2))` from RPRMakie camera, using `horizontalAperture=20.955` like the example and deriving focal length). `clippingRange` from working distance (`norm(eye-lookat)*[0.001, 10]`).
- [ ] **Steps:** failing test → render the M1.2 cube from camera A; re-author camera at pose B (`author_camera!` + `OV.reset!`); re-render; assert the framed content differs substantially (≥ threshold changed px). Subprocess, exit 0. → implement → pass → commit `feat(M1.3): UsdGeomCamera translation`.

**Acceptance:** camera pose drives the rendered viewpoint.

---

## Task M1.4 — Lights translation (`UsdLux`)

**Files:** `src/translation/lights.jl`. Test: `test/m1_lights_test.jl`.

**Interfaces — Produces:** `author_lights!(screen, scene) -> Nothing` from `scene.compute[:lights][]` + `scene.compute[:ambient_color][]`. Mapping to **UsdLux** (ARCHITECTURE §5.1, ovrtx-native — the example stage uses `DomeLight` + `DistantLight`):
`PointLight→UsdLuxSphereLight` (small radius), `DirectionalLight→UsdLuxDistantLight`, `RectLight→UsdLuxRectLight`, `EnvironmentLight→UsdLuxDomeLight`, `AmbientLight→` low-intensity `UsdLuxDomeLight`. Each: `inputs:intensity`, `inputs:color`, + `xformOp:transform`.

- [ ] **Steps:** failing test → render the cube with a single `DistantLight` vs none; assert mean luminance increases with the light. → implement one `usda_light(::T)` per type → pass → commit `feat(M1.4): UsdLux lights translation`.

**Acceptance:** lights affect the image.

---

## Task M1.5 — `to_ovrtx_object(::Mesh)` + displayColor material + `display`/`colorbuffer`/`insert!`

**Files:** `src/translation/meshes.jl`, `src/translation/materials.jl`, `src/screen.jl` (display/colorbuffer/insert!). Test: `test/m1_mesh_render_test.jl`.

**Interfaces — Produces:**
```julia
to_ovrtx_object(screen, scene, plot::Makie.Mesh) -> UInt64   # author UsdGeomMesh ref, return usd handle; record plot2usd
Base.display(screen::Screen, scene::Makie.Scene)             # author root+camera+lights+plots; setup=true
Base.insert!(screen::Screen, scene::Scene, plot::Makie.Plot) # build-once recursion (isempty(plot.plots)⇒atomic); idempotent via plot2usd
Makie.colorbuffer(screen::Screen, fmt=Makie.JuliaNative) -> Matrix{RGBA{N0f8}}
```
- **Key code:**
  - `to_ovrtx_object(::Mesh)`: read `plot[1][]` (GeometryBasics mesh) → `points`, `faceVertexIndices` (triangulate), `faceVertexCounts`, `normals`; `model = plot.model[]`; color → `materials.jl`. Build `usda_mesh(...)` and `OV.add_usd_reference!(screen.renderer, usda, "/World/plot_<id>")`; store handle in `plot2usd[objectid(plot)]`.
  - `materials.jl` `displaycolor_for(plot)`: scalar `Colorant`/Symbol → uniform `primvars:displayColor = [(r,g,b)]`; per-vertex color vector → `primvars:displayColor` with `interpolation="vertex"`; colormap+values → map to per-vertex RGB; `plot.material=` escape hatch reserved for M4 (MaterialX/OmniPBR). (USD-native displayColor — NOT RPR materials.)
  - `display`: `author_render_root!` → `author_camera!` → `author_lights!` → recurse `insert!` over `scene.plots`+children; `setup=true`.
  - `colorbuffer`: if `!setup` call `display(screen, screen.scene)`; render via `OV.render_to_matrix(screen.renderer, screen.product; warmup=config.warmup)` (retaining any open `StepResult` in `open_results` for ordered teardown) and **return its `Matrix{RGBA{N0f8}}` directly** — no alpha drop, no flip, no conversion.
  - `insert!`: `haskey(screen.plot2usd, objectid(plot)) && return`; `isempty(plot.plots) ? (h=to_ovrtx_object(...); !isnothing(h) && (plot2usd[id]=h)) : foreach(p->insert!(screen,scene,p), plot.plots)`.
- [ ] **Steps:** failing test → `fig=Figure(); ax=LScene(fig[1,1]); mesh!(ax, Rect3f(Point3f(0), Vec3f(1)); color=:red)`; `img=Makie.colorbuffer(ax.scene)`; assert `eltype(img)==RGBA{N0f8}`, size matches, non-black; `save(tmp*".png", fig)` writes a valid PNG (right-side-up — spot check a known corner). Subprocess, exit 0. → implement → pass → commit `feat(M1.5): Mesh→UsdGeomMesh + displayColor + colorbuffer/display/insert!`.

**Acceptance:** a real Makie `Scene` with a mesh renders to a correct, right-side-up image; `save` works.

---

## Task M1.6 — `save`/`record` plumbing + offscreen constructors + `backend_showable`

**Files:** `src/screen.jl`. Test: `test/m1_save_record_test.jl`.

**Interfaces — Produces:** `Makie.backend_showable(::Type{Screen}, ::Union{MIME"image/png",MIME"image/jpeg"}) = true`; confirm the offscreen `Screen(scene,config,::IO,::MIME)` / `(…,::ImageStorageFormat)` constructors (from M1.1) route `save`/`record` through `colorbuffer` (Makie's fallback `backend_show` calls `colorbuffer`; per the contract research, no `backend_show` override needed).

- [ ] **Steps:** failing test → `Makie.record(fig, tmp*".mp4", 1:3) do i; rotate!(ax.scene, i*0.1); end` produces a non-empty mp4 (each frame re-authors+renders via `colorbuffer`). Also assert `save("x.png")` and `save("x.jpg")` both work. Subprocess. → implement → pass → commit `feat(M1.6): save/record/offscreen plumbing`.

**Acceptance:** offscreen image + video output work end-to-end.

---

## Task M1.7 — Remaining 3D primitives (scatter, meshscatter, lines, surface)

**Files:** `src/translation/primitives.jl` (+ `meshes.jl`). Test: `test/m1_primitives_test.jl`.

**Interfaces — Produces** one `to_ovrtx_object` per type, USD-native:
- `MeshScatter` → `UsdGeomPointInstancer` (prototype = marker mesh ref; `positions`, `orientations`, `scales`, per-instance `primvars:displayColor`).
- `Scatter` → `UsdGeomPointInstancer` with a `UsdGeomSphere` prototype (sphere fast-path).
- `Surface` → `UsdGeomMesh` (grid re-meshed via `GeometryBasics`/`Tessellation`; read `plot[1..3]` x/y/z).
- `Lines`/`LineSegments` → `UsdGeomBasisCurves` (`points`, `widths`=linewidth, `type="linear"`).
- (`Volume` → `UsdVol` deferred to M4 unless trivial.)

- [ ] **Steps:** failing test → one render per primitive asserting non-black + plausible coverage; a combined `scene` (mesh+scatter+lines) renders. Subprocess. → implement per type → pass → commit `feat(M1.7): scatter/meshscatter/lines/surface → USD`.

**Acceptance:** the v1 static 3D core renders.

---

**M1 GATE:** a `Figure` with `LScene` containing mesh/scatter/lines/surface renders via `colorbuffer`/`save` to a correct PNG; all primitive tests pass; every renderer subprocess exits 0; the shared test harness watchdog is in place. ✅ → M2 (ComputePipeline diff path + dynamic add/delete).

---

## Open assumptions — spike results

A pre-execution spike (`scratchpad/spike/spike_m1.jl`, exit 0, SPIKE_PASS) authored a **complete stage from a string** via `OV.open_usd_string!` — render-config root at the custom path `/Render/OVMakie/RenderProduct`, a from-scratch cube, a `DistantLight` + ambient, the reference camera — and rendered RT2 at 512².

1. **From-scratch stage renders — ✅ PROVEN.** The authored render-root rendered a centered cube (18502 lit px on black, centroid dead-center). The proven minimal skeleton (`scratchpad/spike/stage_Y.usda`) is handed to the M1.2 implementer as the concrete target — mirror it, don't re-derive from `torus-plane.usda`. The RenderProduct must carry the full `OmniRtxSettings*`/`OmniRtxPost*` apiSchemas list and the RenderSettings the two `OmniRtxSettingsGlobal*` schemas (copied verbatim from the reference); a `RenderVar "LdrColor"` under a `Vars` scope; `rel products`/`rel camera`/`rel orderedVars` wired by path.
2. **Custom product path `/Render/OVMakie/RenderProduct` renders — ✅ PROVEN** (not just the magic `OmniverseKit/HydraTextures` path). M1's product path is sound.
3. **`primvars:displayColor` suffices for flat color through RT2 — ✅ PROVEN** for `interpolation="constant"`: `displayColor=[(1,0,0)]` rendered pure red (mean RGB `0.635,0,0`). No MaterialX needed in M1.5. (`interpolation="vertex"` per-vertex color is standard USD, still to confirm in M1.5; `UsdPreviewSurface`/MaterialX stays deferred to M4.)
4. **`upAxis="Z"`** honored by ovrtx — STILL OPEN (spike used `upAxis="Y"` + the reference camera to isolate assumption 1). M1.2 validates `Z`; if ignored, fall back to a Z-up→Y-up root `xformOp:transform`.
5. **`UsdGeomPointInstancer`** honored for scatter — STILL OPEN (M1.7). Else fall back to per-point mesh refs (slower, correct) for M1; revisit at M2/M4.
