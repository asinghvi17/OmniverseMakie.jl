# OmniverseMakie — Architecture & Implementation Plan

A [Makie](https://makie.org) rendering backend that drives **NVIDIA `ovrtx`** (the
Omniverse RTX path tracer) from Julia: translate a Makie `Scene`/`Figure` into an
OpenUSD stage, render it with RTX, stream **minimal** per-frame edits through Makie's
`ComputePipeline` for live/animated rendering, feed input events back so Makie's own
camera controllers work, and display the result in an interactive window with a
GPU-direct (no-CPU-roundtrip) blit.

> Status: **design / pre-implementation.** This document is the agreed architecture.
> The implementation plan (milestones M0–M6) follows in a separate plan doc.

---

## 0. Validation status — what has been proven on this machine

Everything below was run live on this box (**NVIDIA RTX A5000**, driver 595.71.05,
CUDA 13.2-capable; Julia 1.12.6) before committing to the design.

| Claim | Result |
|---|---|
| ovrtx installs & the renderer initializes on the A5000 | ✅ `ovrtx==0.3.0.312915` via `uv` (Python 3.13); `Renderer()` OK |
| Full pipeline: load local USD → step (RT2) → DLPack readback → PNG | ✅ `torus-plane.usda`, 64 steps, `LdrColor` (1080×1920×4 `uint8`), all 2,073,600 px non-black, **~8 s** (`references/validation/torus-A5000.png`) |
| `libOpenGL.so.0` (needed by ovrtx's `usd_resolver` plugin **and** GLMakie/interop) | ✅ present at `/home/juliahub/temp/extra-libs/` and in Julia's GL artifact; fixed with `LD_LIBRARY_PATH`. No sudo/apt needed. |
| Makie drives backends **imperatively** for live add/delete (`insert!`/`delete!` per screen) | ✅ verified in source; **no `scene.plots` observers exist** |
| GLMakie plot add/delete is sound; **subscene** add/remove is the leaky part | ✅ confirmed (the user's instinct points at the real gap) |
| CUDA↔OpenGL interop from Julia for GPU-direct display | ✅ Yellow-Green: CUDA.jl 6.2.0 `functional()`; `cuGraphicsGLRegisterImage` et al. resolve via `CUDA.CUDACore` **and** `@ccall libcuda`; `GLAbstraction.Texture.id` matches |
| ovrtx schema/plugin paths self-register in C (no Python env needed) | ✅ `ovrtx_create_renderer` → `ovrtx_register_schema_paths` via `dladdr` |
| ovrtx C ABI → clean Clang.jl `ccall` bindings, Python-free | ✅ 42 funcs / 58 structs / 24 enums, **0 bogus symbols**, module loads; one flag `skip_static_functions=true` |
| **Full render driven entirely from Julia via `ccall`** (keystone) | ✅ create→open_usd→64× RT2 step→map `LdrColor`→readback→PNG: **2,073,600/2,073,600 px non-black, ~11 s**, generated binding used **verbatim**; + a live `write_attribute(omni:xform)`+`reset` update changes 597,204 px |
| **`:ovrtx_renderobject` diff node + imperative `insert!`/`delete!`** on real Makie internals | ✅ verified with **no window**: `changed`-mask minimal-delta works on a live plot graph; `push!/delete!(scene,plot)` dispatch fires per `current_screens`; recipes call `insert!` once-per-parent (recurse `plot.plots`) |
| Julia process exits cleanly after creating an ovrtx renderer | ⚠️ ovrtx's bundled **carb breakpad crash reporter** hijacks SIGSEGV/SIGABRT → Julia exits 139 at teardown; fixed by snapshot/restore of POSIX signal handlers around `create_renderer` (plan Task M0.4) |

**Net:** the renderer, the USD pipeline, the DLPack readback, the Julia-side interop
primitives, and the Makie backend hooks are all confirmed real and working here. The
risk is integration glue, not feasibility.

---

## 1. The four locked decisions (from design review)

1. **Interactive 3D exploration** is the v1 north-star (live RTX viewport you orbit/zoom),
   built on a static `Scene→USD→image` foundation.
2. **Direct `ccall`** to `libovrtx-dynamic.so` from the start (no Python/PythonCall), in a
   `LibOVRTX` **subpackage**, mirroring RadeonProRender.jl's 3-layer wrap.
3. **3D path-traced core** scope: `Mesh`, `MeshScatter`, `Scatter`, `Surface`, `Lines`,
   `Volume` (2D/text/axes deferred).
4. **GPU-direct display** (ovrtx CUDA frame → GLMakie texture via CUDA-GL interop), CPU
   blit as the v1 fallback. **Streaming/WebRTC is shelved.**

Plus three hard requirements called out in review:
- **Dynamic add *and* delete** of plots (and subscenes) in a live tree — first-class, wired
  correctly (better than GLMakie's subscene story).
- **C zero-copy hot path** for animation — per-frame updates use `map_attribute`/array
  bindings, never USDA re-authoring.
- **`Libdl.dlopen` with an `OVRTX_LIBRARY_PATH` override** — assume a system install; no
  wheel discovery, no JLL.

---

## 2. Two key realizations the whole design rests on

**(a) USD *is* the wire format.** ovrtx has *no* programmatic scene API; it consumes
**OpenUSD exclusively** (file/URL, inline USDA string, or additive references). So
"Makie → ovrtx" is **"Makie `Scene` → USD stage,"** and interactivity is **live edits to
that stage** via the attribute API. The renderer patches its BVH/TLAS incrementally — the
cheap-diff model we want is native.

**(b) The diff engine already exists inside Makie — `ComputePipeline`.** GLMakie and
WGLMakie use one identical pattern: each plot is a *single* compute node
(`:gl_renderobject` / `:wgl_renderobject`) depending on all render-relevant outputs; the
engine hands the callback a **`changed` mask**; the first eval uploads everything, later
evals push *only* changed outputs. We mirror this with an `:ovrtx_renderobject` node that
authors/patches USD. This is literally how WGLMakie sends minimal diffs over its socket —
we send them into a USD stage instead. **That is "respect ComputePipeline and use it."**

---

## 3. Architecture — three layers

```
   Makie Scene/Figure   (Observables + per-plot ComputeGraph in plot.attributes)
        │  collect_atomic_plots →  Mesh/MeshScatter/Scatter/Surface/Lines/Volume
        │  per-plot compute node :ovrtx_renderobject  (the diff driver)
        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Layer 2 — Translation (Makie ⇄ USD)                                         │
│   • static: author USD prim subtree per plot (objectid-keyed path)          │
│   • dynamic: changed-mask → minimal stage edits (xform / points / color)    │
│   • add/delete: add_usd_reference_from_string(handle) / remove_usd(handle)  │
└──────────────────────────────────────────────────────────────────────────┘
        │  OV high-level API  (open_usd / step / write / bind / map / clone / remove)
        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Layer 1 — Binding   OmniverseMakie/lib/LibOVRTX (raw ccall) + OV wrapper    │
│   • Libdl.dlopen("libovrtx-dynamic.so")  ($OVRTX_LIBRARY_PATH override)      │
│   • async lifecycle: enqueue → wait_op → fetch_results → destroy_results     │
│   • DLPack.jl at the tensor boundary                                         │
└──────────────────────────────────────────────────────────────────────────┘
        │  step() → LdrColor render var  (CPU tensor, or CUDA_ARRAY zero-copy)
        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Layer 3 — Presentation                                                      │
│   • Screen <: Makie.MakieScreen  (the backend contract)                     │
│   • on-demand render loop + progressive refinement (RT2 + denoiser)         │
│   • colorbuffer → Matrix{RGB}; record falls out free                        │
│   • display: CUDA-GL interop blit into a GLMakie image! (CPU fallback)      │
│   • events: write scene.events.* → cam3d!/cam2d! controllers work unchanged │
└──────────────────────────────────────────────────────────────────────────┘
```

Package name: **`OmniverseMakie`**. Subpackage: **`LibOVRTX`** at `lib/LibOVRTX/`.

---

## 4. Layer 1 — the `ccall` binding

### 4.1 Three sub-layers (mirrors RadeonProRender.jl)

1. **`LibOVRTX` (subpackage, `lib/LibOVRTX/`)** — Clang.jl-generated 1:1 `ccall` bindings
   from `references/ovrtx/include/` (`ovrtx/ovrtx.h`, `ovrtx_types.h`, `ovrtx_attributes.h`,
   `ovrtx_config.h`, `ovx/types.h`, `ovx/dlpack/dlpack.h`, `ovx/path_dictionary/*`). Opaque
   `ovrtx_renderer_t*`, `uint64` handles (`ovrtx_attribute_binding_handle_t`,
   `ovrtx_step_result_handle_t`, `ovrtx_usd_handle_t`), `ovx_string_t{ptr,len}`, plain
   structs, `CEnum.@cenum` enums, vendored DLPack. Each entry point wrapped in `check_error`
   (status + `ovrtx_get_last_error`, which is thread-local and only valid until the next
   call). **Validated (Clang.jl v0.19 / libclang 18, Julia 1.12):** clean generation —
   42 `ovrtx_*` functions, 58 structs, 24 `@cenum`, 0 bogus symbols — given:
   - `skip_static_functions = true` (the `static inline` `config_entry_*`,
     `is_ovx_string_empty`, and vtable wrappers are **not** exported; build the
     `ovrtx_config_entry_t` structs / reimplement those ~18 helpers directly in Julia);
   - **exclude `ovrtx_attributes.h`** from the umbrella — it's C++ (`nullptr`/`new`/
     brace-init); reimplement its convenience helpers (`ovrtx_set_xform_mat`,
     `ovrtx_set_path_attributes`, …) in Julia;
   - `path_dictionary` is a **vtable of function pointers** — call
     `@ccall $(inst.vtable.field)(...)`, not by symbol name;
   - `ovx_string_t` is passed/returned **by value** and borrows its `ptr` → `GC.@preserve`;
     copy returned error strings immediately (transient, thread-local).
2. **`OV` high-level wrapper (in `OmniverseMakie/src/binding/`)** — GC-aware Julia structs
   (`Renderer`, `RenderProduct`, `AttrBinding`, `StepResult`) with finalizers, plus the
   **async lifecycle** the C API requires (Python hides it; we cannot): every
   `ovrtx_open_usd_from_file`/`clone_usd`/`step` returns an enqueue handle → `ovrtx_wait_op`
   → `ovrtx_fetch_results` → **`ovrtx_destroy_results`** (leaks warn otherwise).
3. **DLPack boundary** — `DLPack.jl` wraps mapped render-var/attribute tensors into Julia
   arrays zero-copy (CPU now; `CUDA_ARRAY` for the GPU-direct display, §7).

### 4.2 Loading & environment (validated, simpler than feared)

- **`.so` discovery:** `Libdl.dlopen` of `libovrtx-dynamic.so`. Resolution order:
  `ENV["OVRTX_LIBRARY_PATH"]` (explicit override) → default soname on the loader path. The
  `.so` lives at `<ovrtx>/bin/libovrtx-dynamic.so` with a **runtime tree beside it**
  (`plugins/ usd_plugins/ mdl/ library/ libs/ rendering-data/ cache/ ovrtx.config.json`).
- **Schema/plugin registration is automatic in C.** `ovrtx_create_renderer` →
  `ovrtx_register_schema_paths`, which finds the runtime root by `dladdr` on the loaded
  `.so` (or `OMNI_USD_PLUGINS_BASE_PATH` / the `binary_package_root_path` config). **We do
  not replicate any Python env logic.** If the install splits the `bin/` tree, pass
  `ovrtx_config_entry_binary_package_root_path`.
- **`libOpenGL.so.0`:** ovrtx's `usd_resolver` plugin needs it (ovrtx does **not** ship it).
  Module `__init__` does `Libdl.dlopen(Libglvnd_jll.libOpenGL_path, RTLD_GLOBAL)` **before**
  loading libovrtx — `Libglvnd_jll` is a dep of `LibOVRTX` and provides `libOpenGL.so` with
  SONAME `libOpenGL.so.0` (verified), so ovrtx's later by-soname plugin `dlopen` resolves to
  the already-loaded image. Override via `$OVRTX_LIBOPENGL_PATH`. Self-contained — no system
  `libglvnd`, no `LD_LIBRARY_PATH` (the validation used `LD_LIBRARY_PATH` before the JLL fix).
- **Headless renderer lifecycle:** create with `keep_system_alive=true` and call
  `ovrtx_initialize()` once before the first `Renderer` to avoid the libEGL teardown crash
  when screens open/close repeatedly. Pair `initialize`/`shutdown`.

### 4.3 Package & workspace layout

```
OmniverseMakie/                      # backend package
├── Project.toml                     # [deps] Makie, GLMakie, GeometryBasics, Colors,
│                                    #   DLPack, CUDA, LibOVRTX (+ Libdl, LinearAlgebra)
│                                    # [sources] LibOVRTX = { path = "lib/LibOVRTX" }
│                                    # [workspace] projects = ["lib/LibOVRTX"]
├── src/
│   ├── OmniverseMakie.jl            # module, __init__ (dlopen, initialize), activate!
│   ├── binding/  OV.jl, dlpack.jl   # high-level GC wrapper + async lifecycle
│   ├── translation/ usd.jl scene.jl meshes.jl scatter.jl lines.jl surface.jl
│   │                volume.jl materials.jl camera.jl lights.jl
│   ├── compute.jl                   # the :ovrtx_renderobject node + push_to_ovrtx!
│   ├── screen.jl                    # Screen <: MakieScreen + contract methods + renderloop
│   ├── events.jl                    # scene.events.* injection, render_tick
│   ├── display.jl                   # GLMakie image! target; CPU + CUDA-GL blit
│   └── settings.jl                  # render mode (RT2/PathTracing/Minimal), samples
├── lib/LibOVRTX/                    # subpackage (raw bindings)
│   ├── Project.toml                 # name=LibOVRTX; deps CEnum, Libdl
│   ├── src/LibOVRTX.jl              # generated + loader (OVRTX_LIBRARY_PATH) + check_error
│   └── gen/  generator.toml generator.jl   # Clang.jl regeneration
├── test/  ext/  docs/  ARCHITECTURE.md
```
Concrete `[sources]`/`[workspace]` TOML and the Clang.jl `generator.jl` are in
`references/notes/clang-libovrtx.md`.

---

## 5. Layer 2a — static translation (Makie `Scene` → USD)

A `to_ovrtx_object(screen, scene, plot)` dispatch per atomic primitive (RPRMakie's
`to_rpr_object` pattern), driven by `collect_atomic_plots(scene)`. Each plot authors a USD
subtree at a **stable path keyed by `objectid(plot)`** (e.g. `/World/Plot_<id>`), added as a
**removable reference** so updates and deletion can target it by handle.

### 5.1 Primitive → USD mapping (v1 3D core)

| Makie primitive | USD prim(s) | Geometry / attrs | Hot-path update |
|---|---|---|---|
| `Mesh` | `UsdGeomMesh` | `points`, `faceVertexCounts/Indices`, `normals`, `primvars:displayColor` | points/normals via `bind_array_attribute`; xform via `omni:xform` |
| `MeshScatter` | `UsdGeomPointInstancer` (prototype = marker mesh) | `positions`, `orientations`, `scales`, per-instance `primvars:displayColor` | array bindings (positions/scales) |
| `Scatter` | `PointInstancer` + **`UsdGeomSphere` prototype** (the **sphere fast path**); `UsdVol.ParticleField` for huge N | `positions`, `scales`, colors | array bindings; splat field for N≫ |
| `Surface` | `UsdGeomMesh` (grid re-meshed) | grid → verts/faces; colormap → `displayColor`/texture | points/displayColor bindings |
| `Lines`/`LineSegments` | `UsdGeomBasisCurves` | `points`, `widths` (= linewidth), `type=linear` | points binding |
| `Volume` | `UsdVol` volume (`VoxelGrid` + volume material) | density/field grid | field attribute writes |

- **Camera** → `UsdGeomCamera` from `scene.camera` (view/projection → camera `omni:xform`
  + focal length/aperture; perspective and orthographic). Updated as `omni:xform` (cheap).
- **Lights** → `UsdLux`: `PointLight`→`SphereLight`, `DirectionalLight`→`DistantLight`,
  `RectLight`→`RectLight`, `EnvironmentLight`→`DomeLight`, `AmbientLight`→low `DomeLight`.
- **Materials** → MDL `OmniPBR` from Makie shading/PBR (metalness/roughness/transparency);
  plain color → `primvars:displayColor`; colormap+scalar → texture or per-vertex
  `displayColor`. Backend-specific `material=` escape hatch (MDL/MaterialX path string) like
  RPRMakie. Runtime swap = write `material:binding` relationship.
- **RenderProduct + RenderVar:** author `/Render/<id>` RenderProduct (`rel camera`,
  `resolution`, `orderedVars`) with a `LdrColor` RenderVar. `step()` takes the **RenderProduct
  path**, never the camera path. Render settings (`omni:rtx:*`) are attributes on the
  RenderProduct.

### 5.2 USD authoring mechanism (decision: **A, growing into C**)

- **Structure (one-time, per plot):** compose an inline USDA string and add it via
  `add_usd_reference_from_string(usda, prefix="/World/Plot_<id>")` → keep the returned
  **handle**. The figure's render config (camera/RenderProduct/RenderVar) is one
  `open_usd_from_string` root (optionally `subLayers` an existing scene).
- **Hot data (every frame):** never re-author USDA — use the C attribute API (§6).
- Option B (linking OpenUSD/`pxr` C++ into Julia) is explicitly deferred; A+C covers v1.

---

## 6. Layer 2b — dynamic diffing (`ComputePipeline` → USD edits) ★ the core

Mirror GLMakie/WGLMakie exactly. At `insert!`, after authoring the prim, register **one**
compute node over the plot's consumed outputs (copy GLMakie's per-primitive input lists:
`:positions_transformed_f32c`, `:model_f32c`, `:faces`, `:normals`, `:texturecoordinates`,
`:scaled_color`, `:alpha_colormap`, `:converted_rotation`, `:quad_scale/offset`, `:visible`,
…):

```julia
register_computation!(plot.attributes, consumed_inputs, [:ovrtx_renderobject]) do args, changed, last
    if isnothing(last)
        prim = author_usd_prim!(screen, plot, args)          # build once (Layer 2a)
        bind_hot_attributes!(screen, prim, args)             # persistent bindings for hot attrs
        robj = OvrtxRObj(prim, handle, bindings)
    else
        robj = last.ovrtx_renderobject
        for name in keys(args)
            changed[name] || continue                         # ← the engine's diff mask
            push_to_ovrtx!(screen, robj, name, args[name])    # minimal stage edit
        end
    end
    screen.requires_update = true
    return (robj,)
end
```

`push_to_ovrtx!` routes each changed output to the right ovrtx write:

| Changed output | USD attribute | ovrtx call (hot path) |
|---|---|---|
| `:model_f32c` (→ `Float64`) | `omni:xform` (4×4 doubles, row-vector, translation in last row) | `map_attribute` **zero-copy** (fixed size) or `ovrtx_set_xform_mat` |
| `:positions_transformed_f32c` | `points` / instancer `positions` (array) | `bind_array_attribute` + `write()` (GPU DLPack tensor, stream-synced) |
| `:faces` | `faceVertexIndices` | `write_array_attribute` |
| `:scaled_color`/colormap | `displayColor` primvar or material param | array binding / `write_attribute` |
| `:visible` | `visibility` | `write_attribute` |
| camera view/proj | camera `omni:xform` | `map_attribute` zero-copy |

**The hot-path tiers (the "C for performance" mandate):**
- **Fixed-size attrs** (`omni:xform`, scalars) → **`map_attribute`** = true zero-copy into
  ovrtx's Fabric buffer. Best for transforms/camera; great for GPU-computed transforms.
- **Array attrs** (`points`, instancer `positions`) are **not mappable** (variable length) →
  **`bind_array_attribute` + `write()`** with a **GPU-resident DLPack tensor** (one GPU→GPU
  copy, `cuda_stream`/`cuda_event` synced). Still no CPU roundtrip.
- Create the binding **once** (locks prim+attr+type; `OVRTX_BINDING_FLAG_OPTIMIZE` on the
  primary hot binding); reuse every frame. `Float32→Float64` convert for `omni:xform`.

**Per frame**, the render loop **pulls** `plot.attributes[:ovrtx_renderobject][]` once
(try/catch → `ComputePipeline.mark_resolved!` on error); a clean graph is a no-op. Any
geometry/camera change → `renderer.reset()` to restart RT2 accumulation.

> **De-risk (M2):** benchmark this path — animate N transforms/points per frame and confirm
> map/bind throughput meets interactive rates. ("Good *if* it's fast" — we measure.)

---

## 7. Dynamic add / delete (first-class) — validated wiring

Makie notifies backends **imperatively** (no `scene.plots`/`scene.children` observers):

- **Add:** `plot!`/`push!(scene, plot)` → `for screen in scene.current_screens:
  insert!(screen, scene, plot)`. We implement `insert!(screen, scene, plot)` (+
  `insertplots!`): walk to atomic leaves (idempotent via `objectid` cache), `add_scene!`
  (author a USD scope for a new subscene), author the prim, register the
  `:ovrtx_renderobject` node, cache it, add to the draw list.
- **Delete:** `delete!(scene, plot)` → `ComputePipeline.unsafe_disconnect_from_parents!` →
  `delete!(screen, scene, plot)` → `free(plot)`. We implement `delete!(screen, scene, plot)`:
  destroy our `AttrBinding`s/mapped buffers, **`remove_usd(handle)`** the plot's prim, drop
  from caches/draw list, and `delete!(plot.attributes, :ovrtx_renderobject)`.
- **Subscenes** (the part GLMakie does poorly): a child Scene copies `current_screens` at
  construction and registers lazily on first plot; `delete!(screen, scene)` recurses
  children/plots. **We fix GLMakie's leaks**: deregister per-scene redraw hooks in
  `delete!(screen, scene)`, refcount shared resources, and reclaim the USD scope.
- Park our listeners in `plot.deregister_callbacks` so `free(plot)` tears them down.

Methods to implement: `insert!`, `insertplots!`, `delete!(screen,scene,plot)`,
`delete!(screen,scene)`, `empty!(screen)`, `push!(screen,scene,robj)`. No observers.

---

## 8. Layer 3 — Screen, render loop, display, events

### 8.1 Screen & the Makie backend contract
`mutable struct Screen <: Makie.MakieScreen` holds the `OV.Renderer`, the USD stage, the
`RenderProduct`, framebuffer size, a `plot→OvrtxRObj` registry, `requires_update`,
samples/mode config, and the display target. Implement (per
`references/notes/makie-backend-contract.md`): the three `Screen(scene, config, …)`
constructors, `display`, `colorbuffer(screen, format)`, `insert!`/`delete!`/`empty!`,
`size`/`resize!`, `close`/`isopen`, `activate!` + `ScreenConfig`, `backend_showable`,
`to_native`. Makie's `record`/`save` route through `colorbuffer` for free.

### 8.2 Render loop & progressive refinement
On-demand loop (GLMakie's `on_demand_renderloop`): when `requires_update` or still
accumulating, `step()` → present. **Interactivity strategy on the A5000:** default
**Real-Time Path-Tracing (RT2)** mode (accumulates across `step()`s, OptiX denoiser) — show
a low-sample denoised frame while the camera moves, keep accumulating when idle; `reset()` on
any change. Offline `record`/`colorbuffer` can switch to `PathTracing` mode
(`samplesPerPixel`) for final quality. Settings are `omni:rtx:rendermode`,
`omni:rtx:rtpt:maxBounces`, `omni:rtx:pt:denoising:*`, etc., on the RenderProduct.

### 8.3 colorbuffer
Render to target samples → map `LdrColor` (CPU) → DLPack → `Matrix{RGB{N0f8}}` with y-flip.

### 8.4 Display — GPU-direct, CPU fallback (validated Yellow-Green)
Display target is a GLMakie window showing a fullscreen `image!`; GLMakie provides the
window **and input capture**.
- **v1 (CPU fallback):** map `LdrColor` → DLPack → update the `image!` data Observable (one
  host roundtrip). Unblocks the interactive viewport immediately.
- **v2 (GPU-direct):** map `LdrColor` in C as **`OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY`** (a
  `CUarray`; Python can't, C can) → `cuGraphicsGLRegisterImage` the `image!`'s
  `GLAbstraction.Texture.id` once → per frame
  `cuGraphicsMapResources`/`SubResourceGetMappedArray`/`cuMemcpy2D`/`Unmap`. **Constraint:**
  all GL+interop must run on **GLMakie's render task** with the GL context current — inject
  via `screen.render_tick`. Sync ovrtx's `cuda_sync.wait_event` before copy; `cuEventRecord`
  → pass to `ovrtx_unmap`. Same device for CUDA+GL (single-GPU here → trivial). Slots in
  behind the same `Texture`; CPU path stays as fallback.

### 8.5 Events — controllers for free
Makie's `Events` struct is GLFW-free. Forward the GLMakie window's input into **our scene's**
`scene.events.*` (mouseposition px/upper-left, mousebutton, scroll, keyboard, resize) and
maintain a `render_tick`. Makie's `cam3d!`/`cam2d!` then orbit/zoom/pan with **zero backend
code**; camera change → update camera `omni:xform` + `reset()`.

---

## 9. Milestones (streaming shelved)

> Reordered 2026-06-28 (option B): examples land before interactivity, and materials are pulled forward — so the example gallery has full materials. Old order was M3 interactive · M4 depth · M5 examples.

- **M0 — binding spike.** Clang.jl-generate `LibOVRTX`; `dlopen` + `OVRTX_LIBRARY_PATH` +
  `libOpenGL` global load; port the C `minimal` path: `init → open_usd(local) → step →
  map LdrColor → PNG` through pure `ccall`. (ovrtx pipeline already validated via Python;
  this re-proves it Julia-native.)
- **M1 — static translation.** `Screen`, `to_ovrtx_object` for the 3D core (start
  Mesh+camera+lights, expand to MeshScatter/Surface/Lines/Volume/Scatter); `colorbuffer`,
  `record`. Author via inline USDA + references.
- **M2 — ComputePipeline diff path.** `:ovrtx_renderobject` node; live attribute/transform/
  color updates via map/array-bindings; accumulation reset; **benchmark the hot path**.
- **M3 — materials.** OmniPBR / MaterialX: Makie shading/PBR (metalness/roughness/
  transparency) → MDL `OmniPBR`; backend `material=` escape hatch (MDL/MaterialX path);
  runtime material swap via `material:binding`. Enables the examples gallery to use full
  materials.
- **M4 — examples gallery.** Adapt the [RPRMakieNotes](https://github.com/lazarusA/RPRMakieNotes)
  and [raydemo](https://github.com/simondanisch/raydemo) scenes into `examples/` (originals
  untouched) — the end-to-end showcase that the backend handles real-world scenes, now with
  full OmniPBR materials available from M3.
- **M5 — interactive viewport.** GLMakie display (CPU blit) + event injection; `cam3d!`
  orbit/zoom; on-demand loop + progressive refinement; dynamic add/delete end-to-end.
- **M6 — GPU-direct + picking + hardening.** GPU-direct CUDA-GL blit (no CPU roundtrip);
  AOVs (Depth/Normal/ID) → picking (`scene.events`); subscene-leak fixes.

(Deferred: 2D/text/axes parity; remote streaming — WGLMakie-style websocket then GStreamer
`webrtcsink` sidecar — see `references/notes/wire-protocol-and-webrtc.md`.)

---

## 10. Risks & open questions

| Risk | Mitigation |
|---|---|
| **carb breakpad crash reporter** hijacks signals → Julia crashes at process exit | Snapshot/restore POSIX signal handlers (SIGSEGV/ABRT/BUS/ILL/FPE) around `ovrtx_create_renderer` (plan Task M0.4); re-verify for the long-lived interactive loop (M5.3). Discovered in Spike A. |
| ovrtx is **preview 0.3** (API stability "later 2026") | Pin a version; isolate churn in `LibOVRTX`; regen via `gen/` |
| Path-tracer interactivity latency on A5000 | RT2 + denoiser + low-sample-while-moving + progressive idle accumulation; benchmark in M2 |
| Hot-path throughput ("fast *if*…") | M2 benchmark of map/array-binding writes for N transforms/points per frame |
| Array attrs not mappable (`points`) | Use `bind_array_attribute` + GPU DLPack write (no CPU roundtrip) |
| CUDA-GL interop glue (context/threading) | Run on GLMakie render task; CPU fallback always available |
| `libOpenGL.so.0` / runtime tree placement | `dlopen(RTLD_GLOBAL)`; `OVRTX_LIBRARY_PATH`; documented `LD_LIBRARY_PATH` |
| NVIDIA-proprietary binaries (can't vendor) | Users install ovrtx; we only `dlopen` |
| Subscene add/delete leaks (inherited from GLMakie) | Explicit deregistration + refcounting in `delete!(screen,scene)` |

---

## 11. Reproducing the validation

```bash
# 1. ovrtx (no sudo; uv provisions Python 3.13 + the wheel from GitHub Releases)
curl -LsSf https://astral.sh/uv/install.sh | sh
cd references/ovrtx/examples/python/minimal

# 2. local render (libOpenGL.so.0 from the pre-staged extra-libs dir)
export LD_LIBRARY_PATH=/home/juliahub/temp/extra-libs:$LD_LIBRARY_PATH
uv run --python 3.13 python <scratchpad>/local_render.py \
    ../../c/minimal/torus-plane.usda \
    /Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0 \
    out.png 64
# → out.png, 1080×1920×4 uint8, ~8 s   (see references/validation/torus-A5000.png)
```
`libovrtx-dynamic.so` → `…/.venv/lib/python3.13/site-packages/ovrtx/bin/`.

---

## 12. Reference notes (under `references/`)

| File | Contents |
|---|---|
| `notes/ovrtx-api.md` | ovrtx scene model, render/update paths, DLPack, C-ABI-vs-Python |
| `notes/rprmakie.md` | RPRMakie backend template (Screen, `to_rpr_object`, render) |
| `notes/makie-backend-contract.md` | the backend contract + `Events` struct + injection |
| `notes/computepipeline.md` | the compute-graph diff pattern backends use |
| `notes/dynamic-add-delete.md` | imperative insert!/delete! wiring + subscene gaps |
| `notes/cuda-gl-interop.md` | CUDA↔GL interop from Julia + ovrtx CUDA_ARRAY handoff |
| `notes/clang-libovrtx.md` | Clang.jl generator config + `[sources]`/`[workspace]` TOML |
| `notes/wire-protocol-and-webrtc.md` | (deferred) streaming design |
| `validation/torus-A5000.png` | proof render from this machine |
| `ovrtx/skills/` | NVIDIA's authoritative ovrtx API skill docs |
```
