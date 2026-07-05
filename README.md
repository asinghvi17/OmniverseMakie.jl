# OmniverseMakie.jl

A [Makie](https://makie.org) rendering backend that drives **NVIDIA `ovrtx`** — the
Omniverse RTX path tracer — from Julia. It translates a Makie `Scene`/`Figure` into an
[OpenUSD](https://openusd.org) stage, renders it with RTX, streams **minimal** per-frame
edits through Makie's `ComputePipeline` for live/animated rendering, and can display the
result in an orbit-able GLMakie window with a GPU-direct (no-CPU-roundtrip) blit.

> **Status: research-grade, under active development.** This is a preview backend built
> against `ovrtx` 0.3 (a preview API). Expect rough edges; interfaces may change. See
> [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design and the milestone history.

---

## What it renders

The 3-D path-traced core: `mesh`, `meshscatter`, `scatter`, `surface`, `lines` /
`linesegments`, and `volume` — with OmniPBR / OmniGlass materials, image textures,
colormaps, cameras, and lights. 2-D primitives, `text`, and axis decorations are not yet
translated (deferred). The port gallery in [`examples/`](examples/README.md) shows 14
real path-traced scenes end-to-end.

---

## Requirements

- **An NVIDIA GPU** (validated on an RTX A5000) and a working NVIDIA driver.
- **An `ovrtx` install** providing `libovrtx-dynamic.so` and its runtime tree. `ovrtx` is
  NVIDIA-proprietary and cannot be vendored — you install it yourself (e.g. the `ovrtx`
  wheel; see the reproduction recipe in [`ARCHITECTURE.md`](ARCHITECTURE.md) §11). The
  backend only `dlopen`s it.
- **Julia 1.12** or newer.
- Optional, for the GPU-direct viewport blit: **CUDA** (via `CUDA.jl`) and **GLMakie**.

### Environment variables

| Variable | When | Purpose |
|---|---|---|
| `OVRTX_LIBRARY_PATH` | always (unless `libovrtx-dynamic.so` is already on the loader path) | Absolute path to `libovrtx-dynamic.so`. `LibOVRTX` `dlopen`s this at load; if unset it falls back to the bare `libovrtx-dynamic.so` soname. |
| `OMNIVERSEMAKIE_INDEX_LIBS` | volume rendering | Path to the `omni.index.libs` extension **root** (the loader appends `/bin/nvindex-libs`). Enables NVIDIA IndeX by synthesizing a carb config. |
| `OMNIVERSEMAKIE_OVRTX_CONFIG` | volume rendering (alternative) | Absolute path to a ready `*.config.json` that already registers `/app/tokens/omni.index.libs`. Takes precedence over `OMNIVERSEMAKIE_INDEX_LIBS`. |
| `OVRTX_LIBOPENGL_PATH` | optional | Override the `libOpenGL.so` `ovrtx`'s `usd_resolver` plugin needs (defaults to `Libglvnd_jll`). |

The volume env-var contract is documented in full in
[`src/binding/index_config.jl`](src/binding/index_config.jl); the `OVRTX_LIBRARY_PATH`
contract used by the test harness is in [`test/helpers.jl`](test/helpers.jl).

---

## Installation

The package is not registered; install it from a checkout.

```bash
git clone <this-repo> OmniverseMakie.jl
cd OmniverseMakie.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
export OVRTX_LIBRARY_PATH=/path/to/ovrtx/bin/libovrtx-dynamic.so
```

`LibOVRTX` and `NanoVDBWriter` are path `[sources]` sub-packages under
[`lib/`](lib) and resolve automatically.

---

## Quickstart

### Offscreen render (no window)

`using OmniverseMakie` registers the backend, so Makie's `save` / `colorbuffer` route
through the RTX path tracer automatically.

```julia
using OmniverseMakie                       # registers the ovrtx backend on load

fig = Figure(; size = (900, 900))
ax  = LScene(fig[1, 1]; show_axis = false)
meshscatter!(ax, rand(Point3f, 500); color = :dodgerblue, markersize = 0.05)

save("scatter.png", fig)                   # RTX-path-traced PNG (via colorbuffer)
```

For direct pixel access, build a `Screen` and call `colorbuffer` — it returns a
`Matrix{RGBA{N0f8}}`, top-left origin:

```julia
screen = OmniverseMakie.Screen(ax.scene)
img    = Makie.colorbuffer(screen)         # Matrix{RGBA{N0f8}}
```

Render quality/mode is set through `activate!` (or per-`Screen` config): `mode`
(`:rt2` default realtime, `:pathtracing` offline), `samples`, `warmup`, `max_bounces`.

```julia
OmniverseMakie.activate!(; mode = :pathtracing, samples = 512)
```

`:pathtracing` switches to the offline path tracer and renders each still at `samples`
samples-per-pixel — slower but higher quality than the default realtime `:rt2` mode
(`samples` is inert in `:rt2`).

### Interactive viewport

`interactive_display` opens an orbit-able GLMakie window showing the live RTX render of a
whole figure; drag orbits and scroll zooms. It needs GLMakie for the window and input; load
CUDA too for the GPU-direct blit.

```julia
using OmniverseMakie, GLMakie              # GLMakie: window + input capture
# using CUDA                               # optional: GPU-direct blit (no CPU roundtrip)

fig = Figure()
ax  = LScene(fig[1, 1])
surface!(ax, -3:0.1:3, -3:0.1:3, (x, y) -> exp(-(x^2 + y^2)))

interactive_display(fig)                   # live, orbit-able RTX viewport
```

### Hybrid figures: `replace_scene!` (RTX 3D + GLMakie 2D diagnostics)

`replace_scene!` replaces ONE scene (an `LScene`, `Axis3`, or `Scene`) in an
already-displayed GLMakie figure with a live raytraced render, leaving the figure's other
axes as ordinary GLMakie 2D plots — the RPRMakie `replace_scene_rpr!` hybrid. The target keeps
its own camera, so you orbit it with normal GLMakie interaction; a hook on the host window
re-renders each frame.

```julia
using OmniverseMakie, GLMakie
GLMakie.activate!()

fig = Figure()
ls  = LScene(fig[1, 1])                     # the 3D panel → RTX raytraced
mesh!(ls, load(assetpath("brain.stl")); color = :bisque)
ax  = Axis(fig[1, 2])                       # a 2D diagnostic → stays GLMakie
lines!(ax, cumsum(randn(200)))

display(fig)                                # show the figure FIRST (GLMakie)
session = replace_scene!(ls)                # ls is now a live RTX viewport, ax untouched
# ... orbit ls, mutate plots; close(session) restores it without closing the window
```

v1 is CPU-blit and one embedded scene per figure; GPU-direct blit and multiple concurrent
embeds are planned follow-ups.

The embedded overlay is pixel-space and does NOT inherit the target scene's transformation,
so a root `rotate!(ls.scene, ...)` (e.g. the Z-up trick) composites correctly.

**Recording a hybrid figure** (e.g. piping frames to ffmpeg): stop the host render loop first —
the per-tick blit otherwise keeps GLMakie's on-demand loop hot and a pipe `write` can starve.
`stop_renderloop!(glscr; close_after_renderloop = false)` (the `false` keeps the screen open),
then `record_frame!(session; ticks = 3)` per frame drives fully synchronous ticks and returns
the composited image; re-apply `update_cam!` each frame if you script the camera. Full recipe
in the `replace_scene!` docstring.

### Realtime-style recording (accumulate across frames)

By default every frame reconverges the path tracer from scratch — correct, but slow for
animations. Set `accumulate_across_frames = true` to instead carry RT2 accumulation across
frames: its temporal reprojection + denoiser absorb the motion the way the interactive
viewport does, so a `record` runs on the order of 10× faster with visually indistinguishable
frames. Only a structural change (adding/removing a plot, a volume data reload) resets;
camera, light, and attribute edits do not.

```julia
using OmniverseMakie
OmniverseMakie.activate!(accumulate_across_frames = true, warmup = 4)

fig = Figure(); ax = LScene(fig[1, 1]); p = scatter!(ax, rand(Point3f, 100))
record(fig, "orbit.mp4", 1:120) do frame
    # move the camera or the data here — no per-frame reconverge
end
```

`warmup` is RTX steps per frame (4 is plenty when accumulating; the default 64 is for
per-frame reconverge). `accumulation_preroll` (default 40) adds steps to the first frame only
so it isn't cold. Best for slow object motion / static-ish cameras; a fast camera fly-through
stresses the reprojection (the same trade-off the viewport makes). If a change ever ghosts,
`OmniverseMakie.reset_accumulation!(screen)` forces one reset.

### Placing USD assets: `usdplot` + `bind_usd!`

`usdplot!` composes an **external USD file** (a DCC export, a vendor asset, a Kit-authored
scene — `.usda` or `.usdc`) into a Makie scene as a first-class atomic plot, rendered through
the path tracer alongside ordinary plots. The file is referenced, not parsed: it brings its
own geometry, payloads, relative textures, and self-contained materials (OmniPBR/MDL and
UsdPreviewSurface both render). `bind_usd!` then ties Julia `Observable`s to prims/attributes
*inside* that file, so an observable update live-updates the render.

```julia
using OmniverseMakie
p = usdplot!(ax, "assets/car.usdc"; bbox = Rect3f(Point3f(-260, -105, 0), Vec3f(520, 210, 150)),
             up = :z)

translate!(p, 0, 0, 100)        # ordinary Makie transforms drive the asset's ROOT transform
p.visible[] = false             # ordinary visibility

wheel = Observable(Makie.rotationmatrix_z(0f0))
bind_usd!(p, "/Chassis/WheelFL", wheel)                    # a prim → its omni:xform (a 4×4 matrix)
bind_usd!(p, "/Body.primvars:displayColor", color_obs)     # an attribute → a typed write
wheel[] = Makie.rotationmatrix_z(0.6f0)                     # live update, no re-author
```

Three rules worth knowing:

- **Bind paths are relative to the file's `defaultPrim`.** A reference pulls in the file's
  `defaultPrim` subtree, so `bind_usd!(p, "/Arm/Geo.primvars:displayColor", …)` addresses
  `Arm/Geo` *under* that prim. Targets split at the first `.` into a prim path (no dot → a
  4×4-matrix transform binding) and an attribute name (`Real` → float, a 3-vector / RGB
  `Colorant` → color3f, a `Vector` of those → the array form). `xformOp:*` targets are
  refused — those are baked at load; bind the **prim** with a matrix instead. A bad target on
  a displayed plot throws immediately (fail-fast); before display it warns and skips.
- **Makie owns the asset's root transform.** `translate!`/`scale!`/`rotate!` on the plot write
  the referenced root's `omni:xform`, *replacing* the file's own root transform. Interior prim
  transforms are untouched. Author units differ per asset — a centimetre `metersPerUnit = 0.01`
  export is your `Makie.scale!`. Pass `up = :y` for a Y-up DCC export (folds a +90° X rotation
  in so it stands upright in the Z-up scene).
- **Needs the ovrtx backend.** `usdplot` renders through `Screen` / `colorbuffer` /
  `interactive_display` / `replace_scene!`. In a plain GLMakie window it renders nothing.

A runnable showcase — the NVIDIA Kit "Zeus ZS300" sedan with all four wheels spun live and
recorded to an `.mp4` — is [`examples/usdplot_zeus_wheels.jl`](examples/usdplot_zeus_wheels.jl).

### Environment lighting: `push_environment_image!` + backgrounds

Image-based lighting through a `UsdLux DomeLight` whose latlong (equirectangular) map you can
set — and **live-replace** — at any time:

```julia
img = fill(RGBf(0.8, 0.9, 1.0), 256, 512)               # any Matrix{<:Colorant}
scene = Scene(lights = [EnvironmentLight(1.0, img)])     # honored at display time
screen = OmniverseMakie.Screen(scene; background = :domelight)  # map also shows as background

push_environment_image!(screen, other_img)               # live swap (Matrix — LDR, clamped)
push_environment_image!(screen, "studio.exr")            # or a file path — true HDR radiance
```

A matrix source is written to a temp PNG (components clamped to `[0,1]`); pass an `.exr`/`.hdr`
**file path** for real HDR. Swaps use the same proven remove+re-reference mechanism as volume
reloads, so in `accumulate_across_frames` mode a swap resets accumulation exactly once. The
`background` screen option selects `:default`, `:domelight` (pin the env map as the visible
background), or `:sky` — note the procedural sky is authored but **not rendered by standalone
ovrtx** (a Kit-runtime feature, like volume colormap colors; a warning says so).

OmniPBR materials also gained UV-projection tiling — textures without hand-authored UVs:

```julia
mesh!(ax, floor_rect; color = grass_texture,
      material = (; project_uvw = true, world_or_object = true, texture_scale = (8, 8)))
```

---

## Feature status

Milestone numbering follows [`ARCHITECTURE.md`](ARCHITECTURE.md) §9; the actual shipped
surface is exercised by [`test/runtests.jl`](test/runtests.jl).

| Milestone | Capability | Status |
|---|---|---|
| **M0** | `LibOVRTX` `ccall` binding; native init → open USD → RT2 step → framebuffer readback | shipped |
| **M1** | Static translation: `Screen`, `mesh`/`meshscatter`/`scatter`/`surface`/`lines`, camera, lights; `save` / `record` / `colorbuffer` | shipped |
| **M2** | `ComputePipeline` `:ovrtx_renderobject` diff path: live camera / light / attribute / transform / color edits; hot-path map & array bindings; leak-free `insert!` / `delete!` / `empty!` | shipped |
| **M3** | Materials: OmniPBR (metallic / roughness / opacity), image textures, OmniGlass refraction, live material edits | shipped |
| **M4** | Colormaps (scatter / lines / mesh), surface textures, and the 14-scene example gallery | shipped |
| **M5** | Interactive GLMakie viewport (CPU blit), event injection, `cam3d!` orbit, on-demand progressive loop, dynamic add/delete | shipped |
| **M6.A** | GPU-direct CUDA↔GL blit (HDR viewport, on-device tonemap; no CPU roundtrip) | shipped |
| **M6.B** | Native ray-query picking + offscreen `select!` selection outline | shipped (live in-viewport outline deferred) |
| **Volumes M1** | NVIDIA IndeX enablement (carb-token) + `author_vdb_volume!` (UsdVol / OpenVDB) | shipped |
| **Volumes M2** | Dense-array `volume!(x, y, z, array)` + live data edits — **grayscale**¹ | shipped |
| **usdplot** | `usdplot!` external USD files (`.usda`/`.usdc`, payloads, textures, materials) as atomic plots + `bind_usd!` observable→prim/attribute bindings | shipped |
| **IBL** | `EnvironmentLight` / `push_environment_image!` (live-swappable DomeLight env map), `background = :domelight`, OmniPBR UV-projection tiling — procedural `:sky` authored but Kit-only² | shipped |

¹ **Volume colormaps render grayscale**, by design, not as a bug. In standalone `ovrtx`
the only bundled volume path is **NVIDIA IndeX Direct**, which renders scalar density as
grayscale and ignores the authored colormap; the color-compositing path lives in a Kit
extension that ships no library here. See the explanation and tripwire test in
[`test/volumes/plot_test.jl`](test/volumes/plot_test.jl) and the IndeX notes in
[`src/binding/index_config.jl`](src/binding/index_config.jl).

² **The procedural sky background is not rendered by standalone `ovrtx`** (verified against
both RT and PT render modes): `background = :sky` authors the correct
`omni:rtx:background:source:type` token — which a Kit/composite runtime honors — but renders
black here and warns once. `:domelight` works natively. Tripwire test in
[`test/offscreen/envlight_test.jl`](test/offscreen/envlight_test.jl).

---

## API surface

- **Exported:** `interactive_display`, `replace_scene!`, `usdplot` / `usdplot!` (place an
  external USD file), `bind_usd!` / `unbind_usd!` (tie observables to prims/attributes inside
  it), `push_environment_image!` (set / live-swap the environment-light map). (Every exported
  Makie name is re-exported too, so `using OmniverseMakie` gives you `Figure`, `mesh!`, `save`,
  etc.)
- **Documented but unexported** (qualify with `OmniverseMakie.`): `Screen`,
  `Makie.colorbuffer`, `select!` (selection outline, offscreen), `author_vdb_volume!`
  (low-level VDB authoring), `reset_accumulation!` (force an RT2 reset in accumulate mode).

---

## Examples, benchmarks, and layout

- **Example gallery** — [`examples/README.md`](examples/README.md): 14 ported RPRMakie
  scenes rendered end-to-end through OmniverseMakie. The gallery is a self-contained Pkg
  environment; [`examples/fetch_assets.jl`](examples/fetch_assets.jl) populates assets and
  [`examples/run_all.jl`](examples/run_all.jl) renders each scene (in an isolated
  subprocess) into [`examples/renders/`](examples/renders) with per-scene asserts.
- **Hot-path benchmark** — [`bench/hot_path.jl`](bench/hot_path.jl), results in
  [`bench/RESULTS.md`](bench/RESULTS.md): measures the per-frame map/array-binding
  throughput that gates interactive animation.
- **Architecture & design** — [`ARCHITECTURE.md`](ARCHITECTURE.md): the three-layer design
  (binding → translation → presentation), locked decisions, milestone plan, and the
  validation recipe.
- **Sub-packages** — [`lib/LibOVRTX/`](lib/LibOVRTX) (raw `ccall` bindings, `dlopen` +
  `OVRTX_LIBRARY_PATH`) and [`lib/NanoVDBWriter/`](lib/NanoVDBWriter) (pure-Julia
  dense-array → `.nvdb` writer; see its README for attribution).
