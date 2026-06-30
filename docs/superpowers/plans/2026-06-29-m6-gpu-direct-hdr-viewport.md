# M6.A — GPU-Direct HDR Viewport (+ extension packaging) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace M5's CPU host-roundtrip blit with a GPU-direct CUDA-GL path that maps ovrtx's `HdrColor` on-device and tonemaps it (live exposure) straight into the GLMakie texture; move GLMakie + CUDA into package extensions so plain `using OmniverseMakie` (offscreen) pulls neither.

**Architecture:** Main `OmniverseMakie` declares method-less generics (`interactive_display`, `present!`) + a pure-FFI `OV.map_cuda_array` + the shared `tonemap` math. A **GLMakie package extension** holds the M5 viewport (now reading `HdrColor` + a host tonemap) and the CPU `present!`. A **CUDA package extension** (triggered by CUDA *and* GLMakie) adds the GPU-direct `present!`: register the GL texture with CUDA once, then per-frame map `HdrColor` as a CUDA buffer, tonemap in a CUDA kernel, and copy into the texture with the map/unmap/event handshake.

**Tech Stack:** Julia 1.12 package extensions (`[weakdeps]`/`[extensions]`) · CUDA.jl (interop via `CUDA.CUDACore.cuGraphics*`, `@ccall libcuda` escape hatch) · GLMakie 0.13.12 · ovrtx `OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY` map mode · the merged M5 viewport.

## Global Constraints

- **Package extensions, not Requires.jl or in-main guards.** GLMakie + CUDA move to `[weakdeps]` + `[extensions]`: `OmniverseMakieGLMakieExt = "GLMakie"`, `OmniverseMakieCUDAExt = ["CUDA", "GLMakie"]`. Main `[deps]` loses GLMakie. Deps via Pkg, never hand-edited TOML — EXCEPT the `[weakdeps]`/`[extensions]` blocks, which Pkg cannot add and MUST be hand-edited (see Task 1).
- **Single-threaded render task.** All CUDA-GL interop + ovrtx steps run on GLMakie's `render_tick` task with the GL context current and CUDA's per-task context initialized there. NO separate render thread.
- **HDR source + shared tonemap → RGBA8.** Both blitters read `HdrColor`; one shared math `tonemap(rgb_linear, exposure) = sRGB_encode(ACES_filmic(exposure·rgb))` → RGBA8. The display texture stays RGBA8 (M5's `image!` display path unchanged).
- **Auto-select + CPU fallback.** `interactive_display(fig; gpu_direct=:auto)`: GPU-direct when the CUDA ext is loaded and `CUDA.functional()`, else CPU. `gpu_direct=true` forces it (error if CUDA unavailable); `false` forces CPU. Setup failure → CPU fallback for the session.
- **Per-frame tonemap kernel** (no resident kernel). CUDA graphs are a FUTURE optimization — do not build them now.
- **Reuse M5 verbatim:** `ViewportSession`, the `render_tick` loop, resize/teardown, and the per-frame error guard are on `main` and move (unchanged in behavior) into the GLMakie ext.
- **Display + CUDA env:** subprocess tests set `DISPLAY=:0`, `XAUTHORITY=/run/user/1000/.mutter-Xwaylandauth.QRQ4Q3`, `XDG_RUNTIME_DIR=/run/user/1000`, `OVRTX_LIBRARY_PATH=/home/juliahub/temp/omniverse-makie/references/ovrtx/examples/python/minimal/.venv/lib/python3.13/site-packages/ovrtx/bin/libovrtx-dynamic.so`. `test/helpers.jl` `run_ovrtx_subprocess` already forwards the X vars. CUDA tests need CUDA.jl + a functional GPU (the A5000 — spike: `CUDA.functional() == true`).
- **The spike is the source of truth for CUDA-GL specifics:** `/home/juliahub/temp/omniverse-makie/references/notes/cuda-gl-interop.md` §1 (CUDA.jl interop symbols + the call sequence), §2 (GL texture id + threading), §3 (ovrtx CUDA-array map + sync), §4 (the per-frame pipeline + pitfalls). The reference C implementation is `references/ovrtx/examples/c/vulkan-interop/src/main.cpp:921-969` + `.../src/cuda/cuda_kernel.cpp:503-562`. Where this plan marks "verify against the spike/C example," confirm the exact API in a REPL before relying on it (a genuine new-dep unknown, as with M5's GLMakie API).
- **Commit trailer:** `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. **Do not push** (commit on a `feat/m6-gpu-direct` branch; the user controls merges).

---

## File Structure

```
src/OmniverseMakie.jl       # MODIFY: drop interactive/* includes + `import GLMakie`; declare interactive_display/present! generics; include tonemap.jl
src/tonemap.jl              # CREATE: shared tonemap math (ACES + sRGB + exposure) — pure, no GLMakie/CUDA
src/binding/OV.jl           # MODIFY: render_to_matrix gains a `var` kwarg; ADD map_cuda_array + unmap_cuda
src/interactive/            # (the 3 files move out — see below)
ext/OmniverseMakieGLMakieExt.jl  # CREATE: the M5 viewport (moved blit.jl/viewport.jl/camera_loop.jl) + CPU present!
ext/OmniverseMakieCUDAExt.jl     # CREATE: GPU-direct present! (register/map/kernel/copy/unmap/sync)
Project.toml               # MODIFY (hand-edit the weakdeps/extensions blocks): GLMakie [deps]→[weakdeps]; add CUDA [weakdeps]; [extensions]
test/m6_tonemap_test.jl    # CREATE: host tonemap unit + host-vs-kernel agreement (subprocess, CUDA)
test/m6_map_cuda_test.jl   # CREATE: OV.map_cuda_array returns a valid CUarray + wait_event (subprocess, CUDA)
test/m6_ext_load_test.jl   # CREATE: extension-loading matrix (offscreen / +GLMakie / +CUDA)
test/m6_gpu_blit_test.jl   # CREATE: GPU-direct present! → non-black + matches CPU (subprocess, CUDA+GL)
test/m6_bench_test.jl      # CREATE: GPU-direct vs CPU blit latency, gate at 4K (subprocess, CUDA+GL)
```

---

### Task 1: Extension repackaging — move the M5 viewport into a GLMakie extension

**Files:**
- Modify: `Project.toml`, `src/OmniverseMakie.jl`
- Create: `ext/OmniverseMakieGLMakieExt.jl`
- Move: `src/interactive/blit.jl`, `src/interactive/viewport.jl`, `src/interactive/camera_loop.jl` → into the ext
- Test: `test/m6_ext_load_test.jl`

**Interfaces:**
- Produces: method-less generics `interactive_display(fig_or_scene; size, steps_per_tick, gpu_direct, exposure)` and `present!(session)` in the main module (the GLMakie ext adds methods; the CUDA ext adds a `present!` method). Calling `interactive_display` with no GLMakie loaded errors helpfully.

- [ ] **Step 1: Write the failing extension-load test** (`test/m6_ext_load_test.jl`)

```julia
using Test
# Offscreen-only: using OmniverseMakie alone must NOT require GLMakie/CUDA, and
# interactive_display must exist but error helpfully until GLMakie is loaded.
const _M6_OFFSCREEN_PROG = """
using OmniverseMakie
println("HAS_INTERACTIVE=", isdefined(OmniverseMakie, :interactive_display))
# A 2-D offscreen render still works with no GLMakie:
scene = Scene(size=(64,64)); cam3d!(scene)
mesh!(scene, Rect3f(Point3f(0), Vec3f(1)); color=:red)
ok = false
try
    OmniverseMakie.interactive_display(scene)   # no GLMakie ext loaded
catch e
    global ok = occursin("GLMakie", sprint(showerror, e))
end
println("ERRORS_WITHOUT_GLMAKIE=", ok)
println("OK_OFFSCREEN")
"""
include("helpers.jl")
@testset "M6 offscreen load (no GLMakie/CUDA needed)" begin
    exitcode, output = run_ovrtx_subprocess(_M6_OFFSCREEN_PROG; timeout=300)
    @info "M6 offscreen output" output
    @test exitcode == 0
    @test contains(output, "HAS_INTERACTIVE=true")
    @test contains(output, "ERRORS_WITHOUT_GLMAKIE=true")
    @test contains(output, "OK_OFFSCREEN")
end
```

- [ ] **Step 2: Run it — RED** (today `interactive_display` only exists when the interactive includes load `using GLMakie`-style; and the offscreen subprocess currently DOES load GLMakie via the module).

Run (export the env vars from Global Constraints first):
`julia --project=. -e 'using Test; include("test/helpers.jl"); include("test/m6_ext_load_test.jl")'`

- [ ] **Step 3: Declare the generics + drop the interactive includes** (`src/OmniverseMakie.jl`). Replace lines 4 (`import GLMakie`) and 22–27 (the `interactive/*` includes + `export interactive_display`) with:

```julia
# (remove `import GLMakie` entirely — GLMakie is now a weakdep, used only in the extension)

# ... keep includes through screen.jl ...
include("tonemap.jl")            # shared HDR tonemap math (Task 2)

# M5/M6 interactive viewport lives in package extensions (GLMakie / CUDA). The main
# module only DECLARES the generics; the GLMakie ext adds the methods.
function interactive_display end
function present! end
export interactive_display
```

- [ ] **Step 4: Add a helpful stub** so calling without GLMakie errors clearly. After the `function present! end` line:

```julia
# Errors helpfully when no GLMakie extension is loaded (no method otherwise).
interactive_display(::Any; kwargs...) =
    error("interactive_display requires GLMakie — run `using GLMakie` (and `using CUDA` for GPU-direct).")
```
(The GLMakie ext defines a MORE specific method `interactive_display(fig_or_scene::Union{Makie.Figure,Makie.Scene}; …)` that wins dispatch when loaded.)

- [ ] **Step 5: Move the three interactive files into the GLMakie extension.** `git mv src/interactive/blit.jl src/interactive/viewport.jl src/interactive/camera_loop.jl` aside (or delete after copying), and create `ext/OmniverseMakieGLMakieExt.jl` that wraps their CURRENT content in a module. The bodies are UNCHANGED except: they now reference main-module internals via the `import` list below (today they are in `module OmniverseMakie` so the names are bare; in the ext they must be imported).

```julia
module OmniverseMakieGLMakieExt

using OmniverseMakie, GLMakie
using OmniverseMakie: Makie, RGBA, N0f8, ColorTypes
# main-module internals the moved code uses (today as bare names):
import OmniverseMakie: interactive_display, present!,
    Screen, OV, _author_screen!, _sync_and_needs_reset!, _camera_snapshot,
    _lights_snapshot, _scene_for_camera, author_root_from_scene!, tonemap
using Makie: Consume, MouseButtonEvent  # event types used by the input forwarders

# ===== moved verbatim from src/interactive/blit.jl =====
#   _orient_for_display, cpu_blit!
# ===== moved verbatim from src/interactive/viewport.jl =====
#   mutable struct ViewportSession (+ its fields), _M5_FORWARDED_EVENTS,
#   interactive_display(fig_or_scene; ...), Base.close(::ViewportSession), resize_viewport!
# ===== moved (and adapted in Task 2) from src/interactive/camera_loop.jl =====
#   _M5_STEP_TIMEOUT_NS, _on_render_tick_impl!, on_render_tick!

end # module
```
Define `ViewportSession` IN the ext (it is GLMakie-only). The `Screen`/`OV`/helpers stay in the main module and are imported. Run `julia --project=. -e 'using OmniverseMakie'` and confirm it loads with NO GLMakie (the `interactive/` includes are gone).

- [ ] **Step 6: Hand-edit `Project.toml`** — move GLMakie to weakdeps + declare the extensions. Pkg cannot author these blocks, so edit by hand. Remove `GLMakie = "e9467ef8-..."` from `[deps]`, then add:

```toml
[weakdeps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
GLMakie = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"

[extensions]
OmniverseMakieGLMakieExt = "GLMakie"
OmniverseMakieCUDAExt = ["CUDA", "GLMakie"]
```
Keep `GLMakie = "=0.13.12"` in `[compat]`; add `CUDA = "5"` to `[compat]`. (Leave `ext/OmniverseMakieCUDAExt.jl` as a one-line stub `module OmniverseMakieCUDAExt end` for now so Pkg's extension check passes; Task 4 fills it.) Resolve: `julia --project=. -e 'using Pkg; Pkg.resolve()'`.

- [ ] **Step 7: Move the M5 interactive tests' env expectation.** The existing `test/m5_*` tests do `using OmniverseMakie, GLMakie` in their subprocess progs — they already load GLMakie explicitly, so they exercise the ext unchanged. No test edit needed; confirm in Step 8.

- [ ] **Step 8: Run the tests — GREEN.** Register `m6_ext_load_test.jl` in `test/runtests.jl`. Run the new test (offscreen) + one M5 viewport test (now via the ext):
`julia --project=. -e 'using Test; include("test/helpers.jl"); include("test/m6_ext_load_test.jl"); include("test/m5_viewport_test.jl")'`
Expected: offscreen errors-without-GLMakie ✓; the M5 viewport test still renders via the ext ✓.

- [ ] **Step 9: Commit**

```bash
git add Project.toml Manifest.toml src/OmniverseMakie.jl ext/OmniverseMakieGLMakieExt.jl ext/OmniverseMakieCUDAExt.jl test/m6_ext_load_test.jl test/runtests.jl
git rm src/interactive/blit.jl src/interactive/viewport.jl src/interactive/camera_loop.jl
git commit -m "refactor(M6.A): move M5 viewport into a GLMakie package extension

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

**Acceptance:** `using OmniverseMakie` pulls no GLMakie/CUDA and renders offscreen; `interactive_display` errors helpfully until `using GLMakie`; the M5 viewport works unchanged via the ext.

---

### Task 2: HDR source + shared host tonemap (CPU path)

**Files:**
- Create: `src/tonemap.jl`
- Modify: `src/binding/OV.jl` (parametrize `render_to_matrix` by render var), `ext/OmniverseMakieGLMakieExt.jl` (the CPU `present!` + the `exposure` field)
- Test: `test/m6_tonemap_test.jl` (host unit part)

**Interfaces:**
- Consumes: `ViewportSession` (Task 1).
- Produces: `tonemap(rgb::NTuple{3,Float32}, exposure::Float32) -> RGBA{N0f8}` and a broadcast `tonemap_frame!(out::Matrix{RGBA{N0f8}}, hdr::Array{Float32,3}, exposure)`; `OV.render_hdr_to_array(r, product; warmup, timeout_ns) -> Array{Float32,3}` ([C,W,H] float); `present!(session)` CPU method; `session.exposure::Float32`.

- [ ] **Step 1: Write the failing host-tonemap unit test** (`test/m6_tonemap_test.jl`, parent process — pure math, no GPU)

```julia
using Test, OmniverseMakie
using OmniverseMakie: tonemap
@testset "M6 host tonemap (ACES + sRGB + exposure)" begin
    # black → black; mid-grey monotonic; exposure brightens.
    @test tonemap((0f0,0f0,0f0), 0f0) == RGBA{N0f8}(0,0,0,1)
    g1 = tonemap((0.18f0,0.18f0,0.18f0), 0f0)
    g2 = tonemap((0.18f0,0.18f0,0.18f0), 1f0)   # +1 stop
    @test Float32(g2.r) > Float32(g1.r)         # more exposure → brighter
    @test Float32(tonemap((10f0,10f0,10f0), 0f0).r) ≈ 1f0 atol=0.02  # highlights clamp near 1
    @test eltype(tonemap((1f0,0f0,0f0),0f0)) == N0f8
end
```

- [ ] **Step 2: Run it — RED** (`tonemap` undefined).

- [ ] **Step 3: Implement the shared tonemap** (`src/tonemap.jl`). ACES filmic (Narkowicz fit) + exposure (a power-of-two stop) + sRGB encode. Pure `Float32`, no GLMakie/CUDA — so the CUDA kernel (Task 4) reuses the SAME scalar functions.

```julia
# Shared HDR → display tonemap.  Pure Float32 scalar math so BOTH the CPU host path
# (Task 2) and the CUDA kernel (Task 4) call the identical functions.
@inline _aces(x::Float32) = clamp((x*(2.51f0x+0.03f0)) / (x*(2.43f0x+0.59f0)+0.14f0), 0f0, 1f0)
@inline _srgb(c::Float32) = c <= 0.0031308f0 ? 12.92f0c : 1.055f0*c^(1f0/2.4f0) - 0.055f0
@inline _u8(c::Float32)   = N0f8(clamp(c, 0f0, 1f0))

"""
    tonemap(rgb::NTuple{3,Float32}, exposure::Float32) -> RGBA{N0f8}

`sRGB( ACES( 2^exposure · rgb ) )`.  `exposure` is in stops (EV); 0 = no change.
"""
@inline function tonemap(rgb::NTuple{3,Float32}, exposure::Float32)
    s = exp2(exposure)
    RGBA{N0f8}(_u8(_srgb(_aces(s*rgb[1]))), _u8(_srgb(_aces(s*rgb[2]))), _u8(_srgb(_aces(s*rgb[3]))), N0f8(1))
end

# Broadcast a [C,W,H] float HDR buffer (channel-fastest, as map_cpu/map_cuda return)
# into an [H,W] RGBA{N0f8} display matrix, top-left origin (matches render_to_matrix).
function tonemap_frame(hdr::AbstractArray{Float32,3}, exposure::Float32)
    C, W, H = size(hdr)
    out = Matrix{RGBA{N0f8}}(undef, H, W)
    @inbounds for j in 1:W, i in 1:H
        out[i, j] = tonemap((hdr[1,j,i], hdr[2,j,i], hdr[3,j,i]), exposure)
    end
    return out
end
```
`include("tonemap.jl")` is already added in Task 1 Step 3.

- [ ] **Step 4: Run it — GREEN.** `julia --project=. -e 'using Test; include("test/m6_tonemap_test.jl")'` → PASS.

- [ ] **Step 5: Add the HDR render-var path to OV.jl.** `render_to_matrix` hardcodes `"LdrColor"` (src/binding/OV.jl). Add a sibling that returns the raw float HDR array (the CPU `present!` tonemaps it). VERIFY against `map_cpu`'s decode that `HdrColor` is float `[C,W,H]` (its `dtype` is `kDLFloat/32`, not `kDLUInt/8`) — confirm in a REPL: map `HdrColor` on CPU and check `eltype`/shape. Then:

```julia
# Map a render var on CPU as a float [C,W,H] array (HdrColor is kDLFloat/32).
function map_cpu_f32(sr::StepResult, name::AbstractString)
    sr.r.alive || error("map_cpu_f32 on a closed Renderer")
    outs = Ref{LibOVRTX.ovrtx_render_product_set_outputs_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_fetch_results(sr.r.ptr, sr.handle, _TIMEOUT_INFINITE, outs), "fetch_results")
    h = _find_var(outs[], name)
    mdesc = Ref(LibOVRTX.ovrtx_map_output_description_t(LibOVRTX.OVRTX_MAP_DEVICE_TYPE_CPU, Csize_t(0)))
    ro = Ref{LibOVRTX.ovrtx_render_var_output_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_map_render_var_output(sr.r.ptr, h, mdesc, _TIMEOUT_INFINITE, ro), "map_render_var_output")
    t0 = unsafe_load(ro[].tensors, 1); dlt = unsafe_load(t0.dl)
    H = Int(unsafe_load(dlt.shape,1)); W = Int(unsafe_load(dlt.shape,2)); C = Int(unsafe_load(dlt.shape,3))
    raw = unsafe_wrap(Array, Ptr{Float32}(dlt.data), (C, W, H); own=false)
    pixels = copy(raw)
    LibOVRTX.ovrtx_unmap_render_var_output(sr.r.ptr, ro[].map_handle, LibOVRTX.NOSYNC)
    return (pixels, W, H)   # [C,W,H] Float32
end

function render_hdr_to_array(r::Renderer, product::AbstractString; warmup::Int=64, timeout_ns::UInt64=_TIMEOUT_INFINITE_NS)
    for s in 1:(warmup-1); sr = step!(r, product; timeout_ns); close(sr); end
    sr = step!(r, product; timeout_ns)
    pixels, W, H = try map_cpu_f32(sr, "HdrColor") finally close(sr) end
    return pixels  # [C,W,H] Float32
end
```
(`_TIMEOUT_INFINITE` is the bare struct const already in OV.jl; `_TIMEOUT_INFINITE_NS` is the UInt64.)

- [ ] **Step 6: Switch the CPU `present!` to HDR + tonemap** (in `ext/OmniverseMakieGLMakieExt.jl`). Add `exposure::Float32` to `ViewportSession` (default `0f0`, threaded from `interactive_display(...; exposure=0f0)`). Rename the M5 `_on_render_tick_impl!`'s blit to go through `present!`:

```julia
# CPU blitter: render HdrColor → host tonemap → image! data.
function OmniverseMakie.present!(session::ViewportSession)
    hdr = OV.render_hdr_to_array(session.screen.renderer, session.screen.product;
                                 warmup = session.steps_per_tick, timeout_ns = _M5_STEP_TIMEOUT_NS)
    frame = tonemap_frame(hdr, session.exposure)          # [H,W] RGBA{N0f8}
    cpu_blit!(session.image_plot, frame)
    return nothing
end
```
In `_on_render_tick_impl!`, replace the `render_to_matrix(...) + cpu_blit!(...)` lines (camera_loop.jl:43–49) with `present!(session); session.samples += session.steps_per_tick`. The reset/sync logic above it is unchanged. The initial frame in `interactive_display` likewise builds via the HDR path (call `present!(session)` once after the listener is registered, or seed the `image!` from `tonemap_frame(render_hdr_to_array(...), exposure)`).

- [ ] **Step 7: Run M5 viewport + camera-loop tests — GREEN.** They now exercise the HDR CPU path. The non-black assertions still hold (a lit scene tonemaps to non-black). Update any M5 test that asserted exact `LdrColor` bytes (there are none — they assert non-black / sample counts).
`julia --project=. -e 'using Test; include("test/helpers.jl"); include("test/m5_viewport_test.jl"); include("test/m5_camera_loop_test.jl"); include("test/m6_tonemap_test.jl")'`

- [ ] **Step 8: Commit**

```bash
git add src/tonemap.jl src/binding/OV.jl ext/OmniverseMakieGLMakieExt.jl test/m6_tonemap_test.jl test/runtests.jl
git commit -m "feat(M6.A): HDR source + shared ACES tonemap with live exposure (CPU path)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

**Acceptance:** the CPU viewport displays the tonemapped `HdrColor`; raising `exposure` brightens the image; the host tonemap is unit-tested.

---

### Task 3: `OV.map_cuda_array` — map a render var as a CUDA array

**Files:**
- Modify: `src/binding/OV.jl`
- Test: `test/m6_map_cuda_test.jl` (subprocess, CUDA)

**Interfaces:**
- Produces: `OV.map_cuda_array(sr::StepResult, name="HdrColor") -> (data::Ptr{Cvoid}, W::Int, H::Int, C::Int, map_handle, wait_event::Csize_t)` and `OV.unmap_cuda(sr, map_handle; done_event=Csize_t(0))`. Returns RAW handles (no CUDA.jl in the main module); the CUDA ext (Task 4) wraps them.

- [ ] **Step 1: VERIFY the CUDA-array map shape against the C example** (REPL + the spike). The map mode is `LibOVRTX.OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY` (= 3, confirmed). The reference is `references/ovrtx/examples/c/vulkan-interop/src/main.cpp:921-969`: `ovrtx_map_render_var_output(..., CUDA_ARRAY, &out)` → `out.tensors[0].dl->data` is a **`CUarray`** and `out.cuda_sync.wait_event` is a `CUevent`. Confirm in a REPL: build a Screen, step, and call `ovrtx_map_render_var_output` with `OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY`; inspect `ro[].cuda_sync.wait_event` (the `ovrtx_render_var_output_t.cuda_sync` field, libovrtx_api.jl:981) and `unsafe_load(t0.dl).data`. Record the exact field path.

- [ ] **Step 2: Write the failing test** (`test/m6_map_cuda_test.jl`)

```julia
using Test
const _M6_MAPCUDA_PROG = """
using OmniverseMakie, CUDA
const OV = OmniverseMakie.OV
OM = OmniverseMakie
OM.activate!(warmup = 8)
scene = Scene(size=(96,96)); cam3d!(scene)
mesh!(scene, Rect3f(Point3f(0), Vec3f(1)); color=:red)
screen = OM.Screen(scene)
OM.author_root_from_scene!(screen, scene; resolution = screen.fb_size)
sr = OV.step!(screen.renderer, screen.product; timeout_ns = UInt64(60_000_000_000))
data, W, H, C, mh, wait_event = OV.map_cuda_array(sr, "HdrColor")
println("CUDA_ARRAY_OK=", data != C_NULL, " W=", W, " H=", H, " C=", C)
OV.unmap_cuda(sr, mh)
close(sr); close(screen)
println("OK_MAP_CUDA")
"""
include("helpers.jl")
@testset "M6 OV.map_cuda_array (subprocess, CUDA)" begin
    exitcode, output = run_ovrtx_subprocess(_M6_MAPCUDA_PROG; timeout=400)
    @info "M6 map_cuda output" output
    @test exitcode == 0
    @test contains(output, "CUDA_ARRAY_OK=true")
    @test contains(output, "OK_MAP_CUDA")
end
```

- [ ] **Step 3: Run it — RED** (`map_cuda_array` undefined).

- [ ] **Step 4: Implement `map_cuda_array` + `unmap_cuda`** (`src/binding/OV.jl`), modeled on `map_cpu` but mode `CUDA_ARRAY`, no copy, returning the raw `CUarray` ptr + the wait event. Use the field path confirmed in Step 1:

```julia
# Map a render var as a CUDA array (device-resident).  Returns RAW handles — the CUDA
# extension wraps `data` as a CUarray and `wait_event` as a CUevent.  Caller MUST
# unmap_cuda when done (gated on a done-event so ovrtx doesn't reclaim mid-copy).
function map_cuda_array(sr::StepResult, name::AbstractString="HdrColor")
    sr.r.alive || error("map_cuda_array on a closed Renderer")
    outs = Ref{LibOVRTX.ovrtx_render_product_set_outputs_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_fetch_results(sr.r.ptr, sr.handle, _TIMEOUT_INFINITE, outs), "fetch_results")
    h = _find_var(outs[], name)
    mdesc = Ref(LibOVRTX.ovrtx_map_output_description_t(LibOVRTX.OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY, Csize_t(0)))
    ro = Ref{LibOVRTX.ovrtx_render_var_output_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_map_render_var_output(sr.r.ptr, h, mdesc, _TIMEOUT_INFINITE, ro), "map_render_var_output(cuda_array)")
    t0 = unsafe_load(ro[].tensors, 1); dlt = unsafe_load(t0.dl)
    H = Int(unsafe_load(dlt.shape,1)); W = Int(unsafe_load(dlt.shape,2)); C = Int(unsafe_load(dlt.shape,3))
    return (dlt.data, W, H, C, ro[].map_handle, ro[].cuda_sync.wait_event)
end

# Unmap a CUDA-array map.  `done_event` (a CUevent cast to Csize_t) gates the unmap so
# ovrtx waits for our copy before reclaiming the buffer (spike §3 step 5).
function unmap_cuda(sr::StepResult, map_handle; done_event::Csize_t = Csize_t(0))
    sync = LibOVRTX.ovrtx_cuda_sync_t(done_event, #= other fields zero — VERIFY the struct in Step 1 =#)
    sr.r.alive && LibOVRTX.ovrtx_unmap_render_var_output(sr.r.ptr, map_handle, sync)
    return nothing
end
```
VERIFY `ovrtx_cuda_sync_t`'s full field list (libovrtx_api.jl:517) in Step 1 and construct it correctly (the `done_event::Csize_t` is the CUevent handle; any other fields zero).

- [ ] **Step 5: Run it — GREEN.** Register in `runtests.jl`. Re-run Step 2's command → `CUDA_ARRAY_OK=true`, `OK_MAP_CUDA`.

- [ ] **Step 6: Commit**

```bash
git add src/binding/OV.jl test/m6_map_cuda_test.jl test/runtests.jl
git commit -m "feat(M6.A): OV.map_cuda_array — map a render var as a CUDA array (raw handles)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

**Acceptance:** `OV.map_cuda_array("HdrColor")` returns a non-null `CUarray` + dims + a wait event; `unmap_cuda` releases it.

---

### Task 4: CUDA extension — GPU-direct `present!` + tonemap kernel + auto-selection

**Files:**
- Create: `ext/OmniverseMakieCUDAExt.jl` (replace the Task-1 stub)
- Modify: `ext/OmniverseMakieGLMakieExt.jl` (the `gpu_direct` selection in `interactive_display`)
- Test: `test/m6_gpu_blit_test.jl` (subprocess, CUDA+GL); extend `test/m6_tonemap_test.jl` (host-vs-kernel agreement)

**Interfaces:**
- Consumes: `OV.map_cuda_array`/`unmap_cuda` (Task 3); `tonemap` scalar funcs (Task 2); `ViewportSession` (Task 1); the spike §1 CUDA-GL symbols.
- Produces: an `OmniverseMakie.present!(session)` method active when CUDA+GLMakie are loaded and the session was built with GPU-direct; a `_gpu_blit_state` cached on the session.

- [ ] **Step 1: VERIFY the CUDA-GL interop symbols + the GL texture id** (REPL + spike §1/§2). Confirm `CUDA.CUDACore.cuGraphicsGLRegisterImage / cuGraphicsMapResources / cuGraphicsSubResourceGetMappedArray / cuGraphicsUnmapResources / cuGraphicsUnregisterResource / cuMemcpy2D_v2 / cuStreamWaitEvent / cuEventRecord` exist (spike §6 confirmed they do in CUDA.jl 6.2.0), and that the `image!` plot's GL texture id is reachable via `GLMakie.plot2robjs(glscreen, image_plot)` → the image uniform `Texture.id`/`.texturetype` (spike §2). Record the exact texture-access path.

- [ ] **Step 2: Write the failing host-vs-kernel agreement test** (extend `test/m6_tonemap_test.jl`, subprocess CUDA)

```julia
const _M6_KERNEL_PROG = """
using OmniverseMakie, CUDA
using OmniverseMakie: tonemap
# Same HDR input, host vs the CUDA tonemap kernel → identical RGBA8.
hdr = Float32[c==1 ? 0.5f0 : (c==2 ? 0.1f0 : 2.0f0) for c in 1:4, x in 1:8, y in 1:6]  # [C,W,H]
host = OmniverseMakie.tonemap_frame(hdr, 0.5f0)
dev  = OmniverseMakieCUDAExt.tonemap_kernel_to_matrix(CuArray(hdr), 0.5f0)  # test-only helper
nmismatch = count(i -> host[i] != dev[i], eachindex(host))
println("KERNEL_MISMATCH=", nmismatch)
println("OK_KERNEL")
"""
@testset "M6 host vs CUDA-kernel tonemap agreement (subprocess, CUDA)" begin
    exitcode, output = run_ovrtx_subprocess(_M6_KERNEL_PROG; timeout=400)
    @info "M6 kernel output" output
    @test exitcode == 0
    @test contains(output, "KERNEL_MISMATCH=0")
end
```

- [ ] **Step 3: Run it — RED** (the CUDA ext + the kernel helper don't exist).

- [ ] **Step 4: Implement the CUDA tonemap kernel + the GPU-direct `present!`** (`ext/OmniverseMakieCUDAExt.jl`). The kernel reuses the SAME scalar `tonemap`. Per Global Constraints, tonemap the HDR `CuArray` into a linear RGBA8 device buffer via a CUDA.jl broadcast (open item #5 option (a)), then `cuMemcpy2D` that into the registered GL texture array. Setup registers the texture once.

```julia
module OmniverseMakieCUDAExt
using OmniverseMakie, CUDA, GLMakie
using OmniverseMakie: tonemap, OV, present!
import OmniverseMakie: present!
const CC = CUDA.CUDACore

# Per-session GPU-direct state (registered GL texture resource + the CUDA stream/event).
mutable struct GPUBlitState
    res::Base.RefValue{CC.CUgraphicsResource}   # registered GL texture
    registered::Bool
    copy_done                                   # CUevent (created lazily)
end

# Tonemap an [C,W,H] HDR CuArray → [H,W] RGBA{N0f8} CuArray, applying the SHARED tonemap.
function _tonemap_dev(hdr::CuArray{Float32,3}, exposure::Float32)
    C, W, H = size(hdr)
    out = CuArray{RGBA{N0f8}}(undef, H, W)
    @inbounds CUDA.@. out = tonemap_tuple(hdr, exposure)   # broadcast kernel — see helper below
    return out
end

# present!: step → map HdrColor as CUDA array → tonemap on device → copy into the GL texture
# array → unmap (event-gated) → stream sync.  VERIFY each cuGraphics* call against spike §1/§4.
function OmniverseMakie.present!(session, ::Val{:gpu})   # method selected via the session's strategy
    # ... register the texture once (Step 1's path) ...
    # sr = OV.step!(...); (data,W,H,C,mh,wait_event) = OV.map_cuda_array(sr,"HdrColor")
    # wrap `data` as a CuArray{Float32,3} (CC.CUarray → unsafe_wrap), tonemap → rgba8 CuArray
    # cuGraphicsMapResources → cuGraphicsSubResourceGetMappedArray(dst)
    # cuStreamWaitEvent(stream, wait_event)
    # cuMemcpy2D_v2(rgba8 → dst, WidthInBytes=W*4, Height=H)
    # cuGraphicsUnmapResources; cuEventRecord(copy_done); OV.unmap_cuda(sr, mh; done_event=copy_done)
    # cuStreamSynchronize(stream); close(sr)
    return nothing
end
end # module
```
This step is the spike's §4 pipeline transcribed with CUDA.jl; build it incrementally in a REPL, verifying each `cuGraphics*`/`cuMemcpy2D` call (the genuine new-dep unknowns). Provide a small `tonemap_kernel_to_matrix(hdr_cuarray, exposure) -> Matrix{RGBA{N0f8}}` test helper (device tonemap + copy back to host) for Step 2.

- [ ] **Step 5: Wire `gpu_direct` selection in `interactive_display`** (`ext/OmniverseMakieGLMakieExt.jl`). Add the kwarg + the strategy resolution:

```julia
function _pick_blitter(gpu_direct::Symbol)
    cuda_ready = false
    try
        cuda_ready = Base.get_extension(OmniverseMakie, :OmniverseMakieCUDAExt) !== nothing &&
                     Base.invokelatest(OmniverseMakie._cuda_functional)   # defined in the CUDA ext
    catch; end
    gpu_direct === :auto ? (cuda_ready ? :gpu : :cpu) :
    gpu_direct === true  ? (cuda_ready ? :gpu : error("gpu_direct=true but CUDA is unavailable")) : :cpu
end
```
`interactive_display(...; gpu_direct=:auto)` stores `session.blitter = _pick_blitter(gpu_direct)`; the loop calls `present!(session, Val(session.blitter))` (CPU method = the Task-2 `present!`, now `present!(session, ::Val{:cpu})`). The CUDA ext defines `OmniverseMakie._cuda_functional() = CUDA.functional()`. On GPU-setup failure, `@warn` once and set `session.blitter = :cpu`.

- [ ] **Step 6: Write the GPU-direct blit test** (`test/m6_gpu_blit_test.jl`)

```julia
using Test
const _M6_GPUBLIT_PROG = """
using OmniverseMakie, GLMakie, CUDA
OM = OmniverseMakie
OM.activate!(warmup = 24)
fig = Figure(); ax = LScene(fig[1,1]; show_axis=false)
mesh!(ax, Rect3f(Point3f(-1), Vec3f(2)); color = :orange)
session = OM.interactive_display(fig; size=(320,240), gpu_direct=true)   # force GPU-direct
buf = GLMakie.colorbuffer(session.glscreen)
nb = count(c -> (Float32(red(c))+Float32(green(c))+Float32(blue(c)))>0.1, buf)
println("GPU_NONBLACK=", nb)
@assert nb > 1000 "GPU-direct viewport black"
OM.close(session)
println("OK_GPU_BLIT")
"""
include("helpers.jl")
@testset "M6 GPU-direct blit (subprocess, CUDA+GL)" begin
    exitcode, output = run_ovrtx_subprocess(_M6_GPUBLIT_PROG; timeout=600)
    @info "M6 gpu blit output" output
    @test exitcode == 0
    @test contains(output, "OK_GPU_BLIT")
    m = match(r"GPU_NONBLACK=(\d+)", output)
    @test m !== nothing && parse(Int, m.captures[1]) > 1000
end
```

- [ ] **Step 7: Run the tests — GREEN.** Register both in `runtests.jl`. Kernel agreement (Step 2) + GPU-direct non-black (Step 6) pass.

- [ ] **Step 8: Commit**

```bash
git add ext/OmniverseMakieCUDAExt.jl ext/OmniverseMakieGLMakieExt.jl test/m6_gpu_blit_test.jl test/m6_tonemap_test.jl test/runtests.jl
git commit -m "feat(M6.A): CUDA-ext GPU-direct present! + tonemap kernel + auto-selection

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

**Acceptance:** with CUDA+GLMakie loaded, `interactive_display(...; gpu_direct=true)` shows a non-black RTX frame via the on-device path; the CUDA kernel tonemap matches the host tonemap pixel-for-pixel; `:auto` picks GPU-direct when `CUDA.functional()`.

---

### Task 5: Benchmark gate + resize re-registration + teardown

**Files:**
- Modify: `ext/OmniverseMakieCUDAExt.jl` (unregister/re-register on resize + teardown), `ext/OmniverseMakieGLMakieExt.jl` (`resize_viewport!`/`close` call the GPU hooks)
- Test: `test/m6_bench_test.jl` (subprocess, CUDA+GL)

**Interfaces:**
- Consumes: the GPU-direct `present!` (Task 4); M5's `resize_viewport!`/`close`.
- Produces: `gpu_unregister!(session)` / re-register on resize; the benchmark.

- [ ] **Step 1: Write the failing benchmark test** (`test/m6_bench_test.jl`) — times `present!` only (excludes the shared RT2 step by pre-warming), GPU-direct vs CPU, at 800×600 and 4K; gates GPU-direct < CPU at 4K.

```julia
using Test
const _M6_BENCH_PROG = """
using OmniverseMakie, GLMakie, CUDA
OM = OmniverseMakie
OM.activate!(warmup = 16)
function blit_latency(sz, mode)
    fig = Figure(); ax = LScene(fig[1,1]; show_axis=false)
    mesh!(ax, Rect3f(Point3f(-1), Vec3f(2)); color = :orange)
    s = OM.interactive_display(fig; size = sz, gpu_direct = mode)
    OM.present!(s, Val(s.blitter))                       # warm
    t = @elapsed (for _ in 1:20; OM.present!(s, Val(s.blitter)); end)
    OM.close(s); return t/20
end
for sz in ((800,600),(3840,2160))
    cpu = blit_latency(sz, false); gpu = blit_latency(sz, true)
    println("BENCH sz=", sz, " cpu_ms=", round(cpu*1e3,digits=3), " gpu_ms=", round(gpu*1e3,digits=3))
end
println("OK_BENCH")
"""
include("helpers.jl")
@testset "M6 GPU-direct vs CPU blit benchmark (subprocess, CUDA+GL)" begin
    exitcode, output = run_ovrtx_subprocess(_M6_BENCH_PROG; timeout=900)
    @info "M6 bench output" output
    @test exitcode == 0
    @test contains(output, "OK_BENCH")
    m4 = match(r"BENCH sz=\(3840, 2160\) cpu_ms=([0-9.]+) gpu_ms=([0-9.]+)", output)
    @test m4 !== nothing
    cpu4k = parse(Float64, m4.captures[1]); gpu4k = parse(Float64, m4.captures[2])
    @test gpu4k < cpu4k          # GPU-direct strictly faster at 4K (gate)
    @info "4K blit latency" cpu_ms=cpu4k gpu_ms=gpu4k speedup=cpu4k/gpu4k
end
```

- [ ] **Step 2: Run it — RED** (no resize/teardown hooks + the bench infra).

- [ ] **Step 3: Implement resize re-registration + teardown** (`ext/OmniverseMakieCUDAExt.jl` + `resize_viewport!`/`close` in the GLMakie ext). On resize, M5 recreates the `image!`/texture; the GPU path must `cuGraphicsUnregisterResource` the old texture and clear `registered` so the next `present!` re-registers the new one. In `Base.close`, unregister + destroy the `copy_done` event before closing the ovrtx Screen. Add `gpu_unregister!(session)` (a no-op for the CPU blitter; the CUDA ext provides the real method) and call it from `resize_viewport!` (after the new `image!`) and `close` (before the ovrtx Screen close — M1 teardown order).

- [ ] **Step 4: Run it — GREEN.** The benchmark prints both resolutions; assert GPU-direct < CPU at 4K. Re-run Step 1's command. (At 800×600 they may be close — that's expected and not gated.)

- [ ] **Step 5: Run the FULL suite — no regression.** `julia --project=. -e 'using Pkg; Pkg.test()'` → "Testing OmniverseMakie tests passed" (M0–M5 offscreen + the M6 ext/GPU tests).

- [ ] **Step 6: Commit**

```bash
git add ext/OmniverseMakieCUDAExt.jl ext/OmniverseMakieGLMakieExt.jl test/m6_bench_test.jl test/runtests.jl
git commit -m "feat(M6.A): GPU-direct benchmark gate + resize re-registration + teardown

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

**Acceptance:** GPU-direct blit latency strictly < CPU at 4K (reported with the actual numbers); resize re-registers the texture; teardown unregisters cleanly; the full `Pkg.test` is green.

---

## Self-Review (completed)

- **Spec coverage:** extension repackaging + GLMakie weak-dep (Task 1); HDR source + shared tonemap + live exposure (Task 2); `map_cuda_array` binding (Task 3); CUDA-ext GPU-direct blit + tonemap kernel + auto-selection + fallback (Task 4); benchmark gate + resize re-registration + teardown (Task 5). The spec's non-goals (picking, subscene hardening, CUDA graphs, float display texture) are correctly ABSENT.
- **Placeholder scan:** the genuinely-novel CUDA-GL/ovrtx FFI calls (the `cuGraphics*` sequence, the exact `ovrtx_cuda_sync_t` fields, the GL texture-id access path) are marked "VERIFY against the spike §X / vulkan-interop C example" with exact file:line references and a REPL step — these are real new-dep unknowns handled by explicit verification (as M5 did for the GLMakie API), not hand-waving. All other code is complete.
- **Type consistency:** `tonemap(::NTuple{3,Float32}, ::Float32) -> RGBA{N0f8}` and `tonemap_frame(::Array{Float32,3}, ::Float32) -> Matrix{RGBA{N0f8}}` are used consistently (Tasks 2, 4). `OV.map_cuda_array -> (data, W, H, C, map_handle, wait_event)` consumed by Task 4. `present!(session, ::Val{:cpu|:gpu})` dispatch is consistent across Tasks 2/4/5. `session.exposure::Float32` / `session.blitter::Symbol` added to `ViewportSession` in Tasks 2/4.
- **Open item carried from the spec:** the kernel→texture-array write uses option (a) (broadcast into a linear RGBA8 buffer → `cuMemcpy2D` into the texture array) — reflected in Task 4 Step 4.
