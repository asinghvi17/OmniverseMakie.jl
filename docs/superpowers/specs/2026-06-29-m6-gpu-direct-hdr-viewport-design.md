# M6.A — GPU‑Direct HDR Viewport (+ extension packaging) — Design

**Date:** 2026-06-29
**Milestone:** M6, sub‑project **A** (the first of M6's pieces). Builds on the merged M5 interactive viewport.
**Status:** design approved in brainstorming; pending spec review → `writing-plans`.

## Goal

Replace M5's CPU host‑roundtrip blit with a **GPU‑direct CUDA‑GL path** that maps ovrtx's frame on‑device and copies it straight into the GLMakie texture — and, while we own the pixel path, display the **HDR** render (`HdrColor`) through our own **tonemap with live exposure**, instead of ovrtx's tone‑mapped `LdrColor`. The same change moves GLMakie and CUDA into **package extensions**, so plain `using OmniverseMakie` (offscreen) pulls neither.

## Scope decision (M6 decomposition)

M6 as a whole is four largely independent pieces: **A. GPU‑direct CUDA‑GL blit** (this spec), **B. AOV picking**, **C. subscene hardening**, and the **GLMakie weak‑dep** packaging follow‑up. They were decomposed during brainstorming; each gets its own spec → plan → build cycle. **This spec is sub‑project A**, which also *absorbs* the GLMakie weak‑dep (D) because the extension mechanism is shared. B and C are out of scope here and follow as later M6 sub‑projects.

## Decisions locked (brainstorming outcomes)

1. **GPU‑direct blit + extension packaging** is the first M6 piece.
2. **Performance bar:** success is gated by a **benchmark** (GPU‑direct vs CPU blit latency), not just "it works".
3. **Automatic selection with CPU fallback:** `interactive_display` uses GPU‑direct when the CUDA extension is loaded and `CUDA.functional()`, else the CPU blit. A `gpu_direct = :auto | true | false` override forces a path (the benchmark needs both).
4. **HDR source + shared tonemap + live exposure:** both blitters read `HdrColor` and apply one shared tonemap (`sRGB(ACES(exposure·hdr))`) → an **RGBA8** display texture. Exposure is a live session knob — the control ovrtx denies via camera exposure.
5. **Per‑frame tonemap kernel** (not a resident/persistent kernel — that would steal SM occupancy from the RT2 path tracer, break the per‑frame CUDA‑GL/ovrtx ownership handshake, and risk the display‑GPU watchdog). **CUDA graphs** are noted as the correct future optimization if launch overhead ever profiles as significant (it won't at these sizes).

## Architecture

**Two Julia package extensions** (native `weakdeps` + `ext/`), not Requires.jl or in‑main `CUDA.functional()` guards:

- **`OmniverseMakie`** (main `src/`) — offscreen rendering (M0–M4) **unchanged**. Declares the generic functions `interactive_display(...)` and the per‑frame `present!(session)` as method‑less stubs that error with a "load GLMakie" message. Also gains the pure‑FFI `OV.map_cuda_array` (no CUDA.jl needed to *declare* it). **No GLMakie, no CUDA in `[deps]`.**
- **`ext/OmniverseMakieGLMakieExt.jl`** — triggered by `GLMakie`. Holds the M5 interactive code (the current `src/interactive/viewport.jl`, `camera_loop.jl`, `blit.jl` move here): `interactive_display`, `ViewportSession`, the `render_tick` loop, resize/teardown, and the **CPU** `present!` (map `HdrColor` → host tonemap → `image!` data).
- **`ext/OmniverseMakieCUDAExt.jl`** — triggered by **both** `CUDA` and `GLMakie` loaded. Holds only the **GPU‑direct** `present!`: register the `image!` GL texture with CUDA once, then per‑frame map `HdrColor` as a CUDA buffer, run the tonemap kernel into the registered texture, and the unmap/sync handshake. Plus the CUDA‑resident state on the session.

**Selection** is resolved once at `interactive_display` time into the session's **blit strategy**; the `render_tick` loop just calls `present!(session)`. `using OmniverseMakie` → offscreen only; `+GLMakie` → CPU viewport; `+GLMakie, CUDA` → GPU‑direct.

## Components

| Unit | Lives in | Responsibility |
|---|---|---|
| generics `interactive_display`, `present!` | `src/` (main) | Declared method‑less; stubs error "load GLMakie". |
| `OV.map_cuda_array(sr, name) -> (cuarray::Ptr, W, H, wait_event::Ptr)` | `src/binding/OV.jl` | New ovrtx FFI: map a render var as `OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY`; returns **raw handles** (CUDA ext wraps them). |
| shared `tonemap` math | `src/` (main, pure) | `sRGB_encode(ACES_filmic(exposure · rgb_linear)) -> RGBA8`. One definition; host + kernel implement the identical math. |
| `ViewportSession` + loop + CPU `present!` | GLMakie ext | M5 viewport, now reading `HdrColor` + host tonemap. |
| GPU‑direct `present!` + CUDA state | CUDA ext | Register GL texture; per‑frame map/kernel/copy/unmap/sync. |

## The new ovrtx binding

`OV.map_cuda_array(sr::StepResult, name="HdrColor") -> (cuarray::Ptr{Cvoid}, W::Int, H::Int, wait_event::Ptr{Cvoid})`. Mirrors `map_cpu` but uses **`OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY`** (a C‑only mode the Python layer doesn't expose). The mapped tensor's `data` is a `CUarray`; `out.cuda_sync.wait_event` is a `CUevent`. Returns raw handles so CUDA.jl stays out of the main module; the CUDA ext wraps them (`CUDA.CUDACore.CUarray` / `CUevent`). An unmap helper closes the map gated on a completion event (below).

## Per‑frame data flow

**CPU `present!(session)`** (GLMakie ext): `render_to_matrix("HdrColor")` → host **tonemap** (broadcast `Float → RGBA8`) → update the `image!` data Observable (M5's `cpu_blit!`, fed a tonemapped frame).

**GPU‑direct `present!(session)`** (CUDA ext):
1. Setup once (render task, GL current, after the texture is realized): `cuGraphicsGLRegisterImage(res, tex.id, tex.texturetype, WRITE_DISCARD)`.
2. `OV.step!` → `OV.map_cuda_array("HdrColor")` → `(srcArray, W, H, wait_event)`.
3. `cuGraphicsMapResources(res, stream)` → `cuGraphicsSubResourceGetMappedArray(dst, res)` (GL texture as a CUarray).
4. `cuStreamWaitEvent(stream, wait_event)` — wait for ovrtx's render to land.
5. **tonemap kernel** (exposure + ACES + sRGB, `Float → RGBA8`) reads `srcArray`, writes `dst` (the registered GL texture) — *replaces* the plain `cuMemcpy2D`; same scaffolding around it.
6. `cuGraphicsUnmapResources(res, stream)`.
7. `cuEventRecord(copy_done, stream)` → ovrtx unmap **gated on `copy_done`** (ovrtx can't reclaim the HDR buffer mid‑kernel).
8. `cuStreamSynchronize(stream)` — v1 GL sync (heavy but race‑free; the spike's recommended first cut).
9. GLMakie draws its fullscreen quad from the updated texture.

**Teardown / resize:** `cuGraphicsUnregisterResource` on close; on resize, M5 already recreates the `image!`/texture, so the path is **unregister → (texture recreated) → re‑register**.

## HDR + tonemapping + exposure

- Both blitters read **`HdrColor`** (linear float radiance).
- One shared tonemap: `display = sRGB_encode( ACES_filmic( exposure · hdr_rgb ) )` → **RGBA8**. The display texture stays RGBA8, so **M5's `image!` display path is unchanged** (no custom GLMakie shader / float‑texture plumbing) — all HDR work lives in the blit step.
- **Exposure** is a live session field (default 0 EV; a kwarg now, scroll/key binding optional later). It re‑tonemaps on the next tick (the loop re‑blits each frame). An idle exposure change can force one extra tick so it shows immediately.
- The tonemap math is defined once in the main module; the **host broadcast** (CPU path) and the **CUDA kernel** (GPU path) implement the identical math, asserted equal by a test.

## Selection & fallback

`interactive_display(fig; gpu_direct = :auto, exposure = 0.0f0, …)`. At session build: `:auto` → GPU‑direct if the CUDA ext is loaded and `CUDA.functional()`, else CPU; `true` forces GPU‑direct (clear error if CUDA unavailable); `false` forces CPU. If GPU‑direct **setup** fails (registration/context error), log once and fall back to the CPU blitter for that session. The M5 per‑frame error guard already wraps `present!`, so transient interop errors keep the window alive.

## Sync & errors

- **Single‑threaded render task:** all CUDA‑GL interop runs on GLMakie's render task with the GL context current and CUDA's per‑task context initialized there (spike §2). **No new thread** (the M0 carb signal‑window risk stays closed).
- **Ownership handshake** is per‑frame and inherent: `cuGraphicsMapResources`/`Unmap` hands the texture between CUDA (write) and GL (read); `cuStreamWaitEvent` + event‑gated ovrtx unmap hands the HDR buffer between ovrtx (render) and CUDA (copy).
- **Graceful degradation:** setup failure → CPU fallback for the session; transient per‑frame error → guarded, window stays alive.

## Benchmark — the success gate

Time the **blit step in isolation** (`present!` minus the shared RT2 step), GPU‑direct vs CPU, at two resolutions:
- **~800×600** — expected close (blit isn't the bottleneck at small sizes).
- **4K (3840×2160)** — where the host roundtrip bites: CPU moves ~66 MB float readback + ~33 MB RGBA8 upload over PCIe per frame; GPU‑direct stays on‑device.

A bench test builds the session both ways (`gpu_direct=true/false`), times `present!` over N frames at each resolution, and **gates on: GPU‑direct blit latency strictly < CPU at 4K (target ≥2×)** — reporting the actual numbers regardless. **Correctness anchor:** GPU‑direct texture readback matches the CPU path's output (within rounding).

## Testing

Renderer/CUDA/GL tests are **subprocess‑isolated** (carb signal handlers + the `:0` display env), as in M0–M5.

- **Tonemap agreement** — host broadcast vs CUDA kernel on the same HDR input → identical RGBA8 (within rounding). Anchors GPU‑direct correctness and the shared‑math claim.
- **`OV.map_cuda_array`** (subprocess, CUDA) — maps `HdrColor` as `CUDA_ARRAY` → a valid `CUarray` + dims + `wait_event`.
- **GPU‑direct blit** (subprocess, CUDA+GL) — register a GL texture, `present!` an ovrtx HDR frame, read back (`GLMakie.colorbuffer`) → non‑black + matches the CPU path.
- **Selection / fallback** — CUDA ext loaded → GPU‑direct chosen; `gpu_direct=false` → CPU; `gpu_direct=true` without CUDA → clear error.
- **Extension loading** — `using OmniverseMakie` alone → offscreen `save`/`colorbuffer` work and `interactive_display` errors helpfully; `+GLMakie` → CPU viewport; `+GLMakie, CUDA` → GPU‑direct.
- **The benchmark** as the perf gate.
- **No regression:** the full `Pkg.test` (M0–M5) stays green after the extension move (every M0–M4 subprocess test still loads `OmniverseMakie` offscreen; the M5 viewport tests now exercise the GLMakie ext).

## Global constraints

- **Package extensions:** GLMakie and CUDA move to `[weakdeps]` + `[extensions]` (`OmniverseMakieGLMakieExt = "GLMakie"`, `OmniverseMakieCUDAExt = ["CUDA", "GLMakie"]`). Main `[deps]` loses GLMakie. Managed via Pkg, never hand‑edited TOML.
- **Single‑threaded render task**, GL + CUDA contexts current there; no separate render thread.
- **Reuse M5 verbatim:** the `ViewportSession`, the `render_tick` loop, resize/teardown, and the per‑frame error guard are on `main`; M6.A adds a blit strategy + the HDR tonemap and relocates the files into the GLMakie ext (no behavior change to the CPU loop beyond the HDR source).
- **Display + CUDA available:** `DISPLAY=:0` + the mutter Xwayland `XAUTHORITY` for GL; `CUDA.functional()` on the RTX A5000 (spike: CUDA.jl 6.2.0 `functional() == true`).
- **Interop via CUDA.jl** (`CUDA.CUDACore.cuGraphics*`, typed + `@checked` + context init), `@ccall libcuda` as a documented escape hatch (spike §1).
- **Commit trailer:** `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. **Do not push** (the user controls merges).

## Task breakdown (one spec, ~5 SDD tasks)

1. **Extension repackaging** — move the M5 interactive files into `ext/OmniverseMakieGLMakieExt.jl`; declare `interactive_display`/`present!` generics + stubs in main; GLMakie → `[weakdeps]`. Gate: full `Pkg.test` green (offscreen unaffected; viewport via the ext).
2. **HDR source + shared host tonemap** — the shared `tonemap` math in main; the CPU `present!` reads `HdrColor` + host tonemap; live `exposure`. Gate: CPU viewport shows the tonemapped HDR; exposure changes the image.
3. **`OV.map_cuda_array` binding** — the `CUDA_ARRAY` map + event‑gated unmap helper.
4. **CUDA ext: GPU‑direct blit** — register/map/kernel/unmap/sync; the tonemap CUDA kernel; auto‑selection + fallback. Gate: GPU‑direct viewport non‑black + matches CPU.
5. **Benchmark gate + resize re‑registration** — the GPU‑vs‑CPU 4K benchmark; unregister/re‑register on resize; teardown. Gate: GPU‑direct < CPU at 4K + clean teardown.

## Non‑goals (this sub‑project)

- **AOV picking** (M6.B) and **subscene hardening** (M6.C) — separate sub‑projects.
- **GPU‑direct sync beyond `cuStreamSynchronize`** — `GL_EXT_semaphore` ↔ `cuSignalExternalSemaphoresAsync` is a later optimization if the sync cost profiles (spike §3).
- **CUDA graphs / resident kernels** — noted as a future optimization only; not built now.
- **A float (HDR) display texture / HDR monitor output** — display stays RGBA8 (tonemapped); the float precision that matters lives in ovrtx's accumulation.

## Risks / open items (resolved in the plan)

1. **CUDA‑GL register timing** — register the texture only after its GL id is realized (after the first draw/upload), not at plot construction (spike §4 pitfall b). M6.A's setup registers on the first GPU‑direct `present!`.
2. **HDR map mode availability** — `HdrColor` is a confirmed ovrtx render var; the `CUDA_ARRAY` map mode is C‑only (verified in the spike). The plan's first GPU step proves `OV.map_cuda_array` returns a valid CUarray before the kernel is written.
3. **Tonemap/exposure look** — ACES filmic + a single exposure scalar is the default; the look is for the interactive viewport and intentionally diverges from the `LdrColor`‑calibrated gallery (separate offscreen path, unaffected).
4. **Extension‑move regressions** — relocating the M5 files into the GLMakie ext must keep the viewport identical; the existing M5 tests (now exercising the ext) are the guard, plus the full `Pkg.test` no‑regression gate.
5. **Kernel → texture‑array write path** — a CUDA kernel cannot write a texture‑backed `CUarray` with a plain store; the GPU `present!` either (a) tonemaps into a linear RGBA8 device buffer (a CUDA.jl broadcast over the HDR `CuArray`) then `cuMemcpy2D`s that linear buffer into the GL texture array, or (b) uses CUDA **surface writes** (`surf2Dwrite`) for a single pass. The plan picks one in the GPU task (lean (a): a CUDA.jl broadcast is simplest and the extra on‑device copy is cheap; revisit (b) only if it profiles). Either keeps the map/unmap/event handshake unchanged.
