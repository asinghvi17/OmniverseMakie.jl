# Milestone M3 — Interactive RTX viewport (bite-sized plan)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes. Grounded in the proven spike `references/notes/cuda-gl-interop.md` (CUDA-GL interop, GLMakie texture, threading) and the live-camera spikes (`write_xform!` on `/World/Camera`, M1.3 disproven).

**Goal:** A real interactive window shows the **live ovrtx path-traced render** of a Makie scene; mouse **orbit/pan/zoom** drives a per-frame ovrtx camera write and **progressive RT2 accumulation** (refine while idle, reset on interaction).

**Architecture:** Host the viewport in a **GLMakie window** displaying one full-viewport `image!` plot. A per-frame hook on GLMakie's render task reads the scene's `Camera3D`, syncs it to the ovrtx camera via M2.1 `sync_camera!` (`write_xform!("/World/Camera", …)` + `reset!`), steps ovrtx on the **open M2 stage**, and **blits** the rendered `LdrColor` frame into the `image!` texture — **CPU readback (v1)**, then **zero-copy CUDA-GL interop (v2)**. Single-threaded on GLMakie's render task so the GL context + CUDA context are co-located (sidesteps the carb-multithread risk).

**Tech Stack:** GLMakie `=0.13.12` (window, events, `Camera3D`, `GLAbstraction.Texture`, `render_tick` — already a registry dep), CUDA.jl v6.2 (`CUDA.CUDACore` GL-interop wrappers, `functional()==true` verified on the A5000), ModernGL, the OmniverseMakie `OV` layer + the M2 open-stage diff path.

> **★ ARCHITECTURE DECISION (ratify in review).** This supersedes the earlier "standalone GLFW window" leaning. We host the interactive viewport **inside a GLMakie window** (ovrtx renders the pixels; GLMakie owns the window / event loop / camera interaction / GL texture). Rationale: `cuda-gl-interop.md` is a thorough project spike built entirely around this path (GLMakie `Texture` + `render_tick` hook + the single-thread GL/CUDA constraint), it reuses M2.1's `sync_camera!` unchanged, and GLMakie's mature windowing/event/camera code is far more robust than a hand-rolled GLFW loop. Cost: GLMakie is a runtime dep of the interactive path (it already is a dep). A standalone GLFW window remains a possible later refinement.

## Global Constraints (M3)

Inherit **all** M0/M1/M2 Global Constraints (Pkg-managed pinned deps; generated bindings `lib/LibOVRTX/src/libovrtx_api.jl` verbatim; `GC.@preserve` on every FFI path; carb `SignalGuard` intact; subprocess-isolated renderer tests + watchdog; `colorbuffer` returns `Matrix{RGBA{N0f8}}` **right-side-up, NO flip**; open-stage M2 model — author once, live-diff camera/lights/plots). Plus M3-specific:

- **Single-threaded render task.** All `ovrtx_step` + map/readback + CUDA-GL interop run on **GLMakie's render task** with the GL context current (`cuda-gl-interop.md §2`: `@async renderloop` does not migrate threads; interop must be on that task). **Do NOT add a separate render thread** — that would re-open the M0 carry (the create-time `SIGSEGV→SIG_DFL` window colliding with Julia GC safepoints under multithreading). If threading is ever needed, it gets its own carb re-validation spike first.
- **CPU-fallback display FIRST, GPU-direct second** (`cuda-gl-interop.md §5` then `§4`). v1 = host roundtrip (map `LdrColor` → `Matrix{RGBA{N0f8}}` → texture); v2 = on-device CUDA-array→GL-texture copy. The GPU-direct path slots in behind the same `Texture`.
- **Bounded `enqueue_wait` in the UI loop** (M1 forward-carry): the interactive step must use a **bounded** ovrtx wait, never `OVRTX_TIMEOUT_INFINITE`, so a stuck step can't freeze the window. Add `OV.step!(r; timeout_ns)` / a bounded variant.
- **Live camera via the PROVEN path** (M1.3 "ignored" is DISPROVEN): read `Camera3D` (`eyeposition`/`lookat`/`upvector`/`fov`) → `sync_camera!` → `write_xform!("/World/Camera", camera_to_world(...))` + `OV.reset!`. **Never re-author the root on a camera move.**
- **Headless dev box.** This machine has no X display (the renderer logs "failed to open the default display"). A *visible* GLMakie window needs a display → run the interactive integration under **`xvfb-run`** (virtual framebuffer) on this box, or on a machine with a display. **Component logic** (camera→ovrtx mapping, the blit, the accumulation state machine) is testable WITHOUT a visible window (GLMakie offscreen `Screen(visible=false)` or pure functions). Each task states which mode its test uses.
- **M2 dependencies:** the per-frame camera write uses M2.4's persistent `map_attribute` binding when available (zero-copy fixed-size xform) — M3 falls back to the M0 `write_attribute` path if M2.4 isn't merged. M2.6's benchmark sets the interactivity bar (if CPU-side writes miss interactive rates, M3.4 GPU-direct becomes mandatory, not optional).

## File structure (M3 adds)

```
src/
  interactive/
    viewport.jl     # NEW: interactive_display(fig/scene)->ViewportSession; the GLMakie host window + image! + render_tick hook
    blit.jl         # NEW: cpu_blit!(tex, frame) (v1); cuda_gl_blit!(session, …) (v2, M3.4)
    camera_loop.jl  # NEW: on_render_tick!(session) — read Camera3D → sync_camera! → reset-on-change → step (bounded) → blit; accumulation state
  binding/OV.jl     # MODIFY: bounded step!(r; timeout_ns); CUDA-array map mode (OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY) for M3.4
  screen.jl         # MODIFY: ViewportSession owns the open-stage Screen + the GLMakie screen + the render_tick listener (teardown order)
test/
  m3_camera_loop_test.jl   m3_blit_test.jl   m3_viewport_xvfb_test.jl   m3_interop_test.jl
bench/
  interactive.jl    # NEW (M3.3/M3.4): sustained interactive FPS under an orbit, CPU vs GPU-direct
```

---

## Task M3.1 — Viewport window + static ovrtx frame on screen (CPU blit) ★

**Files:** `src/interactive/viewport.jl`, `src/interactive/blit.jl`. Test: `test/m3_blit_test.jl` (offscreen, no window), `test/m3_viewport_xvfb_test.jl` (xvfb integration).

**Interfaces — Produces:**
```julia
struct ViewportSession            # owns both screens + the open M2 stage + interop state
    screen::Screen                # OmniverseMakie open-stage Screen (the ovrtx renderer + USD stage)
    glscreen                      # GLMakie.Screen (the window)
    image_plot                    # the full-viewport `image!` plot (its texture is the blit target)
    fb_size::Tuple{Int,Int}
    tick_listener                 # the render_tick Observable listener (for teardown)
end
interactive_display(fig_or_scene) -> ViewportSession      # opens the window, authors the stage, blits the first frame
cpu_blit!(image_plot, frame::Matrix{RGBA{N0f8}})          # update the image! texture from a host frame
```
- **Approach:** build the open-stage `Screen` for the scene (M2 `colorbuffer` path authors once + renders). Open a GLMakie window whose root scene holds a single `image!(ax, frame)` filling the viewport (the display surface). `cpu_blit!` updates that plot's data Observable (`image_plot[1][] = frame`) — GLMakie re-uploads (`cuda-gl-interop.md §5`). Render one ovrtx frame (`OV.render_to_matrix`) and blit it. **Validate-first (Step 2): does a GLMakie window open on this box under `xvfb-run`?** If not, report (fallback: pure-offscreen development + a display machine for the visible window).
- [ ] **Step 1 (failing offscreen test, blit):** `test/m3_blit_test.jl` — make a `400×400` `RGBA{N0f8}` frame (red top half, blue bottom), an offscreen GLMakie `Screen(visible=false)` with `image!`, call `cpu_blit!`, read the texture back (`GLMakie.GLAbstraction.gpu_data(tex)` or `colorbuffer(glscreen)`), assert the top is red / bottom blue (and orientation matches our top-left-origin, no flip). RED (`cpu_blit!` undefined).
- [ ] **Step 2 (validate window, xvfb):** `test/m3_viewport_xvfb_test.jl` (run via `xvfb-run -a julia …` in `run_ovrtx_subprocess`) — `interactive_display` of a 1-mesh `LScene`; assert the window opened (`glscreen` is open) and `colorbuffer(glscreen)` is non-black (the blitted RTX frame is visible). RED.
- [ ] **Step 3:** run both → FAIL.
- [ ] **Step 4:** implement `cpu_blit!` + `interactive_display` (build `Screen`; `OV.render_to_matrix`; open GLMakie window with `image!`; blit).
- [ ] **Step 5:** run → PASS (offscreen blit asserts colors+orientation; xvfb asserts a non-black window).
- [ ] **Step 6:** commit `feat(M3.1): interactive viewport window + CPU blit of the ovrtx frame`.

**Acceptance:** a window (under xvfb on this box) shows the static RTX render of a Makie scene; `cpu_blit!` puts a host frame on screen right-side-up.

---

## Task M3.2 — Live camera interaction loop (orbit/pan/zoom → ovrtx) ★

**Files:** `src/interactive/camera_loop.jl`, `src/binding/OV.jl` (bounded `step!`). Test: `test/m3_camera_loop_test.jl` (offscreen/component), `m3_viewport_xvfb_test.jl` (synthetic-event integration).

**Interfaces — Produces:**
```julia
OV.step!(r::Renderer; timeout_ns::UInt64 = UInt64(2_000_000_000))   # bounded wait (M1 carry; default 2 s)
on_render_tick!(session::ViewportSession)   # one frame: sync_camera! → reset-if-moved → bounded step → cpu_blit!
```
- **Approach:** register `on_render_tick!` on `session.glscreen.render_tick` (fired per frame on the render task — `cuda-gl-interop.md §2`). Each tick: `cam_changed = sync_camera!(session.screen, cam_scene)` (M2.1, reuses the proven `write_xform!("/World/Camera", …)`); if changed → `OV.reset!`; `OV.step!` (bounded); `frame = OV.map LdrColor → Matrix`; `cpu_blit!`. The user orbits/pans/zooms with GLMakie's normal mouse controls (they mutate the scene's `Camera3D`); the hook propagates that to ovrtx, so the RTX view reframes live. The listener must **not** `Consume(true)`.
- [ ] **Step 1 (failing component test, no window):** `test/m3_camera_loop_test.jl` (subprocess) — build a `ViewportSession` offscreen; record `frame_A` via one `on_render_tick!`; mutate `cam.eyeposition[]` (180° orbit); `on_render_tick!` again → `frame_B`; assert (a) the red/blue centroid swapped (camera write took effect via the loop), (b) exactly one `write_xform!` to `/World/Camera` fired (instrument), (c) `reset!` fired on the moved frame and NOT on an unmoved repeat. RED.
- [ ] **Step 2 (bounded step):** assert `OV.step!(r; timeout_ns=1)` returns/raises a *bounded* timeout (never hangs) — a unit test that a tiny timeout doesn't block forever. RED (`step!` kw undefined).
- [ ] **Step 3:** run → FAIL.
- [ ] **Step 4:** implement bounded `OV.step!` (bounded `enqueue_wait`); implement `on_render_tick!`; wire it as the `render_tick` listener in `interactive_display`.
- [ ] **Step 5:** run → PASS. Add an xvfb integration check: drive synthetic GLFW drag events, assert `colorbuffer(glscreen)` changes between frames.
- [ ] **Step 6:** commit `feat(M3.2): live camera interaction loop (orbit→write_xform!→step→blit) + bounded step`.

**Acceptance:** dragging the mouse orbits the live RTX view (camera write per frame on the open stage, no re-author); the step is bounded (a stuck step can't freeze the window).

---

## Task M3.3 — Progressive RT2 accumulation (refine while idle, reset on interaction)

**Files:** `src/interactive/camera_loop.jl`. Test: `test/m3_camera_loop_test.jl` (extend, component).

- **Approach:** add an accumulation state to `ViewportSession`: when nothing changed this tick (camera/lights/plots clean), DO NOT `reset!` — just `OV.step!` once more and blit, so RT2 keeps accumulating and the image converges (noise drops frame over frame). On ANY change (`sync_camera!`/`sync_lights!`/plot-node pull returns changed), `reset!` then step. Track a `samples` counter for optional display.
- [ ] **Step 1 (failing test):** subprocess, offscreen — hold the camera fixed; call `on_render_tick!` 16× without mutation; assert the per-pixel variance / mean-noise **decreases** across frames (accumulation), and `reset!` fired **zero** times after the first; then mutate the camera once → assert exactly one `reset!`. RED.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** implement the idle-accumulate / change-reset state machine.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** commit `feat(M3.3): progressive RT2 accumulation — idle-refine, reset-on-interaction`.

**Acceptance:** a still view visibly refines over frames; any interaction restarts accumulation.

---

## Task M3.4 — GPU-direct CUDA-GL interop blit (zero host roundtrip) ★

**Files:** `src/interactive/blit.jl`, `src/binding/OV.jl` (CUDA-array map mode). Test: `test/m3_interop_test.jl` (xvfb — needs a real GL context + CUDA).

**Interfaces — Produces:**
```julia
# OV: map LdrColor as a CUDA array (C-only mode; Python can't) — returns (CUarray, wait_event::CUevent)
OV.map_render_var_cuda_array(r, product, var="LdrColor") -> (src_cuarray, wait_event)
OV.unmap_render_var(r, handle; wait_event)                # event-gated handback
cuda_gl_blit!(session)                                    # register-once → per-frame map/copy/unmap into the GL texture
```
- **Key code** (`cuda-gl-interop.md §4`, proven call sequence): once (render task, GL current) `cuGraphicsGLRegisterImage(res, tex.id, tex.texturetype, WRITE_DISCARD)`; per frame `cuGraphicsMapResources`→`cuGraphicsSubResourceGetMappedArray(dst)`→`cuStreamWaitEvent(stream, wait_event)`→`cuMemcpy2DAsync(src CUarray → dst CUarray, W*4, H)`→`cuGraphicsUnmapResources`→`cuEventRecord(copy_done)`→ovrtx `unmap(wait_event=copy_done)`→`cuStreamSynchronize` (v1 GL sync). All via `CUDA.CUDACore.*` (fallback: `@ccall libcuda.*` — every symbol resolves, note §1/§6). Register **after** the texture id is realized (after the first CPU-blit upload), not at plot construction (note §3 pitfall b).
- ⚠️ **Validate-first (Step 1):** the whole interop in the *live loop* is the note's flagged risk ("integration glue"). Step 1 proves a single map→copy→unmap puts the right pixels in the texture (compare against the CPU-blit frame). If it doesn't (context/sync/format), STAY on the CPU blit (fully working from M3.1) and record the gap for a follow-up — do NOT block the milestone on GPU-direct.
- [ ] **Step 1 (failing interop test, xvfb):** `test/m3_interop_test.jl` — render one frame; `cuda_gl_blit!` it; read the texture back and assert it matches the CPU-blit of the same frame within tolerance (correct pixels via the on-device path). RED.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** implement `OV.map_render_var_cuda_array`/`unmap_render_var` (CUDA-array mode binding, `GC.@preserve` the descriptors) + `cuda_gl_blit!` (register-once, map/copy/unmap, event-gated, `cuStreamSynchronize`); make the blit path selectable (`session.blit = :cpu | :cuda`), default `:cpu` until this passes.
- [ ] **Step 4:** run → PASS (GPU-direct frame == CPU frame).
- [ ] **Step 5:** commit `feat(M3.4): GPU-direct CUDA-GL interop blit (CUDA-array → GL texture)`.

**Acceptance:** frames reach the screen with no host roundtrip (on-device CUDA→GL copy), pixel-equivalent to the CPU path; CPU blit remains the guaranteed fallback.

---

## Task M3.5 — Resize, teardown, robustness

**Files:** `src/interactive/viewport.jl`, `src/screen.jl`. Test: `test/m3_viewport_xvfb_test.jl` (extend).

- **Approach:** on viewport resize (`glscreen` size change): `cuGraphicsUnregisterResource` → recreate the `image!`/texture at the new size → re-register (note §4) → re-author the render product resolution on the ovrtx `Screen`. Teardown (`close(session)`): deregister the `render_tick` listener, `cuGraphicsUnregisterResource` (if registered), close the GLMakie screen, then close the ovrtx `Screen` (StepResults before renderer — M1 order). Wrap the per-frame loop body so a transient ovrtx/CUDA error logs (`@warn maxlog=…`) and keeps the window alive rather than crashing the render task.
- [ ] **Step 1 (failing test, xvfb):** open a viewport, resize the window (set `glscreen` size), assert the next frame renders non-black at the new size (no stale-texture crash); then `close(session)` and assert both the GLMakie screen and the ovrtx renderer are closed and the `render_tick` listener is gone (no leaked listener). RED.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** implement resize (unregister→recreate→re-register→re-author resolution) + `close(session)` teardown + the loop-body error guard.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** commit `feat(M3.5): viewport resize + leak-free teardown + render-loop error guard`.

**Acceptance:** resizing keeps rendering correctly; closing leaves no leaked listener/resource/renderer; a transient frame error doesn't kill the window.

---

## Task M3.6 — Interactive throughput benchmark (gate)

**Files:** `bench/interactive.jl`. Test: a threshold assertion (xvfb).

- **Approach:** under xvfb, drive a scripted continuous orbit for N seconds; measure sustained **frames/sec** and **camera-write→visible-frame latency**, CPU-blit vs GPU-direct, on the A5000. Report to `bench/interactive.jl` output / `bench/RESULTS.md`.
- [ ] **Step 1:** write the benchmark (reuse the session; scripted orbit; measure FPS + latency for `:cpu` and `:cuda`).
- [ ] **Step 2:** run; record FPS + latency.
- [ ] **Step 3:** a test asserting **≥ interactive rate** (target **≥24 FPS** for a default-size viewport on the A5000) for at least one blit path; if CPU misses it but GPU-direct meets it, that's the documented justification for GPU-direct; if BOTH miss, record the gap + escalate (lower sample-per-frame / DLSS settings / resolution scaling).
- [ ] **Step 4:** commit `bench(M3.6): interactive throughput (FPS + latency, CPU vs GPU-direct)`.

**Acceptance:** the interactive loop sustains an interactive frame rate on the A5000 (or the shortfall is measured + escalated).

---

**M3 GATE:** a Makie scene opens in an interactive window (xvfb on this box; a display elsewhere) showing the live ovrtx path-traced render; mouse orbit/pan/zoom reframes it live via per-frame `write_xform!` on the open stage; the image progressively refines when idle and resets on interaction; frames reach the screen via CPU blit (guaranteed) and GPU-direct CUDA-GL interop (when it passes); resize + teardown are clean; throughput meets interactive rates. ✅ → M4 (the next milestone — e.g. richer materials / textures / volume; streaming remains SHELVED).

---

## Open assumptions this plan validates early (with fallbacks)
1. **A visible GLMakie window opens under `xvfb-run` on this headless box** (M3.1 Step 2). Fallback: pure-offscreen component development + a display machine for the visible window.
2. **CUDA-GL interop works inside the live render-task loop** (M3.4 Step 1 — the note's flagged "integration glue" risk). Fallback: ship the CPU blit (fully working from M3.1); GPU-direct is a follow-up.
3. **ovrtx exposes the `CUDA_ARRAY` map mode over the C ABI from Julia** (M3.4) — the note says yes (C ABI == whole API; `OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY`, Python can't but ccall can). Fallback: CUDA-linear (`Device.CUDA`) device→array copy, or the CPU blit.
4. **A bounded `enqueue_wait` exists / can be added** without destabilizing the step (M3.2). Fallback: poll `ovrtx_get_status` with a deadline.
5. **GLMakie's `Camera3D` mouse interaction + `render_tick` hook compose with our blit** (M3.2) — i.e. driving the camera and injecting per-frame work via `render_tick` (no `Consume`) behaves. Fallback: a custom GLMakie `renderloop` (note §2) or feed GLFW events to `Camera3D` manually.
6. **(M2 dependency)** M2.4's persistent `map_attribute` binding for the camera xform lands — M3 uses it for the per-frame write; until then M3 uses the M0 `write_attribute` path (correct, non-zero-copy).
