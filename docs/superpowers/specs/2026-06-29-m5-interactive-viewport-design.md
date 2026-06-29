# M5 — Interactive RTX Viewport — Design

**Date:** 2026-06-29
**Milestone:** M5 — the first interactive milestone (the project's v1 "north‑star": a live, orbit‑able RTX viewport). Builds directly on the merged M0–M4 backend.
**Status:** design approved in brainstorming; pending spec review → `writing-plans`.

## Goal

A real window shows the **live ovrtx path‑traced render** of a Makie scene. Mouse **orbit / pan / zoom** drives a per‑frame ovrtx camera write and **progressive RT2 accumulation** — the image refines while the user is idle and resets the instant they interact. This turns the M0–M4 "build a scene → save a PNG" backend into an interactive 3‑D explorer.

## Scope decision (M5 vs M6)

The pre‑reorder `M5_PLAN.md` bundled the GPU‑direct CUDA‑GL blit in as task 5.4, but the milestone reorder put GPU‑direct (with picking and subscene hardening) in M6. **Resolved: M5 ships the CPU‑blit interactive viewport; GPU‑direct moves fully to M6.** The CPU blit (read the finished frame to the host, upload to the GL texture each frame) fully delivers the interactive experience — the RT2 path‑trace *step* dominates per‑frame time, so the ~2 MB/frame readback is minor overhead. GPU‑direct removes that roundtrip but isn't required to *be* interactive, and the CUDA‑GL interop is the spike's flagged‑risk "integration glue." A direct consequence: **M5 adds only GLMakie as a new dependency — not CUDA** (CUDA enters with GPU‑direct in M6).

## Non‑goals (M5)

- **No GPU‑direct CUDA‑GL blit** (→ M6). CPU blit is the M5 display path.
- **No picking / object selection** (→ M6, AOV‑based).
- **No subscene‑hardening work** (→ M6).
- **No separate render thread.** Everything runs single‑threaded on GLMakie's render task; a second thread re‑opens the M0 carb create‑time signal window vs Julia GC safepoints and would need its own carb re‑validation spike first.
- **No standalone GLFW window.** The viewport is hosted inside a GLMakie window (a hand‑rolled window remains a possible later refinement).

## Architecture

Host the viewport **inside a GLMakie window** that displays a single **full‑viewport `image!` plot**; that plot's GL texture is the blit target for ovrtx frames. GLMakie owns the window, event loop, and `Camera3D` interaction; ovrtx renders the pixels on the **open M2 stage**. A per‑frame hook on GLMakie's `render_tick` runs the whole loop **single‑threaded on the render task**, so the GL context is co‑located and the carb multithread risk is avoided.

Rationale for GLMakie‑host over a hand‑rolled GLFW window: the `references/notes/cuda-gl-interop.md` spike is built entirely around this path (GLMakie `Texture` + `render_tick` hook + the single‑thread GL constraint); it reuses the merged M2.1 `sync_camera!` unchanged; and GLMakie's windowing/event/camera code is far more robust than a hand‑rolled loop.

### The per‑frame loop (`on_render_tick!`)

Each `render_tick` (fired per frame on the render task):

1. Read the scene's `Camera3D`; `cam_changed = sync_camera!(session.screen, cam_scene)` — the merged M2.1 path (`write_xform!("/World/Camera", …)`, using M2.4's persistent `map_attribute` binding for the zero‑copy fixed‑size xform when present). Also pull lights/plot diffs.
2. If anything changed → `OV.reset!` (restart RT2 accumulation); otherwise leave accumulation running.
3. `OV.step!(r; timeout_ns)` — a **bounded** step (never `OVRTX_TIMEOUT_INFINITE`), so a stuck frame can't freeze the window.
4. Read `LdrColor` → `Matrix{RGBA{N0f8}}` (right‑side‑up, no flip) → `cpu_blit!` into the `image!` texture.

The listener must **not** `Consume(true)` (it observes, doesn't swallow events). Orbit/pan/zoom with GLMakie's normal mouse controls mutate the scene's `Camera3D`; the hook propagates that to ovrtx, so the RTX view reframes live. Sitting idle lets RT2 converge (noise drops frame‑over‑frame).

## Components

New under `src/interactive/`:

- **`viewport.jl`** — `ViewportSession` (holds the ovrtx `Screen`, the GLMakie window/`Screen`, the full‑viewport `image!` plot, and the accumulation state) + `interactive_display(fig_or_scene) -> ViewportSession`: builds the open‑stage ovrtx `Screen` (the M2 `colorbuffer` authoring path), opens a GLMakie window whose root scene holds one `image!` filling the viewport, renders the first ovrtx frame, and blits it.
- **`blit.jl`** — `cpu_blit!(image_plot, frame::Matrix{RGBA{N0f8}})`: update the `image!` texture from a host frame via the plot's data Observable (GLMakie re‑uploads). Orientation matches our top‑left origin (no flip).
- **`camera_loop.jl`** — `on_render_tick!(session::ViewportSession)`: the per‑frame loop above, plus the accumulation state machine (reset‑on‑change vs keep‑accumulating; a `samples` counter for optional display).
- **`binding/OV.jl`** (modify) — `OV.step!(r; timeout_ns)`: a bounded `enqueue_wait` variant.

Each unit has one clear purpose and a small interface: `interactive_display` is the entry point, `cpu_blit!` is a pure texture update, `on_render_tick!` is the loop body, `step!` is the bounded ovrtx advance.

## Task breakdown (4 tasks, SDD)

1. **M5.1 — Viewport window + static ovrtx frame on screen (CPU blit).** `ViewportSession`, `interactive_display`, `cpu_blit!`. Acceptance: a real window shows the static RTX render of a Makie scene; `cpu_blit!` puts a host frame on screen right‑side‑up.
2. **M5.2 — Live camera interaction loop.** `on_render_tick!` wired to `render_tick`; bounded `OV.step!`. Acceptance: dragging the mouse orbits the live RTX view (per‑frame camera write on the open stage, no re‑author); the step is bounded.
3. **M5.3 — Progressive RT2 accumulation.** Idle‑refine, reset‑on‑interaction. Acceptance: the idle image converges (noise drops); any interaction resets accumulation.
4. **M5.4 — Resize, teardown, robustness.** Recreate the `image!`/texture and re‑author the render‑product resolution on resize; clean teardown (deregister the `render_tick` listener, close the GLMakie screen, then close the ovrtx `Screen` — StepResults before renderer, M1 order); wrap the per‑frame body so a transient ovrtx error logs and keeps the window alive rather than crashing the render task.

## Dependencies

Add **GLMakie `=0.13.12`** (registry dep, pinned to match `Makie =0.24.12`) via Pkg — it provides the window, events, `Camera3D`, `GLAbstraction.Texture`, and the `render_tick` hook. **No CUDA in M5.** Managed via Pkg, never hand‑edited TOML. GLMakie is a runtime dependency only of the interactive path.

## Testing

- **Component logic without a window** (clean for CI): camera→ovrtx mapping, the blit, and the accumulation state machine are testable via offscreen `GLMakie.Screen(visible=false)` or as pure functions. E.g. blit a known 2‑colour frame and read the texture back (assert colours + orientation); drive `on_render_tick!` twice with a camera mutation between and assert the frame reframed, exactly one `write_xform!` fired, and `reset!` fired on the moved frame but not on an unmoved repeat.
- **One real‑window integration test** (subprocess): `interactive_display` of a small scene; assert the window opened and `colorbuffer(glscreen)` is non‑black (the blitted RTX frame is visible); optionally drive synthetic drag events and assert the frame changes.
- Renderer work stays **subprocess‑isolated** (carb signal handlers), as in M0–M4. Each task states which mode its tests use.

## Global constraints

- **Single‑threaded render task** — all ovrtx step + readback + blit run on GLMakie's render task with the GL context current. No separate render thread.
- **Bounded `enqueue_wait`** in the UI loop — the interactive step uses a bounded ovrtx wait, never `OVRTX_TIMEOUT_INFINITE`.
- **Display available** — the box has a display (the ovrtx "failed to open the default display" log is just its platform check; it renders offscreen via CUDA regardless), so the viewport opens a real visible window.
- **Reuse the merged backend** — `sync_camera!` / `sync_lights!` / the per‑plot diff nodes / the M2.4 persistent bindings are all on `main`; M5 consumes them unchanged (no "if merged" caveats).
- **`colorbuffer` orientation** — frames are right‑side‑up `Matrix{RGBA{N0f8}}`; the blit preserves that.

## Risks / open items (resolved in the plan)

1. **GLMakie GL context vs ovrtx.** GLMakie owns the window's GL context; ovrtx renders via CUDA offscreen and we upload its frame into a GLMakie texture — they don't share a GL render target, so they coexist. The spike validated the `render_tick` + texture‑upload path. Mitigation: M5.1's window integration test proves a non‑black RTX frame reaches the GLMakie texture before any camera work.
2. **Per‑frame CPU‑blit cost.** A ~2 MB readback + texture upload per frame is small next to the RT2 step. If a future scene makes the blit the bottleneck, that's exactly what M6's GPU‑direct path removes — M5 doesn't need it.
3. **`render_tick` hook semantics** (firing cadence, not consuming events, deregistration on teardown) — pinned down in M5.2/M5.4 against the spike's notes.
4. **Bounded step tuning.** `timeout_ns` must be long enough for a normal RT2 step but short enough that a stuck step doesn't visibly hang; the value is set empirically in M5.2 with a safe default.

## Reconciliation

`M5_PLAN.md` (the pre‑reorder bite‑sized plan) will be regenerated by `writing-plans` from this spec: drop task 5.4 (GPU‑direct → M6), renumber resize/teardown to 5.4, and remove the "if M2.x merged" caveats now that M2–M4 are on `main`.
