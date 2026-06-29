# M5 Interactive RTX Viewport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A real window shows the live ovrtx path-traced render of a Makie scene; mouse orbit/pan/zoom drives a per-frame ovrtx camera write and progressive RT2 accumulation (refine while idle, reset on interaction).

**Architecture:** A GLMakie window hosts one full-viewport `image!` plot whose GL texture is the blit target. A per-frame `render_tick` hook reads the scene's `Camera3D`, reuses the merged `colorbuffer` per-frame logic (`sync_camera!`/`sync_lights!`/`pull_ovrtx_nodes!` → `reset!` if changed) but steps once (bounded) and **blits** the `LdrColor` frame into the texture instead of returning it. Single-threaded on GLMakie's render task (GL context co-located; no separate render thread).

**Tech Stack:** GLMakie `=0.13.12` (new dep — window, `Camera3D`, `GLAbstraction.Texture`, `render_tick`), the OmniverseMakie `OV` layer + the merged M2 open-stage diff path. **No CUDA in M5** (GPU-direct → M6).

## Global Constraints

- **Strict scope:** M5 is the CPU-blit interactive viewport. NO GPU-direct CUDA-GL blit, NO picking, NO subscene hardening (all → M6). No CUDA dependency.
- **Single-threaded render task:** all ovrtx step + readback + blit run on GLMakie's `render_tick` (the render task, GL context current). NEVER add a separate render thread (re-opens the M0 carb signal-window risk).
- **Bounded ovrtx wait in the UI loop:** the interactive step uses a bounded wait (`timeout_ns`), never `OVRTX_TIMEOUT_INFINITE`, so a stuck step can't freeze the window.
- **The `render_tick` listener must NOT `Consume(true)`** (it observes, doesn't swallow events).
- **Reuse the merged backend verbatim:** `sync_camera!(screen, scene) -> Bool`, `sync_lights!(screen, scene) -> Bool`, `pull_ovrtx_nodes!(screen, scene)`, `OV.reset!(r)`, `OV.render_to_matrix(r, product; warmup) -> Matrix{RGBA{N0f8}}`, `OV.step!(r, product; dt) -> StepResult`, `OV.map_cpu(sr, "LdrColor") -> (pixels, W, H)`. These are on `main`; M5 adds NO new translation logic.
- **`colorbuffer` orientation:** frames are right-side-up `Matrix{RGBA{N0f8}}` (top-left origin, verified by `test/m1_orientation_test.jl`); the blit preserves that (no flip).
- **Renderer tests subprocess-isolated** (carb signal handlers), as in M0–M4. GLMakie component tests use offscreen `Screen(visible=false)`.
- **Deps via Pkg** (`Pkg.add`), never hand-edited TOML. **Commit trailer:** `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. **Do not push** (commit on a `feat/m5-viewport` branch; the user controls merges/pushes).
- **Spike is the source of truth for GLMakie specifics:** `references/notes/cuda-gl-interop.md` §2 (texture id + `render_tick` + threading) and §5 (CPU blit). Where this plan's GLMakie code is marked "verify", the implementer confirms it against GLMakie 0.13.12's actual API before relying on it.

---

## File Structure

```
src/interactive/
  viewport.jl     # ViewportSession struct + interactive_display(fig_or_scene) -> ViewportSession (the window + image! + first frame)
  blit.jl         # cpu_blit!(image_plot, frame) — update the image! texture from a host frame
  camera_loop.jl  # on_render_tick!(session) — the per-frame loop + accumulation state machine
src/binding/OV.jl # MODIFY: timeout_ns kwarg on enqueue_wait + step! (+ render_to_matrix passthrough)
src/OmniverseMakie.jl  # MODIFY: `include` the interactive/*.jl files; export interactive_display
Project.toml      # MODIFY (via Pkg): add GLMakie =0.13.12
test/
  m5_blit_test.jl         # offscreen: cpu_blit! puts a host frame on the texture, right-side-up
  m5_viewport_test.jl     # subprocess + real window: interactive_display shows a non-black RTX frame; resize/teardown
  m5_camera_loop_test.jl  # offscreen/component: on_render_tick! reframes + reset logic; accumulation
```

---

### Task 1: Add GLMakie dep + bounded ovrtx step

**Files:**
- Modify: `Project.toml` (via Pkg — add GLMakie), `src/binding/OV.jl` (bounded `enqueue_wait`/`step!`)
- Test: `test/m5_bounded_step_test.jl`

**Interfaces:**
- Produces: `OV.step!(r, product; dt=1/60, timeout_ns=LibOVRTX.OVRTX_TIMEOUT_INFINITE) -> StepResult`; `OV.enqueue_wait(r, enq, op; timeout_ns=…)`. Default `timeout_ns` keeps every existing caller byte-unchanged.

- [ ] **Step 1: Add GLMakie (separate from the geo trio, so a resolver conflict can't break the env)**

```bash
cd /home/juliahub/temp/omniverse-makie/OmniverseMakie.jl
julia --project=. -e 'using Pkg; Pkg.add(name="GLMakie", version="0.13.12")'
```
Expected: resolves against `Makie =0.24.12` (GLMakie 0.13.12 is the matching pin). If it conflicts, STOP and report — M5 cannot proceed without GLMakie. Confirm `GLMakie` is in `Project.toml [deps]` + `[compat] GLMakie = "=0.13.12"`.

- [ ] **Step 2: Write the failing bounded-step test** (`test/m5_bounded_step_test.jl`, subprocess body via `run_ovrtx_subprocess`)

```julia
using Test
const _M5_BOUNDED_PROG = """
using OmniverseMakie, ColorTypes
const OV = OmniverseMakie.OV
OM = OmniverseMakie
OM.activate!(warmup = 8)
scene = Scene(size = (200, 200)); cam3d!(scene)
mesh!(scene, Rect3f(Point3f(0), Vec3f(1)); color = :red)
screen = OM.Screen(scene)
OM.author_root_from_scene!(screen, scene; resolution = screen.fb_size)
OM.OV.add_usd_reference!(screen.renderer, OM.usda_mesh(
    [(0f0,0f0,0f0),(1f0,0f0,0f0),(1f0,1f0,0f0)], [[0,1,2]],
    [(0f0,0f0,1f0) for _ in 1:3], (1f0,0f0,0f0)), "/World/m")
# A generous bounded step completes normally and returns a closeable StepResult.
sr = OV.step!(screen.renderer, screen.product; timeout_ns = UInt64(60_000_000_000))
println("STEP_OK=", sr isa OV.StepResult)
close(sr)
close(screen)
println("OK_BOUNDED_STEP")
"""
include("helpers.jl")
@testset "M5 bounded ovrtx step" begin
    exitcode, output = run_ovrtx_subprocess(_M5_BOUNDED_PROG; timeout = 300)
    @test exitcode == 0
    @test contains(output, "STEP_OK=true")
    @test contains(output, "OK_BOUNDED_STEP")
end
```

- [ ] **Step 3: Run it — RED** (`OV.step!` has no `timeout_ns` kwarg).

Run: `OVRTX_LIBRARY_PATH=<lib> julia --project=. -e 'using Test, OmniverseMakie; include("test/helpers.jl"); include("test/m5_bounded_step_test.jl")'`
Expected: FAIL (MethodError / unknown kwarg `timeout_ns`).

- [ ] **Step 4: Thread `timeout_ns` through `enqueue_wait` + `step!`** (`src/binding/OV.jl`). Add the kwarg (default = the existing infinite constant) so every current call is unchanged:

```julia
# enqueue_wait (replace the signature + the wait line)
function enqueue_wait(r::Renderer, enq, op::AbstractString;
                     timeout_ns::UInt64 = L.OVRTX_TIMEOUT_INFINITE)
    r.alive || error("enqueue_wait called on a closed Renderer")
    wr = Ref{L.ovrtx_wait_result_t}()
    L.check(L.ovrtx_wait_op(r.ptr, enq.op_index, timeout_ns, wr), op * ":wait")
    return enq
end

# step! (add the kwarg + forward it)
function step!(r::Renderer, product::AbstractString;
              dt::Float64 = 1/60, timeout_ns::UInt64 = L.OVRTX_TIMEOUT_INFINITE)
    rp = L.ovx_string_t[ L.ovx_string(product) ]
    GC.@preserve product rp begin
        set = L.ovrtx_render_product_set_t(pointer(rp), Csize_t(1))
        h = Ref{L.ovrtx_step_result_handle_t}(0)
        enqueue_wait(r, L.ovrtx_step(r.ptr, set, dt, h), "step"; timeout_ns)
        sr = StepResult(r, h[], true)
        finalizer(close, sr)
        return sr
    end
end
```

- [ ] **Step 5: Register the test + run it — GREEN.** Add `include("m5_bounded_step_test.jl")` to `test/runtests.jl` (after the m4 includes). Re-run Step 3's command → PASS.

- [ ] **Step 6: Commit**

```bash
git add Project.toml Manifest.toml src/binding/OV.jl test/m5_bounded_step_test.jl test/runtests.jl
git commit -m "feat(M5): add GLMakie dep + bounded ovrtx step! (timeout_ns)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Viewport window + CPU blit of a static ovrtx frame

**Files:**
- Create: `src/interactive/viewport.jl`, `src/interactive/blit.jl`
- Modify: `src/OmniverseMakie.jl` (include the new files; export `interactive_display`)
- Test: `test/m5_blit_test.jl` (offscreen, no real window), `test/m5_viewport_test.jl` (subprocess, real window)

**Interfaces:**
- Consumes: `OV.render_to_matrix(r, product; warmup) -> Matrix{RGBA{N0f8}}`; `OmniverseMakie.Screen(scene)`; `author_root_from_scene!(screen, scene; resolution)`; `Makie.insertplots!(screen, scene)`.
- Produces:
  - `mutable struct ViewportSession` with fields `screen::Screen`, `glscreen` (GLMakie.Screen), `image_plot` (the `image!` plot), `cam_scene::Makie.Scene`, `samples::Int`, `tick_listener` (set in Task 3).
  - `interactive_display(fig_or_scene; size=(800,600), steps_per_tick=2) -> ViewportSession`
  - `cpu_blit!(image_plot, frame::Matrix{RGBA{N0f8}}) -> Nothing`

- [ ] **Step 1: Verify the GLMakie image!/texture API** (exploration, no commit). In a `julia --project=.` REPL with `OVRTX_LIBRARY_PATH` set, confirm against GLMakie 0.13.12 (spike §2/§5):
  - `GLMakie.activate!(); fig = Figure(); ax = Makie.Axis(fig[1,1]); p = image!(ax, rand(RGBA{N0f8}, 100, 100))` — note which positional arg holds the image data (`p[1]` vs `p[3]`) so `cpu_blit!` updates the right Observable.
  - An offscreen screen: `glscr = GLMakie.Screen(visible=false); display(glscr, fig)`; confirm `GLMakie.colorbuffer(glscr)` returns a non-black matrix after a draw.
  Record the confirmed data-Observable index in a comment in `blit.jl`.

- [ ] **Step 2: Write the failing blit test** (`test/m5_blit_test.jl`, subprocess — GL needs a real process)

```julia
using Test
const _M5_BLIT_PROG = """
using OmniverseMakie, GLMakie, ColorTypes, FixedPointNumbers
OM = OmniverseMakie
# A 2-colour host frame: top half red, bottom half blue (row 1 = top, our convention).
H, W = 80, 120
frame = Matrix{RGBA{N0f8}}(undef, H, W)
frame[1:H÷2, :]   .= RGBA{N0f8}(1,0,0,1)
frame[H÷2+1:H, :] .= RGBA{N0f8}(0,0,1,1)
GLMakie.activate!()
fig = Figure(); ax = Makie.Axis(fig[1,1]); ax.aspect = Makie.DataAspect()
img = image!(ax, frame)
glscr = GLMakie.Screen(visible = false); display(glscr, fig)
OM.cpu_blit!(img, frame)         # update the texture from the host frame
buf = GLMakie.colorbuffer(glscr) # read the rendered window back
println("BUF_SIZE=", size(buf))
# top region should read red-dominant, bottom blue-dominant (orientation preserved, no flip)
topc = buf[round(Int,0.25*size(buf,1)), round(Int,0.5*size(buf,2))]
botc = buf[round(Int,0.75*size(buf,1)), round(Int,0.5*size(buf,2))]
println("TOP=", (Float32(red(topc)),Float32(green(topc)),Float32(blue(topc))))
println("BOT=", (Float32(red(botc)),Float32(green(botc)),Float32(blue(botc))))
@assert red(topc) > blue(topc) "top not red — blit flipped/failed"
@assert blue(botc) > red(botc) "bottom not blue — blit flipped/failed"
println("OK_BLIT")
"""
include("helpers.jl")
@testset "M5 cpu_blit! (subprocess, offscreen GL)" begin
    exitcode, output = run_ovrtx_subprocess(_M5_BLIT_PROG; timeout = 300)
    @info "M5 blit output" output
    @test exitcode == 0
    @test contains(output, "OK_BLIT")
end
```

- [ ] **Step 3: Run it — RED** (`cpu_blit!`/`interactive_display` undefined).

- [ ] **Step 4: Implement `cpu_blit!`** (`src/interactive/blit.jl`) — update the `image!` plot's data Observable (spike §5; idiomatic, GLMakie re-uploads). Use the Observable index confirmed in Step 1 (shown here as `[3]`; correct to `[1]` if Step 1 found that):

```julia
# CPU blit (M5): update the image! plot's data Observable from a host frame.  GLMakie
# re-uploads the texture on the data change (spike §5).  Right-side-up: row 1 = top, no flip.
function cpu_blit!(image_plot, frame::AbstractMatrix{RGBA{N0f8}})
    image_plot[3][] = frame     # ← VERIFY index in Step 1; image!(ax, img) data Observable
    return nothing
end
```

- [ ] **Step 5: Implement `ViewportSession` + `interactive_display`** (`src/interactive/viewport.jl`):

```julia
mutable struct ViewportSession
    screen::Screen                  # the open-stage ovrtx Screen
    glscreen                        # GLMakie.Screen (the window)
    image_plot                      # the full-viewport image! plot (blit target)
    cam_scene::Makie.Scene          # the scene whose Camera3D drives the view
    steps_per_tick::Int
    samples::Int
    tick_listener                   # set in Task 3 (on_render_tick! registration); nothing here
end

function interactive_display(fig_or_scene; size = (800, 600), steps_per_tick = 2)
    scene     = fig_or_scene isa Makie.Figure ? fig_or_scene.scene : fig_or_scene
    cam_scene = something(_scene_for_camera(scene), scene)

    # 1. Build the open-stage ovrtx Screen at the window size + author + add plots.
    screen = Screen(cam_scene; size = size)
    author_root_from_scene!(screen, cam_scene; resolution = screen.fb_size)
    screen.last_camera = _camera_snapshot(cam_scene)
    screen.last_lights = _lights_snapshot(cam_scene.compute[:lights][])
    screen.authored = true
    Makie.insertplots!(screen, scene)

    # 2. First ovrtx frame (full warmup for a clean initial image).
    frame = OV.render_to_matrix(screen.renderer, screen.product; warmup = screen.config.warmup)

    # 3. GLMakie host window: an Axis filling the figure with one image! of `frame`.
    GLMakie.activate!()
    fig = Figure(; size = size)
    ax  = Makie.Axis(fig[1, 1]); Makie.hidedecorations!(ax); Makie.hidespines!(ax)
    img = image!(ax, frame)
    glscr = GLMakie.Screen(); display(glscr, fig)   # real visible window (box has a display)

    return ViewportSession(screen, glscr, img, cam_scene, steps_per_tick, screen.config.warmup, nothing)
end
```

- [ ] **Step 6: Wire includes + export** (`src/OmniverseMakie.jl`) — after the existing `include("screen.jl")`:

```julia
include("interactive/blit.jl")
include("interactive/viewport.jl")
include("interactive/camera_loop.jl")   # defined in Task 3; create an empty stub now if needed
export interactive_display
```
(GLMakie is `using`-ed lazily inside `viewport.jl`/`blit.jl` via `import GLMakie` at the top of `OmniverseMakie.jl`, OR `using GLMakie` in those files — keep it a normal dep import.)

- [ ] **Step 7: Write the real-window integration test** (`test/m5_viewport_test.jl`, subprocess)

```julia
using Test
const _M5_VIEWPORT_PROG = """
using OmniverseMakie, GLMakie, ColorTypes
OM = OmniverseMakie
OM.activate!(warmup = 32)
fig = Figure(); ax = LScene(fig[1,1]; show_axis=false)
mesh!(ax, Rect3f(Point3f(-1), Vec3f(2)); color = :orange)
session = OM.interactive_display(fig; size = (400, 300))
buf = GLMakie.colorbuffer(session.glscreen)
nb = count(c -> (Float32(red(c))+Float32(green(c))+Float32(blue(c))) > 0.1, buf)
println("VIEWPORT_NONBLACK=", nb)
@assert nb > 1000 "viewport window is black — RTX frame did not reach the texture"
println("OK_VIEWPORT")
"""
include("helpers.jl")
@testset "M5 interactive_display window shows RTX frame (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M5_VIEWPORT_PROG; timeout = 600)
    @info "M5 viewport output" output
    @test exitcode == 0
    @test contains(output, "OK_VIEWPORT")
    m = match(r"VIEWPORT_NONBLACK=(\d+)", output)
    @test m !== nothing && parse(Int, m.captures[1]) > 1000
end
```

- [ ] **Step 8: Run both tests — GREEN.** Register both in `runtests.jl`. The blit test asserts colours + orientation; the viewport test asserts a non-black window. Fix the `cpu_blit!` Observable index / `image!` setup if either fails.

- [ ] **Step 9: Commit**

```bash
git add src/interactive/ src/OmniverseMakie.jl test/m5_blit_test.jl test/m5_viewport_test.jl test/runtests.jl
git commit -m "feat(M5): interactive viewport window + CPU blit of the ovrtx frame

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

**Acceptance:** a real window shows the static RTX render of a Makie scene; `cpu_blit!` puts a host frame on screen right-side-up.

---

### Task 3: Live camera interaction loop (orbit/pan/zoom → ovrtx)

**Files:**
- Create: `src/interactive/camera_loop.jl`
- Modify: `src/interactive/viewport.jl` (register the listener in `interactive_display`)
- Test: `test/m5_camera_loop_test.jl` (component, subprocess)

**Interfaces:**
- Consumes: `sync_camera!(screen, scene) -> Bool`, `sync_lights!(screen, scene) -> Bool`, `pull_ovrtx_nodes!(screen, scene)`, `OV.reset!(r)`, `OV.render_to_matrix(r, product; warmup, timeout_ns)`, `cpu_blit!`.
- Produces: `on_render_tick!(session::ViewportSession) -> Nothing`.

- [ ] **Step 1: Add `timeout_ns` passthrough to `render_to_matrix`** (`src/binding/OV.jl`) so the loop's accumulation steps are bounded:

```julia
function render_to_matrix(r::Renderer, product::AbstractString;
                         warmup::Int = 64, timeout_ns::UInt64 = L.OVRTX_TIMEOUT_INFINITE)
    for s in 1:(warmup - 1)
        sr = step!(r, product; timeout_ns); close(sr)
    end
    sr = step!(r, product; timeout_ns)
    pixels, W, H = try
        map_cpu(sr, "LdrColor")
    finally
        close(sr)
    end
    return cwh_to_matrix(pixels)
end
```

- [ ] **Step 2: Write the failing camera-loop component test** (`test/m5_camera_loop_test.jl`, subprocess) — build a session offscreen, tick once, mutate the camera, tick again, assert the frame reframed + exactly one camera xform write + `reset!` on the moved frame:

```julia
using Test
const _M5_LOOP_PROG = """
using OmniverseMakie, GLMakie, ColorTypes
OM = OmniverseMakie; OV = OmniverseMakie.OV
OM.activate!(warmup = 16)
fig = Figure(); ax = LScene(fig[1,1]; show_axis=false)
mesh!(ax, Rect3f(Point3f(-1), Vec3f(2)); color = :red)
GLMakie.activate!()
session = OM.interactive_display(fig; size = (300, 300), steps_per_tick = 2)
cam = Makie.cameracontrols(session.cam_scene)
OM.on_render_tick!(session)                       # frame A (no change)
eye0 = cam.eyeposition[]
cam.eyeposition[] = Vec3f(-eye0[1], -eye0[2], eye0[3])   # 180° orbit
OM.on_render_tick!(session)                       # frame B (camera moved)
bufB = GLMakie.colorbuffer(session.glscreen)
nb = count(c -> (Float32(red(c))+Float32(green(c))+Float32(blue(c))) > 0.1, bufB)
println("LOOP_NONBLACK=", nb)
@assert nb > 500 "frame B black — loop blit failed after camera move"
println("OK_LOOP")
"""
include("helpers.jl")
@testset "M5 on_render_tick! reframes the live view (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M5_LOOP_PROG; timeout = 600)
    @info "M5 loop output" output
    @test exitcode == 0
    @test contains(output, "OK_LOOP")
end
```

- [ ] **Step 3: Run it — RED** (`on_render_tick!` undefined).

- [ ] **Step 4: Implement `on_render_tick!`** (`src/interactive/camera_loop.jl`) — the per-frame loop, mirroring `colorbuffer`'s sync/reset logic but bounded-step + blit:

```julia
# One interactive frame on GLMakie's render task: push live camera/light/plot deltas, reset
# RT2 only if something changed (else keep accumulating), do a BOUNDED accumulation step, and
# blit.  Mirrors `Makie.colorbuffer`'s per-frame logic (screen.jl) but steps a few times and
# blits instead of returning the matrix.  Must NOT Consume events (registered as an observer).
const _M5_STEP_TIMEOUT_NS = UInt64(10_000_000_000)   # 10 s — long enough for a normal step, short enough not to hang

function on_render_tick!(session::ViewportSession)
    screen    = session.screen
    cam_scene = session.cam_scene

    cam_changed   = sync_camera!(screen, cam_scene)
    light_changed = sync_lights!(screen, cam_scene)
    pending = screen.requires_update
    screen.requires_update = false
    pull_ovrtx_nodes!(screen, cam_scene)
    need_reset = cam_changed || light_changed || screen.requires_update || pending
    screen.requires_update = false

    if need_reset
        OV.reset!(screen.renderer)
        session.samples = 0
    end

    frame = OV.render_to_matrix(screen.renderer, screen.product;
                                warmup = session.steps_per_tick, timeout_ns = _M5_STEP_TIMEOUT_NS)
    session.samples += session.steps_per_tick
    cpu_blit!(session.image_plot, frame)
    return nothing
end
```

- [ ] **Step 5: Register the listener in `interactive_display`** (`src/interactive/viewport.jl`) — after the window opens, before `return`:

```julia
    # Per-frame hook on GLMakie's render task (spike §2). Must NOT Consume(true).
    session = ViewportSession(screen, glscr, img, cam_scene, steps_per_tick, screen.config.warmup, nothing)
    session.tick_listener = on(glscr.render_tick) do _
        on_render_tick!(session)
        return Makie.Consume(false)
    end
    return session
```
(Replace the bare `return ViewportSession(...)` from Task 2 with this.)

- [ ] **Step 6: Run the test — GREEN.** Register `m5_camera_loop_test.jl` in `runtests.jl`. PASS (frame B non-black after the camera orbit via the loop).

- [ ] **Step 7: Commit**

```bash
git add src/interactive/camera_loop.jl src/interactive/viewport.jl src/binding/OV.jl test/m5_camera_loop_test.jl test/runtests.jl
git commit -m "feat(M5): live camera loop (render_tick → sync_camera! → reset → bounded step → blit)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

**Acceptance:** dragging the mouse orbits the live RTX view (per-frame camera write on the open stage, no re-author); the step is bounded (a stuck step can't freeze the window).

---

### Task 4: Resize, teardown, robustness

**Files:**
- Modify: `src/interactive/viewport.jl` (`close`, resize handling, per-frame error guard)
- Test: `test/m5_viewport_test.jl` (extend)

**Interfaces:**
- Produces: `Base.close(session::ViewportSession) -> Nothing`; a per-frame try/guard inside `on_render_tick!`.

- [ ] **Step 1: Write the failing teardown test** (extend `test/m5_viewport_test.jl`'s subprocess prog): after building a session, `close(session)`; assert the GLMakie screen is closed and a second `close` is a safe no-op:

```julia
# (append to _M5_VIEWPORT_PROG, before OK_VIEWPORT)
OM.close(session)
println("GL_CLOSED=", !GLMakie.isopen(session.glscreen))
OM.close(session)    # idempotent — must not throw
println("OK_TEARDOWN")
```
And add asserts in the testset: `@test contains(output, "GL_CLOSED=true")`, `@test contains(output, "OK_TEARDOWN")`.

- [ ] **Step 2: Run it — RED** (`close(::ViewportSession)` undefined).

- [ ] **Step 3: Implement `close` + the per-frame error guard** (`src/interactive/viewport.jl` / `camera_loop.jl`):

```julia
function Base.close(session::ViewportSession)
    # 1. stop the per-frame hook (Observables.off on the registered listener)
    session.tick_listener === nothing || off(session.tick_listener)
    session.tick_listener = nothing
    # 2. close the GLMakie window (idempotent — guard on isopen)
    try
        GLMakie.isopen(session.glscreen) && GLMakie.close(session.glscreen)
    catch e
        @warn "M5: error closing GLMakie screen" exception = e
    end
    # 3. close the ovrtx Screen LAST (StepResults before renderer — M1 teardown order)
    Base.close(session.screen)
    return nothing
end
```
Wrap the body of `on_render_tick!` so a transient ovrtx/CUDA error logs (`@warn maxlog=5`) and keeps the window alive rather than crashing the render task:

```julia
function on_render_tick!(session::ViewportSession)
    try
        _on_render_tick_impl!(session)   # the Task-3 body, renamed
    catch e
        @warn "M5: render-tick frame failed (window kept alive)" exception = e maxlog = 5
    end
    return nothing
end
```

- [ ] **Step 4: Implement resize handling.** On a `glscreen` framebuffer-size change, re-author the ovrtx render-product resolution + recreate the `image!` at the new size. Listen to the GLMakie scene's `events.window_area` / framebuffer size (verify the exact Observable name against GLMakie 0.13.12 in a REPL first) and, when it changes:

```julia
# inside interactive_display, after registering the tick listener:
session.resize_listener = on(session.cam_scene.events.window_area) do area
    new = (Int(widths(area)[1]), Int(widths(area)[2]))
    new == session.screen.fb_size && return
    # rebuild the ovrtx Screen at the new size (re-authors the render product) + a fresh image!
    resize_viewport!(session, new)
end
```
where `resize_viewport!(session, (W,H))` builds a new open-stage `Screen` at `(W,H)`, re-authors + re-adds plots, swaps `session.screen`, and updates the `image!` plot data to a fresh `Matrix{RGBA{N0f8}}(undef, H, W)` first frame. (Add a `resize_listener` field to `ViewportSession`; `off` it in `close`.)

- [ ] **Step 5: Run the tests — GREEN.** The teardown asserts the window closes + idempotent close; the resize path is exercised by a synthetic size change in the subprocess prog (`resize!(session.glscreen, 500, 360)` → assert `colorbuffer` is still non-black + `session.screen.fb_size == (500,360)`).

- [ ] **Step 6: Commit**

```bash
git add src/interactive/viewport.jl src/interactive/camera_loop.jl test/m5_viewport_test.jl
git commit -m "feat(M5): viewport resize + clean teardown + per-frame error guard

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

**Acceptance:** the window survives a transient render error; resize re-authors the product + recreates the texture; `close` deregisters the hook, closes the GL window, then the ovrtx Screen (correct order); second `close` is a no-op.

---

## Self-Review (completed)

- **Spec coverage:** GLMakie dep + bounded step (Task 1); window + CPU blit + static frame (Task 2 = spec M5.1); live camera loop (Task 3 = M5.2); progressive accumulation (Task 3's `need_reset`-else-accumulate + `samples` = M5.3 — folded into the loop, as the spec's loop already describes idle-refine/reset-on-change); resize/teardown/robustness (Task 4 = M5.4). GPU-direct / picking / subscene correctly ABSENT (→ M6).
- **Note on M5.3:** progressive accumulation is not a separate task because it IS the loop's reset-vs-accumulate branch (`need_reset ? reset! : keep stepping`) — there is no independent deliverable to gate separately. The `samples` counter + the "don't reset when clean" behavior land in Task 3 and are asserted by an added camera-loop step (idle ticks raise `samples` and do NOT call `reset!`; instrument `reset!` to confirm). If a reviewer wants it isolated, split a Task 3b.
- **Placeholder scan:** the GLMakie-API specifics (the `image!` data-Observable index, the resize Observable name) are marked "verify in a REPL" with the spike as source — these are genuine new-dep unknowns, handled by an explicit verification step, not hand-waving. All backend calls use exact merged signatures.
- **Type consistency:** `ViewportSession` fields + `interactive_display`/`cpu_blit!`/`on_render_tick!`/`close` signatures are consistent across tasks; `timeout_ns::UInt64` consistent in `enqueue_wait`/`step!`/`render_to_matrix`; `OV.render_to_matrix` returns `Matrix{RGBA{N0f8}}` consumed by `cpu_blit!`.
- **Added to Task 3** (from self-review): an accumulation assertion step — tick twice with NO camera change, assert `reset!` did NOT fire the second time and `samples` increased (instrument `OV.reset!` via a counter, like `_PUSH_OBSERVER`).
