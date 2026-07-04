module OmniverseMakieGLMakieExt

using OmniverseMakie, GLMakie
using OmniverseMakie: Makie, RGBA, N0f8, ColorTypes
# main-module internals used by the moved src/interactive code (bare names):
import OmniverseMakie: Screen, OV, _author_screen!, _sync_and_needs_reset!,
    _scene_for_camera, interactive_display, present!, on_render_tick!,
    attach_picking!, detach_picking!, _pick_at!,  # M6.B Task 5: picking
    replace_scene!, record_frame!, ScreenConfig    # hybrid embedded viewport
using OmniverseMakie: tonemap
using Makie: Consume, MouseButtonEvent  # event types for the input forwarders

# A USDPlot is a CHILDLESS recipe, so GLMakie's insert! routes it to the atomic branch
# (plot-primitives.jl: `isempty(x.plots)` → draw_atomic) — without a method here,
# `display(fig)` MethodErrors, and that display is the replace_scene! PRECONDITION.  GL cannot
# render a USD reference, so draw NOTHING (the usdplot docstring's contract; same shape as
# GLMakie's own `draw_atomic(::Screen, ::Scene, ::PlotList) = nothing` for childless
# PlotLists) — the asset appears only through an ovrtx path (offscreen colorbuffer,
# interactive_display, or the replace_scene! overlay).  Deletion is already safe: GLMakie's
# `delete!(screen, scene, plot)` tolerates a plot with no cached renderobject.
GLMakie.draw_atomic(::GLMakie.Screen, ::Makie.Scene, ::OmniverseMakie.USDPlot) = nothing

# ===== moved verbatim from src/interactive/blit.jl =====
# CPU blit (M5): set the image! plot's data Observable from a frame; GLMakie
# re-uploads the texture on the change (idiomatic Makie path).
#
# Orientation: the ovrtx frame is Matrix{RGBA{N0f8}}[H,W], row 1 = top.  Makie
# image! is [x, y] with y increasing UPWARD, so map frame→display via
# reverse(permutedims(frame), dims=2): permutedims → [W,H]; reverse dim 2 → y-up
# (data[col,H]=frame[1,col] at TOP, data[col,1]=frame[H,col] at BOTTOM).
# image! data is Observable [3] (x=[1], y=[2]) — REPL-verified.

# The one host-HDR→display orientation is the fused tonemap+orient loop
# `_tonemap_orient!` below: present!, interactive_display, and resize_viewport! all
# fill their session-owned [W,H] display buffer through it — reproducing
# reverse(permutedims(tonemap_frame([C,W,H] hdr, exposure)), dims=2) via out[j, H+1-i]
# (the SAME fused indexing the CUDA ext's oriented copy uses).

# Fused host tonemap + display-orient: write the [C,W,H] linear-HDR `hdr` INTO the
# pre-oriented [W,H] RGBA{N0f8} buffer `out` in place — one pass, no intermediate
# frame / permutedims / reverse.  out[j, H+1-i] reproduces
# reverse(permutedims(tonemap_frame(hdr, exposure)), dims=2) exactly.  scale =
# exp2(exposure) is hoisted out of the W×H loop (once per tick); the per-pixel
# `tonemap` is the SHARED scalar (src/tonemap.jl), so host and CUDA kernel agree.
# `hdr` is eltype-generic (`<:Real`) with a per-element `Float32(...)` promotion, EXACTLY
# like the CUDA oriented kernel (C3, `_tonemap_orient_kernel!`): it accepts either the Float32
# [C,W,H] from render_hdr_to_array (interactive_display / resize_viewport! warmups) OR — for the
# INT-2 zero-copy present! — the still-mapped Float16 HdrColor view DIRECTLY (no Float32 transient).
# The Float32 path stays byte-identical (`Float32(::Float32)` is identity, goldens/C2/C3 pin it);
# Float16→Float32 widening is exact, so the zero-copy Float16 path matches the widen-first Float32
# path pixel-for-pixel.
#
# Threaded over columns j: each j writes a DISJOINT output column (out[j, ·]) and every pixel is
# independent scalar math with NO reductions, so the result is BYTE-IDENTICAL at any thread count
# (the CUDA twin's ≤1-LSB gate is preserved).  At nthreads()==1 an explicit serial branch avoids
# `Threads.@threads`'s task-setup allocation, keeping a warmed `_cpu_present!` tick inside the C2
# absolute allocation gate (test/viewport/present_cpu_test.jl).  The subprocess test harness spawns
# single-threaded children, so tests exercise the serial branch; users get the speedup with
# `julia -t auto` (measured 17.9 ms → parallel at 1.14 Mpx, the 1080p present's #2 cost).
function _tonemap_orient!(out::AbstractMatrix{RGBA{N0f8}},
                          hdr::AbstractArray{<:Real,3}, exposure::Float32)
    C, W, H = size(hdr)
    # The loop bounds come from `hdr` and the writes are `@inbounds`, so `out` MUST be the
    # oriented [W,H] — guard it explicitly (the CUDA twin `_tonemap_orient_kernel!` carries the
    # same guard per-thread).  Callers that only @warn on a size mismatch (resize_viewport!)
    # would otherwise turn a mismatch into an out-of-bounds write instead of a clean error.
    size(out) == (W, H) ||
        throw(DimensionMismatch("_tonemap_orient!: out is $(size(out)), need ($W, $H) for the [C=$C,W,H] hdr"))
    scale = exp2(exposure)                          # once per tick, hoisted out of the W×H loop
    if Threads.nthreads() == 1
        @inbounds for j in 1:W, i in 1:H
            out[j, H + 1 - i] = tonemap((Float32(hdr[1, j, i]), Float32(hdr[2, j, i]), Float32(hdr[3, j, i])), scale)
        end
    else
        Threads.@threads for j in 1:W
            @inbounds for i in 1:H
                out[j, H + 1 - i] = tonemap((Float32(hdr[1, j, i]), Float32(hdr[2, j, i]), Float32(hdr[3, j, i])), scale)
            end
        end
    end
    return out
end

# ===== moved verbatim from src/interactive/viewport.jl =====
# Viewport session: a GLMakie window showing the live ovrtx RTX frame.
#
# Display (M5): one bare `campixel!` Scene holding a full-viewport `image!` (the
# blit target).  No Axis (its 2-D zoom/pan/limits are meaningless for a raw
# framebuffer).  The path tracer renders OFFSCREEN; each tick uploads its frame
# into the image texture (`present!`).
#
# Interaction (M5): campixel! has no 3-D camera, so FORWARD window input (mouse
# button/position/scroll/keyboard) into `cam_scene.events`.  cam_scene owns the
# `Camera3D` that `on_render_tick!` reads, so drag orbits / scroll zooms live.

"""
    mutable struct ViewportSession

All state for an interactive RTX viewport window (M5).

Fields:
  - `screen::Screen`        — open-stage ovrtx Screen (stage, renderer, plots)
  - `glscreen`              — GLMakie.Screen (the host window)
  - `glscene::Makie.Scene`  — the campixel! display Scene (image + input source)
  - `image_plot`            — full-viewport image! plot (present! target)
  - `present_buf`           — `[W,H]` pre-oriented `RGBA{N0f8}` display buffer; IS the
                              image! data array (CPU `present!` writes it in place + notify)
  - `cam_scene::Makie.Scene`— scene whose Camera3D drives the ovrtx view
  - `steps_per_tick::Int`   — ovrtx accumulation steps per render tick
  - `samples::Int`          — RT2 accumulation counter (0 on reset, += steps_per_tick)
  - `tick_listener`         — the render_tick per-frame hook
  - `input_listeners::Vector`— glscene→cam_scene forwarders (orbit/zoom)
  - `resize_listener`       — window-resize hook, or `nothing`
  - `exposure::Float32`     — EV exposure stops (0 = none) for the HDR tonemap
  - `blitter::Symbol`       — `:cpu` or `:gpu` — the present! strategy
  - `gpu_state`             — CUDA-ext GPUBlitState (lazy), or `nothing`; reset to
                              `:cpu` on GPU-setup failure (graceful CPU fallback)
  - `gpu_forced::Bool`      — gpu_direct=true was forced; surface GPU errors, no fallback
"""
mutable struct ViewportSession
    screen::Screen                  # the open-stage ovrtx Screen
    glscreen                        # GLMakie.Screen (the window)
    glscene::Makie.Scene            # the campixel! display Scene
    image_plot                      # full-viewport image! plot (blit target)
    present_buf::Matrix{RGBA{N0f8}} # [W,H] pre-oriented display buffer = image_plot's data array
    cam_scene::Makie.Scene          # the scene whose Camera3D drives the view
    steps_per_tick::Int
    samples::Int
    tick_listener                   # render_tick listener
    input_listeners::Vector         # glscene→cam_scene forwarding listeners
    resize_listener                 # window-resize hook, or nothing
    exposure::Float32               # EV stops for ACES tonemap
    blitter::Symbol                 # :cpu or :gpu — present! strategy
    gpu_state                       # CUDA-ext GPUBlitState (lazy) or nothing
    gpu_forced::Bool                # gpu_direct=true forced; no CPU fallback
end

# Input events forwarded from the display window into the camera scene, so the
# existing Camera3D orbits/zooms in response to the window's mouse/keyboard.
const _M5_FORWARDED_EVENTS = (:mousebutton, :mouseposition, :scroll, :keyboardbutton)

"""
    _pick_blitter(gpu_direct::Symbol_or_Bool) -> :gpu | :cpu

Resolve the per-frame blit strategy for `interactive_display`.  GPU-direct is
available only when the CUDA package extension is loaded AND `CUDA.functional()`.

  - `:auto`  → `:gpu` if CUDA is functional, else `:cpu` (the default).
  - `true`   → `:gpu` if CUDA is functional, else `error` (the caller demanded GPU).
  - `false`  → `:cpu` (always).
"""
function _pick_blitter(gpu_direct)
    cuda_ready = false
    try
        cuda_ready = Base.get_extension(OmniverseMakie, :OmniverseMakieCUDAExt) !== nothing &&
                     Base.invokelatest(OmniverseMakie._cuda_functional)
    catch
        cuda_ready = false
    end
    gpu_direct === :auto ? (cuda_ready ? :gpu : :cpu) :
    gpu_direct === true  ? (cuda_ready ? :gpu : error("gpu_direct=true but CUDA is unavailable (load `using CUDA` and ensure CUDA.functional())")) :
    :cpu
end

"""
    interactive_display(fig_or_scene; size=(800,600), steps_per_tick=2,
                        exposure=0f0, gpu_direct=:auto, selection_outline=false) -> ViewportSession

Open an orbit-able GLMakie window showing the live ovrtx RTX render of `fig_or_scene`.

Authors the open ovrtx stage once (mirrors `colorbuffer`'s open-once path), renders a
warmup frame into a single `campixel!` Scene as a full-viewport `image!` (no Axis), and
forwards window input to the scene's `Camera3D` so drag orbits / scroll zooms the live
view; a `render_tick` hook re-accumulates each frame (`on_render_tick!`).  Render
resolution is set to `size` by resizing the input scene's root before building `Screen`.

`gpu_direct` picks the blit path (M6.A): `:auto` = GPU-direct when the CUDA ext is
loaded and functional else CPU; `true` = force GPU-direct (errors if unavailable);
`false` = CPU.  GPU-setup failure degrades gracefully to `:cpu`.
`selection_outline=true` is NOT yet supported (the HdrColor viewport can't carry the
LdrColor-only outline) — it throws; use offscreen `select!` for outline images.

Calls `GLMakie.activate!()`; call `OmniverseMakie.activate!()` afterwards to restore the
ovrtx backend for offscreen `save`/`colorbuffer`.
"""
function interactive_display(fig_or_scene::Union{Makie.Figure,Makie.Scene}; size = (800, 600), steps_per_tick = 2, exposure = 0f0, gpu_direct = :auto, selection_outline::Bool = false)
    # M6.B (scope): the viewport presents HdrColor, but a selection-outline
    # Screen is LdrColor-only (no HdrColor AOV) — both present paths map
    # HdrColor and would throw.  Rather than build an LdrColor live-present
    # now, refuse selection_outline=true with an actionable error (offscreen
    # select! + render_to_matrix/colorbuffer give outline images).  false
    # (default) keeps a normal HdrColor Screen — byte-identical to before.
    selection_outline && throw(ArgumentError("interactive_display(; selection_outline=true) is not yet supported: the live viewport presents via the HdrColor path, but the selection outline requires an LdrColor-only Screen. Use offscreen `select!` + `Makie.colorbuffer`/`render_to_matrix` for outline images, or pass selection_outline=false. (LdrColor live-present is a planned follow-up.)"))
    scene     = fig_or_scene isa Makie.Figure ? fig_or_scene.scene : fig_or_scene
    cam_scene = something(_scene_for_camera(scene), scene)

    # 1. Resize root so Screen() (size(Makie.root(scene))) renders at `size`.
    Makie.resize!(Makie.root(cam_scene), size...)

    # 2. Build + author open-stage ovrtx Screen (colorbuffer open-once path).
    screen = Screen(cam_scene)
    screen.fb_size == size ||
        error("interactive_display: expected fb_size $(size), got $(screen.fb_size)")
    _author_screen!(screen, cam_scene, scene)

    # 3. First ovrtx frame (full warmup for a clean initial image) via HDR path.
    warmup_hdr = OV.render_hdr_to_array(screen.renderer, screen.product;
                                        warmup = screen.config.warmup)
    # The eager insertplots! flipped requires_update; the warmup already drew
    # that geometry, so consume the flag (as colorbuffer does) — else the first
    # on_render_tick! would redundantly OV.reset! and discard this clean frame.
    screen.requires_update = false

    # 4. Display: one campixel! Scene with a full-viewport image! whose data array IS the
    #    session-owned present buffer, filled by the fused tonemap+orient loop (the one
    #    orientation → frame row 1 = top).  present! writes this buffer in place each tick.
    GLMakie.activate!()
    W, H = size
    glscene = Makie.Scene(size = size)
    Makie.campixel!(glscene)
    present_buf = Matrix{RGBA{N0f8}}(undef, W, H)
    _tonemap_orient!(present_buf, warmup_hdr, exposure)
    img = image!(glscene, 0 .. W, 0 .. H, present_buf; interpolate = false)
    glscr = GLMakie.Screen()
    display(glscr, glscene)

    # 5. Forward window input to cam_scene so its Camera3D orbits/zooms
    #    (campixel! has no 3-D camera).  One-way; listeners observe (never
    #    Consume), so GLMakie still processes events.
    input_listeners = [on(getproperty(glscene.events, f)) do v
        getproperty(cam_scene.events, f)[] = v
        return Makie.Consume(false)
    end for f in _M5_FORWARDED_EVENTS]

    # Blit strategy: :gpu when CUDA loaded + functional, else :cpu.
    blitter = _pick_blitter(gpu_direct)
    # Only literal `true` is forced; :auto/false allow graceful CPU fallback.
    gpu_forced = (gpu_direct === true)

    session = ViewportSession(screen, glscr, glscene, img, present_buf, cam_scene,
                              steps_per_tick, screen.config.warmup,
                              nothing, input_listeners, nothing, exposure,
                              blitter, nothing, gpu_forced)

    # 6. Per-frame hook on GLMakie's render task. Must NOT Consume(true).
    session.tick_listener = on(glscr.render_tick) do _
        on_render_tick!(session)
        return Makie.Consume(false)
    end

    # 7. Window-resize hook (resize_viewport!): rebuild the ovrtx renderer +
    #    image! at the new size.  Fires on glscene.events.window_area (GLFW size
    #    callback); the initial fire carries fb_size and is a no-op.
    #    Guarded like on_render_tick!: `notify(::Observable)` does NOT catch listener
    #    errors, so an uncaught throw here (e.g. the intermittent ovrtx startup crash
    #    while rebuilding the Screen) would kill GLMakie's @async render task and freeze
    #    the whole window.  On failure the session keeps the OLD screen (warn + carry on).
    resize_inflight = Ref(false)   # re-entrancy guard: two entries past the fb_size check
                                   # would double-build/double-free the ovrtx Screen
    session.resize_listener = on(glscene.events.window_area) do area
        w, h = Int.(widths(area))
        (w, h) == session.screen.fb_size && return   # no change — skip
        w > 0 && h > 0                  || return   # skip degenerate sizes
        resize_inflight[] && return
        resize_inflight[] = true
        try
            resize_viewport!(session, (w, h))
        catch e
            @warn "M5: viewport resize failed (window kept alive at the old size)" exception=(e, catch_backtrace()) maxlog=5
        finally
            resize_inflight[] = false
        end
        return nothing
    end

    return session
end

# ------------------------------------------------------------------
# Teardown
# ------------------------------------------------------------------

"""
    Base.close(session::ViewportSession) -> Nothing

Tear down the interactive viewport: detach all listeners, close the GLMakie window,
then close the underlying ovrtx `Screen` (renderer + bindings).

Idempotent: a second `close` is a safe no-op (listener guards + `renderer.alive`
check in `Base.close(::Screen)` absorb repeated calls).
"""
function Base.close(session::ViewportSession)
    session.tick_listener   === nothing || off(session.tick_listener)
    session.resize_listener === nothing || off(session.resize_listener)
    for l in session.input_listeners; off(l); end
    empty!(session.input_listeners)
    session.tick_listener   = nothing
    session.resize_listener = nothing
    # M6.A: STOP the render loop and JOIN the render task BEFORE any teardown.
    # The task fires render_tick→on_render_tick!→present!, which for GPU-direct
    # runs raw CUDA→GL interop on ovrtx/CUDA resources — tearing those down
    # mid-tick is a use-after-free (segfault).  stop_renderloop! joins the task
    # (called off it), so no present! races the teardown below.
    try
        isopen(session.glscreen) && GLMakie.stop_renderloop!(session.glscreen)
    catch e
        @warn "M6: error stopping GLMakie render loop" exception=e
    end
    # M6.A: unregister the GPU-direct GL texture resource (texture still alive,
    # loop stopped) — MUST run BEFORE close(glscreen) destroys the texture.
    if session.gpu_state !== nothing
        try
            Base.invokelatest(OmniverseMakie._gpu_teardown!, session.gpu_state)
        catch e
            @warn "M6: error tearing down GPU-direct blit state" exception=e
        end
        session.gpu_state = nothing
    end
    try
        isopen(session.glscreen) && close(session.glscreen)
    catch e
        @warn "M5: error closing GLMakie screen" exception=e
    end
    Base.close(session.screen)   # ovrtx Screen LAST (bindings→renderer; M1)
    return nothing
end

# ------------------------------------------------------------------
# Window resize
# ------------------------------------------------------------------

"""
    resize_viewport!(session::ViewportSession, (W, H)::Tuple{Int,Int}) -> Nothing

Rebuild the ovrtx renderer at `(W, H)` and refresh the displayed image.  Called by
the `resize_listener` on `glscene.events.window_area`.

Resize root → build+author a new ovrtx `Screen` + warmup → swap it in, then close the
OLD `Screen` (avoid a GPU leak) → replace the `image!` plot at the new size (delete +
recreate, avoiding image! range-change edge cases).
"""
function resize_viewport!(session::ViewportSession, (W, H)::Tuple{Int,Int})
    cam_scene = session.cam_scene
    root_scene = Makie.root(cam_scene)  # root scene passed to insertplots!

    # 1. Resize root so Screen() picks up the new resolution.
    Makie.resize!(root_scene, W, H)

    # 2. Build + author the new ovrtx Screen (same as interactive_display).
    new_screen = Screen(cam_scene)
    if new_screen.fb_size != (W, H)
        @warn "M5 resize: expected fb_size $((W, H)), got $(new_screen.fb_size)"
    end
    _author_screen!(new_screen, cam_scene, root_scene)

    # 3. Render a warmup frame at the new size via HDR path.
    warmup_hdr = OV.render_hdr_to_array(new_screen.renderer, new_screen.product;
                                        warmup = new_screen.config.warmup)

    # 4. Swap in the new Screen FIRST (session stays self-consistent), then
    #    free the OLD ovrtx renderer (avoid a GPU leak).  Reset the counter.
    old_screen      = session.screen
    session.screen  = new_screen
    session.samples = new_screen.config.warmup
    # Consume the warmup flag (as interactive_display does) so the first tick
    # does not redundantly reset and discard this clean frame.
    new_screen.requires_update = false
    Base.close(old_screen)

    # 5. Replace the image!: delete the wrong-size one, allocate a fresh [W,H] present
    #    buffer filled by the fused tonemap+orient loop (the one orientation), and add an
    #    image! backed by it spanning 0..W,0..H (campixel! coords cover that after resize)
    #    — so the next present! writes it in place (its size guard passes).
    # M6.A: unregister the OLD CUDA-GL resource BEFORE delete! (texture still
    # alive → no leak).  GL may RECYCLE the freed texture id, so explicit
    # unregister + the present! `!st.registered` re-register guard make resize
    # id-recycle-proof (A-2).  No-op for CPU / no-CUDA (gpu_state === nothing).
    session.gpu_state === nothing || Base.invokelatest(OmniverseMakie.gpu_unregister!, session)
    delete!(session.glscene, session.image_plot)
    new_buf = Matrix{RGBA{N0f8}}(undef, W, H)
    _tonemap_orient!(new_buf, warmup_hdr, session.exposure)
    new_img = image!(session.glscene, 0 .. W, 0 .. H, new_buf; interpolate = false)
    session.image_plot  = new_img
    session.present_buf = new_buf

    return nothing
end

# ===== moved verbatim from src/interactive/camera_loop.jl =====
# Camera loop (M5 Task 3): per-frame hook on GLMakie's render task — push live
# camera/light/plot deltas, reset RT2 accumulation only on a change, step, blit.
# Mirrors colorbuffer's sync/reset (screen.jl:342-358) but bounded-steps, blits
# (not returns).  Must NOT Consume (registered as an observer, not a handler).
#
# Accumulation state machine:
#   - idle tick (no delta): samples += steps_per_tick (RT2 keeps refining)
#   - change tick (sync_camera!/sync_lights!/pull_ovrtx_nodes!): OV.reset!,
#     samples = 0, then += steps_per_tick

"""Bounded step timeout: 10 s — long enough for a normal RT2 step, short enough not to hang."""
const _M5_STEP_TIMEOUT_NS = UInt64(10_000_000_000)

# ------------------------------------------------------------------
# present! — CPU HDR blit: HdrColor → host tonemap → image! data
# ------------------------------------------------------------------

"""
    OmniverseMakie.present!(session::ViewportSession, ::Val{:cpu}) -> Nothing

`:cpu` blit strategy: step ovrtx `steps_per_tick` times (host twin of the CUDA ext's
`_gpu_present!` step loop — drop each intermediate `StepResult`, keep the final `sr`), then map
its `HdrColor` (float16) and in ONE fused pass tonemap (ACES + sRGB + `session.exposure`) AND
orient it STRAIGHT from the still-mapped Float16 view into the session's cached `[W,H]` display
buffer (`present_buf`) IN PLACE, then `notify` the image! plot's data Observable so GLMakie
re-uploads the texture.  Steady state = ZERO full-frame allocations: the buffer is reused and
(INT-2) the tonemap reads the mapped Float16 directly, so the Float32 `[C,W,H]` HDR transient that
`render_hdr_to_array` used to materialize is gone.  Everything touching the mapped view stays
INSIDE the `with_mapped_hdr` closure (the mapping dies on unmap at its return); the loop fully
materializes into `present_buf` (owned — NOT a view over the mapping), so nothing lazy escapes.  A
size guard reallocates + re-seats the buffer after a resize (Screen rebuilt).  The `:gpu` strategy
(on-device tonemap + CUDA→GL copy, no host roundtrip) lives in the CUDA extension; both are selected
via `present!(session, Val(session.blitter))`.
"""
OmniverseMakie.present!(session::ViewportSession, ::Val{:cpu}) = _cpu_present!(session)

# Shared CPU blit, duck-typed over any session with `screen`/`steps_per_tick`/`present_buf`/
# `image_plot`/`exposure` — used by BOTH the standalone ViewportSession and the embedded
# `replace_scene!` session, so there is exactly one host tonemap-present path (no drift).
function _cpu_present!(session)
    screen = session.screen
    # Host twin of the CUDA ext's _gpu_present! step loop: run `steps_per_tick` bounded RT2 steps,
    # closing each dropped StepResult; KEEP the final `sr` to map its HdrColor.  (Was
    # OV.render_hdr_to_array, which ADDITIONALLY materialized a Float32 [C,W,H] HDR copy per tick —
    # the transient INT-2 removes; render_hdr_to_array itself stays for colorbuffer/HDR callers.)
    for _ in 1:(session.steps_per_tick - 1)
        sr_drop = OV.step!(screen.renderer, screen.product; timeout_ns = _M5_STEP_TIMEOUT_NS)
        close(sr_drop)
    end
    sr = OV.step!(screen.renderer, screen.product; timeout_ns = _M5_STEP_TIMEOUT_NS)
    try
        # INT-2 zero-copy present: tonemap+orient STRAIGHT from the still-mapped Float16 HdrColor
        # view into the cached buffer — no Float32 HDR transient.  CORRECTNESS: everything that
        # touches `raw16` stays INSIDE this closure (the mapping dies on unmap at with_mapped_hdr's
        # return); the loop fully materializes into `buf` (== session.present_buf, owned — NOT a
        # view over raw16), so nothing lazy over the mapping escapes.  The closure captures only
        # `session` (read + field-mutated, never rebound → not boxed); `buf` is closure-local.
        OV.with_mapped_hdr(sr) do raw16, W, H
            # Size guard: a resize rebuilds the Screen (+ image!) at a new size — reallocate the
            # cached buffer and re-seat it AS the image Observable's data array when it no longer
            # fits (interactive_display / resize_viewport! seed it; this is belt-and-suspenders).
            buf = session.present_buf
            if size(buf) != (W, H)
                buf = Matrix{RGBA{N0f8}}(undef, W, H)
                session.present_buf = buf
                session.image_plot[3][] = buf
            end
            # Fuse tonemap+orient straight into the cached buffer (in place), then re-seat it as the
            # image plot's data so GLMakie re-uploads the texture from that SAME array — zero
            # steady-state display garbage.
            #
            # ★ Use ASSIGNMENT (`[3][] = buf`), NOT `notify(image_plot[3])`: `image_plot[3]` is a
            # ComputePipeline `Computed` node, and `Base.notify(::Computed)` is a NO-OP
            # (Makie compute-plots.jl: `notify(computed) = computed`).  A bare notify never marks
            # the graph dirty, so GLMakie's `poll_updates`→`:gl_renderobject`→`update_robjs!`
            # texture upload never fires and the composited frame stays FROZEN at the warmup image
            # (only `present_buf` — mutated in place — advances).  `setindex!` routes through
            # `mark_dirty!`; because `is_same` conservatively returns `false` for the same mutable
            # array object, re-assigning the in-place-mutated `buf` still dirties the node and
            # triggers the re-upload, with no fresh allocation (INT-2 zero-copy preserved).
            _tonemap_orient!(buf, raw16, session.exposure)
            session.image_plot[3][] = buf
            return nothing
        end
    finally
        close(sr)
    end
    return nothing
end

"""
    _on_render_tick_impl!(session::ViewportSession) -> Nothing

Core of the per-frame live camera loop (M5 Task 3), called from `on_render_tick!`.
Pushes live camera/light/plot deltas, resets RT2 accumulation only on a change (else
keeps accumulating), runs `steps_per_tick` bounded steps, and blits to the image plot.
"""
function _on_render_tick_impl!(session::ViewportSession)
    screen    = session.screen
    cam_scene = session.cam_scene

    # Push live camera/light/plot deltas; reset RT2 accumulation on any change.
    if _sync_and_needs_reset!(screen, cam_scene)
        OV.reset!(screen.renderer)
        session.samples = 0
    end

    # HDR step + tonemap + blit via the selected strategy (:cpu / :gpu).  The
    # CUDA ext's GPU present! falls back to :cpu (session.blitter) on failure.
    present!(session, Val(session.blitter))
    session.samples += session.steps_per_tick
    return nothing
end

"""
    on_render_tick!(session::ViewportSession) -> Nothing

Per-frame live camera loop (M5 Task 3), fired on every GLMakie `render_tick`.  Wraps
`_on_render_tick_impl!` in a try/catch (a bad frame warns, `maxlog=5`, but does NOT
crash the window).  Must NOT Consume — the caller returns `Makie.Consume(false)`.
"""
function OmniverseMakie.on_render_tick!(session::ViewportSession)
    try
        _on_render_tick_impl!(session)
    catch e
        @warn "M5: render-tick frame failed (window kept alive)" exception=(e, catch_backtrace()) maxlog=5
    end
    return nothing
end

# ===== M6.B Task 5 — attachable picking interaction =====
# Wire a viewport click to a native AOV pick.  The pick CORE is in the main
# module (`pick_hit`, no GLMakie/CUDA dep); this ext only adds input wiring.
# Data flow: click → `_pick_at!` → `pick_hit(session.screen, xy)` →
# `PickHandle.selected[]` + `on_hit`.  Works on the HdrColor viewport.
#
# In-viewport OUTLINE is DEFERRED (see interactive_display's selection_outline
# guard): the viewport is always a normal HdrColor Screen
# (config.selection_outline == false), so attach_picking!(outline=true) degrades
# (@warn maxlog=1 → outline=false).  The _pick_at! outline branch is kept (gated
# on config.selection_outline) for the LdrColor follow-up.

"""
    mutable struct PickHandle

Handle from [`attach_picking!`](@ref): the click listener + picking state.

Fields:
  - `session`      — the `ViewportSession` this handle picks on
  - `listener`     — the `glscene.events.mousebutton` listener (or `nothing` once detached)
  - `on_hit`       — user callback `hit -> …` run after each pick (or `nothing`)
  - `outline::Bool`— draw a hit outline (degrades to `false` unless the Screen was built
                     with `selection_outline=true` — deferred)
  - `selected::Observable{Any}` — last hit `(; plot, index, world_position, normal)`, or
                     `nothing` over background; observe it to react to picks
  - `last_plot`    — currently-outlined plot (or `nothing`); cleared on next pick / detach
"""
mutable struct PickHandle
    session
    listener
    on_hit
    outline::Bool
    selected::Observable{Any}
    last_plot
end

# Max press→release travel (px) still counted as a CLICK, not a drag.  A
# left-drag orbits the Camera3D (M5 forwarding); a pick fires only on a click.
const _PICK_CLICK_PX = 5.0

# Cursor position on scene as (Float64, Float64).  `mouseposition` may hold a
# Point2 or a tuple (M5 forwarders assign tuples); indexing works for both.
_mouse_xy(session) = let mp = session.glscene.events.mouseposition[]
    (Float64(mp[1]), Float64(mp[2]))
end

# Did press→release travel far enough to be a drag (orbit) rather than a click?
# `press_pos` is a Ref captured by the attach_picking! listener (picking state
# lives on the handle/closure, off the ViewportSession).  NaN press ⇒ a click.
function _was_dragging(session, press_pos::Ref)
    px, py = press_pos[]
    isnan(px) && return false
    rx, ry = _mouse_xy(session)
    return hypot(rx - px, ry - py) > _PICK_CLICK_PX
end

"""
    _pick_at!(session, h::PickHandle, xy) -> hit | nothing

Pick at display pixel `xy` and publish: set `h.selected[]`, invoke `h.on_hit`.  Also the
test entry point (drive synchronously to assert wiring without the flaky GLFW event thread).
`pick_hit` is a renderer query, safe from the click listener (GLMakie runs handlers on the
render task, same as `present!`).  The outline branch is a no-op on the current HdrColor
viewport (`config.selection_outline == false`); kept for the deferred LdrColor path.
"""
function _pick_at!(session, h::PickHandle, xy)
    hit = OmniverseMakie.pick_hit(session.screen, xy)
    if h.outline && session.screen.config.selection_outline
        h.last_plot === nothing || OmniverseMakie.clear_selection!(session.screen, h.last_plot)
        h.last_plot = hit === nothing ? nothing : hit.plot
        hit === nothing || OmniverseMakie.select!(session.screen, hit.plot)
    end
    h.selected[] = hit
    h.on_hit === nothing || h.on_hit(hit)
    return hit
end

"""
    attach_picking!(session; on_hit=nothing, outline=false, button=Makie.Mouse.left) -> PickHandle

Attach click-to-pick to a live viewport `session`.  A click (press+release of `button`
without a drag, so it does not fight the M5 left-drag orbit) runs a native AOV pick and
publishes the hit on the handle's `selected` Observable, also invoking `on_hit(hit)`.
`hit` is `(; plot, index, world_position, normal)` or `nothing` over background.

`outline=true` DEGRADES (warns once): the HdrColor viewport can't carry the LdrColor-only
outline, so no highlight is drawn (pick data still works) — for outline IMAGES use offscreen
`select!` + `render_to_matrix`/`colorbuffer`.

[`detach_picking!`](@ref) removes it (also torn down on window close — the listener lives on
`glscene.events`).
"""
function attach_picking!(session; on_hit = nothing, outline::Bool = false, button = Makie.Mouse.left)
    if outline && !session.screen.config.selection_outline
        @warn "attach_picking!(outline=true) but the viewport was built without selection_outline=true; \
               no highlight will be drawn (the live in-viewport outline is a deferred follow-up — use \
               offscreen select! + render_to_matrix/colorbuffer for outline images)" maxlog=1
        outline = false
    end
    h = PickHandle(session, nothing, on_hit, outline, Observable{Any}(nothing), nothing)
    # Click = press+release without a drag.  Track press pos in a closure Ref
    # so a moved release (orbit drag) does not fire a pick.  Never Consume.
    press_pos = Ref((NaN, NaN))
    h.listener = on(session.glscene.events.mousebutton) do ev
        if ev.button == button
            if ev.action == Makie.Mouse.press
                press_pos[] = _mouse_xy(session)
            elseif ev.action == Makie.Mouse.release && !_was_dragging(session, press_pos)
                _pick_at!(session, h, _mouse_xy(session))
            end
        end
        return Makie.Consume(false)
    end
    return h
end

"""
    detach_picking!(h::PickHandle) -> Nothing

Remove the click listener attached by [`attach_picking!`](@ref) and clear any selection outline
the handle drew.  Idempotent (a second call is a safe no-op).
"""
function detach_picking!(h::PickHandle)
    h.listener === nothing || off(h.listener)
    h.listener = nothing
    h.last_plot === nothing || OmniverseMakie.clear_selection!(h.session.screen, h.last_plot)
    h.last_plot = nothing
    return nothing
end

# ==================================================================
# replace_scene! — embed a live ovrtx render of ONE scene inside an
# already-displayed GLMakie figure (RPRMakie replace_scene_rpr! pattern).
#
# The other axes stay GLMakie 2D diagnostics; the target 3D scene is replaced by a live
# raytraced image.  Unlike `interactive_display` (its own window, whole-figure blit), this
# attaches to the HOST window's render loop and overlays a pixel-space `image!` pinned to the
# target scene's viewport rectangle.  The target keeps its OWN Camera3D, so the user orbits it
# with normal GLMakie interaction — no input forwarding.
#
# v1 = CPU blit, one embedded scene.  The opaque ovrtx image occludes the GL 3D plots beneath
# (we do NOT toggle `plot.visible`: the open ovrtx stage live-diffs that same observable, so
# hiding for GL would also hide in the render — occlusion sidesteps the conflict).  GPU-direct
# blit and multiple concurrent embeds are follow-ups.
# ==================================================================

mutable struct EmbeddedSession
    screen::Screen                  # the embedded ovrtx Screen (renders at the sub-scene size)
    parent_glscreen                 # host figure's GLMakie.Screen — NOT owned; never closed here
    sub_scene::Makie.Scene          # pixel-space overlay child, pinned to target_scene.viewport
    image_plot                      # image! in sub_scene = the blit target (fields present! reads)
    present_buf::Matrix{RGBA{N0f8}} # [W,H] display buffer = image_plot's data array
    cam_scene::Makie.Scene          # the target's Camera3D scene (drives the ovrtx view)
    target_scene::Makie.Scene       # the scene being replaced
    steps_per_tick::Int
    exposure::Float32
    blitter::Symbol                 # :cpu (v1)
    tick_listener
    resize_listener
end

# Resolve the user's argument to the Scene to render: an LScene / Axis3 expose `.scene`; a raw
# Scene is itself.
_embed_target_scene(s::Makie.Scene) = s
_embed_target_scene(x) = hasproperty(x, :scene) ? x.scene :
    throw(ArgumentError("replace_scene!: expected an LScene, Axis3, or Scene, got $(typeof(x))"))

"""
    replace_scene!(target; steps_per_tick=2, exposure=0f0, accumulate=true) -> EmbeddedSession

Replace `target` (an `LScene`, `Axis3`, or `Scene`) inside an ALREADY-DISPLAYED GLMakie figure
with a live ovrtx RTX render, leaving the figure's other axes as GLMakie 2D diagnostics — the
RPRMakie `replace_scene_rpr!` hybrid.

Requires the figure to be displayed in a GLMakie window first (`GLMakie.activate!(); display(fig)`)
so the target scene has a laid-out pixel viewport and a host render loop to attach to.  The
target keeps its own `Camera3D`, so ordinary GLMakie orbit/zoom drives the raytraced view; a
per-frame hook on the host window re-renders.  `accumulate=true` (default) uses the
across-frames accumulation mode (realtime-style; only structural edits reset) — set `false` for
a full per-frame reconverge.  `steps_per_tick` bounds RTX steps per frame; `exposure` is EV
stops.  v1 is CPU-blit and one embedded scene per figure.

Returns an `EmbeddedSession`; `close(session)` detaches the hooks, removes the overlay, and
frees the ovrtx renderer WITHOUT touching the host GLMakie window.

## Recording (scripted / offscreen)

The embedded blit marks the host scene dirty on every tick, so once a session is attached
GLMakie's on-demand render loop self-sustains at full rate — a `write` to a child-process pipe
(e.g. ffmpeg rawvideo) from the same task can then starve forever (libuv never gets a loop
turn).  For scripted recording, STOP the loop and drive frames synchronously:

    glscr = GLMakie.Screen(; visible = false, px_per_unit = 1, scalefactor = 1)
    display(glscr, fig.scene)
    GLMakie.colorbuffer(glscr)                                        # layout pass
    GLMakie.stop_renderloop!(glscr; close_after_renderloop = false)   # keep the screen OPEN
    session = replace_scene!(ls; steps_per_tick = 8)
    for t in timesteps
        # ... update observables, then re-apply update_cam! (see below) ...
        img = record_frame!(session; ticks = 3)     # 3 sync ticks = 24 accumulated samples
        write(vio, permutedims(img))                # pipe writes now complete
    end

Two footguns: `stop_renderloop!`'s default `close_after_renderloop = true` CLOSES the screen
(after which `replace_scene!` errors) — pass `false`; and the loop must only be stopped, the
screen kept open, which still satisfies the `replace_scene!` precondition.  Each
[`record_frame!`](@ref) tick is a fully synchronous `GLMakie.colorbuffer` (pollevents →
render_tick → embedded step+blit → texture upload → composite), so this is also ~10% faster
than recording against a live loop.

Scripted cameras: the per-tick sync follows the target's `Camera3D` LIVE — anything that
touches it after your `update_cam!` (an LScene re-fit around display, a recipe's own camera
logic) silently wins and BOTH views jump.  Re-apply `update_cam!` per recorded frame; it is
cheap (accumulation only resets on an actual change).

Debugging a wrong composite: save (a) the pure-GL baseline (`colorbuffer` before
`replace_scene!`), (b) the native ground truth (`Makie.colorbuffer(OmniverseMakie.Screen(ls.scene))`),
and (c) `permutedims(session.present_buf)`.  Composite == GL baseline ⇒ the overlay is not
drawing; present_buf correct but wrong view ⇒ a camera fight.
"""
# Typed narrower than the main module's `replace_scene!(::Any)` fallback so it OVERRIDES (not
# overwrites) it: LScene / Axis3 are `Makie.Block`s, a raw target is a `Makie.Scene`.
function replace_scene!(target::Union{Makie.Scene,Makie.Block};
                        steps_per_tick::Int = 2, exposure = 0f0, accumulate::Bool = true)
    tscene = _embed_target_scene(target)
    parent = Makie.getscreen(Makie.root(tscene), GLMakie)
    parent === nothing && throw(ArgumentError(
        "replace_scene!: the figure is not shown in a GLMakie window. Run `GLMakie.activate!(); \
         display(fig)` before replace_scene! (the target needs a laid-out viewport on an OPEN \
         GLMakie screen — the render loop may be stopped, see the docstring's recording recipe)."))
    cam_scene = something(_scene_for_camera(tscene), tscene)

    # Render size = the target scene's pixel rectangle (blitted 1:1 into that rectangle).
    rect = tscene.viewport[]
    W, H = Int.(widths(rect))
    (W > 0 && H > 0) || throw(ArgumentError(
        "replace_scene!: target has a degenerate viewport $(rect) — display/lay out the figure first."))

    # Build an ovrtx Screen at the sub-scene size (fb_size override), accumulate for smooth
    # realtime streaming (exactly RPR's 1-sample/frame accumulate model).
    cfg = Makie.merge_screen_config(ScreenConfig,
        Dict{Symbol,Any}(:accumulate_across_frames => accumulate))
    screen = Screen(cam_scene, cfg; fb_size = (W, H))
    _author_screen!(screen, cam_scene, tscene)
    warmup_hdr = OV.render_hdr_to_array(screen.renderer, screen.product; warmup = cfg.warmup)
    screen.requires_update = false          # the warmup already drew the eager insertplots!

    # Overlay: a pixel-space child scene SHARING the target's viewport Observable (so it tracks
    # the rectangle across layout/resize), with our opaque image! drawn over the GL 3D beneath.
    # ★ The overlay must NOT inherit the target's transformation: Makie's child-Scene constructor
    # defaults `transformation = Transformation(parent)` (scenes.jl), so a root transform on the
    # target — e.g. `rotate!(ls.scene, Q_X90)`, the usual Z-up trick — would rotate the blit quad
    # edge-on and the composite would silently show the plain GL render underneath.  An unparented
    # identity Transformation keeps the quad pixel-space forever (covers _resize_embedded! too,
    # which re-creates the image! in this same sub-scene).
    GLMakie.activate!()
    sub = Makie.Scene(tscene; viewport = tscene.viewport, clear = false,
                      transformation = Makie.Transformation())
    Makie.campixel!(sub)
    present_buf = Matrix{RGBA{N0f8}}(undef, W, H)
    _tonemap_orient!(present_buf, warmup_hdr, Float32(exposure))
    # `image!` on a child of the already-displayed target scene auto-registers the sub-scene +
    # plot with the host GL screen (verified: it renders on top of the GL 3D in the rectangle);
    # an explicit insertplots! here would DOUBLE-register and error on the compute edge.
    img = image!(sub, 0 .. W, 0 .. H, present_buf; interpolate = false)

    session = EmbeddedSession(screen, parent, sub, img, present_buf, cam_scene, tscene,
                              steps_per_tick, Float32(exposure), :cpu, nothing, nothing)

    # Hook the HOST window's render loop (never open our own).  Observer only — Consume(false).
    # Guarded: `notify(::Observable)` does NOT catch listener errors, and the host loop is a
    # plain @async task with no guard of its own — an uncaught throw from a bad frame would
    # kill the render task and silently freeze the ENTIRE host figure (all GL plots too).
    session.tick_listener = on(parent.render_tick) do _
        try
            _embedded_tick!(session)
        catch e
            @warn "replace_scene!: embedded tick failed (host window kept alive)" exception=(e, catch_backtrace()) maxlog=5
        end
        return Makie.Consume(false)
    end

    # Track the target rectangle: a layout/window resize changes tscene.viewport → rebuild the
    # ovrtx render + image at the new size.  Guarded for the same reason as the tick above —
    # the rebuild runs ovrtx Screen creation (the historically crash-prone window); on failure
    # the session keeps rendering at the OLD size instead of killing the host render task.
    embed_resize_inflight = Ref(false)   # re-entrancy guard (same rationale as the standalone viewport)
    session.resize_listener = on(tscene.viewport) do rect2
        w, h = Int.(widths(rect2))
        ((w, h) == session.screen.fb_size || w <= 0 || h <= 0) && return nothing
        embed_resize_inflight[] && return nothing
        embed_resize_inflight[] = true
        try
            _resize_embedded!(session, (w, h))
        catch e
            @warn "replace_scene!: embedded resize failed (kept the old render size)" exception=(e, catch_backtrace()) maxlog=5
        finally
            embed_resize_inflight[] = false
        end
        return nothing
    end

    return session
end

"""
    record_frame!(session::EmbeddedSession; ticks = 3)

Drive `ticks` synchronous host frames and return the composited figure image (`[H,W]`,
`Makie.JuliaNative`) — the scripted-recording companion to [`replace_scene!`](@ref); see its
docstring's recording recipe (stop the render loop first with
`GLMakie.stop_renderloop!(glscr; close_after_renderloop = false)`).

A recorded frame accumulates `ticks × steps_per_tick` RTX samples (a moved camera/scene resets
accumulation once, then the steps refine).  Only the LAST tick's image is returned, so the
intermediate ticks' per-frame present + composite + readback (host tonemap + HdrColor map + GL
composite, ≈ 26.6 ms/tick at 1080p) is discarded — for `ticks > 1` those are skipped: sync ONCE,
run `(ticks-1) × steps_per_tick` BARE ovrtx steps (no present/composite), then a single presenting
`GLMakie.colorbuffer` for the returned frame.  Same sample count, pixel-equivalent
(profiler-verified: bare-step vs all-present frame, nonblack ratio 1.0000).  `ticks == 1` is one
presenting colorbuffer, unchanged.

The bare steps run on the CALLER's task (not the render task) — safe under the stopped-loop
recording recipe (no @async loop competes) and cooperatively serialized with a live @async loop
(a step yields at its boundary; the final tick's own sync is then a no-op re-check).

Each presenting tick is one `GLMakie.colorbuffer` on the host screen: pollevents fires
`render_tick` → the embedded hook steps ovrtx `steps_per_tick` times and blits → the dirtied
texture uploads → the frame composites, all synchronously.  The returned image is GLMakie's frame
cache — copy it (or feed it straight to a `write`/`permutedims`, which copies) before the next
frame.
"""
function record_frame!(session::EmbeddedSession; ticks::Int = 3)
    ticks >= 1 || throw(ArgumentError("record_frame!: need ticks >= 1, got $ticks"))
    ticks == 1 && return Makie.colorbuffer(session.parent_glscreen)
    # ticks > 1: sync once, then (ticks-1)·steps_per_tick bare steps (no present/composite), then
    # ONE presenting colorbuffer.  Its _embedded_tick! syncs again (a no-op — nothing changed) and
    # steps the final steps_per_tick before compositing, so the total is ticks·steps_per_tick.
    _embedded_sync!(session)
    for _ in 1:((ticks - 1) * session.steps_per_tick)
        sr = OV.step!(session.screen.renderer, session.screen.product; timeout_ns = _M5_STEP_TIMEOUT_NS)
        close(sr)
    end
    return Makie.colorbuffer(session.parent_glscreen)
end

# Per-frame embedded SYNC: push live camera/light/plot deltas, reset RT2 accumulation only on a
# real change (accumulate mode → only a structural edit).  Factored out of _embedded_tick! so
# record_frame!'s bare-step path syncs through the SAME code (the two cannot drift).
function _embedded_sync!(session::EmbeddedSession)
    _sync_and_needs_reset!(session.screen, session.cam_scene) &&
        OV.reset!(session.screen.renderer)
    return nothing
end

# Per-frame embedded hook: sync (above), step, and CPU-blit into the overlay image.  Runs on the
# host window's render task.
function _embedded_tick!(session::EmbeddedSession)
    _embedded_sync!(session)
    _cpu_present!(session)
    return nothing
end

# Rebuild the embedded ovrtx render + overlay image at a new target-rectangle size.
function _resize_embedded!(session::EmbeddedSession, (W, H)::Tuple{Int,Int})
    cfg        = session.screen.config
    new_screen = Screen(session.cam_scene, cfg; fb_size = (W, H))
    _author_screen!(new_screen, session.cam_scene, session.target_scene)
    warmup_hdr = OV.render_hdr_to_array(new_screen.renderer, new_screen.product; warmup = cfg.warmup)
    new_screen.requires_update = false
    close(session.screen)                    # free the old renderer
    session.screen = new_screen
    delete!(session.sub_scene, session.image_plot)   # Makie's delete!(::Scene, ::Plot)
    buf = Matrix{RGBA{N0f8}}(undef, W, H)
    _tonemap_orient!(buf, warmup_hdr, session.exposure)
    # The sub-scene is still GL-registered; adding the replacement image to it auto-inserts the
    # new plot (same path as the initial image!).
    session.image_plot  = image!(session.sub_scene, 0 .. W, 0 .. H, buf; interpolate = false)
    session.present_buf = buf
    return nothing
end

"""
    Base.close(session::EmbeddedSession) -> Nothing

Detach the render-loop + resize hooks, QUIESCE the host render loop across the ovrtx
teardown, remove the overlay image, and free the embedded ovrtx renderer.  Does NOT close
the host GLMakie window (the figure lives on; a live loop is restarted).  Idempotent.
"""
function Base.close(session::EmbeddedSession)
    session.tick_listener   === nothing || off(session.tick_listener)
    session.resize_listener === nothing || off(session.resize_listener)
    session.tick_listener   = nothing     # no more ticks BEFORE freeing the ovrtx Screen
    session.resize_listener = nothing
    # Quiesce the HOST loop across the teardown (mirror ViewportSession.close): stop + JOIN
    # so no in-flight embedded tick can overlap the Screen free — `Base.close(::Renderer)`
    # destroys the ptr before flipping `alive`, so a mid-tick `step!` would be a
    # use-after-free under any future threaded renderloop (today's @async loop masks it).
    # Restart only if WE stopped a running loop: a stopped-loop recording session
    # (record_frame! recipe) must stay stopped.
    parent = session.parent_glscreen
    was_running = false
    try
        if parent !== nothing && isopen(parent) && GLMakie.renderloop_running(parent)
            was_running = true
            GLMakie.stop_renderloop!(parent; close_after_renderloop = false)   # joins the task
        end
    catch e
        @warn "replace_scene!: error stopping the host render loop for teardown" exception = e
    end
    try
        # Remove the overlay sub-scene's GL renderobjects from the host screen (the GL 3D beneath
        # reappears), then drop the sub-scene from the target's children.
        delete!(parent, session.sub_scene)
        filter!(c -> c !== session.sub_scene, session.target_scene.children)
    catch e
        @warn "replace_scene!: error removing overlay" exception = e
    end
    close(session.screen)                 # ovrtx renderer + bindings; host window untouched
    was_running && try
        GLMakie.start_renderloop!(parent) # the figure lives on — resume its loop
    catch e
        @warn "replace_scene!: error restarting the host render loop" exception = e
    end
    return nothing
end

end # module OmniverseMakieGLMakieExt
