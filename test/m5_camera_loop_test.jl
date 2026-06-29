using Test

# M5 Task 3: live camera loop (on_render_tick! — sync_camera! + bounded step + cpu_blit!)
#
# Two assertions per the M5.3 requirement (Correction D):
#   A. Progressive accumulation: idle ticks strictly increase session.samples
#      (no reset while scene is static — keeps accumulating / refining).
#   B. Camera-move tick resets accumulation: session.samples drops back to
#      steps_per_tick (2) after sync_camera! sees a change.
#   C. Frame non-black after camera orbit (live blit of the reframed RTX view).
#
# Deterministic tick sequence (listener detached before manual ticks — no GLMakie auto-ticks):
#   Tick 0: first pull_ovrtx_nodes! flips requires_update → RESET → samples = steps_per_tick (2)
#   Tick 1: idle → samples = 4 (baseline samples_settle)
#   Tick 2: idle → samples = 6 (samples_idle); assert samples_idle > samples_settle
#   Camera move → Tick 3: RESET → samples = 2; assert samples_after_move == steps_per_tick == 2
#   GLMakie colorbuffer after Tick 3: assert non-black
const _M5_LOOP_PROG = """
using OmniverseMakie, GLMakie, ColorTypes
OM = OmniverseMakie; OV = OmniverseMakie.OV
OM.activate!(warmup = 16)
fig = Figure(); ax = LScene(fig[1,1]; show_axis=false)
mesh!(ax, Rect3f(Point3f(-1), Vec3f(2)); color = :red)
GLMakie.activate!()
session = OM.interactive_display(fig; size = (300, 300), steps_per_tick = 2)
cam = Makie.cameracontrols(session.cam_scene)

# Detach the live render_tick listener so GLMakie's own auto-ticks cannot
# interleave with the manual sequence below.  Production behavior is unchanged
# (listener remains registered for real sessions); this is test-only teardown.
off(session.tick_listener)

# --- Deterministic tick sequence ---
# Tick 0: first pull_ovrtx_nodes! legitimately flips requires_update → reset → samples = steps_per_tick
OM.on_render_tick!(session)   # tick 0: first tick, reset expected
println("SAMPLES_0=", session.samples)
# Tick 1: idle (no change) → samples grows by steps_per_tick
OM.on_render_tick!(session)   # tick 1: idle accumulation
samples_settle = session.samples
println("SAMPLES_SETTLE=", samples_settle)

# --- Idle-accumulation assertion (M5.3): second tick after settling must grow samples ---
OM.on_render_tick!(session)   # tick 2: idle — no camera/scene change
samples_idle = session.samples
println("SAMPLES_IDLE=", samples_idle)
@assert samples_idle > samples_settle "idle tick did not accumulate: samples went \$samples_settle → \$samples_idle (expected increase)"

# --- Camera-move assertion: samples resets to steps_per_tick on a change tick ---
eye0 = cam.eyeposition[]
cam.eyeposition[] = Vec3f(-eye0[1], -eye0[2], eye0[3])   # 180° orbit
OM.on_render_tick!(session)   # tick 3: camera moved → sync_camera! returns true → reset
samples_after_move = session.samples
println("SAMPLES_AFTER_MOVE=", samples_after_move)
@assert samples_after_move == 2 "camera-move tick did not reset accumulation: got \$samples_after_move, expected 2 (steps_per_tick)"

# --- Frame non-black after camera orbit ---
bufB = GLMakie.colorbuffer(session.glscreen)
nb = count(c -> (Float32(red(c))+Float32(green(c))+Float32(blue(c))) > 0.1, bufB)
println("LOOP_NONBLACK=", nb)
@assert nb > 500 "frame B black — loop blit failed after camera move"
println("OK_LOOP")
"""

include("helpers.jl")
@testset "M5 on_render_tick! reframes and accumulates (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M5_LOOP_PROG; timeout = 600)
    @info "M5 camera loop output" output
    @test exitcode == 0
    @test contains(output, "OK_LOOP")

    # Verify idle accumulation: samples_idle > samples_settle
    m_settle = match(r"SAMPLES_SETTLE=(\d+)", output)
    m_idle   = match(r"SAMPLES_IDLE=(\d+)", output)
    if m_settle !== nothing && m_idle !== nothing
        samples_settle = parse(Int, m_settle.captures[1])
        samples_idle   = parse(Int, m_idle.captures[1])
        @test samples_idle > samples_settle
    end

    # Verify reset on camera move: samples == steps_per_tick == 2
    m_move = match(r"SAMPLES_AFTER_MOVE=(\d+)", output)
    if m_move !== nothing
        @test parse(Int, m_move.captures[1]) == 2
    end

    # Verify frame non-black
    m = match(r"LOOP_NONBLACK=(\d+)", output)
    @test m !== nothing && parse(Int, m.captures[1]) > 500
end
