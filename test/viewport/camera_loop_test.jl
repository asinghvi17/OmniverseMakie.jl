using Test

# Live camera loop (on_render_tick! — sync_camera! + bounded step + CPU
# present).  Invariants (not exact counts — the start tracks `warmup`):
#   A. idle ticks strictly increase session.samples (no reset while static);
#   B. a camera-move tick resets samples back to steps_per_tick (2) after
#      sync_camera! sees the change;
#   C. the frame is non-black after the camera orbit (live blit).
# Deterministic sequence: the tick listener is detached before manual ticks
# (no GLMakie auto-ticks), and interactive_display's warmup consumes the
# initial `requires_update` (see viewport.jl), so a static tick doesn't reset.
const _M5_LOOP_PROG = """
using OmniverseMakie, GLMakie, ColorTypes
OM = OmniverseMakie; OV = OmniverseMakie.OV
OM.activate!(warmup = 16)
fig = Figure(); ax = LScene(fig[1,1]; show_axis=false)
mesh!(ax, Rect3f(Point3f(-1), Vec3f(2)); color = :red)
GLMakie.activate!()
# steps_per_tick < 1 desyncs the samples counter — interactive_display must
# reject it with an ArgumentError BEFORE building any ovrtx Screen (no leak).
_spt_ok = false
try
    OM.interactive_display(fig; size = (300, 300), steps_per_tick = 0)
catch e
    global _spt_ok = e isa ArgumentError
end
println("STEPS_PER_TICK_GUARD=", _spt_ok)
session = OM.interactive_display(fig; size = (300, 300), steps_per_tick = 2)
cam = Makie.cameracontrols(session.cam_scene)

# Detach the live render_tick listener so GLMakie's own auto-ticks cannot
# interleave with the manual sequence below.  Production behavior is unchanged
# (listener remains registered for real sessions); this is test-only teardown.
off(session.tick_listener)

# --- Deterministic tick sequence ---
# Tick 0: settle tick — interactive_display's warmup already consumed the
# initial build flag, so idle.
OM.on_render_tick!(session)   # tick 0: settle (idle, no reset)
println("SAMPLES_0=", session.samples)
# Tick 1: idle (no change) → samples grows by steps_per_tick
OM.on_render_tick!(session)   # tick 1: idle accumulation
samples_settle = session.samples
println("SAMPLES_SETTLE=", samples_settle)

# --- Idle accumulation: the 2nd tick after settling must grow samples ---
OM.on_render_tick!(session)   # tick 2: idle — no camera/scene change
samples_idle = session.samples
println("SAMPLES_IDLE=", samples_idle)
@assert samples_idle > samples_settle "idle tick did not accumulate: samples went \$samples_settle → \$samples_idle (expected increase)"

# --- Camera move: samples resets to steps_per_tick on a change tick ---
eye0 = cam.eyeposition[]
cam.eyeposition[] = Vec3f(-eye0[1], -eye0[2], eye0[3])   # 180° orbit
OM.on_render_tick!(session)   # tick 3: camera moved → sync_camera! true → reset
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

include(joinpath(@__DIR__, "..", "helpers.jl"))
@testset "on_render_tick! reframes and accumulates (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M5_LOOP_PROG; timeout = 600, retries = 2, ready_marker = "SAMPLES_0=")
    @info "M5 camera loop output" output
    @test exitcode == 0
    @test contains(output, "OK_LOOP")
    # steps_per_tick < 1 is rejected at the interactive_display entry point.
    @test contains(output, "STEPS_PER_TICK_GUARD=true")

    # Verify idle accumulation: samples_idle > samples_settle
    m_settle = match(r"SAMPLES_SETTLE=(\d+)", output)
    m_idle   = match(r"SAMPLES_IDLE=(\d+)", output)
    if m_settle !== nothing && m_idle !== nothing
        samples_settle = parse(Int, m_settle.captures[1])
        samples_idle   = parse(Int, m_idle.captures[1])
        @test samples_idle > samples_settle
    else
        @test false   # SAMPLES_SETTLE / SAMPLES_IDLE line missing
    end

    # Verify reset on camera move: samples == steps_per_tick == 2
    m_move = match(r"SAMPLES_AFTER_MOVE=(\d+)", output)
    if m_move !== nothing
        @test parse(Int, m_move.captures[1]) == 2
    else
        @test false   # SAMPLES_AFTER_MOVE line missing
    end

    # Verify frame non-black
    m = match(r"LOOP_NONBLACK=(\d+)", output)
    @test m !== nothing && parse(Int, m.captures[1]) > 500
end
