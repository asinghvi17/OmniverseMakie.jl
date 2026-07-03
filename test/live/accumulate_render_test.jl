# Accumulate-across-frames — behavioural render test (subprocess, GPU).
#
# Proves the realtime-style recording mode end to end:
#   • accumulate=true: six per-frame CAMERA moves fire ZERO RT2 resets (non-structural), yet the
#     rendered image still changes (the live diff writes are applied) — the lit centroid MOVES;
#   • the first-frame pre-roll lands a converged (non-black) frame 1;
#   • a mid-sequence structural change (insert! a new plot) fires EXACTLY ONE reset;
#   • default mode (accumulate=false) resets on EVERY one of the six camera moves (the control that
#     proves the gate changed nothing when the flag is off).
# Reset count is observed exactly via OV._RESET_OBSERVER.

using Test
include(joinpath(@__DIR__, "..", "helpers.jl"))

const _ACCUM_PROG = """
using OmniverseMakie, ColorTypes, FixedPointNumbers
const OM = OmniverseMakie
using OmniverseMakie: OV

$(PROG_PIXEL_HELPERS)

# Off-centre red box + one directional light: a camera orbit visibly moves its screen position.
function build_scene()
    scene = Scene(size = (300, 300); lights = AbstractLight[
        DirectionalLight(RGBf(1, 1, 1), Vec3f(-1, -1, -0.4), false)])
    cam3d!(scene)
    mesh!(scene, Rect3f(Point3f(1.0, -0.5, -0.5), Vec3f(1)); color = :red)
    return scene
end
orbit!(scene, t) = update_cam!(scene,
    Vec3f(8cos(0.5t), 8sin(0.5t), 5), Vec3f(0, 0, 0), Vec3f(0, 0, 1))

# Wrapped in a function (not a top-level loop) so the accumulator isn't a soft-scope local.
function run_motion(screen, scene, nframes)
    local img
    for t in 1:nframes
        orbit!(scene, t)
        img = Makie.colorbuffer(screen)
    end
    return img
end

resets = Ref(0)
OV._RESET_OBSERVER[] = () -> (resets[] += 1)

# ---- accumulate mode ----
scene  = build_scene(); orbit!(scene, 0)
screen = OM.Screen(scene; accumulate_across_frames = true, warmup = 4, accumulation_preroll = 12)
img1   = Makie.colorbuffer(screen)          # authors (one structural reset) + pre-roll folded in
c1     = lit_centroid(img1)
resets[] = 0                                 # ignore the author-time reset; count only motion
imgN = run_motion(screen, scene, 6)
cN   = lit_centroid(imgN)
println("ACC_PREROLL_NONBLACK=", c1.nb)
println("ACC_MOTION_RESETS=", resets[])
println("ACC_MOVED=", abs(c1.ccol - cN.ccol) + abs(c1.crow - cN.crow))

# structural: a mid-sequence insert must reset exactly once
resets[] = 0
p = mesh!(scene, Rect3f(Point3f(-3.0, -0.5, -0.5), Vec3f(1)); color = :blue)
insert!(screen, scene, p)
Makie.colorbuffer(screen)
println("ACC_STRUCTURAL_RESETS=", resets[])
close(screen)

# ---- default mode (control): every camera move resets ----
scene2  = build_scene(); orbit!(scene2, 0)
screen2 = OM.Screen(scene2; warmup = 4)      # accumulate defaults false
Makie.colorbuffer(screen2)
resets[] = 0
run_motion(screen2, scene2, 6)
println("DEFAULT_MOTION_RESETS=", resets[])
close(screen2)

OV._RESET_OBSERVER[] = nothing
println("OK_ACCUMULATE")
"""

@testset "accumulate-across-frames: reset suppression + structural + preroll (subprocess)" begin
    # ovrtx has a known intermittent startup crash (GeometryGroup::attachToContext); retry until
    # the render markers appear (E3 harness pattern).
    _, out = run_ovrtx_subprocess(_ACCUM_PROG; timeout = 600, retries = 4,
                                  ready_marker = "OK_ACCUMULATE")
    contains(out, "OK_ACCUMULATE") || @info "accumulate render output" out
    @test contains(out, "OK_ACCUMULATE")

    # First frame converged (pre-roll landed): non-black (LIT_PX_MIN is spliced into the child, so
    # the parent asserts the literal 300, as the other render tests do).
    m_pre = match(r"ACC_PREROLL_NONBLACK=(\d+)", out)
    @test m_pre !== nothing && parse(Int, m_pre.captures[1]) > 300

    # Six camera moves in accumulate mode → ZERO resets (the whole point).
    m_acc = match(r"ACC_MOTION_RESETS=(\d+)", out)
    @test m_acc !== nothing && parse(Int, m_acc.captures[1]) == 0

    # ...yet the image still changed — the per-frame diff writes were applied (centroid moved).
    m_mov = match(r"ACC_MOVED=([-\d.]+)", out)
    @test m_mov !== nothing && parse(Float64, m_mov.captures[1]) > 3.0

    # A structural change (insert!) still resets exactly once.
    m_str = match(r"ACC_STRUCTURAL_RESETS=(\d+)", out)
    @test m_str !== nothing && parse(Int, m_str.captures[1]) == 1

    # Control: default mode resets on every one of the six moves (gate is a no-op when off).
    m_def = match(r"DEFAULT_MOTION_RESETS=(\d+)", out)
    @test m_def !== nothing && parse(Int, m_def.captures[1]) == 6
end
