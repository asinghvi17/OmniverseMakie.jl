using Test

# interactive_display lifecycle (one subprocess — the GLMakie+ovrtx startup
# is amortized):
#   1. the window shows a non-black RTX frame (CPU present reached the GL
#      texture);
#   2. resize!(glscreen, …) → GLFW size callback → window_area →
#      resize_viewport! rebuilds the ovrtx renderer at the new size and the
#      new frame is non-black;
#   3. Base.close(session) detaches listeners + closes the GL window + the
#      ovrtx Screen; a second close is a safe no-op.
#
# NOTE: colorbuffer is called after resize to flush GLFW events (pollevents
# fires the window_area callback).  The tick_listener is detached before the
# resize to avoid interference from auto-ticks.
const _VIEWPORT_PROG = """
using OmniverseMakie, GLMakie, ColorTypes
OM = OmniverseMakie
OM.activate!(warmup = 32)
fig = Figure(); ax = LScene(fig[1,1]; show_axis=false)
mesh!(ax, Rect3f(Point3f(-1), Vec3f(2)); color = :orange)
session = OM.interactive_display(fig; size = (400, 300))

# --- 1. live frame reaches the GL texture ---
buf = GLMakie.colorbuffer(session.glscreen)
nb = count(c -> (Float32(red(c))+Float32(green(c))+Float32(blue(c))) > 0.1, buf)
println("VIEWPORT_NONBLACK=", nb)
@assert nb > 1000 "viewport window is black — RTX frame did not reach the texture"

# --- 2. resize rebuilds the renderer at the new size ---
off(session.tick_listener)       # avoid auto-tick interference during resize
session.tick_listener = nothing  # …and keep close() from double-off'ing it
resize!(session.glscreen, 500, 360)
# The GLFW size callback lands on the render task ASYNCHRONOUSLY relative to
# this script, so poll (colorbuffer runs pollevents → window_area →
# resize_listener) until the rebuild takes, bounded — a single flush can
# still read the OLD fb_size (race).
deadline = time() + 15
while session.screen.fb_size != (500, 360) && time() < deadline
    GLMakie.colorbuffer(session.glscreen)
    sleep(0.05)
end
println("RESIZE_FB=", session.screen.fb_size)
@assert session.screen.fb_size == (500, 360)
buf2 = GLMakie.colorbuffer(session.glscreen)   # render with the new image plot
nb2 = count(c -> (Float32(red(c)) + Float32(green(c)) + Float32(blue(c))) > 0.1, buf2)
println("RESIZE_NONBLACK=", nb2)
@assert nb2 > 500

# --- 3. teardown: close detaches + closes; second close is a safe no-op ---
close(session)
println("GL_CLOSED=", !isopen(session.glscreen))
close(session)
println("OK_VIEWPORT")
"""

include(joinpath(@__DIR__, "..", "helpers.jl"))
@testset "interactive_display: frame + resize + teardown (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_VIEWPORT_PROG; timeout = 600, retries = 2, ready_marker = "VIEWPORT_NONBLACK=")
    @info "viewport output" output
    @test exitcode == 0
    @test contains(output, "OK_VIEWPORT")
    m = match(r"VIEWPORT_NONBLACK=(\d+)", output)
    @test m !== nothing && parse(Int, m.captures[1]) > 1000
    mf = match(r"RESIZE_FB=\((\d+), (\d+)\)", output)
    @test mf !== nothing && parse(Int, mf.captures[1]) == 500 && parse(Int, mf.captures[2]) == 360
    m2 = match(r"RESIZE_NONBLACK=(\d+)", output)
    @test m2 !== nothing && parse(Int, m2.captures[1]) > 500
    @test contains(output, "GL_CLOSED=true")
end

# A build failure AFTER the ovrtx Screen is created (but before it is seated on
# the returned session) must close the Screen and rethrow, or the renderer
# leaks (SyncScopeIds are machine-wide, ~7).  Inject a throw into
# _author_screen! (which interactive_display calls right after building the
# Screen), capturing the Screen so we can assert it was closed on the error.
const _LEAK_GUARD_PROG = """
using OmniverseMakie, GLMakie, ColorTypes
OM = OmniverseMakie
OM.activate!(warmup = 16)
fig = Figure(); ax = LScene(fig[1,1]; show_axis=false)
mesh!(ax, Rect3f(Point3f(-1), Vec3f(2)); color = :orange)

const CAP = Ref{Any}(nothing)
@eval OmniverseMakie function _author_screen!(screen::Screen, cam_scene, plot_scene)
    (\$CAP)[] = screen
    error("injected author failure")
end

threw = false
try
    OM.interactive_display(fig; size = (200, 200))
catch e
    global threw = true
end
println("BUILD_FAILED_RETHROWN=", threw)
println("SCREEN_CAPTURED=", CAP[] !== nothing)
println("SCREEN_CLOSED_ON_ERROR=", CAP[] !== nothing && !OM.isopen(CAP[]))
println("OK_LEAK_GUARD")
"""

@testset "interactive_display: build failure closes the ovrtx Screen (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_LEAK_GUARD_PROG; timeout = 600, retries = 4,
                                            ready_marker = "OK_LEAK_GUARD")
    @info "leak-guard output" output
    @test exitcode == 0
    @test contains(output, "BUILD_FAILED_RETHROWN=true")   # the error propagated
    @test contains(output, "SCREEN_CAPTURED=true")          # the Screen was built
    @test contains(output, "SCREEN_CLOSED_ON_ERROR=true")   # …and closed, not leaked
end

# GLMakie.stop_renderloop!'s default close_after_renderloop=true CLOSES the
# joined screen — which is why close(::ViewportSession) (and the recording
# recipe) pass false, so the GL image texture stays alive for the CUDA-GL
# unregister / _gpu_teardown! before close(glscreen).  Demonstrate both.
const _STOP_RENDERLOOP_PROG = """
using GLMakie
scene = Scene(size = (120, 90))
s1 = GLMakie.Screen(; visible = false)
display(s1, scene); GLMakie.colorbuffer(s1)
GLMakie.stop_renderloop!(s1)                                  # default kwarg
println("DEFAULT_KWARG_CLOSED=", !isopen(s1))
s2 = GLMakie.Screen(; visible = false)
display(s2, scene); GLMakie.colorbuffer(s2)
GLMakie.stop_renderloop!(s2; close_after_renderloop = false)  # keep it open
println("FALSE_KWARG_OPEN=", isopen(s2))
close(s2)
println("OK_STOP_RENDERLOOP")
"""

@testset "stop_renderloop! default kwarg closes the GL screen (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_STOP_RENDERLOOP_PROG; timeout = 300,
                                            retries = 2, ready_marker = "OK_STOP_RENDERLOOP")
    @info "stop_renderloop kwarg output" output
    @test exitcode == 0
    @test contains(output, "DEFAULT_KWARG_CLOSED=true")   # default closes the screen
    @test contains(output, "FALSE_KWARG_OPEN=true")       # false keeps it open
end
