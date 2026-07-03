using Test

# interactive_display lifecycle (formerly m5_viewport_test.jl, three subprocesses merged
# into ONE — the GLMakie+ovrtx startup is amortized):
#   1. the window shows a non-black RTX frame (the CPU present reached the GL texture);
#   2. resize!(glscreen, …) → GLFW size callback → window_area → resize_viewport!
#      rebuilds the ovrtx renderer at the new size and the new frame is non-black;
#   3. Base.close(session) detaches listeners + closes the GL window + the ovrtx Screen,
#      and a second close is a safe no-op.
#
# NOTE: colorbuffer is called after resize to flush GLFW events (pollevents is called
# internally, which fires the window_area callback).  The tick_listener is detached
# before the resize to avoid interference from auto-ticks.
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
off(session.tick_listener)          # avoid auto-tick interference during the resize
session.tick_listener = nothing     # …and keep the later close() from double-off'ing it
resize!(session.glscreen, 500, 360)
# The GLFW size callback can land on the render task ASYNCHRONOUSLY relative to this
# script, so poll (colorbuffer runs pollevents → window_area → resize_listener) until the
# rebuild takes, bounded — a single flush read the OLD fb_size in a real run (race).
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
    exitcode, output = run_ovrtx_subprocess(_VIEWPORT_PROG; timeout = 600, retries = 2, ready_marker = "OK_VIEWPORT")
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
