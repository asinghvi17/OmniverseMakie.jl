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

# M5 Task 4 — teardown (close) + idempotency.
#
# Proves that Base.close(session) detaches all listeners, closes the GLMakie window,
# and closes the underlying ovrtx Screen — and that a second close is a safe no-op.
const _M5_TEARDOWN_PROG = """
using OmniverseMakie, GLMakie, ColorTypes
OM = OmniverseMakie
OM.activate!(warmup = 16)
fig = Figure(); ax = LScene(fig[1,1]; show_axis = false)
mesh!(ax, Rect3f(Point3f(-1), Vec3f(2)); color = :teal)
session = OM.interactive_display(fig; size = (300, 300))

# First close: should detach listeners + close GL window + close ovrtx screen.
close(session)
println("GL_CLOSED=", !isopen(session.glscreen))

# Second close: must be a safe no-op (no throw, no crash).
close(session)
println("OK_TEARDOWN")
"""

# M5 Task 4 — window resize.
#
# Proves that resize!(glscreen, W, H) → GLFW resize callback → window_area update
# → resize_viewport! rebuilds the ovrtx renderer at the new size.
#
# After resize, session.screen.fb_size == (500, 360) and the displayed frame is
# non-black (a warmup render was done at the new size).
#
# NOTE: colorbuffer is called AFTER resize to flush GLFW events (pollevents is
# called internally by colorbuffer, which fires the window_area callback).
# The tick_listener is detached first to avoid interference from auto-ticks.
const _M5_RESIZE_PROG = """
using OmniverseMakie, GLMakie, ColorTypes
OM = OmniverseMakie
OM.activate!(warmup = 16)
fig = Figure(); ax = LScene(fig[1,1]; show_axis = false)
mesh!(ax, Rect3f(Point3f(-1), Vec3f(2)); color = :crimson)
session = OM.interactive_display(fig; size = (300, 300))

# Detach tick_listener to prevent auto-ticks from interfering.
off(session.tick_listener)

# Trigger a window resize to 500×360.
resize!(session.glscreen, 500, 360)

# colorbuffer flushes GLFW events (pollevents internally), which triggers the
# GLFW window-size callback → updates glscene.events.window_area → fires
# session.resize_listener → calls resize_viewport!(session, (500, 360)).
_ = GLMakie.colorbuffer(session.glscreen)

println("RESIZE_FB=", session.screen.fb_size)
@assert session.screen.fb_size == (500, 360)

# Verify the new frame is non-black (warmup render at new size was performed).
# A second colorbuffer call is needed to let GLMakie render with the new image plot.
buf2 = GLMakie.colorbuffer(session.glscreen)
nb2 = count(c -> (Float32(red(c)) + Float32(green(c)) + Float32(blue(c))) > 0.1, buf2)
println("RESIZE_NONBLACK=", nb2)
@assert nb2 > 500

println("OK_RESIZE")
"""

include("helpers.jl")
@testset "M5 interactive_display window shows RTX frame (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M5_VIEWPORT_PROG; timeout = 600, retries = 2, ready_marker = "OK_VIEWPORT")
    @info "M5 viewport output" output
    @test exitcode == 0
    @test contains(output, "OK_VIEWPORT")
    m = match(r"VIEWPORT_NONBLACK=(\d+)", output)
    @test m !== nothing && parse(Int, m.captures[1]) > 1000
end

@testset "M5 Base.close(session) teardown + idempotency (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M5_TEARDOWN_PROG; timeout = 600, retries = 2, ready_marker = "OK_TEARDOWN")
    @info "M5 teardown output" output
    @test exitcode == 0
    @test contains(output, "GL_CLOSED=true")
    @test contains(output, "OK_TEARDOWN")
end

@testset "M5 resize_viewport! rebuilds renderer at new size (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M5_RESIZE_PROG; timeout = 600, retries = 2, ready_marker = "OK_RESIZE")
    @info "M5 resize output" output
    @test exitcode == 0
    @test contains(output, "OK_RESIZE")
    m = match(r"RESIZE_FB=\((\d+), (\d+)\)", output)
    @test m !== nothing && parse(Int, m.captures[1]) == 500 && parse(Int, m.captures[2]) == 360
    m2 = match(r"RESIZE_NONBLACK=(\d+)", output)
    @test m2 !== nothing && parse(Int, m2.captures[1]) > 500
end
