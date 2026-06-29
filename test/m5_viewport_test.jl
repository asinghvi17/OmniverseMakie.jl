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
