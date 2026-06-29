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
