using Test
# Offscreen-only: using OmniverseMakie alone must NOT require GLMakie/CUDA, and
# interactive_display must exist but error helpfully until GLMakie is loaded.
const _M6_OFFSCREEN_PROG = """
using OmniverseMakie
println("HAS_INTERACTIVE=", isdefined(OmniverseMakie, :interactive_display))
# A 2-D offscreen render still works with no GLMakie:
scene = Scene(size=(64,64)); cam3d!(scene)
mesh!(scene, Rect3f(Point3f(0), Vec3f(1)); color=:red)
ok = false
try
    OmniverseMakie.interactive_display(scene)   # no GLMakie ext loaded
catch e
    global ok = occursin("GLMakie", sprint(showerror, e))
end
println("ERRORS_WITHOUT_GLMAKIE=", ok)
println("OK_OFFSCREEN")
"""
include("helpers.jl")
@testset "M6 offscreen load (no GLMakie/CUDA needed)" begin
    exitcode, output = run_ovrtx_subprocess(_M6_OFFSCREEN_PROG; timeout=300)
    @info "M6 offscreen output" output
    @test exitcode == 0
    @test contains(output, "HAS_INTERACTIVE=true")
    @test contains(output, "ERRORS_WITHOUT_GLMAKIE=true")
    @test contains(output, "OK_OFFSCREEN")
end
