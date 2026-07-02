using Test

const _RENDERER_OV_JL = joinpath(@__DIR__, "..", "src", "binding", "OV.jl")

# Subprocess: include OV.jl (which brings SignalGuard via its own include("signals.jl")),
# create two Renderers with a close between them, print "OK".  The child does `using LibOVRTX`
# first so OV's `using ..LibOVRTX` resolves to Main.LibOVRTX.  "OK" guards against early-exit-0.
const _RENDERER_PROG = """
using LibOVRTX
include($(repr(_RENDERER_OV_JL)))
r1 = OV.Renderer(); close(r1)
r2 = OV.Renderer(); close(r2)
println("OK")
"""

@testset "M0.5 OV.Renderer create/close lifecycle" begin
    exitcode, output = run_ovrtx_subprocess(_RENDERER_PROG)
    @test exitcode == 0          # without the signal guard this is 139
    @test contains(output, "OK") # guard against silent early-exit-0
end
