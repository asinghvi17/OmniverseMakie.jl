using Test
@testset "M0.1 workspace loads" begin
    # Both packages must import without error (stubs at this point).
    @test (using LibOVRTX; true)
    @test (using OmniverseMakie; true)
end

include("libovrtx_struct_test.jl")
