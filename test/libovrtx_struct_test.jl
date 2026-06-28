using Test, CEnum
module _Probe
    using CEnum
    include(joinpath(@__DIR__, "..", "lib", "LibOVRTX", "src", "libovrtx_api.jl"))
end
@testset "M0.2 generated ABI" begin
    @test sizeof(_Probe.ovx_string_t) == 16
    @test sizeof(_Probe.ovrtx_xform_matrix44d_t) == 128
    @test sizeof(_Probe.DLTensor) == 48
    @test Int(_Probe.OVRTX_API_SUCCESS) == 0
    @test Int(_Probe.kDLCUDA) == 2
end
