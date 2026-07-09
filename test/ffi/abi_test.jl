using Test, CEnum

# Generated-binding ABI layout + real-library link. Pure — no GPU subprocess.

# Struct sizes / enum values of the GENERATED bindings, included into a
# throwaway module (no .so load) — catches regen drift (wrong layout, enum
# reorder) without the GPU.
module _Probe
    using CEnum
    include(joinpath(@__DIR__, "..", "..", "lib", "LibOVRTX", "src", "libovrtx_api.jl"))
end
@testset "generated ABI layout" begin
    @test sizeof(_Probe.ovx_string_t) == 16
    @test sizeof(_Probe.ovrtx_xform_matrix44d_t) == 128
    @test sizeof(_Probe.DLTensor) == 48
    @test Int(_Probe.OVRTX_API_SUCCESS) == 0
    @test Int(_Probe.kDLCUDA) == 2
end

# The real shared library dlopens at __init__ AND its runtime version()
# matches the pinned build — the regression anchor for an ovrtx
# upgrade/mismatch.
@testset "LibOVRTX loads + links" begin
    @eval using LibOVRTX
    @test LibOVRTX.version() == (UInt32(0), UInt32(3), UInt32(0))
end
