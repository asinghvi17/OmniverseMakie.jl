using Test, OmniverseMakie
using ColorTypes, FixedPointNumbers
using OmniverseMakie: tonemap

# Host tonemap scalar (formerly m6_tonemap_test.jl testset 1) — pure, no GPU.
# The old testset 2 (host-vs-device agreement of a NON-oriented kernel) was retired: that
# kernel had no production caller, and present_gpu_test.jl proves the SHIPPED oriented
# kernel byte-equals the host chain, including a random HDR sweep.
@testset "host tonemap (ACES + sRGB, linear scale)" begin
    # black → black; mid-grey monotonic; larger scale brightens.  `scale` is a LINEAR
    # multiplier (= exp2(EV)); callers convert from stops (tonemap_frame / CUDA launcher).
    @test tonemap((0f0,0f0,0f0), 1f0) == RGBA{N0f8}(0,0,0,1)
    g1 = tonemap((0.18f0,0.18f0,0.18f0), exp2(0f0))   # scale 1  (0 EV)
    g2 = tonemap((0.18f0,0.18f0,0.18f0), exp2(1f0))   # scale 2  (+1 stop)
    @test Float32(g2.r) > Float32(g1.r)               # more scale → brighter
    @test Float32(tonemap((10f0,10f0,10f0), 1f0).r) ≈ 1f0 atol=0.02  # highlights clamp near 1
    @test eltype(tonemap((1f0,0f0,0f0),1f0)) == N0f8
end
