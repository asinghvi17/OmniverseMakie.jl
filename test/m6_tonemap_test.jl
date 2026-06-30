using Test, OmniverseMakie
using ColorTypes, FixedPointNumbers
using OmniverseMakie: tonemap
@testset "M6 host tonemap (ACES + sRGB + exposure)" begin
    # black → black; mid-grey monotonic; exposure brightens.
    @test tonemap((0f0,0f0,0f0), 0f0) == RGBA{N0f8}(0,0,0,1)
    g1 = tonemap((0.18f0,0.18f0,0.18f0), 0f0)
    g2 = tonemap((0.18f0,0.18f0,0.18f0), 1f0)   # +1 stop
    @test Float32(g2.r) > Float32(g1.r)         # more exposure → brighter
    @test Float32(tonemap((10f0,10f0,10f0), 0f0).r) ≈ 1f0 atol=0.02  # highlights clamp near 1
    @test eltype(tonemap((1f0,0f0,0f0),0f0)) == N0f8
end
