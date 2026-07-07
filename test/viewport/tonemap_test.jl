using Test, OmniverseMakie
using ColorTypes, FixedPointNumbers
using OmniverseMakie: tonemap

# Host tonemap scalar — pure, no GPU.  Host-vs-device agreement of the
# shipped oriented kernel is proven in present_gpu_test.jl.
@testset "host tonemap (ACES + sRGB, linear scale)" begin
    # black → black; mid-grey monotonic; larger scale brightens.  `scale` is
    # a LINEAR multiplier (= exp2(EV)); callers convert from stops
    # (tonemap_frame / CUDA launcher).
    @test tonemap((0f0,0f0,0f0), 1f0) == RGBA{N0f8}(0,0,0,1)
    g1 = tonemap((0.18f0,0.18f0,0.18f0), exp2(0f0))   # scale 1  (0 EV)
    g2 = tonemap((0.18f0,0.18f0,0.18f0), exp2(1f0))   # scale 2  (+1 stop)
    @test Float32(g2.r) > Float32(g1.r)               # more scale → brighter
    # highlights clamp near 1
    @test Float32(tonemap((10f0,10f0,10f0), 1f0).r) ≈ 1f0 atol=0.02
    @test eltype(tonemap((1f0,0f0,0f0),1f0)) == N0f8
end

# A NaN HDR sample (one bad pixel in the HdrColor AOV) must NOT throw:
# `round(UInt8, NaN)` is an InexactError on the host and a trap in the CUDA
# kernel, so `_u8` maps NaN→0 before rounding.  Alpha stays fully opaque.
@testset "tonemap is NaN-safe (no InexactError; NaN → black)" begin
    black = RGBA{N0f8}(0, 0, 0, 1)
    @test tonemap((NaN32, NaN32, NaN32), 1f0) == black
    # a partial-NaN pixel: NaN channels → 0, finite channels tonemap normally
    px = tonemap((NaN32, 0f0, 0f0), 1f0)
    @test px.r == N0f8(0) && px.g == N0f8(0) && px.b == N0f8(0) && px.alpha == N0f8(1)
    # Inf propagates to NaN through ACES (Inf/Inf) → also clamps to black.
    @test tonemap((Inf32, Inf32, Inf32), 1f0) == black
end
