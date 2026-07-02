using Test, OmniverseMakie
using ColorTypes, FixedPointNumbers
using OmniverseMakie: tonemap
@testset "M6 host tonemap (ACES + sRGB, linear scale)" begin
    # black → black; mid-grey monotonic; larger scale brightens.  `scale` is a LINEAR
    # multiplier (= exp2(EV)); callers convert from stops (tonemap_frame / CUDA launcher).
    @test tonemap((0f0,0f0,0f0), 1f0) == RGBA{N0f8}(0,0,0,1)
    g1 = tonemap((0.18f0,0.18f0,0.18f0), exp2(0f0))   # scale 1  (0 EV)
    g2 = tonemap((0.18f0,0.18f0,0.18f0), exp2(1f0))   # scale 2  (+1 stop)
    @test Float32(g2.r) > Float32(g1.r)               # more scale → brighter
    @test Float32(tonemap((10f0,10f0,10f0), 1f0).r) ≈ 1f0 atol=0.02  # highlights clamp near 1
    @test eltype(tonemap((1f0,0f0,0f0),1f0)) == N0f8
end

# M6.A Task 4 — host vs CUDA-kernel tonemap agreement.
#
# The device tonemap kernel reuses the SHARED scalar `tonemap` (after Task 4 made its N0f8
# quantization non-throwing → GPU-compilable AND byte-identical to the old form), so the
# host `tonemap_frame` and the device `tonemap_kernel_to_matrix` produce identical RGBA8.
# The CUDA ext requires both CUDA and GLMakie loaded, so the subprocess loads both.
const _M6_KERNEL_PROG = """
using OmniverseMakie, GLMakie, CUDA
using OmniverseMakie: tonemap
const Ext = Base.get_extension(OmniverseMakie, :OmniverseMakieCUDAExt)
println("EXT_LOADED=", Ext !== nothing)
# Same HDR input, host vs the CUDA tonemap kernel → identical RGBA8.
hdr = Float32[c==1 ? 0.5f0 : (c==2 ? 0.1f0 : 2.0f0) for c in 1:4, x in 1:8, y in 1:6]  # [C,W,H]
host = OmniverseMakie.tonemap_frame(hdr, 0.5f0)
dev  = Ext.tonemap_kernel_to_matrix(CuArray(hdr), 0.5f0)  # test-only helper
nmismatch = count(i -> host[i] != dev[i], eachindex(host))
println("KERNEL_MISMATCH=", nmismatch)
# Larger random sweep across the HDR range to stress the transcendentals.
using Random; Random.seed!(7)
hdr2 = rand(Float32, 4, 96, 72) .* 6.0f0
host2 = OmniverseMakie.tonemap_frame(hdr2, 0.0f0)
dev2  = Ext.tonemap_kernel_to_matrix(CuArray(hdr2), 0.0f0)
println("KERNEL_MISMATCH_RAND=", count(i -> host2[i] != dev2[i], eachindex(host2)), " / ", length(host2))
println("OK_KERNEL")
"""

include("helpers.jl")

@testset "M6 host vs CUDA-kernel tonemap agreement (subprocess, CUDA)" begin
    exitcode, output = run_ovrtx_subprocess(_M6_KERNEL_PROG; timeout=400)
    @info "M6 kernel output" output
    @test exitcode == 0
    @test contains(output, "EXT_LOADED=true")
    @test contains(output, "KERNEL_MISMATCH=0")
    @test contains(output, "KERNEL_MISMATCH_RAND=0 / ")   # device == host pixel-for-pixel
    @test contains(output, "OK_KERNEL")
end
