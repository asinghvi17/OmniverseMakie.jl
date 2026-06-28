using Test
@testset "M0.3 LibOVRTX loads + links" begin
    ENV["OVRTX_LIBRARY_PATH"] = get(ENV, "OVRTX_LIBRARY_PATH",
        "/home/juliahub/temp/omniverse-makie/references/ovrtx/examples/python/minimal/.venv/lib/python3.13/site-packages/ovrtx/bin/libovrtx-dynamic.so")
    @eval using LibOVRTX
    @test LibOVRTX.version() == (UInt32(0), UInt32(3), UInt32(0))
    @test sizeof(LibOVRTX.ovx_string_t) == 16
end
