using Test

# Set OVRTX_LIBRARY_PATH before any `using LibOVRTX` so __init__ can dlopen it.
# The package itself does NOT hardcode this path; only the test harness sets the default.
get!(ENV, "OVRTX_LIBRARY_PATH",
    "/home/juliahub/temp/omniverse-makie/references/ovrtx/examples/python/minimal/.venv/lib/python3.13/site-packages/ovrtx/bin/libovrtx-dynamic.so")

@testset "M0.1 workspace loads" begin
    # Both packages must import without error (stubs at this point).
    @test (using LibOVRTX; true)
    @test (using OmniverseMakie; true)
end

include("libovrtx_struct_test.jl")
include("libovrtx_load_test.jl")
include("m0_signals_test.jl")
include("m0_renderer_test.jl")
