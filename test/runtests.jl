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
include("m0_render_test.jl")
include("m0_update_test.jl")

# Shared subprocess runner (M1+)
include("helpers.jl")
include("m1_screen_test.jl")
include("m1_usd_test.jl")
include("m1_camera_test.jl")
include("m1_lights_test.jl")
include("m1_mesh_render_test.jl")
include("m1_orientation_test.jl")
include("m1_save_record_test.jl")
include("m1_primitives_test.jl")

# M2.1 — open-stage Screen + live camera/lights + imperative insert!
include("m2_openstage_test.jl")
include("m2_insert_test.jl")
include("m2_rendercfg_test.jl")

# M2.2 — :ovrtx_renderobject diff node + push_to_ovrtx! minimal edits
include("m2_diffnode_test.jl")

# M2.3 — USD subscene grouping (def Scope hierarchy mirroring the Makie scene tree)
include("m2_subscene_test.jl")

# M2.4 — persistent hot-path bindings (map_attribute / bind_array_attribute)
include("m2_binding_test.jl")

# M2.5 — leak-free delete! / delete!(scene) / empty! teardown
include("m2_delete_test.jl")

# M2.6 — hot-path throughput benchmark + gate (≥30 Hz for 10^4 xforms OR 10^5 points)
include("m2_bench_test.jl")
