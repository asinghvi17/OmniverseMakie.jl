# Tests the manifest + the :copy path (no network). Run with the examples env.
include(joinpath(@__DIR__, "fetch_assets.jl"))   # defines MANIFEST, fetch_one, ASSETS_DIR, RPRN
using Test

@testset "fetch_assets manifest + copy" begin
    scenes = unique(first.(MANIFEST))
    # every ported scene that needs assets is represented
    for s in ["reflections_glass_material","materials_julia_room","helix","transparentMaterial",
              "uberMExample","earth_ina_julia_box","twoEarths","earthquakes","earthquakesLight",
              "submarineCables"]
        @test s in scenes
    end
    # the :copy sources exist under references
    for (scene, dest, kind, src) in MANIFEST
        kind === :copy && @test isfile(joinpath(RPRN, src))
    end
    # exercise one :copy end-to-end
    p = fetch_one("reflections_glass_material", "envLightImage.exr", :copy, "lights/envLightImage.exr")
    @test isfile(p)
end
println("FETCH_TESTS_OK")
