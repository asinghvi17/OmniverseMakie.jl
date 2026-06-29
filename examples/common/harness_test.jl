# Unit tests for harness helpers (no GPU/render). Run with the examples env.
include(joinpath(@__DIR__, "harness.jl"))
using Test, ColorTypes, FixedPointNumbers

@testset "harness helpers" begin
    black = fill(RGBA{N0f8}(0,0,0,1), 10, 10)
    red   = fill(RGBA{N0f8}(1,0,0,1), 10, 10)
    blue  = fill(RGBA{N0f8}(0,0,1,1), 10, 10)
    @test nonblack_count(black) == 0
    @test nonblack_count(red)   == 100
    @test_throws AssertionError assert_nonblack(black, "x")
    @test assert_nonblack(red, "x") == 100
    @test color_fraction(red,  :red)  > 0.99
    @test color_fraction(red,  :blue) < 0.01
    @test color_fraction(blue, :blue) > 0.99
    # asset() errors clearly when missing
    @test_throws ErrorException asset("nope", "nope.png")
end
println("HARNESS_TESTS_OK")
