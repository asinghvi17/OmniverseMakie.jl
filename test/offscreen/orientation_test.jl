using Test
include(joinpath(@__DIR__, "..", "helpers.jl"))

# ---------------------------------------------------------------------------
# M1.5 orientation — determine and lock the vertical orientation of the
# ovrtx LdrColor readback via an ASYMMETRIC, color-coded fixture.
#
# Strategy: render TWO small meshes that differ only in world-Z:
#   RED box  at world Z ≈ +1.0 (high)  → should appear near the TOP of img
#   BLUE box at world Z ≈ -1.6 (low)   → should appear near the BOTTOM of img
#
# The default LScene camera (Camera3D) is Z-up, looking at the origin from a
# 3/4 vantage (roughly +X+Y+Z side).  World +Z projects toward the TOP of the
# rendered image for a right-side-up (top-left-origin) buffer.
#
# Decision rule: compute centroid ROW of red-dominant pixels and of blue-dominant
# pixels.  In a right-side-up image row indices INCREASE downward, so
#   red_row < blue_row  ⟹  red is ABOVE blue  ⟹  top-left-origin (no flip needed)
#   red_row > blue_row  ⟹  red is BELOW blue  ⟹  bottom-left-origin (flip needed)
# ---------------------------------------------------------------------------

const _M15_ORIENT_PROG = """
using OmniverseMakie, ColorTypes, FixedPointNumbers

OmniverseMakie.activate!()

fig = Figure()
ax  = LScene(fig[1, 1])

# RED box high in world Z  (should project to the TOP of the image)
mesh!(ax, Rect3f(Point3f(-0.3f0, -0.3f0, 0.8f0), Vec3f(0.6f0, 0.6f0, 0.5f0));
      color = :red)

# BLUE box low in world Z  (should project to the BOTTOM of the image)
mesh!(ax, Rect3f(Point3f(-0.3f0, -0.3f0, -1.6f0), Vec3f(0.6f0, 0.6f0, 0.5f0));
      color = :blue)

img = Makie.colorbuffer(ax.scene; warmup = 64)

H, W = size(img)
println("ORIENT_SIZE=", (H, W))

# Wrap in a function to avoid Julia top-level soft-scope warnings/errors.
function color_centroids(img)
    H, W = size(img)
    red_sum_row  = 0.0; red_n  = 0
    blue_sum_row = 0.0; blue_n = 0
    for h in 1:H, w in 1:W
        c = img[h, w]
        r = Float32(red(c)); g = Float32(green(c)); b = Float32(blue(c))
        if r > g && r > b && r > 0.1f0
            red_sum_row  += h; red_n  += 1
        end
        if b > r && b > g && b > 0.1f0
            blue_sum_row += h; blue_n += 1
        end
    end
    red_row  = red_n  > 0 ? red_sum_row  / red_n  : -1.0
    blue_row = blue_n > 0 ? blue_sum_row / blue_n : -1.0
    return (red_n = red_n, red_row = red_row, blue_n = blue_n, blue_row = blue_row)
end

st = color_centroids(img)
println("RED_N=",  st.red_n,  "  RED_ROW=",  st.red_row)
println("BLUE_N=", st.blue_n, "  BLUE_ROW=", st.blue_row)

# Hard assertion: world +Z (red) must appear ABOVE world -Z (blue).
# In a top-left-origin matrix, smaller row = higher in the image.
@assert st.red_n  > 50 "too few red-dominant pixels (got \$(st.red_n)) — red box not rendered"
@assert st.blue_n > 50 "too few blue-dominant pixels (got \$(st.blue_n)) — blue box not rendered"
@assert st.red_row < st.blue_row "ORIENTATION WRONG: red_row=\$(st.red_row) >= blue_row=\$(st.blue_row) (image is vertically flipped)"

println("ORIENT_OK")
"""

@testset "M1.5 orientation — red above blue (top-left-origin)" begin
    exitcode, output = run_ovrtx_subprocess(_M15_ORIENT_PROG; timeout = 900, retries = 2, ready_marker = "ORIENT_OK")
    @info "M1.5 orientation subprocess output" output
    @test exitcode == 0
    @test contains(output, "ORIENT_OK")

    # size ≥ 300²
    ms = match(r"ORIENT_SIZE=\((\d+), (\d+)\)", output)
    if ms !== nothing
        @test parse(Int, ms.captures[1]) >= 300
        @test parse(Int, ms.captures[2]) >= 300
    else
        @test false   # ORIENT_SIZE line missing
    end

    # red centroid row < blue centroid row (red is above blue → top-left-origin)
    mr = match(r"RED_ROW=([0-9.]+)", output)
    mb = match(r"BLUE_ROW=([0-9.]+)", output)
    if mr !== nothing && mb !== nothing
        red_row  = parse(Float64, mr.captures[1])
        blue_row = parse(Float64, mb.captures[1])
        @test red_row < blue_row
    else
        @test false   # ROW lines missing
    end
end
