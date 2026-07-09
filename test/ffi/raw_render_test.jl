using Test
include(joinpath(@__DIR__, "..", "helpers.jl"))

# Raw OV-layer render + live xform smoke, below Makie/Screen, in one
# subprocess (render startup amortized):
#   1. OV.Renderer() + open_usd! on the vendored fixture render a lit 320×240
#      frame.
#   2. write_xform! moves /World/FeatureBox + reset! restarts RT2 accumulation
#      → a meaningful number of pixels change by ≥8/255 per channel.

const _RAW_OV_JL   = joinpath(@__DIR__, "..", "..", "src", "binding", "OV.jl")
const _RAW_PRODUCT = "/Render/OVMakie/RenderProduct"
const _RAW_WARMUP  = 64

const _RAW_RENDER_PROG = """
using LibOVRTX
include($(repr(_RAW_OV_JL)))
using ColorTypes

const USDA    = ENV["OM_USDA"]
const PRODUCT = $(repr(_RAW_PRODUCT))
const WARMUP  = $(_RAW_WARMUP)
const PRIM    = "/World/FeatureBox"

r = OV.Renderer()
OV.open_usd!(r, USDA)

# --- frame 1: fixture render, non-black coverage ---
img1 = OV.render_to_matrix(r, PRODUCT; warmup=WARMUP)
H, W = size(img1)
nb = count(c -> (red(c) + green(c) + blue(c)) > 0, img1)
println("SIZE=", size(img1), " NONBLACK=", nb)
@assert size(img1) == (240, 320) "unexpected image size: \$(size(img1))"
@assert nb >= 1000 "image mostly black: \$nb / \$(H * W)"

# --- move torus (row-vector: translation in LAST row), reset, re-render ---
M = Float64[
    1.0  0.0  0.0  0.0
    0.0  1.0  0.0  0.0
    0.0  0.0  1.0  0.0
    180.0 0.0 0.0 1.0
]
OV.write_xform!(r, PRIM, M)
OV.reset!(r)
img2 = OV.render_to_matrix(r, PRODUCT; warmup=WARMUP)

# Count pixels where any channel changed by >= 8/255 (excludes RT2
# stochastic noise).
threshold = 8
changed = let c = 0
    for h in 1:H, w in 1:W
        c1 = img1[h, w]
        c2 = img2[h, w]
        dr = abs(Float32(red(c1))   - Float32(red(c2)))
        dg = abs(Float32(green(c1)) - Float32(green(c2)))
        db = abs(Float32(blue(c1))  - Float32(blue(c2)))
        if dr*255 >= threshold || dg*255 >= threshold || db*255 >= threshold
            c += 1
        end
    end
    c
end
println("CHANGED_PIXELS=", changed, " / ", H*W)
@assert changed >= 1000 "xform write did not move geometry: changed=\$changed < 1000"

close(r)
println("OK_RAW_RENDER")
"""

@testset "raw OV render + write_xform! (subprocess)" begin
    # Renderer shader compile (~30–60 s) + three 64-warmup renders; 600 s
    # is ample.
    exitcode, output = run_ovrtx_subprocess(_RAW_RENDER_PROG; timeout = 600, retries = 2,
                                            ready_marker = "SIZE=")
    @test exitcode == 0
    @test contains(output, "OK_RAW_RENDER")
    @test contains(output, "SIZE=(240, 320)")

    m = match(r"NONBLACK=(\d+)", output)
    @test m !== nothing && parse(Int, m.captures[1]) >= 1000

    mc = match(r"CHANGED_PIXELS=(\d+)", output)
    @test mc !== nothing && parse(Int, mc.captures[1]) >= 1000
end
