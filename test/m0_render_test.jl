using Test

# Paths / constants spliced into the subprocess prog.
const _RENDER_OV_JL   = joinpath(@__DIR__, "..", "src", "binding", "OV.jl")
const _RENDER_PRODUCT = "/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0"
const _RENDER_WARMUP  = 64

# The subprocess script:
# - uses LibOVRTX directly (OV.jl does `using ..LibOVRTX`, which resolves to Main.LibOVRTX)
# - includes OV.jl (which in turn includes dlpack.jl via its own include)
# - creates a Renderer, opens the USDA (OM_USDA, set by the harness), runs render_to_matrix
#   with warmup frames
# - asserts size == (1080,1920) and that the image is substantially non-black
#   (tolerates up to 100 edge/border pixels that may be black after 64 frames)
# - exits normally (clean teardown verified: no _exit needed)
const _RENDER_PROG = """
using LibOVRTX
include($(repr(_RENDER_OV_JL)))
using ColorTypes

const USDA    = ENV["OM_USDA"]
const PRODUCT = $(repr(_RENDER_PRODUCT))
const WARMUP  = $(_RENDER_WARMUP)

r = OV.Renderer()
OV.open_usd!(r, USDA)
img = OV.render_to_matrix(r, PRODUCT; warmup=WARMUP)

nonblack = count(c -> (red(c) + green(c) + blue(c)) > 0, img)
println("SIZE=", size(img), " NONBLACK=", nonblack)

@assert size(img) == (1080, 1920) "unexpected image size: \$(size(img))"
# Allow up to 100 edge/border pixels to be black (RT2 may not converge all
# border samples within 64 warmup frames; >=2073500 out of 2073600 is fine).
total = 1080 * 1920
@assert nonblack >= total - 100 "image mostly black: \$nonblack / \$total"

close(r)
println("OK")
"""

@testset "M0.6 Julia-native render (torus, LdrColor, non-black)" begin
    # Renderer shader compile (~30–60 s) + 64 warmup frames (~11 s); the 300 s default is ample.
    exitcode, output = run_ovrtx_subprocess(_RENDER_PROG)

    @test exitcode == 0
    @test contains(output, "OK")
    @test contains(output, "SIZE=(1080, 1920)")
    # Parse NONBLACK rather than exact-match the string so a few black edge pixels (up to 100)
    # still pass.  NONBLACK must be at least 2073500 (out of 2073600).
    m = match(r"NONBLACK=(\d+)", output)
    if m !== nothing
        nb = parse(Int, m.captures[1])
        @test nb >= 1080 * 1920 - 100
    else
        @test false  # sentinel — NONBLACK line missing entirely
    end
end
