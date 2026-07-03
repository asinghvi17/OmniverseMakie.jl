using Test
include(joinpath(@__DIR__, "..", "helpers.jl"))

# ---------------------------------------------------------------------------
# M1.2 — USD render-root authoring + OV reference add/remove wrappers
#
# Tests:
#   1. author_render_root! + add_usd_reference! + render → 256×256, non-black > 1000
#   2. remove_usd! + reset! + re-render → non-black drops sharply (< 500)
# ---------------------------------------------------------------------------

# Hardcoded cube USDA layer (geometry reused from m1.2-proven-stage.usda).
# Defined here so repr() can embed it safely into the subprocess program string.
# NOTE: reference layers must NOT include upAxis — the root stage's upAxis governs.
# Including upAxis = "Z" here caused ovrtx to produce a black render (confirmed by
# M1.2 diagnostics).  The root stage authored by author_render_root! sets upAxis="Z".
const _M12_CUBE_USDA = """#usda 1.0
( defaultPrim = "cube" )
def Mesh "cube"
{
    int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
    int[] faceVertexIndices = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 3, 7, 6, 2, 0, 4, 7, 3, 1, 2, 6, 5]
    normal3f[] normals = [(0, 0, -1), (0, 0, -1), (0, 0, -1), (0, 0, -1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, -1, 0), (0, -1, 0), (0, -1, 0), (0, -1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), (-1, 0, 0), (-1, 0, 0), (-1, 0, 0), (-1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0)] (
        interpolation = "faceVarying"
    )
    point3f[] points = [(-100.0, -100.0, -100.0), (100.0, -100.0, -100.0), (100.0, 100.0, -100.0), (-100.0, 100.0, -100.0), (-100.0, -100.0, 100.0), (100.0, -100.0, 100.0), (100.0, 100.0, 100.0), (-100.0, 100.0, 100.0)]
    color3f[] primvars:displayColor = [(1, 0, 0)] (
        interpolation = "constant"
    )
    uniform token subdivisionScheme = "none"
}
"""

# Build the subprocess program.  repr(_M12_CUBE_USDA) produces a single-line
# Julia string literal with all quotes / newlines properly escaped, so it is
# valid Julia code when embedded verbatim in the program string.
const _M12_USD_PROG = """
using OmniverseMakie, ColorTypes

# 1. Author the render root and add the cube as a reference.
scene  = Scene(size = (256, 256))
screen = OmniverseMakie.Screen(scene)
OmniverseMakie.author_render_root!(screen; resolution=(256, 256))

cube_usda = $(repr(_M12_CUBE_USDA))

handle = OmniverseMakie.OV.add_usd_reference!(screen.renderer, cube_usda, "/World/cube")

# 2. Render with warmup=32 and assert non-black.
img1 = OmniverseMakie.OV.render_to_matrix(screen.renderer, screen.product; warmup=32)
nonblack1 = count(c -> (red(c) + green(c) + blue(c)) > 0.0f0, img1)
println("SIZE=", size(img1), " NONBLACK1=", nonblack1)
@assert size(img1) == (256, 256) "size mismatch: \$(size(img1))"
@assert nonblack1 > 1000 "cube did not render: nonblack1=\$nonblack1 (expected > 1000)"

# 3. Remove the cube reference, reset, re-render, assert non-black drops.
OmniverseMakie.OV.remove_usd!(screen.renderer, handle)
OmniverseMakie.OV.reset!(screen.renderer)
img2 = OmniverseMakie.OV.render_to_matrix(screen.renderer, screen.product; warmup=32)
nonblack2 = count(c -> (red(c) + green(c) + blue(c)) > 0.0f0, img2)
println("NONBLACK2=", nonblack2)
@assert nonblack2 < 500 "cube did not disappear: nonblack2=\$nonblack2 (expected < 500)"

close(screen)
println("OK_USD_REF")
"""

@testset "M1.2 USD render-root + reference add/remove (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M12_USD_PROG; timeout = 600, retries = 2, ready_marker = "OK_USD_REF")
    @info "M1.2 subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_USD_REF")
    # Parse and validate NONBLACK1 > 1000
    m1 = match(r"NONBLACK1=(\d+)", output)
    if m1 !== nothing
        @test parse(Int, m1.captures[1]) > 1000
    else
        @test false  # NONBLACK1 line missing
    end
    # Parse and validate NONBLACK2 < 500
    m2 = match(r"NONBLACK2=(\d+)", output)
    if m2 !== nothing
        @test parse(Int, m2.captures[1]) < 500
    else
        @test false  # NONBLACK2 line missing
    end
end
