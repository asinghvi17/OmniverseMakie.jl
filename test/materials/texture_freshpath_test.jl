# Image-texture fresh path (subprocess behavioral proof).
#
# An image `color` writes a temp PNG for the OmniPBR `diffuse_texture`.  The
# path must be FRESH and unique per write: re-authoring the SAME plot onto a
# stable path overwrites the on-disk file, which ovrtx flags as a "video
# texture" and DISABLES (and races its async read → "Corrupt PNG"), so only
# the first still keeps its texture.
# (The pure `_texture_asset_for` fresh-path unit lives in
# authoring/usd_hygiene_test.jl.)

using Test
include(joinpath(@__DIR__, "..", "helpers.jl"))

const _TEXFRESH_PROG = """
using OmniverseMakie, ColorTypes, FixedPointNumbers, GeometryBasics
const OM = OmniverseMakie

# A flat quad with UVs, textured by an all-RED image.
positions = Point3f[(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
uvs = Vec2f[(0, 0), (1, 0), (1, 1), (0, 1)]; normals = fill(Vec3f(0, 0, 1), 4)
qf  = [GeometryBasics.GLTriangleFace(1, 2, 3), GeometryBasics.GLTriangleFace(1, 3, 4)]
quad = GeometryBasics.Mesh(positions, qf; uv = uvs, normal = normals)
redimg = fill(RGBf(1, 0, 0), 32, 32)

fig = Figure(); ax = LScene(fig[1, 1]; show_axis = false)
plt = mesh!(ax, quad; color = redimg)
Makie.update_cam!(ax.scene, Vec3f(0, 0, 6), Vec3f(0, 0, 0), Vec3f(0, 1, 0))

function red_frac(im)
    nb = 0; nr = 0
    for c in im
        r = Float32(red(c)); g = Float32(green(c)); b = Float32(blue(c))
        (0.21f0r + 0.72f0g + 0.07f0b) > 0.02f0 || continue
        nb += 1
        r > b + 0.15f0 && r >= g && (nr += 1)
    end
    return (nb, nr)
end

# Render the SAME plot with TWO Screens in ONE process (same objectid → a
# stable temp path would be overwritten).  Both must render the red texture.
s1 = OM.Screen(ax.scene); a = red_frac(Makie.colorbuffer(s1)); close(s1)
s2 = OM.Screen(ax.scene); b = red_frac(Makie.colorbuffer(s2)); close(s2)
println("STILL_A_NONBLACK=", a[1]); println("STILL_A_RED=", a[2])
println("STILL_B_NONBLACK=", b[1]); println("STILL_B_RED=", b[2])
println("OK_TEXFRESH")
"""

@testset "image-texture survives re-author (subprocess)" begin
    exitcode, out = run_ovrtx_subprocess(_TEXFRESH_PROG; timeout = 600, retries = 4,
                                         ready_marker = "OK_TEXFRESH")
    contains(out, "OK_TEXFRESH") || @info "texture fresh-path output" out
    @test exitcode == 0   # a teardown segfault after OK_TEXFRESH must not pass
    @test contains(out, "OK_TEXFRESH")
    # BOTH stills of the same plot render the red texture (still 2 is not
    # disabled).
    for still in ("A", "B")
        m_nb = match(Regex("STILL_$(still)_NONBLACK=(\\d+)"), out)
        m_r  = match(Regex("STILL_$(still)_RED=(\\d+)"), out)
        @test m_nb !== nothing && parse(Int, m_nb.captures[1]) > 300
        # red-dominant = texture applied
        @test m_r  !== nothing && parse(Int, m_r.captures[1])  > 300
    end
end
