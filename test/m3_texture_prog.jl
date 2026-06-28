# Subprocess body for the M3.3 image-texture INTEGRATION render (read + run by
# test/m3_texture_test.jl via run_ovrtx_subprocess).  Standalone .jl so the scene setup
# needs no escaping.
#
# ★ VALIDATE-FIRST (load-bearing): proves an image `color` is SAMPLED as a texture
# across the surface, not collapsed to a flat average.  A quad `mesh!` is textured with a
# 2-colour (red/blue) checker through the FULL Screen/colorbuffer pipeline:
#   - `mesh!(…; color = checker_img)` is MATERIALIZED → `author_root_from_scene!`
#     PRE-AUTHORS an OmniPBR material with `asset inputs:diffuse_texture = @<temp PNG>@`
#     (+ `project_uvw = 0`) into /World/Looks, and the Mesh build branch authors the
#     `texCoord2f[] primvars:st` UV primvar from `:texturecoordinates` + binds the
#     material → the quad shows the checker.
# The render must contain BOTH a clearly RED-dominant region AND a clearly BLUE-dominant
# region (the texture is sampled across the surface).  A failed texture load (flat /
# black / averaged purple) shows NEITHER → the assertion fails.

using OmniverseMakie, ColorTypes, FixedPointNumbers, GeometryBasics
const OM = OmniverseMakie

OM.activate!(warmup = 120)

# A 2-colour red/blue checker — a 2×2 LOGICAL checker SCALED to solid blocks so each
# colour owns a large pure region robust to bilinear texture filtering.
function checker(nblocks::Int, block::Int)
    W   = nblocks * block
    img = Matrix{RGBf}(undef, W, W)
    @inbounds for i in 1:W, j in 1:W
        bi = (i - 1) ÷ block
        bj = (j - 1) ÷ block
        img[i, j] = iseven(bi + bj) ? RGBf(1, 0, 0) : RGBf(0, 0, 1)
    end
    return img
end
checkimg = checker(2, 128)   # 256×256, four solid quadrants (red/blue/blue/red)

# A flat 3-D quad (z = 0) carrying per-vertex UVs + normals → Makie surfaces the UVs as
# `:texturecoordinates` (the `st` source).  3-D positions so `usda_mesh` authors point3f.
positions = Point3f[(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
uvs       = Vec2f[(0, 0), (1, 0), (1, 1), (0, 1)]
normals   = Vec3f[(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)]
qfaces    = [GeometryBasics.GLTriangleFace(1, 2, 3), GeometryBasics.GLTriangleFace(1, 3, 4)]
quad      = GeometryBasics.Mesh(positions, qfaces; uv = uvs, normal = normals)

# Render the textured quad through the real pipeline.  A fresh Figure/LScene → its own
# Screen + stage; look straight down the quad normal (+z) so it fills the frame face-on.
fig = Figure()
ax  = LScene(fig[1, 1]; show_axis = false)
plt = mesh!(ax, quad; color = checkimg)
Makie.update_cam!(ax.scene, Vec3f(0, 0, 6), Vec3f(0, 0, 0), Vec3f(0, 1, 0))
img = Makie.colorbuffer(ax.scene; warmup = 120)

# Classify each lit pixel as red-dominant / blue-dominant (a flat average is NEITHER).
function classify(im)
    nonblack = 0; nred = 0; nblue = 0
    for c in im
        r = Float32(red(c)); g = Float32(green(c)); b = Float32(blue(c))
        L = 0.2126f0 * r + 0.7152f0 * g + 0.0722f0 * b
        L > 0.02f0 || continue
        nonblack += 1
        if     r > b + 0.15f0 && r >= g; nred  += 1
        elseif b > r + 0.15f0 && b >= g; nblue += 1
        end
    end
    return (nonblack = nonblack, nred = nred, nblue = nblue)
end

# The pre-authored Looks scope must carry the diffuse_texture asset (open-time resolve).
looks = OM.materialized_looks_usda(ax.scene)
has_difftex = occursin("inputs:diffuse_texture = @", looks)
texpath = OM._texture_asset_for(OM._plot_color(plt), plt)

s = classify(img)
println("ELTYPE=", eltype(img))
println("SIZE=", size(img))
println("HAS_DIFFUSE_TEXTURE=", has_difftex)
println("TEXPATH=", texpath, " EXISTS=", isfile(texpath))
println("NONBLACK=", s.nonblack)
println("RED_DOMINANT=", s.nred)
println("BLUE_DOMINANT=", s.nblue)

@assert eltype(img) == RGBA{N0f8} "eltype is $(eltype(img)) (expected RGBA{N0f8})"
@assert has_difftex "Looks scope missing diffuse_texture (texture not pre-authored)"
@assert isfile(texpath) "temp texture asset not on disk: $(texpath)"
@assert s.nonblack > 1000 "textured quad render is (near) black: nonblack=$(s.nonblack)"
# BOTH checker colours must appear — the texture is SAMPLED, not a flat average.
@assert s.nred  > 200 "no red-dominant region: red=$(s.nred) (texture not sampled?)"
@assert s.nblue > 200 "no blue-dominant region: blue=$(s.nblue) (texture not sampled?)"

println("OK_TEXTURE")
