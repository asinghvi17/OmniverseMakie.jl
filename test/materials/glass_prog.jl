# Subprocess body for the M4 refractive-glass follow-up (read + run by test/m4_glass_test.jl).
#
# Proves `material=(; glass=true, …)` authors + binds a TRUE OmniGlass material that REFRACTS.
# A clear glass sphere in front of a bright-red wall has the unmistakable glass SIGNATURE in its
# body: a sharp bright SPECULAR highlight AND DARK refraction/total-internal-reflection — both
# present at once.  A flat opaque sphere (or the surrounding red wall) shows neither.  Rendered
# through the full Screen/colorbuffer pipeline.

using OmniverseMakie, ColorTypes, FixedPointNumbers, GeometryBasics
const OM = OmniverseMakie

OM.activate!(warmup = 48)

SphereTess(o, r) = uv_normal_mesh(Tesselation(Sphere(Point3f(o...), Float32(r)), 64))

fig    = Figure(; size = (400, 400))
lights = [AmbientLight(RGBf(0.6, 0.6, 0.6)), PointLight(RGBf(8, 8, 8), Vec3f(2, 2, 5))]
ax     = LScene(fig[1, 1]; show_axis = false, scenekw = (; lights = lights))

mesh!(ax, Rect3f(Vec3f(-3, -3, -2), Vec3f(6, 6, 0.1)); color = RGBf(0.9, 0.05, 0.05))  # red wall
mesh!(ax, SphereTess((0, 0, 0.5), 0.9); material = (; glass = true, ior = 1.5f0))       # glass sphere

update_cam!(ax.scene, Vec3f(0, 0, 4), Vec3f(0, 0, 0), Vec3f(0, 1, 0))

screen = OM.Screen(ax.scene)
img    = Makie.colorbuffer(screen)
println("ELTYPE=", eltype(img))
println("SIZE=", size(img))

lum(c) = Float32(red(c)) + Float32(green(c)) + Float32(blue(c))
nb = count(c -> lum(c) > 0.06f0, img)

# Glass signature in the central sphere region: a SHARP bright specular highlight (near-white,
# lum≈3) AND DARK refraction/TIR (lum<0.4) both present.  The red wall (lum≈1.0) triggers
# neither, so this isolates the glass body.
function center_signature(img)
    H, W = size(img); bright = 0; dark = 0
    for h in round(Int, 0.30H):round(Int, 0.70H), w in round(Int, 0.30W):round(Int, 0.70W)
        l = lum(img[h, w])
        l > 2.2f0 && (bright += 1)
        l < 0.40f0 && (dark += 1)
    end
    return bright, dark
end
bright, dark = center_signature(img)
println("NONBLACK=", nb)
println("CENTER_BRIGHT=", bright)
println("CENTER_DARK=", dark)

@assert eltype(img) == RGBA{N0f8} "eltype is $(eltype(img))"
@assert nb > 1000 "glass scene rendered near-black: nonblack=$nb"
@assert bright > 3 "no glass specular highlight (center bright=$bright) — material not glossy/bound?"
@assert dark > 50 "no glass refraction/TIR darkening (center dark=$dark) — sphere opaque, not glass?"

Base.close(screen)
println("OK_GLASS")
