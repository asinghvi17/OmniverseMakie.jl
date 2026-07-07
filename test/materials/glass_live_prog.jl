# Subprocess body for the glass LIVE-edit test (read + run by
# materials/glass_test.jl).
#
# A LIVE `plot.material[]` edit on a GLASS plot routes to OmniGlass input
# names: a clear glass sphere renders (frame A), its material is edited live
# to strongly RED-tinted glass, and frame B must DIFFER (the edit wrote
# `glass_color`; OmniPBR names on an OmniGlass shader are a silent no-op).

using OmniverseMakie, ColorTypes, FixedPointNumbers, GeometryBasics
const OM = OmniverseMakie

OM.activate!(warmup = 48)

SphereTess(o, r) = uv_normal_mesh(Tesselation(Sphere(Point3f(o...), Float32(r)), 64))

fig    = Figure(; size = (400, 400))
lights = [AmbientLight(RGBf(0.7, 0.7, 0.7)), PointLight(RGBf(8, 8, 8), Vec3f(2, 2, 5))]
ax     = LScene(fig[1, 1]; show_axis = false, scenekw = (; lights = lights))

# grey wall + clear glass sphere
mesh!(ax, Rect3f(Vec3f(-3, -3, -2), Vec3f(6, 6, 0.1)); color = RGBf(0.85, 0.85, 0.85))
gp = mesh!(ax, SphereTess((0, 0, 0.5), 0.9); material = (; glass = true, ior = 1.5f0))

update_cam!(ax.scene, Vec3f(0, 0, 4), Vec3f(0, 0, 0), Vec3f(0, 1, 0))

screen = OM.Screen(ax.scene)
imgA   = Makie.colorbuffer(screen)

# LIVE glass material edit — strongly tint the glass RED.  (plot.material[]
# live-set needs a Makie.Attributes, not a bare NamedTuple.)
gp.material[] = Makie.Attributes(; glass = true, ior = 1.5f0, base_color = RGBf(1.0, 0.0, 0.0))
imgB = Makie.colorbuffer(screen)

lum(c) = Float32(red(c)) + Float32(green(c)) + Float32(blue(c))
function meanabsdiff(a, b)
    s = 0.0
    @inbounds for i in eachindex(a, b)
        s += abs(Float32(red(a[i]))   - Float32(red(b[i]))) +
             abs(Float32(green(a[i])) - Float32(green(b[i]))) +
             abs(Float32(blue(a[i]))  - Float32(blue(b[i])))
    end
    return s / (3 * length(a))
end
# red-dominant pixels in the central glass region (the red tint should
# raise this A→B)
function red_center(img)
    H, W = size(img); cnt = 0
    for h in round(Int, 0.35H):round(Int, 0.65H), w in round(Int, 0.35W):round(Int, 0.65W)
        c = img[h, w]; r, g, b = Float32(red(c)), Float32(green(c)), Float32(blue(c))
        (r > 0.2f0 && r > 1.3f0 * g && r > 1.3f0 * b) && (cnt += 1)
    end
    return cnt
end

nbB = count(c -> lum(c) > 0.06f0, imgB)
mad = meanabsdiff(imgA, imgB)
rcA = red_center(imgA)
rcB = red_center(imgB)
println("NONBLACK_B=", nbB)
println("MEANABSDIFF=", mad)
println("RED_CENTER_A=", rcA)
println("RED_CENTER_B=", rcB)

@assert nbB > 1000 "frame B near-black — live glass edit crashed/blanked? ($nbB)"
@assert mad > 0.01 "live glass material edit had no effect (mad=$mad) — misrouted to a non-OmniGlass input?"
@assert rcB > rcA + 50 "live glass red tint did not take (red_center A=$rcA B=$rcB) — glass_color not written?"

Base.close(screen)
println("OK_GLASS_LIVE")
