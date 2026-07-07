# Subprocess body for the surface!-texture test (read + run by
# materials/surface_texture_test.jl via run_ovrtx_subprocess).
#
# A TEXTURED `surface!` (image `color`) samples the grid's `st` UVs: a mesh
# emitted WITHOUT `st` gives the bound `diffuse_texture` nothing to sample
# and the surface renders WHITE.  A flat grid surface textured with a 2×2
# red/blue checker must show BOTH colours sampled across it.

using OmniverseMakie, ColorTypes, FixedPointNumbers, GeometryBasics
const OM = OmniverseMakie

OM.activate!(warmup = 48)

xs  = range(-1, 1; length = 24)
ys  = range(-1, 1; length = 24)
zs  = zeros(Float32, 24, 24)
# 2×2 red/blue checker
img = [RGBf(1, 0, 0) RGBf(0, 0, 1); RGBf(0, 0, 1) RGBf(1, 0, 0)]

fig = Figure(; size = (400, 400))
ax  = LScene(fig[1, 1]; show_axis = false,
    scenekw = (; lights = [AmbientLight(RGBf(0.9, 0.9, 0.9)), PointLight(RGBf(4, 4, 4), Vec3f(0, 0, 5))]))
surface!(ax, xs, ys, zs; color = img)
update_cam!(ax.scene, Vec3f(0, 0, 3.5), Vec3f(0, 0, 0), Vec3f(0, 1, 0))

screen  = OM.Screen(ax.scene)
img_out = Makie.colorbuffer(screen)
println("ELTYPE=", eltype(img_out))
println("SIZE=", size(img_out))

# Count strictly red-dominant and blue-dominant pixels — BOTH must appear
# (texture sampled, not a white/flat surface).
function dominant(img, which)
    count(img) do c
        r, g, b = Float32(red(c)), Float32(green(c)), Float32(blue(c))
        m = max(r, g, b)
        m <= 0.2f0 && return false
        which === :red ? (r > 1.3f0 * g && r > 1.3f0 * b) : (b > 1.3f0 * g && b > 1.3f0 * r)
    end
end
nr = dominant(img_out, :red)
nb = dominant(img_out, :blue)
println("RED_DOMINANT=", nr)
println("BLUE_DOMINANT=", nb)

@assert eltype(img_out) == RGBA{N0f8} "eltype is $(eltype(img_out))"
@assert nr > 200 "textured surface: red not sampled ($nr) — surface rendered white (no st UVs)?"
@assert nb > 200 "textured surface: blue not sampled ($nb)"

Base.close(screen)
println("OK_SURFACE_TEXTURE")
