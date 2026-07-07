# Subprocess body for the numeric-color+colormap test (read + run by
# materials/colormap_test.jl via run_ovrtx_subprocess).
#
# The backend colormap-maps a NUMERIC `color` vector on meshscatter! and
# linesegments!.  Renders a ring of colormapped points + colormapped line
# segments and asserts the render is non-black and shows MANY distinct
# colours (the colormap gradient was sampled, not collapsed to one colour).

using OmniverseMakie, ColorTypes, FixedPointNumbers, GeometryBasics
const OM = OmniverseMakie

OM.activate!(warmup = 48)

n  = 60
θ  = range(0, 2π; length = n)
pts  = [Point3f(cos(t), sin(t), 0) for t in θ]
vals = collect(Float32.(1:n))                       # numeric colour → colormap

segpts = Point3f[]
for i in 1:(n - 1)
    push!(segpts, pts[i]); push!(segpts, pts[i + 1])
end

fig = Figure(; size = (500, 500))
ax  = LScene(fig[1, 1]; show_axis = false,
    scenekw = (; lights = [AmbientLight(RGBf(0.7, 0.7, 0.7)), PointLight(RGBf(6, 6, 6), Vec3f(0, 0, 6))]))
meshscatter!(ax, pts; markersize = 0.11, color = vals, colormap = :plasma)
linesegments!(ax, segpts; color = Float32.(1:length(segpts)), colormap = :viridis, linewidth = 3)
update_cam!(ax.scene, Vec3f(0, 0, 4), Vec3f(0, 0, 0), Vec3f(0, 1, 0))

screen = OM.Screen(ax.scene)
img    = Makie.colorbuffer(screen)
println("ELTYPE=", eltype(img))
println("SIZE=", size(img))

lum(c) = Float32(red(c)) + Float32(green(c)) + Float32(blue(c))
nb = count(c -> lum(c) > 0.06f0, img)
println("NONBLACK=", nb)

# Distinct colour buckets among lit pixels — a colormap gradient spans many;
# a single flat colour (or a failure) would not.
function color_buckets(img)
    s = Set{NTuple{3,Int}}()
    for c in img
        lum(c) > 0.12f0 || continue
        push!(s, (round(Int, Float32(red(c)) * 6),
                  round(Int, Float32(green(c)) * 6),
                  round(Int, Float32(blue(c)) * 6)))
    end
    return length(s)
end
nbuck = color_buckets(img)
println("COLOR_BUCKETS=", nbuck)

@assert eltype(img) == RGBA{N0f8} "eltype is $(eltype(img))"
@assert nb > 1000 "colormapped scatter+lines rendered near-black: nonblack=$nb"
@assert nbuck >= 6 "expected a varied colormap gradient, got only $nbuck colour buckets"

Base.close(screen)
println("OK_COLORMAP")
