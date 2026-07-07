# Subprocess body for the numeric-colormap LIVE-edit test (read + run by
# materials/colormap_test.jl via run_ovrtx_subprocess).
#
# The LIVE diff path colormap-maps a numeric `scaled_color`: a live
# `plot.color[]` edit (new numeric vector) on a colour-mapped meshscatter
# re-renders through push_to_ovrtx!→_push_displaycolor!→_scaled_to_display.
# Renders frame A, edits the colour values live, renders frame B; B must be
# non-black and DIFFER from A.

using OmniverseMakie, ColorTypes, FixedPointNumbers, GeometryBasics
const OM = OmniverseMakie

OM.activate!(warmup = 48)

n   = 40
pts = [Point3f(cos(t), sin(t), 0) for t in range(0, 2π; length = n)]

fig = Figure(; size = (400, 400))
ax  = LScene(fig[1, 1]; show_axis = false,
    scenekw = (; lights = [AmbientLight(RGBf(0.8, 0.8, 0.8)), PointLight(RGBf(6, 6, 6), Vec3f(0, 0, 6))]))
p = meshscatter!(ax, pts; markersize = 0.12, color = Float32.(1:n), colormap = :plasma)
update_cam!(ax.scene, Vec3f(0, 0, 4), Vec3f(0, 0, 0), Vec3f(0, 1, 0))

screen = OM.Screen(ax.scene)
imgA   = Makie.colorbuffer(screen)

# LIVE numeric-colour edit — reverse the per-point values so the colormap
# maps each point to a different colour.
p.color[] = Float32.(n:-1:1)
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

nbA = count(c -> lum(c) > 0.06f0, imgA)
nbB = count(c -> lum(c) > 0.06f0, imgB)
mad = meanabsdiff(imgA, imgB)
println("NONBLACK_A=", nbA)
println("NONBLACK_B=", nbB)
println("MEANABSDIFF=", mad)

@assert nbA > 1000 "frame A rendered near-black: $nbA"
@assert nbB > 1000 "frame B near-black — live numeric-colour edit crashed/blanked? ($nbB)"
@assert mad > 0.004 "live numeric-colour edit had no effect (mad=$mad) — colormap not re-applied on the live path?"

Base.close(screen)
println("OK_COLORMAP_LIVE")
