# Subprocess body for the material-composition INTEGRATION render (read + run
# by materials/material_test.jl via run_ovrtx_subprocess).  Standalone .jl so
# the Makie scene setup needs no escaping.
#
# End to end THROUGH the real Screen / colorbuffer pipeline:
#   - `mesh!(…; color, material=(; metallic, roughness))` is MATERIALIZED →
#     `author_root_from_scene!` PRE-AUTHORS its OmniPBR material into
#     /World/Looks and the Mesh build branch BINDS it (`OV.bind_material!`) +
#     omits displayColor → the sphere reads METALLIC (sharp specular
#     highlight over a dark body).
#   - `mesh!(…; color)` (no material) stays on the displayColor path →
#     renders the flat diffuse colour (regression: displayColor is unbroken).
# The metallic render is compared to the SAME sphere rendered plain-diffuse:
# a much higher luminance contrast + a substantial pixel-wise difference
# proves the pre-authored material BOUND through the full pipeline.

using OmniverseMakie, ColorTypes, FixedPointNumbers, GeometryBasics
const OM = OmniverseMakie

OM.activate!(warmup = 120)

sph = GeometryBasics.normal_mesh(GeometryBasics.Tesselation(Sphere(Point3f(0), 1.0f0), 96))
basecol = RGBf(0.72, 0.20, 0.20)

# --- metrics ------------------------------------------------------------------
lum(c) = 0.2126f0 * Float32(red(c)) + 0.7152f0 * Float32(green(c)) + 0.0722f0 * Float32(blue(c))
function stats(img)
    L   = vec([lum(c) for c in img])
    lit = sort(L[L .> 0.02f0])
    isempty(lit) && return (nonblack = 0, meanlit = 0.0f0, maxl = 0.0f0,
                            p50 = 0.0f0, p95 = 0.0f0, contrast = 0.0f0,
                            mr = 0.0f0, mg = 0.0f0, mb = 0.0f0)
    q(p) = lit[clamp(round(Int, p * length(lit)), 1, length(lit))]
    p50, p95 = q(0.50), q(0.95)
    # mean RGB over lit pixels (red-dominance check for the plain render)
    sr = 0.0f0; sg = 0.0f0; sb = 0.0f0; n = 0
    for c in img
        if lum(c) > 0.02f0
            sr += Float32(red(c)); sg += Float32(green(c)); sb += Float32(blue(c)); n += 1
        end
    end
    n = max(n, 1)
    return (nonblack = length(lit), meanlit = sum(lit) / length(lit), maxl = lit[end],
            p50 = p50, p95 = p95, contrast = p95 / (p50 + 1.0f-3),
            mr = sr / n, mg = sg / n, mb = sb / n)
end
function meanabsdiff(a, b)
    s = 0.0
    @inbounds for i in eachindex(a, b)
        s += abs(Float32(red(a[i]))   - Float32(red(b[i]))) +
             abs(Float32(green(a[i])) - Float32(green(b[i]))) +
             abs(Float32(blue(a[i]))  - Float32(blue(b[i])))
    end
    return s / (3 * length(a))
end

# Render a fresh scene holding ONE sphere built by `addplot!(ax)`.  The
# material is baked into the stage at open time, so each render gets its own
# Screen + stage (fully isolating the materialized and plain renders).  The
# Screen is created explicitly and `close`d after the frame is read — an
# implicit `colorbuffer(scene)` Screen never closes, leaking a renderer.
function render_scene(addplot!)
    fig = Figure()
    ax  = LScene(fig[1, 1])
    addplot!(ax)
    screen = OM.Screen(ax.scene; warmup = 120)
    try
        return Makie.colorbuffer(screen)
    finally
        close(screen)
    end
end

# Plain: displayColor path (regression baseline).
imgP = render_scene(ax -> mesh!(ax, sph; color = basecol))
# Materialized: OmniPBR metallic via the `material=` escape hatch.
imgM = render_scene(ax -> mesh!(ax, sph; color = basecol,
                                material = (; metallic = 1.0, roughness = 0.08)))

sP  = stats(imgP)
sM  = stats(imgM)
mad = meanabsdiff(imgP, imgM)
cr  = sM.contrast / (sP.contrast + 1.0f-3)

println("ELTYPE=", eltype(imgM))
println("SIZE=", size(imgM))
println("PLAIN_STATS nonblack=$(sP.nonblack) meanlit=$(sP.meanlit) p50=$(sP.p50) p95=$(sP.p95) contrast=$(sP.contrast) mrgb=($(sP.mr),$(sP.mg),$(sP.mb))")
println("METALLIC_STATS nonblack=$(sM.nonblack) meanlit=$(sM.meanlit) p50=$(sM.p50) p95=$(sM.p95) contrast=$(sM.contrast)")
println("MEANABSDIFF=$(mad)")
println("CONTRAST_RATIO=$(cr)")

@assert eltype(imgM) == RGBA{N0f8} "eltype is $(eltype(imgM)) (expected RGBA{N0f8})"
# Both renders are non-black.
@assert sP.nonblack > 1000 "plain render is (near) black: nonblack=$(sP.nonblack)"
@assert sM.nonblack > 1000 "metallic render is (near) black: nonblack=$(sM.nonblack)"
# Regression: the plain displayColor render is RED-dominant.
@assert sP.mr > sP.mg && sP.mr > sP.mb "plain render not red-dominant: mrgb=($(sP.mr),$(sP.mg),$(sP.mb))"
# The materialized render differs SUBSTANTIALLY from the plain render — the
# pre-authored OmniPBR material BOUND through the full pipeline.  Thresholds
# sit well below the metallic signal and above flat-diffuse noise, so
# path-traced noise stays clear of them.
@assert mad > 0.02 "metallic render too similar to plain (mad=$(mad)): material:binding did not take"
# Metallic SIGNATURE: a concentrated specular highlight over a dark body →
# a much higher luminance contrast than the flat diffuse sphere.
@assert sM.contrast > 2.0 "no metallic specular signature: metallic contrast=$(sM.contrast)"
@assert cr > 1.5 "metallic not distinct enough from plain diffuse: contrast ratio=$(cr)"

println("OK_COMPOSE")
