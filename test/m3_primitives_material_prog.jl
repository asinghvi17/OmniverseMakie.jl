# Subprocess body for the M3.5 primitive-coverage material INTEGRATION renders
# (read + run by test/m3_material_test.jl via run_ovrtx_subprocess).  Standalone .jl
# so the Makie scene setup needs no escaping.
#
# Proves M3.5 end to end THROUGH the real Screen / colorbuffer pipeline that the OmniPBR
# `material=` escape hatch now applies to the OTHER primitive types:
#   - MeshScatter: `material=(; metallic, roughness)` → the instances render METALLIC
#     (sharp specular over a dark body), DISTINCT from the same meshscatter shaded
#     plain-diffuse.  (A materialized instancer is rendered as a merged UsdGeomMesh, since
#     ovrtx does not honor materials on a PointInstancer.)
#   - Surface: `material=(; metallic, roughness)` → a materialized `UsdGeomMesh` (NO
#     displayColor) that BINDS + renders, substantially different from the plain surface.
#   - Lines: `material=(; emissive=(1,0,0))` → an emissive RED curve (validates the
#     `bool enable_emission` + `emissive_intensity` fix — without it emission is OFF and
#     the curve reads OmniPBR's near-grey default, NOT red).  Red-dominance is measured by
#     a RED-PIXEL COUNT (luminance under-weights red: a full-red pixel has luminance only
#     0.21, so a luminance-mean is swamped by the brighter background).

using OmniverseMakie, ColorTypes, FixedPointNumbers, GeometryBasics
const OM = OmniverseMakie

OM.activate!(warmup = 64)

# --- metrics -----------------------------------------------------------------
lum(c) = 0.2126f0 * Float32(red(c)) + 0.7152f0 * Float32(green(c)) + 0.0722f0 * Float32(blue(c))
function stats(img)
    L   = sort([lum(c) for c in img if lum(c) > 0.02f0])
    isempty(L) && return (nonblack = 0, contrast = 0.0f0)
    q(p) = L[clamp(round(Int, p * length(L)), 1, length(L))]
    return (nonblack = length(L), contrast = q(0.95) / (q(0.50) + 1.0f-3))
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
# Count strongly RED-dominant pixels (the emissive curve): red high AND clearly above
# both other channels.  A plain (non-emissive) curve / the blue-ish background score ~0.
function redcount(img; thr = 0.35f0)
    n = 0
    for c in img
        r = Float32(red(c)); g = Float32(green(c)); b = Float32(blue(c))
        (r > thr && r > g + 0.12f0 && r > b + 0.12f0) && (n += 1)
    end
    return n
end

function render_scene(build!; setcam! = nothing)
    fig = Figure(); ax = LScene(fig[1, 1])
    build!(ax)
    setcam! === nothing || setcam!(ax)
    return Makie.colorbuffer(ax.scene; warmup = 64)
end

basecol = RGBf(0.72, 0.20, 0.20)

# === MeshScatter: metallic instances vs plain diffuse ========================
msmarkers = [Point3f(x, 0.0, 0.0) for x in -1.5:1.0:1.5]
imgMSd = render_scene(ax -> meshscatter!(ax, msmarkers; markersize = 0.45, color = basecol))
imgMSm = render_scene(ax -> meshscatter!(ax, msmarkers; markersize = 0.45, color = basecol,
                                         material = (; metallic = 1.0, roughness = 0.08)))
sMSd = stats(imgMSd); sMSm = stats(imgMSm)
madMS = meanabsdiff(imgMSd, imgMSm); crMS = sMSm.contrast / (sMSd.contrast + 1.0f-3)
println("MESHSCATTER_DIFFUSE nonblack=$(sMSd.nonblack) contrast=$(sMSd.contrast)")
println("MESHSCATTER_METALLIC nonblack=$(sMSm.nonblack) contrast=$(sMSm.contrast)")
println("MESHSCATTER_MEANABSDIFF=$(madMS)")
println("MESHSCATTER_CONTRAST_RATIO=$(crMS)")

# === Surface: materialized (metallic) vs plain ===============================
framecam!(ax) = Makie.update_cam!(ax.scene, Vec3f(9, 9, 9), Vec3f(0, 0, 0))
surf_xs = -3:0.3:3; surf_ys = -3:0.3:3
surf_f(x, y) = sin(x) * cos(y)
imgSd = render_scene(ax -> surface!(ax, surf_xs, surf_ys, surf_f); setcam! = framecam!)
imgSm = render_scene(ax -> surface!(ax, surf_xs, surf_ys, surf_f;
                                    material = (; metallic = 1.0, roughness = 0.2));
                     setcam! = framecam!)
sSd = stats(imgSd); sSm = stats(imgSm); madS = meanabsdiff(imgSd, imgSm)
println("SURFACE_PLAIN nonblack=$(sSd.nonblack) contrast=$(sSd.contrast)")
println("SURFACE_METALLIC nonblack=$(sSm.nonblack) contrast=$(sSm.contrast)")
println("SURFACE_MEANABSDIFF=$(madS)")

# === Lines: emissive red vs plain white ======================================
helix = [Point3f(cos(t), sin(t), t / 4 - 1.5) for t in range(0, 4pi, length = 80)]
imgLe = render_scene(ax -> lines!(ax, helix; linewidth = 6, material = (; emissive = (1, 0, 0))))
imgLw = render_scene(ax -> lines!(ax, helix; linewidth = 6, color = :white))
rcE = redcount(imgLe); rcW = redcount(imgLw)
println("LINES_EMISSIVE redpix=$(rcE) nonblack=$(stats(imgLe).nonblack)")
println("LINES_WHITE redpix=$(rcW)")

# --- assertions --------------------------------------------------------------
@assert sMSd.nonblack > 300 "diffuse meshscatter (near) black: $(sMSd.nonblack)"
@assert sMSm.nonblack > 300 "metallic meshscatter (near) black: $(sMSm.nonblack)"
@assert madMS > 0.02 "metallic meshscatter too similar to diffuse (mad=$(madMS)): bind did not take"
@assert crMS > 1.3 "metallic meshscatter no specular signature: contrast ratio=$(crMS)"

@assert sSd.nonblack > 2000 "plain surface (near) black: $(sSd.nonblack)"
@assert sSm.nonblack > 2000 "metallic surface (near) black: $(sSm.nonblack)"
@assert madS > 0.02 "metallic surface too similar to plain (mad=$(madS)): bind did not take"

# Lines emissive: MANY red-dominant pixels (the curve glows red) and FAR more than a
# plain white curve — the bool enable_emission + emissive_intensity fix renders emission.
@assert rcE > 500 "emissive lines not visibly red: redpix=$(rcE)"
@assert rcE > 20 * (rcW + 1) "emissive lines not decisively redder than plain: $(rcE) vs white $(rcW)"

println("OK_PRIMITIVES_MATERIAL")
