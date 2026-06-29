# Ported from references/RPRMakieNotes/scripts/submarineCables.jl (Lazaro Alonso).
# Textured earth globe with submarine-cable landing points (meshscatter, :plasma) + cable
# routes (lines!, cycled DEFAULT_PALETTES) + ground plane.
# DiffuseMaterial on ground plane → dropped (color=:gainsboro only, USD displayColor matte).
# PointLight arg order: color-first (OmniverseMakie convention), position-second.
# GeoInterface.features → GeoInterface.getfeature (modern API, returns generator).
using OmniverseMakie, GeometryBasics, Colors, FileIO, GeoMakie, GeoInterface, GeoJSON

"""
    toCartesian(lon, lat; r, cxyz) -> (x, y, z)

Project geographic coordinates (degrees) onto a sphere of radius `r` centred at `cxyz`.
"""
function toCartesian(lon, lat; r = 1.02, cxyz = (0, 0, 0))
    x = cxyz[1] + r * cosd(lat) * cosd(lon)
    y = cxyz[2] + r * cosd(lat) * sind(lon)
    z = cxyz[3] + r * sind(lat)
    return (x, y, z)
end

function scene_submarineCables()
    # ── GeoJSON (local assets — no Downloads at render time) ──────────────────
    landPoints = GeoJSON.read(read(asset("submarineCables", "landing-point-geo.json"), String))
    landCables = GeoJSON.read(read(asset("submarineCables", "cable-geo.json"), String))

    # Landing points → Vector{Point2}
    toPoints = GeoMakie.geo2basic(landPoints)

    # Cable routes → collect features, then extract MultiLineString coordinate arrays.
    # GeoInterface 1.x uses getfeature (no-index form returns a generator); GeoJSON
    # FeatureCollection is also directly iterable.
    feat = GeoInterface.getfeature(landCables)          # lazy generator over features
    toLines = [GeoInterface.coordinates(GeoInterface.geometry(f))
               for f in feat
               if !isnothing(GeoInterface.geometry(f))]  # skip null-geometry features

    # ── Sphere surface coordinates (lon/lat parametric) ───────────────────────
    n  = 1024 ÷ 4
    θ  = LinRange(0, pi, n)
    φ  = LinRange(-pi, pi, 2 * n)
    xe = [cos(φ) * sin(θ) for θ in θ, φ in φ]
    ye = [sin(φ) * sin(θ) for θ in θ, φ in φ]
    ze = [cos(θ)           for θ in θ, φ in φ]

    # ── Convert landing points to 3-D Cartesian ────────────────────────────────
    toPoints3D = [Point3f([toCartesian(point[1], point[2])...]) for point in toPoints]

    # ── Convert cable MultiLineStrings to 3-D line-segment arrays ─────────────
    # toLines[i]    = Vector{Vector{NTuple{2,Float64}}} (MultiLineString coords)
    # toLines[i][j] = Vector{NTuple{2,Float64}}         (one LineString segment)
    # toLines[i][j][k] = NTuple{2,Float64}              = (lon, lat)
    splitLines3D = Vector{Vector{Point3f}}()
    for i in eachindex(toLines)
        isnothing(toLines[i]) && continue                # skip null coordinates
        for j in eachindex(toLines[i])
            ptsLines = toLines[i][j]
            tmp3D    = Vector{Point3f}()
            for k in eachindex(ptsLines)
                x, y = ptsLines[k]
                xi, yi, zi = toCartesian(x, y)
                push!(tmp3D, Point3f(xi, yi, zi))
            end
            isempty(tmp3D) || push!(splitLines3D, tmp3D)
        end
    end

    # ── Earth texture ─────────────────────────────────────────────────────────
    earth_img = FileIO.load(asset("submarineCables", "earth.jpg"))

    # ── Lights ────────────────────────────────────────────────────────────────
    # EnvironmentLight: 1×1 grey90 swatch (neutral dome; full EXR not needed here).
    img    = [colorant"grey90" for _ in 1:1, _ in 1:1]
    # ★ PointLight: color-first (OmniverseMakie convention), position-second
    lights = [EnvironmentLight(1.0, img'), PointLight(RGBf(8.0, 6.0, 5.0), Vec3f(2, 0, 2.0))]

    plane  = Rect3f(Vec3f(-5, -2, -1.05), Vec3f(10, 4, 0.05))

    # ── Figure + LScene ───────────────────────────────────────────────────────
    fig = Figure(; size = (1000, 1000))
    ax  = LScene(fig[1, 1]; show_axis = false, scenekw = (; lights = lights))

    # Earth sphere via surface! — color=earth_img → an OmniPBR diffuse_texture sampled over the
    # grid's parametric st UVs (textured globe), with the coloured cable routes overlaid.
    surface!(ax, xe, ye, ze; color = earth_img)

    # Landing points: colored by index with :plasma colormap
    meshscatter!(ax, toPoints3D;
        color      = 1:length(toPoints3D),
        colormap   = :plasma,
        markersize = 0.005)

    # Cable routes: one lines! call per segment, colours cycled from the default palette
    colors = Makie.DEFAULT_PALETTES.color[]
    c      = Iterators.cycle(colors)
    foreach(((l, col),) -> lines!(ax, l; linewidth = 2, color = col), zip(splitLines3D, c))

    # Ground plane: RPR.DiffuseMaterial → dropped (plain color= / USD displayColor matte)
    mesh!(ax, plane; color = :gainsboro)

    # Camera: eyeposition=Vec3f(1.5) per original (all 3 components = 1.5)
    cam = cameracontrols(ax.scene)
    cam.eyeposition[] = Vec3f(1.5)
    update_cam!(ax.scene, cam)

    return fig
end

function assert_submarineCables(img)
    assert_nonblack(img, "submarineCables"; frac = 0.03)
    bf = color_fraction(img, :blue)
    @assert bf > 0.004 "FAIL: submarineCables expected textured globe; blue_fraction=$(bf)"
    println("  blue_fraction=", bf)
end
