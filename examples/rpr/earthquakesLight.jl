# Ported from references/RPRMakieNotes/scripts/earthquakesLight.jl (Lazaro
# Alonso). Earthquake globe (meshscatter + :nuuk colormap) + earth-textured
# sphere + emissive indicator boxes. PointLight arg order swapped: original
# was position-first; OmniverseMakie wants color-first. Earth sphere
# activated via SphereTess (was commented-out surface! in original).
# EmissiveMaterial boxes → material=(; emissive=c). DiffuseMaterial planes
# → drop material=.
using OmniverseMakie, GeometryBasics, Colors, CSV, DataFrames, FileIO

SphereTess(; o = Point3f(0), r = 1, tess = 64) = uv_normal_mesh(Tesselation(Sphere(o, r), tess))

## depth unit: km; projects (lon,lat,depth_km) onto unit sphere surface
function toCartesian(lon, lat; r = 1.02, cxyz = (0, 0, 0))
    x = cxyz[1] + (r + 1_500_000) * cosd(lat) * cosd(lon)
    y = cxyz[2] + (r + 1_500_000) * cosd(lat) * sind(lon)
    z = cxyz[3] + (r + 1_500_000) * sind(lat)
    return (x, y, z) ./ 1_500_000
end

function scene_earthquakesLight()
    # Load earthquake CSVs (two half-year chunks, 2021-01 → 2022-01)
    earthquakes1 = DataFrame(CSV.File(asset("earthquakesLight", "2021_01_2021_05.csv")))
    earthquakes2 = DataFrame(CSV.File(asset("earthquakesLight", "2021_06_2022_01.csv")))
    earthquakes  = vcat(earthquakes1, earthquakes2)

    lons  = earthquakes.longitude
    lats  = earthquakes.latitude
    depth = earthquakes.depth
    mag   = earthquakes.mag

    # Project quake locations onto (slightly indented) unit sphere
    toPoints3D = [Point3f([toCartesian(lons[i], lats[i]; r = -depth[i] * 1000)...])
                  for i in 1:length(lons)]

    # Magnitude → normalised marker size (same formula as original)
    ms = (exp.(mag) .- minimum(exp.(mag))) ./ maximum(exp.(mag) .- minimum(exp.(mag)))

    # Earth texture
    earth_img = FileIO.load(asset("earthquakesLight", "earth.jpg"))

    # 1×1 grey10 env image (background/sky colour as in original)
    grey10 = [colorant"grey10" for _ in 1:1, _ in 1:1]

    # PointLight: color-first (OmniverseMakie convention), position-second
    lights = [
        EnvironmentLight(1.0, grey10'),
        PointLight(RGBf(8.0, 6.0, 5.0), Vec3f(2, 2, 0.0)),
    ]

    plane = Rect3f(Vec3f(-5, -2, -1.05), Vec3f(10, 4, 0.05))

    fig = Figure(; size = (1080, 1080))
    ax  = LScene(fig[1, 1]; show_axis = false, scenekw = (; lights = lights))

    # Earthquake scatter: numeric colour → :nuuk colormap, sized by
    # normalised magnitude
    meshscatter!(ax, toPoints3D;
        markersize = ms ./ 14 .+ 0.004,
        color      = mag,
        colormap   = :nuuk)

    # Earth sphere: color=earth_img → auto diffuse_texture in RTX backend
    mesh!(ax, SphereTess(); color = earth_img)

    # Room planes: DiffuseMaterial → drop material= (USD displayColor matte)
    mesh!(ax, plane; color = :gainsboro)
    mesh!(ax, Rect3f(Vec3f(-5, -2, 2.05), Vec3f(10, 4, 0.05)); color = :white)
    mesh!(ax, Rect3f(Vec3f(-5, -2, -1.05), Vec3f(10, 0.05, 2.15)); color = :gainsboro)

    # Emissive indicator boxes: RPR.EmissiveMaterial(matsys) →
    # material=(; emissive=c)
    mesh!(ax, Rect3f(Vec3f(-3, -1, -0.5), Vec3f(0.5, 0.5, 0.5));
        material = (; emissive = 10colorant"orange"))
    mesh!(ax, Rect3f(Vec3f(3, -1, -0.5), Vec3f(0.5, 0.5, 0.5));
        material = (; emissive = 10colorant"red"))

    # Camera: pulled back from the original's eyeposition=(2,2,1.5) (too
    # close → the floor plane occluded the globe) to frame the earth +
    # indicator lights from outside the room.
    update_cam!(ax.scene, Vec3f(4.0, 4.0, 2.6), Vec3f(0, 0, 0.2), Vec3f(0, 0, 1))

    return fig
end

function assert_earthquakesLight(img)
    assert_nonblack(img, "earthquakesLight")
    rf = color_fraction(img, :red)
    @assert rf > 0.0012 "FAIL: earthquakesLight expected emissive indicators; red_fraction=$(rf)"
    println("  red_fraction=", rf)
end
