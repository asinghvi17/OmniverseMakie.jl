# Ported from references/RPRMakieNotes/scripts/earthquakes.jl (Lazaro Alonso).
# 3-D globe of earthquake points (meshscatter, colormap=:nuuk, sized by magnitude)
# + a glass box frame (3 × mesh! with opacity material).
# Earth jpg was loaded-but-unused in the original → omitted here.
# PointLight arg order swapped: color-first (OmniverseMakie convention), position-second.
using OmniverseMakie, GeometryBasics, Colors, CSV, DataFrames

## depth unit: km; projects (lon,lat,depth_km) onto unit sphere surface
function toCartesian(lon, lat; r = 1.02, cxyz = (0, 0, 0))
    x = cxyz[1] + (r + 1_500_000) * cosd(lat) * cosd(lon)
    y = cxyz[2] + (r + 1_500_000) * cosd(lat) * sind(lon)
    z = cxyz[3] + (r + 1_500_000) * sind(lat)
    return (x, y, z) ./ 1_500_000
end

function scene_earthquakes()
    # Load earthquake CSVs (two half-year chunks, 2021-01 → 2022-01)
    earthquakes1 = DataFrame(CSV.File(asset("earthquakes", "2021_01_2021_05.csv")))
    earthquakes2 = DataFrame(CSV.File(asset("earthquakes", "2021_06_2022_01.csv")))
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

    # 1×1 dark env image — a subtle dome fill so the globe of points reads against black
    # (the original's EXR is not honored by our backend; the warm PointLight is the key).
    img = [colorant"grey15" for _ in 1:1, _ in 1:1]

    # ★ PointLight: color-first (OmniverseMakie convention), position-second
    lights = [
        EnvironmentLight(1.0, img'[end:-1:1, :]),
        PointLight(RGBf(65.0, 50.0, 35.0), Vec3f(1, 0.25, 0.0)),
    ]

    fig = Figure(; size = (1080, 1080))
    ax  = LScene(fig[1, 1]; show_axis = false, scenekw = (; lights = lights))

    # Earthquake scatter: numeric color → :nuuk colormap
    meshscatter!(ax, toPoints3D;
        markersize = ms ./ 12 .+ 0.006,
        color      = mag,
        colormap   = :nuuk)

    # NOTE: the original's glass box frame is dropped here — our OmniPBR `opacity` glass
    # renders as a bright opaque panel (no true refraction) that washes out the globe of
    # points. The earthquake globe is the subject; it reads cleanly against the dark dome.

    # Frame the globe + glass frame from OUTSIDE the box (the default LScene camera,
    # without a display(), sits inside the box → a white void).
    update_cam!(ax.scene, Vec3f(3.4, 3.4, 2.2), Vec3f(0, 0, 0), Vec3f(0, 0, 1))

    return fig
end

function assert_earthquakes(img)
    # A sparse globe of earthquake dots — expect at least 1 % non-black pixels
    assert_nonblack(img, "earthquakes"; frac = 0.01)
end
