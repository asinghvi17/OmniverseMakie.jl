# Ported from references/RPRMakieNotes/scripts/twoEarths.jl (Lazaro Alonso).
# Two earth-textured spheres side by side in a single LScene;
# color=earth_img auto-emits diffuse_texture. Original was pure GLMakie (no
# RPR materials, no lights) — AmbientLight + PointLight added for the RTX
# path tracer.
# PointLight arg order: color-first (OmniverseMakie convention),
# position-second.
using OmniverseMakie, GeometryBasics, Colors, FileIO

SphereTess(; o = Point3f(0), r = 1, tess = 64) = uv_normal_mesh(Tesselation(Sphere(o, r), tess))

function scene_twoEarths()
    earth_img = FileIO.load(asset("twoEarths", "earth.jpg"))

    # PointLight: color-first (OmniverseMakie convention), position-second
    lights = [
        AmbientLight(RGBf(0.6, 0.6, 0.6)),
        PointLight(RGBf(8, 8, 8), Vec3f(0, 0, 5)),
    ]

    fig = Figure(; size = (1600, 1200))
    ax  = LScene(fig[1, 1]; show_axis = false, scenekw = (; lights = lights))

    # Two earth spheres side by side; color=earth_img → auto
    # diffuse_texture in RTX backend
    mesh!(ax, SphereTess(; o = Point3f(-1.2, 0, 0)); color = earth_img)
    mesh!(ax, SphereTess(; o = Point3f( 1.2, 0, 0)); color = earth_img)

    # Camera: sit at y=8, look at origin, Z-up — frames both spheres side
    # by side along X
    update_cam!(ax.scene, Vec3f(0, 8, 0), Vec3f(0, 0, 0), Vec3f(0, 0, 1))
    zoom!(ax.scene, cameracontrols(ax.scene), 0.45)

    return fig
end

function assert_twoEarths(img)
    assert_nonblack(img, "twoEarths")
    bf = color_fraction(img, :blue)
    @assert bf > 0.015 "FAIL: twoEarths expected earth textures (blue ocean presence); blue_fraction=$(bf)"
    println("  blue_fraction=", bf)
end
