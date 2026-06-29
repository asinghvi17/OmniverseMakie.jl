# Ported from references/RPRMakieNotes/scripts/transparentMaterial.jl (Lazaro Alonso).
# Earth-textured transparent sphere (opacity=0.75) over a white inner sphere + gainsboro plane.
# UberMaterial(transparency=Vec4f(0.25)) → material=(; opacity=0.75f0); earth texture via color=earth_img'.
using OmniverseMakie, GeometryBasics, Colors, FileIO

SphereTess(; o = Point3f(0), r = 1, tess = 64) = uv_normal_mesh(Tesselation(Sphere(o, r), tess))

function scene_transparentMaterial()
    # 1×1 white env map (neutral dome)
    white = [colorant"white" for _ in 1:1, _ in 1:1]
    # ★ PointLight fix: color first, then position (OmniverseMakie API)
    lights = [EnvironmentLight(1.0, white'), PointLight(RGBf(5.0, 5.0, 5.0), Vec3f(2, 2, 3))]

    plane     = Rect3f(Vec3f(-5, -2, -1.05), Vec3f(10, 4, 0.05))
    earth_img = FileIO.load(asset("transparentMaterial", "earth.jpg"))

    fig = Figure(; size = (900, 900))
    ax  = LScene(fig[1, 1]; show_axis = false, scenekw = (; lights = lights))

    # Inner white sphere — DiffuseMaterial → plain color= (no material= kwarg)
    mesh!(ax, SphereTess(; o = Point3f(0), r = 0.75); color = 0.85colorant"white")

    # Outer earth sphere — UberMaterial(transparency=Vec4f(0.25)) → opacity=0.75; texture via color=img'
    mesh!(ax, SphereTess(); color = earth_img', material = (; opacity = 0.75f0))

    # Ground plane — DiffuseMaterial → plain color= (no material= kwarg)
    mesh!(ax, plane; color = :gainsboro)

    zoom!(ax.scene, cameracontrols(ax.scene), 0.22)
    return fig
end

function assert_transparentMaterial(img)
    assert_nonblack(img, "transparentMaterial")
    @assert color_fraction(img, :blue) > 0.01 "FAIL: transparentMaterial expected earth texture"
    println("  blue_fraction=", color_fraction(img, :blue))
end
