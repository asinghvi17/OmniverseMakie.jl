# Ported from references/RPRMakieNotes/scripts/transparentM.jl (Lazaro Alonso).
# Nested transparent spheres over a gainsboro plane: outer shell (opacity=0.2) + inner sphere (opacity=0.95).
# UberMaterial transparency mapped as opacity = 1 - transparency. DiffuseMaterial plane dropped to plain color=.
using OmniverseMakie, GeometryBasics, Colors

SphereTess(; o = Point3f(0), r = 1, tess = 64) = uv_normal_mesh(Tesselation(Sphere(o, r), tess))

function scene_transparentM()
    grey   = [colorant"grey90" for _ in 1:1, _ in 1:1]
    lights = [EnvironmentLight(1.0, grey'), PointLight(RGBf(8.0, 6.0, 5.0), Vec3f(2, 0, 2.0))]
    plane  = Rect3f(Vec3f(-5, -2, -1.05), Vec3f(10, 4, 0.05))

    fig = Figure(; size = (900, 900))
    ax  = LScene(fig[1, 1]; show_axis = false, scenekw = (; lights = lights))

    # Outer sphere: UberMaterial(transparency=0.8) → opacity = 1 - 0.8 = 0.2
    mesh!(ax, SphereTess();
        color    = RGB(0.082, 0.643, 0.918),
        material = (; opacity = 0.2f0, roughness = 0.1f0))

    # Inner sphere: UberMaterial(transparency=0.05) → opacity = 1 - 0.05 = 0.95
    mesh!(ax, SphereTess(; o = Point3f(0), r = 0.5);
        color    = RGB(0.91, 0.122, 0.361),
        material = (; opacity = 0.95f0))

    # Plane: DiffuseMaterial → drop material=, keep color=
    mesh!(ax, plane; color = :gainsboro)

    zoom!(ax.scene, cameracontrols(ax.scene), 0.22)
    cam = cameracontrols(ax.scene)
    cam.eyeposition[] = Vec3f(0.73402506, 13.08092, 2.8793263)
    cam.lookat[]      = Vec3f(0.0, 0.0, 0.0)
    update_cam!(ax.scene, cam)

    return fig
end

function assert_transparentM(img)
    assert_nonblack(img, "transparentM"; frac = 0.03)
    @assert color_fraction(img, :blue) > 0.01 "FAIL: transparentM expected the blue transparent sphere; blue_fraction=$(color_fraction(img, :blue))"
    println("  blue_fraction=", color_fraction(img, :blue))
end
