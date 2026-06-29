# Ported from references/RPRMakieNotes/scripts/sphere_source_light.jl (Lazaro Alonso).
# Yellow diffuse sphere + three emissive sphere lights (white, red, dodgerblue) on a gainsboro plane.
# DiffuseMaterial → color= only; EmissiveMaterial → material=(; emissive=c).
using OmniverseMakie, GeometryBasics, Colors

SphereTess(; o = Point3f(0), r = 1, tess = 64) = uv_normal_mesh(Tesselation(Sphere(o, r), tess))

function scene_sphere_source_light()
    grey   = [colorant"grey20" for _ in 1:1, _ in 1:1]
    lights = [EnvironmentLight(1.0, grey'), PointLight(RGBf(8.0, 6.0, 5.0), Vec3f(2.25, 0, 0.5))]
    plane  = Rect3f(Vec3f(-5, -2, -1.05), Vec3f(10, 4, 0.05))
    fig = Figure(; size = (900, 900))
    ax  = LScene(fig[1, 1]; show_axis = false, scenekw = (; lights = lights))
    # diffuse spheres / plane — DiffuseMaterial dropped, color= retained
    mesh!(ax, SphereTess(); color = RGB(0.929, 0.773, 0.0))
    mesh!(ax, plane; color = :gainsboro)
    # emissive source spheres — EmissiveMaterial → material=(; emissive=c)
    mesh!(ax, SphereTess(; o = Point3f(0, 0, 2), r = 0.2);
          material = (; emissive = RGBf(1, 1, 1)))
    mesh!(ax, SphereTess(; o = Point3f(0, 2, 0.5), r = 0.1);
          material = (; emissive = RGBf(1, 0, 0)))
    mesh!(ax, SphereTess(; o = Point3f(-3, -0.75, -1), r = 0.1);
          material = (; emissive = RGBf(colorant"dodgerblue")))
    zoom!(ax.scene, cameracontrols(ax.scene), 0.22)
    return fig
end

function assert_sphere_source_light(img)
    assert_nonblack(img, "sphere_source_light")
    @assert color_fraction(img, :red) > 0.003 || color_fraction(img, :blue) > 0.003 "FAIL: sphere_source_light expected coloured emissive sources"
end
