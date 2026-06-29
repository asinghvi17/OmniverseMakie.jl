# Ported from references/RPRMakieNotes/scripts/sphere_plane_greysky.jl (Lazaro Alonso).
# Blue diffuse sphere on a gainsboro plane under a grey dome + warm point light.
# DiffuseMaterial → plain color= (USD displayColor matte).
using OmniverseMakie, GeometryBasics, Colors

SphereTess(; o = Point3f(0), r = 1, tess = 64) = uv_normal_mesh(Tesselation(Sphere(o, r), tess))

function scene_sphere_plane_greysky()
    grey   = [colorant"grey90" for _ in 1:1, _ in 1:1]
    lights = [EnvironmentLight(1.0, grey'), PointLight(RGBf(8.0, 6.0, 5.0), Vec3f(2, 0, 2.0))]
    plane  = Rect3f(Vec3f(-5, -2, -1.05), Vec3f(10, 4, 0.05))
    fig = Figure(; size = (900, 900))
    ax  = LScene(fig[1, 1]; show_axis = false, scenekw = (; lights = lights))
    mesh!(ax, SphereTess(); color = RGB(0.082, 0.643, 0.918))
    mesh!(ax, plane; color = :gainsboro)
    zoom!(ax.scene, cameracontrols(ax.scene), 0.22)
    return fig
end

function assert_sphere_plane_greysky(img)
    assert_nonblack(img, "sphere_plane_greysky"; frac = 0.05)
    bf = color_fraction(img, :blue)
    @assert bf > 0.01 "FAIL: sphere_plane_greysky expected a blue sphere; blue_fraction=$(bf)"
    println("  blue_fraction=", bf)
end
