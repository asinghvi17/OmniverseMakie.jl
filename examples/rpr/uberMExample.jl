# Ported from references/RPRMakieNotes/scripts/uberMExample.jl (Lazaro Alonso).
# Earth-textured sphere (UberMaterial metallic=0/roughness=0.1) on a white plane under grey dome.
# RPR.UberMaterial → material=(; metallic, roughness); color=earth_img auto-emits diffuse_texture.
# PointLight arg order swapped: original was position-first; OmniverseMakie wants color-first.
using OmniverseMakie, GeometryBasics, Colors, FileIO

SphereTess(; o = Point3f(0), r = 1, tess = 64) = uv_normal_mesh(Tesselation(Sphere(o, r), tess))

function scene_uberMExample()
    grey98   = [colorant"grey98" for _ in 1:1, _ in 1:1]
    # ★ PointLight: color-first (OmniverseMakie convention), position-second
    lights   = [EnvironmentLight(1.0, grey98'), PointLight(RGBf(15, 15, 15), Vec3f(3, 3, 3.0))]
    plane    = Rect3f(Vec3f(-5, -2, -1.05), Vec3f(10, 4, 0.05))
    earth_img = FileIO.load(asset("uberMExample", "earth.jpg"))

    fig = Figure(; size = (900, 900))
    ax  = LScene(fig[1, 1]; show_axis = false, scenekw = (; lights = lights))

    # Earth sphere: UberMaterial(reflection_metalness=0.0, reflection_roughness=0.1) →
    # material=(; metallic=0.0f0, roughness=0.1f0); color=earth_img → auto diffuse_texture
    mesh!(ax, SphereTess(); color = earth_img, material = (; metallic = 0.0f0, roughness = 0.1f0))

    # Plane: DiffuseMaterial → drop material= entirely (USD displayColor matte)
    mesh!(ax, plane; color = :white)

    zoom!(ax.scene, cameracontrols(ax.scene), 0.22)
    return fig
end

function assert_uberMExample(img)
    assert_nonblack(img, "uberMExample")
    bf = color_fraction(img, :blue)
    @assert bf > 0.008 "FAIL: uberMExample expected earth texture (blue ocean presence); blue_fraction=$(bf)"
    println("  blue_fraction=", bf)
end
