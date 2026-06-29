# Ported from references/RPRMakieNotes/scripts/earth_ina_julia_box.jl (Lazaro Alonso).
# Earth-textured sphere inside a coloured Cornell box with 4 emissive corner lights.
# EmissiveMaterial corners → material=(; emissive=c); earth texture via circshift+color=img'.
using OmniverseMakie, GeometryBasics, Colors, FileIO

SphereTess(; o = Point3f(0), r = 1, tess = 64) = uv_normal_mesh(Tesselation(Sphere(o, r), tess))

function scene_earth_ina_julia_box()
    # 1×1 grey environment map (neutral dome)
    grey = [colorant"grey90" for _ in 1:1, _ in 1:1]
    # ★ PointLight fix: color-first, position-second (OmniverseMakie convention)
    lights = [EnvironmentLight(1.0, grey'), PointLight(RGBf(1.0, 1.0, 1.0), Vec3f(4, 0, 0.85))]

    earth_img = FileIO.load(asset("earth_ina_julia_box", "earth.jpg"))

    fig = Figure(; size = (950, 950))
    ax  = LScene(fig[1, 1]; show_axis = false, scenekw = (; lights = lights))

    # Box walls — DiffuseMaterial → color= only (no material= kwarg)
    mesh!(ax, Rect3(Vec3f(-1, -1, -1.1), Vec3f(2, 2, 0.1));
          color = RGB(0.082, 0.643, 0.918))                          # floor, blue
    mesh!(ax, Rect3(Vec3f(-1, -1.1, -1.1), Vec3f(2, 0.1, 2.2));
          color = RGB(0.929, 0.773, 0.0))                            # front wall, yellow
    mesh!(ax, Rect3(Vec3f(-1, 1.0, -1.1), Vec3f(2, 0.1, 2.2));
          color = RGB(0.588, 0.196, 0.722))                          # back wall, purple
    mesh!(ax, Rect3(Vec3f(-1, -1, 1.0), Vec3f(2, 2, 0.1));
          color = RGB(0.361, 0.722, 0.361))                          # ceiling, green
    mesh!(ax, Rect3(Vec3f(-1, -1, -1.1), Vec3f(0.1, 2, 2.2));
          color = RGB(0.522, 0.522, 0.522))                          # side wall, grey

    # Earth sphere — texture via circshift + color = img' (backend auto-emits diffuse_texture + st)
    mesh!(ax, SphereTess(; o = Point3f(0.5, 0, 0), r = 0.85);
          color = circshift(earth_img, (0, 3800))')

    # Emissive corner spheres — EmissiveMaterial → material=(; emissive=c)
    # HDR emissive colour (65× white); plain white for displayColor
    emissive_c = RGBf(65.0f0, 65.0f0, 65.0f0)
    mesh!(ax, SphereTess(; o = Point3f(0.9, -0.95, 0.95), r = 0.05);
          color = colorant"white", material = (; emissive = emissive_c))
    mesh!(ax, SphereTess(; o = Point3f(0.9, 0.95, 0.95), r = 0.05);
          color = colorant"white", material = (; emissive = emissive_c))
    mesh!(ax, SphereTess(; o = Point3f(-0.9, -0.95, -0.95), r = 0.05);
          color = colorant"white", material = (; emissive = emissive_c))
    mesh!(ax, SphereTess(; o = Point3f(-0.9, 0.95, -0.95), r = 0.05);
          color = colorant"white", material = (; emissive = emissive_c))

    # Camera: explicit eyeposition, lookat, fov per spec
    cam = cameracontrols(ax.scene)
    cam.eyeposition[] = Vec3f(9.5, 0.0, 0.5)
    cam.lookat[]      = Vec3f(0.0, 0.0, 0.0)
    cam.fov[]         = 12.0f0
    update_cam!(ax.scene, cam)

    return fig
end

function assert_earth_ina_julia_box(img)
    assert_nonblack(img, "earth_ina_julia_box")
    @assert color_fraction(img, :blue) > 0.005 "FAIL: earth_ina_julia_box expected earth texture"
    println("  blue_fraction=", color_fraction(img, :blue))
end
