# Ported from references/RPRMakieNotes/scripts/reflections_glass_material.jl
# (Lazaro Alonso). Room box with a glass sphere (Cornell-box style).
# RPR.Glass → true OmniGlass (refractive).
# Env-light images ARE honored (scene EnvironmentLight images are authored —
# src/screen.jl `_author_env_light!`); this port uses a neutral 1×1 grey60
# dome instead of the original ./lights/envLightImage.exr — swap in the
# asset EXR for true IBL.
using OmniverseMakie, GeometryBasics, Colors

SphereTess(; o = Point3f(0), r = 1, tess = 64) = uv_normal_mesh(Tesselation(Sphere(o, r), tess))

function scene_reflections_glass_material()
    grey   = [colorant"grey60" for _ in 1:1, _ in 1:1]
    lights = [EnvironmentLight(1.0, grey'),
              PointLight(RGBf(8.0, 6.0, 5.0), Vec3f(0, 0, 1.0))]

    fig = Figure(; size = (950, 950))
    ax  = LScene(fig[1, 1]; show_axis = false, scenekw = (; lights = lights))

    # room box (diffuse walls — drop material=, keep color=)
    mesh!(ax, Rect3(Vec3f(-1, -1, -1.1), Vec3f(2, 2, 0.1)); color = :white)
    mesh!(ax, Rect3(Vec3f(-1, -1.1, -1.1), Vec3f(2, 0.1, 2.2));
          color = RGB(0.929, 0.773, 0.0))
    mesh!(ax, Rect3(Vec3f(-1, 1.0, -1.1), Vec3f(2, 0.1, 2.2));
          color = RGB(0.588, 0.196, 0.722))
    mesh!(ax, Rect3(Vec3f(-1, -1, 1.0), Vec3f(2, 2, 0.1));
          color = RGB(0.361, 0.722, 0.361))
    mesh!(ax, Rect3(Vec3f(-1, -1, -1.1), Vec3f(0.1, 2, 2.2));
          color = RGB(0.522, 0.522, 0.522))

    # glass sphere — RPR.Glass → TRUE refractive OmniGlass
    # (material=(; glass=true, ior))
    mesh!(ax, SphereTess(; o = Point3f(0.5, 0, 0), r = 0.5);
          material = (; glass = true, ior = 1.5f0))

    cam = cameracontrols(ax.scene)
    cam.eyeposition[] = Vec3f(10.0, 0.0, 0.5)
    cam.lookat[]      = Vec3f(0.0, 0.0, 0.0)
    cam.fov[]         = 12.0f0
    update_cam!(ax.scene, cam)

    return fig
end

function assert_reflections_glass_material(img)
    assert_nonblack(img, "reflections_glass_material"; frac = 0.05)
end
