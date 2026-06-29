# Ported from references/RPRMakieNotes/scripts/materials_julia_room.jl (Lazaro Alonso).
# Julia "colour room": 5 brand-colour diffuse spheres, 5 emissive spheres, walls/floor/ceiling,
# emissive ceiling panel, two image-textured quads (Makie logo + portrait), and a Glass mirror.
# PointLight arg order swapped (original: position-first → OmniverseMakie: color-first).
# zoom_mult[] skipped (GLMakie-specific); upvector preserved via cam.upvector[].
using OmniverseMakie, GeometryBasics, Colors, FileIO

function SphereTess(o = Point3f(0), r = 1.0f0; tess = 64)
    return uv_normal_mesh(Tesselation(Sphere(o, r), tess))
end

function scene_materials_julia_room()
    # 1×1 white environment image (neutral dome — no HDR map needed here)
    white = [colorant"white" for _ in 1:1, _ in 1:1]
    # ★ PointLight: color-first, position-second (OmniverseMakie convention)
    lights = [EnvironmentLight(1.0, white'), PointLight(RGBf(5.0, 5.0, 5.0), Vec3f(0, -2, 1.99))]

    # Julia / Makie brand colour palette (index 1 = black; 2–6 = brand; 7 = grey)
    jlmkecs = [
        RGB(0.0,   0.0,   0.0),
        RGB(0.082, 0.643, 0.918),   # blue
        RGB(0.91,  0.122, 0.361),   # red/magenta
        RGB(0.929, 0.773, 0.0),     # yellow
        RGB(0.588, 0.196, 0.722),   # purple
        RGB(0.361, 0.722, 0.361),   # green
        RGB(0.522, 0.522, 0.522),   # grey
    ]

    # Pre-compute 2.5× emissive variants (HDR float colours — values > 1.0 are intentional)
    ec = [RGB(red(c) * 2.5, green(c) * 2.5, blue(c) * 2.5) for c in jlmkecs]

    # Textured assets — loaded once at scene-build time
    logoMakie = FileIO.load(asset("materials_julia_room", "makie_logo.png"))
    thandle   = FileIO.load(asset("materials_julia_room", "lazaro2.png"))

    # Room geometry
    plane    = Rect3f(Vec3f(-3, -9, -1.05), Vec3f(8, 17, 0.05))   # floor
    planetop = Rect3f(Vec3f(-3, -9,  4.0),  Vec3f(8, 17, 0.05))   # ceiling

    fig = Figure(; size = (1650, 950))
    ax  = LScene(fig[1, 1]; show_axis = false, scenekw = (; lights = lights))

    # ── 5 colour diffuse spheres (DiffuseMaterial → drop material=, keep color=) ────────────
    mesh!(ax, SphereTess(Point3f(0.0, -4.5,  0.0), 1.0f0); color = jlmkecs[2])
    mesh!(ax, SphereTess(Point3f(0.0, -2.25, 0.0), 1.0f0); color = jlmkecs[3])
    mesh!(ax, SphereTess(Point3f(0.0,  0.0,  0.0), 1.0f0); color = jlmkecs[4])
    mesh!(ax, SphereTess(Point3f(0.0,  2.25, 0.0), 1.0f0); color = jlmkecs[5])
    mesh!(ax, SphereTess(Point3f(0.0,  4.5,  0.0), 1.0f0); color = jlmkecs[6])

    # ── 5 emissive spheres (EmissiveMaterial(c) → material=(; emissive=c)) ─────────────────
    mesh!(ax, SphereTess(Point3f(0.75,  5.5, -1.0 + 0.15), 0.15f0);
          color = ec[6], material = (; emissive = ec[6]))
    mesh!(ax, SphereTess(Point3f(0.75,  3.0, -1.0 + 0.15), 0.15f0);
          color = ec[5], material = (; emissive = ec[5]))
    mesh!(ax, SphereTess(Point3f(0.75,  1.0, -1.0 + 0.15), 0.15f0);
          color = ec[4], material = (; emissive = ec[4]))
    mesh!(ax, SphereTess(Point3f(0.75, -1.5, -1.0 + 0.15), 0.15f0);
          color = ec[3], material = (; emissive = ec[3]))
    mesh!(ax, SphereTess(Point3f(0.75, -5.5, -1.0 + 0.15), 0.15f0);
          color = ec[2], material = (; emissive = ec[2]))

    # ── Walls / floor / ceiling (DiffuseMaterial → drop material=) ──────────────────────────
    mesh!(ax, plane;    color = :gainsboro)
    mesh!(ax, planetop; color = :gainsboro)
    mesh!(ax, Rect3f(Vec3f(-3, -9, -1.05), Vec3f(0.05, 17, 5.05)); color = :grey)

    # White diffuse rect at PointLight position (DiffuseMaterial → drop material=)
    mesh!(ax, Rect3f(Vec3f(-1, -3, 2.01), Vec3f(2, 2.0, 0.05)); color = :white)

    # Emissive ceiling panel (EmissiveMaterial → material=(; emissive=c))
    mesh!(ax, Rect3f(Vec3f(-1, 2, 2.0), Vec3f(2, 2.5, 0.05));
          color = RGBf(3, 3, 3), material = (; emissive = RGBf(3, 3, 3)))

    # ── Image-textured quads (color = img → backend auto-emits diffuse_texture + st) ────────
    # Makie logo quad — no transform needed
    mesh!(ax, Rect3f(Vec3f(2, 3, -1.0), Vec3f(1.2455, 0.5, 0.05));
          color = logoMakie)

    # Portrait quad — rotate then translate (backend honors plot.model)
    mlazaro = mesh!(ax, Rect3f(Vec3f(-0.5, -0.5, -1.0), Vec3f(0.2, 0.2, 0.05));
                    color = thandle)
    rotate!(mlazaro, Vec3f(1, 0, 0), π / 2)
    translate!(mlazaro, Vec3f(6.025, 4.15, 0))

    # ── Glass mirror (Glass → material=(; opacity, roughness, metallic)) ────────────────────
    mesh!(ax, Rect3f(Vec3f(-3, -7, -1.05), Vec3f(8, 0.05, 5.5));
          color = :white,
          material = (; opacity = 0.15f0, roughness = 0.0f0, metallic = 0.0f0))

    # ── Camera (eyeposition/lookat/upvector from original; zoom_mult[] is GLMakie-only → skip) ─
    cam = cameracontrols(ax.scene)
    cam.eyeposition[] = Vec3f(7.2984867f0,  9.654725f0,   0.6902726f0)
    cam.lookat[]      = Vec3f(-0.81408334f0, 0.84149307f0, 0.575948f0)
    cam.upvector[]    = Vec3f(0.24841104f0,  0.26986563f0, 0.9303031f0)
    update_cam!(ax.scene, cam)
    zoom!(ax.scene, cameracontrols(ax.scene), 1.1)

    return fig
end

function assert_materials_julia_room(img)
    assert_nonblack(img, "materials_julia_room"; frac = 0.05)
    total_color = color_fraction(img, :red) + color_fraction(img, :green) + color_fraction(img, :blue)
    @assert total_color > 0.005 "FAIL: materials_julia_room expected a colourful room (got $(total_color))"
    println("  color_fractions r+g+b=", total_color)
end
