# Ported from references/RPRMakieNotes/scripts/helix.jl (Lazaro Alonso).
# DNA double-helix: two strands of coloured spheres (A/T/C/G base-pairs),
# connecting two-tone rods, letter-glyph legend meshes (Luxor), and an
# image-textured portrait quad. PointLight arg order swapped: original was
# position-first; OmniverseMakie wants color-first.
# GLMakie.rotate!/RPRMakie.rotate! → Makie.rotate! (re-exported via
# OmniverseMakie). DiffuseMaterial → drop material=;
# UberMaterial(metalness, roughness) → material=(; metallic, roughness).
using OmniverseMakie, GeometryBasics, Colors, FileIO
using Random

include(joinpath(@__DIR__, "_pointsfont.jl"))

function SphereTess(; o=Point3f(0), r=1, tess=64)
    return uv_normal_mesh(Tesselation(Sphere(o, r), tess))
end

function scene_helix()
    Random.seed!(123)

    thandle = FileIO.load(asset("helix", "lazaro2.png"))

    # ── Letter glyph meshes for A, T, C, G ─────────────────────────────
    letterMesh = []
    letters = ["A", "T", "C", "G"]
    for l in letters
        mletter     = 0.65pointsfont(l)
        top_poly    = [Point3f(2.5 .+ p[1]/30, p[2]/30, 0.15) for p in mletter if !isnan(p[1])]
        bottom_poly = [Point3f(2.5 .+ p[1]/30, p[2]/30, 0.01) for p in mletter if !isnan(p[1])]
        push!(letterMesh, getMesh(top_poly, bottom_poly))
    end

    # ── Helix geometry ─────────────────────────────────────────────────
    npairs, θinit = 40, 10*π/180
    z  = 1:npairs
    θ  = z .* θinit
    xr(θ; r=5) = r * cos(θ)
    yr(θ; r=5) = r * sin(θ)
    x1, y1 = xr.(θ), yr.(θ)
    x2, y2 = xr.(θ .+ π), yr.(θ .+ π)
    colors1 = rand(1:4, npairs)   # random base-pair assignment (seeded above)
    # complementary pairs
    colors2 = [i == 1 ? 2 : i == 2 ? 1 : i == 3 ? 4 : 3 for i in colors1]
    # Julia / Makie brand colours: blue, red, yellow, purple
    jlmkecs = [
        RGB(0.082, 0.643, 0.918),
        RGB(0.91,  0.122, 0.361),
        RGB(0.929, 0.773, 0.0),
        RGB(0.588, 0.196, 0.722),
    ]
    lseg   = Point3f[]
    colors = Int64[]
    for i in 1:npairs
        push!(lseg, Point3f(x1[i], z[i], y1[i]))
        push!(lseg, Point3f(x2[i], z[i], y2[i]))
        push!(colors, colors1[i])
        push!(colors, colors2[i])
    end

    # ── Lights ─────────────────────────────────────────────────────────
    # 1×1 grey90 env image (neutral dome — no HDR map needed here)
    bg  = [colorant"grey90" for _ in 1:1, _ in 1:1]
    # PointLight: color-first (OmniverseMakie convention), position-second
    lights = [
        EnvironmentLight(1, bg'),
        PointLight(RGBf(100.0, 100.0, 100.0), Vec3f(0, 42, 5.0)),
    ]

    # ── Scene geometry ─────────────────────────────────────────────────
    plane  = Rect3f(Vec3f(-20, -10, -5.5),  Vec3f(60, 55, 0.05))   # floor
    # emissive back panel
    planeL = Rect3f(Vec3f(-10,  -2, 17.5),  Vec3f(20, 30, 0.05))

    fig = Figure(; size=(1600, 800))
    ax  = LScene(fig[1, 1]; show_axis=false, scenekw=(; lights=lights))

    # Floor: DiffuseMaterial → drop material=
    mesh!(ax, plane; color=:gainsboro)

    # Emissive back panel: EmissiveMaterial(2.5 white) → material=(; emissive=c)
    mesh!(ax, planeL;
          color    = RGBf(2.5, 2.5, 2.5),
          material = (; emissive = RGBf(2.5, 2.5, 2.5)))

    # ── Helix strand 1 spheres ─────────────────────────────────────────
    # UberMaterial(reflection_metalness=0.0, reflection_roughness=0.1)
    #   → material=(; metallic=0.0f0, roughness=0.1f0)
    for i in 1:40
        meshscatter!(ax, Point3f(x1[i], z[i], y1[i]);
            color     = jlmkecs[colors1[i]],
            markersize = 0.35,
            material   = (; metallic = 0.0f0, roughness = 0.1f0))
    end

    # ── Helix strand 2 spheres ─────────────────────────────────────────
    for i in 1:40
        meshscatter!(ax, Point3f(x2[i], z[i], y2[i]);
            color     = jlmkecs[colors2[i]],
            markersize = 0.35,
            material   = (; metallic = 0.0f0, roughness = 0.1f0))
    end

    # ── Connecting rods (two-tone lines per base-pair) ─────────────────
    for i in 1:2:80
        mid = (lseg[i] .+ lseg[i+1]) / 2
        lines!(ax, [lseg[i],   mid], linewidth=20, color=jlmkecs[colors[i]])
        lines!(ax, [mid, lseg[i+1]], linewidth=20, color=jlmkecs[colors[i+1]])
    end

    # ── Legend: 4 coloured spheres down the left side ──────────────────
    posk = tuple.(-8, 3:10:42, -4.85)
    for i in 1:4
        meshscatter!(ax, [Point3f(posk[i]...)];
            color     = jlmkecs[i],
            markersize = 0.5,
            material   = (; metallic = 0.0f0, roughness = 0.1f0))
    end

    # ── Letter glyph meshes (A/T/C/G) — DiffuseMaterial → drop material= ──
    for i in 1:4
        letterx = mesh!(ax, letterMesh[i]; color=jlmkecs[i])
        rotate!(letterx, Vec3f(0, 0, 1), -π/2)
        translate!(letterx, Vec3f(-10, 7 + (i-1)*10, -5.0))
    end

    # ── Portrait quad (DiffuseMaterial → drop material=, color=img) ────
    mlazaro = mesh!(ax, Rect3f(Vec3f(-0.5, -0.5, -1.0), Vec3f(1.5, 2, 0.25));
                    color=thandle)
    rotate!(mlazaro, Vec3f(0, 0, 1), -π/2)
    translate!(mlazaro, Vec3f(-11, 40.5, -4.5))

    # ── Camera ─────────────────────────────────────────────────────────
    zoom!(ax.scene, cameracontrols(ax.scene), 0.36)
    cam = cameracontrols(ax.scene)
    cam.eyeposition[] = Vec3f(-70.38632, 46.678234, 45.849705)
    cam.lookat[]      = Vec3f(-0.80685806, 20.650236, -1.1407485)
    update_cam!(ax.scene, cam)

    return fig
end

function assert_helix(img)
    assert_nonblack(img, "helix"; frac=0.03)
end
