# Textured road-strip prototype: a slightly bumpy road authored from X/Y/Z
# matrices, with a Tron-like grid baked into a road UV texture.
#
# Run:
#   julia --project=examples examples/tron_textured_road_strip.jl
#
# Env overrides:
#   TRON_ROAD_OUT, TRON_ROAD_NS, TRON_ROAD_NT

using OmniverseMakie, ColorTypes, FixedPointNumbers, GeometryBasics
using FileIO, ImageIO
using LinearAlgebra: cross, norm

const OM = OmniverseMakie
using Makie: Scene, Vec2f, Vec3f, Point3f, RGBf, AbstractLight,
             AmbientLight, DirectionalLight, EnvironmentLight, PointLight,
             RectLight, cam3d!, update_cam!

const OUT = get(ENV, "TRON_ROAD_OUT",
                normpath(joinpath(@__DIR__, "..", "recordings", "tron_textured_road_strip.png")))
const ROAD_NS = parse(Int, get(ENV, "TRON_ROAD_NS", "192"))
const ROAD_NT = parse(Int, get(ENV, "TRON_ROAD_NT", "30"))

const ROAD_MATERIAL = (; metallic = 0.34f0, roughness = 0.014f0)

function road_strip_matrices(; ns = ROAD_NS, nt = ROAD_NT,
                             length = 4200f0, width = 620f0)
    ns >= 2 || error("TRON_ROAD_NS must be at least 2.")
    nt >= 2 || error("TRON_ROAD_NT must be at least 2.")

    X = Matrix{Float32}(undef, ns, nt)
    Y = similar(X)
    Z = similar(X)

    for is in 1:ns
        u = Float32((is - 1) / (ns - 1))
        s = length * u

        cx = 185f0 * sin(s / 520f0) + 72f0 * sin(s / 1250f0 + 0.45f0)
        cy = s - length / 2f0
        dxds = 185f0 / 520f0 * cos(s / 520f0) +
               72f0 / 1250f0 * cos(s / 1250f0 + 0.45f0)
        tangent_len = Float32(hypot(dxds, 1f0))
        right_x = 1f0 / tangent_len
        right_y = -dxds / tangent_len

        center_z = 26f0 * sin(2.2f0 * Float32(pi) * u - 0.35f0) +
                   6f0 * sin(s / 180f0)
        bank = 0.034f0 * sin(2.0f0 * Float32(pi) * u + 0.6f0)

        for it in 1:nt
            v = Float32((it - 1) / (nt - 1))
            lateral = (v - 0.5f0) * width
            crown = -5.2f0 * (lateral / (width / 2f0))^2
            ripple = 2.4f0 * sin(0.019f0 * s + 0.018f0 * lateral) +
                     1.3f0 * sin(0.041f0 * s - 0.011f0 * lateral)

            X[is, it] = cx + lateral * right_x
            Y[is, it] = cy + lateral * right_y
            Z[is, it] = center_z + bank * lateral + crown + ripple
        end
    end

    return X, Y, Z
end

point_at(X, Y, Z, is, it) = Point3f(X[is, it], Y[is, it], Z[is, it])

function road_normal(X, Y, Z, is, it)
    ns, nt = size(X)
    is0 = max(is - 1, 1)
    is1 = min(is + 1, ns)
    it0 = max(it - 1, 1)
    it1 = min(it + 1, nt)
    ds = point_at(X, Y, Z, is1, it) - point_at(X, Y, Z, is0, it)
    dt = point_at(X, Y, Z, is, it1) - point_at(X, Y, Z, is, it0)
    n = cross(dt, ds)
    len = norm(n)
    return len > 0 ? Vec3f(n ./ len) : Vec3f(0, 0, 1)
end

function road_mesh_from_matrices(X, Y, Z)
    size(X) == size(Y) == size(Z) || error("X, Y, and Z must have identical sizes.")
    ns, nt = size(X)

    pts = Point3f[]
    uvs = Vec2f[]
    normals = Vec3f[]
    sizehint!(pts, ns * nt)
    sizehint!(uvs, ns * nt)
    sizehint!(normals, ns * nt)

    for is in 1:ns, it in 1:nt
        push!(pts, point_at(X, Y, Z, is, it))
        push!(uvs, Vec2f((is - 1) / (ns - 1), (it - 1) / (nt - 1)))
        push!(normals, road_normal(X, Y, Z, is, it))
    end

    idx(is, it) = (is - 1) * nt + it
    faces = TriangleFace{Int}[]
    sizehint!(faces, 2 * (ns - 1) * (nt - 1))
    for is in 1:(ns - 1), it in 1:(nt - 1)
        a = idx(is, it)
        b = idx(is + 1, it)
        c = idx(is + 1, it + 1)
        d = idx(is, it + 1)
        push!(faces, TriangleFace(a, b, c))
        push!(faces, TriangleFace(a, c, d))
    end

    return GeometryBasics.Mesh(pts, faces; uv = uvs, normal = normals)
end

function grid_distance(x, spacing = 1f0)
    y = mod(x, spacing)
    return min(y, spacing - y)
end

function tron_road_texture(width = 2048, height = 512;
                           length_cells = 28f0, width_cells = 4f0)
    img = Matrix{RGBf}(undef, height, width)

    for j in 1:height, i in 1:width
        u = Float32((i - 1) / max(width - 1, 1))
        v = Float32((j - 1) / max(height - 1, 1))
        su = u * length_cells
        tv = (v - 0.5f0) * width_cells

        long_line = exp(-(grid_distance(su) / 0.022f0)^2)
        cross_line = exp(-(grid_distance(tv) / 0.028f0)^2)
        major_long = exp(-(grid_distance(su, 4f0) / 0.030f0)^2)
        shoulder = exp(-((abs(v - 0.5f0) - 0.485f0) / 0.010f0)^2)

        grid = clamp(0.58f0 * max(long_line, cross_line) +
                     0.44f0 * max(major_long, shoulder), 0f0, 1f0)
        halo = clamp(0.35f0 * max(exp(-(grid_distance(su) / 0.060f0)^2),
                                  exp(-(grid_distance(tv) / 0.070f0)^2)), 0f0, 1f0)
        grain = 0.5f0 + 0.5f0 * sin(54f0 * u + 11f0 * sin(18f0 * v))

        base = 0.0035f0 + 0.0025f0 * grain
        cyan = clamp(0.36f0 * halo + 0.96f0 * grid, 0f0, 1f0)
        img[j, i] = RGBf(clamp(base + 0.010f0 * grid, 0f0, 1f0),
                         clamp(base + 0.92f0 * cyan, 0f0, 1f0),
                         clamp(base + 1.00f0 * cyan, 0f0, 1f0))
    end

    return img
end

function build_scene()
    X, Y, Z = road_strip_matrices()
    road = road_mesh_from_matrices(X, Y, Z)
    road_texture = tron_road_texture()
    sky = fill(RGBf(0.018f0, 0.020f0, 0.022f0), 16, 32)

    lights = AbstractLight[
        EnvironmentLight(0.78f0, sky),
        AmbientLight(RGBf(0.24f0, 0.24f0, 0.24f0)),
        DirectionalLight(RGBf(0.88f0, 0.88f0, 0.88f0), Vec3f(-0.30, -0.45, -1.0), true),
        RectLight(RGBf(2.85f0, 2.85f0, 2.85f0), Point3f(-260, -880, 620),
                  Vec3f(920, 0, 0), Vec3f(0, 520, 0), Vec3f(0.20, 0.48, -1.0)),
        PointLight(RGBf(0.95f0, 1.06f0, 1.12f0), Vec3f(380, -620, 280)),
    ]

    scene = Scene(size = (1280, 720); lights = lights)
    cam3d!(scene)
    mesh!(scene, road; color = road_texture, material = ROAD_MATERIAL)

    update_cam!(scene, Vec3f(-720, -2020, 430), Vec3f(65, -150, 8), Vec3f(0, 0, 1))
    return scene
end

function main()
    mkpath(dirname(abspath(OUT)))
    OM.activate!(mode = :rt2,
                 warmup = 1,
                 samples = 64,
                 background = :domelight,
                 accumulate_across_frames = true,
                 accumulation_preroll = 16)

    scene = build_scene()
    screen = OM.Screen(scene; background = :domelight)
    img = Makie.colorbuffer(screen)
    close(screen)
    FileIO.save(OUT, img)
    println("PNG=", OUT)
    println("OK_TRON_TEXTURED_ROAD_STRIP")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
