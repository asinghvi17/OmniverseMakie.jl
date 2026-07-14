# Tron-grid drive showcase -- NVIDIA's ConceptCar01 hero asset on a
# fictional, glassy Grid trajectory.
#
# The car stays a referenced USD (`usdplot!`).  Makie primitives build the
# environment: a reflective black floor, curved translucent arena glass,
# subtle embedded LED lines, a visible dome sky, and flat wheel-side light
# disks.  The trajectory is scripted, not simulated.
#
# Run it (GPU; serialize on the shared lock as every ovrtx job does):
#   flock -w 3600 /tmp/omniversemakie-gpu.lock -c \
#     'OVRTX_LIBRARY_PATH=<...>/libovrtx-dynamic.so JULIA_CUDA_USE_COMPAT=false \
#      julia --project=examples examples/tron_conceptcar_drive.jl'
#
# Env overrides: CONCEPTCAR_USD (asset path), TRON_MP4 (output path),
# TRON_SECONDS, TRON_FRAMES (low-level override), TRON_FPS, TRON_SPEED_MPS,
# TRON_SKY_IMAGE, TRON_SKY_EXPOSURE, TRON_SKY_SATURATION, TRON_SKY_ROTATION,
# TRON_RENDER_MODE (:rt2 or :pathtracing), TRON_WARMUP, TRON_SAMPLES.

using OmniverseMakie, ColorTypes, FixedPointNumbers, GeometryBasics
using FileIO, ImageIO
const OM = OmniverseMakie
using Makie: Scene, Vec3f, Point3f, Rect3f, RGBf, Observable, AbstractLight,
             AmbientLight, DirectionalLight, EnvironmentLight, PointLight, RectLight,
             cam3d!, update_cam!, translationmatrix, rotationmatrix_x,
             rotationmatrix_y, rotationmatrix_z

const CONCEPTCAR_DEFAULT =
    "/home/juliahub/temp/digital-twins-for-fluid-simulation/stages/layers/CarHero/" *
    "ConceptCar/ConceptCar01_Adjust.usd"

const CAR    = get(ENV, "CONCEPTCAR_USD", CONCEPTCAR_DEFAULT)
const MP4    = get(ENV, "TRON_MP4",
                   normpath(joinpath(@__DIR__, "..", "recordings", "conceptcar_tron_drive.mp4")))
const FPS = parse(Int, get(ENV, "TRON_FPS", "30"))
const DURATION_SECONDS = parse(Float32, get(ENV, "TRON_SECONDS", "6.0"))
const FRAMES = haskey(ENV, "TRON_FRAMES") ?
    parse(Int, ENV["TRON_FRAMES"]) :
    max(1, Int(round(DURATION_SECONDS * FPS)))
const SPEED_MPS = parse(Float32, get(ENV, "TRON_SPEED_MPS", "2.0"))
const SKY_IMAGE = get(ENV, "TRON_SKY_IMAGE", "")
const SKY_EXPOSURE = parse(Float32, get(ENV, "TRON_SKY_EXPOSURE", "0.40"))
const SKY_SATURATION = parse(Float32, get(ENV, "TRON_SKY_SATURATION", "0.16"))
const SKY_ROTATION = parse(Float32, get(ENV, "TRON_SKY_ROTATION", "0.25"))
const RENDER_MODE = let raw = lowercase(get(ENV, "TRON_RENDER_MODE", "rt2"))
    Symbol(startswith(raw, ":") ? raw[2:end] : raw)
end
const WARMUP = parse(Int, get(ENV, "TRON_WARMUP", RENDER_MODE === :pathtracing ? "48" : "1"))
const SAMPLES = parse(Int, get(ENV, "TRON_SAMPLES", RENDER_MODE === :pathtracing ? "256" : "64"))

const GRAPHITE    = RGBf(0.0015f0, 0.0020f0, 0.0030f0)
const GLASS_CYAN  = RGBf(0.024f0, 0.215f0, 0.255f0)
const LED_CYAN    = RGBf(0.008f0, 0.135f0, 0.160f0)
const SKY_TOP     = RGBf(0.026f0, 0.060f0, 0.125f0)
const SKY_HORIZON = RGBf(0.110f0, 0.42f0, 0.52f0)

const BLACK_FLOOR_MATERIAL = (; metallic = 0.46f0, roughness = 0.018f0)
const GLASS_LINE_MATERIAL = (; glass = true, ior = 1.505f0,
                             roughness = 0.0f0, thin_walled = true)
const DIM_LED_MATERIAL = (; emissive = RGBf(0.0f0, 0.072f0, 0.090f0),
                          opacity = 0.34f0)
const GRID_LED_MATERIAL = (; emissive = RGBf(0.0f0, 0.28f0, 0.34f0),
                           opacity = 0.66f0,
                           metallic = 0.04f0,
                           roughness = 0.006f0)
const GRID_LED_MAJOR_MATERIAL = (; emissive = RGBf(0.0f0, 0.46f0, 0.56f0),
                                 opacity = 0.74f0,
                                 metallic = 0.04f0,
                                 roughness = 0.004f0)
const OCEAN_Z = -185f0
const OCEAN_MATERIAL = (; metallic = 0.0f0, roughness = 0.032f0)
const SCENE_UNITS_PER_METER = 100f0
const CAMERA_RESPONSE_HZ = 5.0f0

# ConceptCar01_Adjust.usd is Y-up, authored in centimetres, but the working
# CarComponents path drives it with `up = :z` and an explicit root rotation.
# The asset nose is local -Z; this maps local -Z to scene +X and local +Y to
# scene +Z.
const CAR_BBOX = Rect3f(Point3f(-112, 0, -101), Vec3f(224, 138, 506))
const CONCEPT_BASE = rotationmatrix_x(Float32(pi / 2)) * rotationmatrix_y(Float32(-pi / 2))

# ConceptCar wheel-local data from the CarComponents memory.
const L_D = Float32[
    1 0 0 0;
    0 1 0 0;
    0 0 1 0;
    0 0 0 1
]
const L_P = Float32[
    -0.999995 0 0.003051 0;
    0 1 0 0;
    0.003051 0 0.999995 0;
    0 0 0 1
]
concept_wheel_mat(hub, steer, spin, localmat) =
    translationmatrix(hub) * rotationmatrix_y(Float32(steer)) *
    rotationmatrix_x(Float32(spin)) * translationmatrix(-hub) * localmat

const WHEELS = [
    (target = "/root/Wheels/wheel_DF", hub = Vec3f( 91.68, 38.37,  141.04), front = true,  localmat = L_D, radius = 38.27f0),
    (target = "/root/Wheels/wheel_PF", hub = Vec3f(-91.70, 38.40,  141.20), front = true,  localmat = L_P, radius = 38.27f0),
    (target = "/root/Wheels/wheel_DB", hub = Vec3f( 94.50, 38.90, -154.20), front = false, localmat = L_D, radius = 39.00f0),
    (target = "/root/Wheels/wheel_PB", hub = Vec3f(-94.40, 38.90, -154.20), front = false, localmat = L_P, radius = 39.00f0),
]

const WHEEL_DISK_POINTS = Point3f[
    Point3f( -10, -92, 39), Point3f( -10,  92, 39),
    Point3f(-305,  95, 39), Point3f(-305, -95, 39),
]

function route_point(t; z = 0f0)
    tt = Float32(t)
    rmod = 1.0f0 + 0.075f0 * sin(3f0 * tt + 0.6f0)
    ymod = 1.0f0 + 0.04f0 * cos(2f0 * tt)
    return Point3f(920f0 * rmod * cos(tt),
                   560f0 * ymod * sin(tt) - 90f0,
                   z)
end

function route_derivative(t)
    tt = Float32(t)
    rmod = 1.0f0 + 0.075f0 * sin(3f0 * tt + 0.6f0)
    dr = 0.225f0 * cos(3f0 * tt + 0.6f0)
    ymod = 1.0f0 + 0.04f0 * cos(2f0 * tt)
    dy = -0.08f0 * sin(2f0 * tt)
    dxdt = 920f0 * (dr * cos(tt) - rmod * sin(tt))
    dydt = 560f0 * (dy * sin(tt) + ymod * cos(tt))
    return dxdt, dydt
end

const ROUTE = [route_point(2f0 * Float32(pi) * k / 360f0) for k in 0:360]

const ROUTE_TABLE = let
    ts = [2f0 * Float32(pi) * k / 2048f0 for k in 0:2048]
    cum = Float32[0]
    prev = route_point(first(ts))
    total = 0f0
    for t in ts[2:end]
        p = route_point(t)
        d = p - prev
        total += Float32(hypot(d[1], d[2]))
        push!(cum, total)
        prev = p
    end
    (; ts, cum, total)
end

function mix_rgb(a::RGBf, b::RGBf, t)
    s = Float32(clamp(t, 0, 1))
    return RGBf((1 - s) * red(a) + s * red(b),
                (1 - s) * green(a) + s * green(b),
                (1 - s) * blue(a) + s * blue(b))
end

function tron_sky(width = 768, height = 384)
    img = Matrix{RGBf}(undef, height, width)
    for j in 1:height, i in 1:width
        u = Float32((i - 1) / max(width - 1, 1))
        v = Float32((j - 1) / max(height - 1, 1))
        horizon = exp(-((v - 0.56f0) / 0.15f0)^2)
        zenith = clamp((0.72f0 - v) / 0.72f0, 0f0, 1f0)
        scan = 0.5f0 + 0.5f0 * sin(2f0 * Float32(pi) * (u * 12f0 + v * 0.7f0))
        base = mix_rgb(SKY_TOP, SKY_HORIZON, 0.70f0 * horizon + 0.26f0 * zenith)
        img[j, i] = RGBf(red(base) + 0.035f0 * horizon + 0.006f0 * scan * horizon,
                         green(base) + 0.070f0 * horizon + 0.010f0 * scan * horizon,
                         blue(base) + 0.110f0 * horizon + 0.016f0 * scan * horizon)
    end
    return img
end

function rotate_sky_image(img)
    width = size(img, 2)
    shift = Int(round(SKY_ROTATION * width))
    return shift == 0 ? img : circshift(img, (0, shift))
end

function grade_sky_pixel(c; exposure, saturation)
    r = Float32(red(c))
    g = Float32(green(c))
    b = Float32(blue(c))
    l = clamp(0.2126f0 * r + 0.7152f0 * g + 0.0722f0 * b, 0f0, 1f0)
    contrast = 1.08f0
    gray = clamp((l - 0.5f0) * contrast + 0.5f0, 0f0, 1f0)
    rr = clamp((r - 0.5f0) * contrast + 0.5f0, 0f0, 1f0)
    gg = clamp((g - 0.5f0) * contrast + 0.5f0, 0f0, 1f0)
    bb = clamp((b - 0.5f0) * contrast + 0.5f0, 0f0, 1f0)
    sat = clamp(Float32(saturation), 0f0, 1f0)
    gain = max(Float32(exposure), 0f0)
    return RGBf(clamp(gain * ((1f0 - sat) * gray + sat * rr), 0f0, 1f0),
                clamp(gain * ((1f0 - sat) * gray + sat * gg), 0f0, 1f0),
                clamp(gain * ((1f0 - sat) * gray + sat * bb), 0f0, 1f0))
end

function grade_sky_image(img; exposure, saturation)
    out = Matrix{RGBf}(undef, size(img, 1), size(img, 2))
    @inbounds for j in axes(img, 1), i in axes(img, 2)
        out[j, i] = grade_sky_pixel(img[j, i]; exposure = exposure, saturation = saturation)
    end
    return out
end

function load_sky_source()
    isempty(SKY_IMAGE) && return tron_sky(1024, 512)
    isfile(SKY_IMAGE) || error("TRON_SKY_IMAGE does not exist: $SKY_IMAGE")
    return FileIO.load(SKY_IMAGE)
end

function scene_sky_image()
    source = rotate_sky_image(load_sky_source())
    return grade_sky_image(source; exposure = SKY_EXPOSURE,
                           saturation = SKY_SATURATION)
end

function disk_mesh(center::Point3f, radius; n = 72)
    pts = Point3f[center]
    for k in 0:(n - 1)
        t = Float32(2pi * k / n)
        push!(pts, Point3f(center[1] + radius * cos(t), center[2], center[3] + radius * sin(t)))
    end
    faces = TriangleFace{Int}[]
    for k in 1:n
        a = k + 1
        b = (k == n ? 2 : k + 2)
        push!(faces, TriangleFace(1, a, b))
        push!(faces, TriangleFace(1, b, a))
    end
    normal = Vec3f(0, center[2] >= 0 ? 1 : -1, 0)
    normals = fill(normal, length(pts))
    return GeometryBasics.Mesh(pts, faces; normal = normals)
end

function ocean_height(x, y)
    xx = Float32(x)
    yy = Float32(y)
    return OCEAN_Z +
           8.0f0 * sin(0.0045f0 * xx + 0.7f0) +
           5.0f0 * sin(0.0037f0 * yy - 1.2f0) +
           3.0f0 * sin(0.0028f0 * (xx + yy) + 1.9f0)
end

function ocean_normal(x, y)
    xx = Float32(x)
    yy = Float32(y)
    dzdx = 8.0f0 * 0.0045f0 * cos(0.0045f0 * xx + 0.7f0) +
           3.0f0 * 0.0028f0 * cos(0.0028f0 * (xx + yy) + 1.9f0)
    dzdy = 5.0f0 * 0.0037f0 * cos(0.0037f0 * yy - 1.2f0) +
           3.0f0 * 0.0028f0 * cos(0.0028f0 * (xx + yy) + 1.9f0)
    n = Vec3f(-dzdx, -dzdy, 1f0)
    return n / sqrt(n[1]^2 + n[2]^2 + n[3]^2)
end

function ocean_mesh(; half = 6800f0, n = 88)
    pts = Point3f[]
    normals = Vec3f[]
    colors = RGBf[]
    for iy in 0:n, ix in 0:n
        x = -half + 2f0 * half * Float32(ix / n)
        y = -half + 2f0 * half * Float32(iy / n)
        z = ocean_height(x, y)
        ripple = 0.5f0 + 0.5f0 * sin(0.006f0 * x - 0.004f0 * y)
        push!(pts, Point3f(x, y, z))
        push!(normals, ocean_normal(x, y))
        push!(colors, RGBf(0.0020f0 + 0.0020f0 * ripple,
                           0.0075f0 + 0.0045f0 * ripple,
                           0.0100f0 + 0.0065f0 * ripple))
    end

    idx(ix, iy) = iy * (n + 1) + ix + 1
    faces = TriangleFace{Int}[]
    for iy in 0:(n - 1), ix in 0:(n - 1)
        a = idx(ix, iy)
        b = idx(ix + 1, iy)
        c = idx(ix + 1, iy + 1)
        d = idx(ix, iy + 1)
        push!(faces, TriangleFace(a, b, c))
        push!(faces, TriangleFace(a, c, d))
    end
    return GeometryBasics.Mesh(pts, faces; normal = normals, color = colors)
end

function box!(scene, center::Point3f, size::Vec3f; color, material = (;))
    lo = Point3f(center[1] - size[1] / 2, center[2] - size[2] / 2, center[3] - size[3] / 2)
    return mesh!(scene, Rect3f(lo, size); color = color, material = material)
end

function grid_block!(scene, center::Point3f, size::Vec3f; color, material)
    return box!(scene, center, size; color = color, material = material)
end

function arc_points(radius, a0, a1; n = 96, yscale = 0.68f0, yoffset = -90f0, z = 8f0)
    return [Point3f(radius * cos(t), yscale * radius * sin(t) + yoffset, z)
            for t in range(Float32(a0), Float32(a1), length = n)]
end

function bridge_deck_mesh(radius, width, a0, a1; n = 56, yscale = 0.68f0,
                          yoffset = -90f0, z = 120f0)
    left = Point3f[]
    right = Point3f[]
    for t in range(Float32(a0), Float32(a1), length = n)
        tx = -radius * sin(t)
        ty = yscale * radius * cos(t)
        invlen = inv(Float32(hypot(tx, ty)))
        nx = -ty * invlen
        ny = tx * invlen
        cx = radius * cos(t)
        cy = yscale * radius * sin(t) + yoffset
        halfw = Float32(width) / 2
        push!(left, Point3f(cx + halfw * nx, cy + halfw * ny, z))
        push!(right, Point3f(cx - halfw * nx, cy - halfw * ny, z))
    end

    pts = vcat(left, right)
    faces = TriangleFace{Int}[]
    for i in 1:(n - 1)
        push!(faces, TriangleFace(i, i + 1, n + i + 1))
        push!(faces, TriangleFace(i, n + i + 1, n + i))
    end
    normals = fill(Vec3f(0, 0, 1), length(pts))
    return GeometryBasics.Mesh(pts, faces; normal = normals), left, right
end

function glassline!(scene, pts; linewidth = 0.45, color = GLASS_CYAN)
    return lines!(scene, pts; linewidth = linewidth, color = color,
                  material = GLASS_LINE_MATERIAL)
end

function ledline!(scene, pts; linewidth = 0.22, color = LED_CYAN)
    return lines!(scene, pts; linewidth = linewidth, color = color,
                  material = DIM_LED_MATERIAL)
end

function add_floor!(scene)
    box!(scene, Point3f(0, 0, -5), Vec3f(2700, 2300, 10);
         color = GRAPHITE,
         material = BLACK_FLOOR_MATERIAL)

    grid_z = 0.55f0
    grid_h = 0.70f0
    grid_color = RGBf(0.0f0, 0.23f0, 0.28f0)
    for x in -1260f0:180f0:1260f0
        grid_block!(scene, Point3f(x, 0, grid_z), Vec3f(5.5f0, 2160f0, grid_h);
                    color = grid_color,
                    material = GRID_LED_MATERIAL)
    end
    for y in -1080f0:180f0:1080f0
        grid_block!(scene, Point3f(0, y, grid_z), Vec3f(2520f0, 5.5f0, grid_h);
                    color = grid_color,
                    material = GRID_LED_MATERIAL)
    end

    major_color = RGBf(0.0f0, 0.34f0, 0.40f0)
    for x in -1080f0:360f0:1080f0
        grid_block!(scene, Point3f(x, 0, grid_z + 0.10f0), Vec3f(10f0, 2000f0, grid_h);
                    color = major_color,
                    material = GRID_LED_MAJOR_MATERIAL)
    end
    for y in -900f0:360f0:900f0
        grid_block!(scene, Point3f(0, y, grid_z + 0.10f0), Vec3f(2380f0, 10f0, grid_h);
                    color = major_color,
                    material = GRID_LED_MAJOR_MATERIAL)
    end
end

function add_ocean!(scene)
    mesh!(scene, ocean_mesh();
          color = RGBf(0.0030f0, 0.0100f0, 0.0140f0),
          material = OCEAN_MATERIAL)
end

function add_bridge_deck!(scene, radius, width, a0, a1, z; n = 56)
    deck, left, right = bridge_deck_mesh(radius, width, a0, a1; n = n, z = z)
    mesh!(scene, deck;
          color = RGBf(0.0018f0, 0.0030f0, 0.0044f0),
          material = (; metallic = 0.38f0, roughness = 0.010f0))
    for edge in (left, right)
        glassline!(scene, edge; linewidth = 0.42, color = RGBf(0.010f0, 0.100f0, 0.122f0))
        ledline!(scene, edge; linewidth = 0.18, color = RGBf(0.0f0, 0.072f0, 0.088f0))
    end
    for k in 6:10:(length(left) - 5)
        glassline!(scene, [left[k], right[k]];
                   linewidth = 0.28, color = RGBf(0.006f0, 0.052f0, 0.066f0))
    end
end

function add_background_bridges!(scene)
    add_bridge_deck!(scene, 1280f0, 92f0, 0.15pi, 0.90pi, 118f0; n = 72)
    add_bridge_deck!(scene, 1510f0, 76f0, 0.18pi, 0.82pi, 238f0; n = 62)
    add_bridge_deck!(scene, 1760f0, 64f0, 0.21pi, 0.76pi, 354f0; n = 54)

    for (r, z1, lw) in ((1120f0, 285f0, 0.48), (1320f0, 420f0, 0.36))
        for z in range(66f0, z1, length = 5)
            glassline!(scene, arc_points(r, 0.16pi, 0.94pi; n = 120, z = z);
                       linewidth = lw, color = RGBf(0.008f0, 0.095f0, 0.115f0))
        end
        for t in range(Float32(0.18pi), Float32(0.92pi), length = 16)
            x = r * cos(t)
            y = 0.68f0 * r * sin(t) - 90f0
            glassline!(scene, [Point3f(x, y, 46f0), Point3f(x, y, z1)];
                       linewidth = lw * 0.76, color = RGBf(0.006f0, 0.070f0, 0.085f0))
        end
    end
end

function add_city_silhouette!(scene)
    for x in -1280:160:1280
        h = 170f0 + 190f0 * (0.5f0 + 0.5f0 * sin(Float32(x) * 0.013f0 + 0.5f0))
        w = 46f0 + 24f0 * (0.5f0 + 0.5f0 * cos(Float32(x) * 0.019f0))
        box!(scene, Point3f(x, 1285, h / 2), Vec3f(w, 38, h);
             color = RGBf(0.004f0, 0.011f0, 0.018f0),
             material = (; metallic = 0.55f0, roughness = 0.055f0))
    end
    for z in (74f0, 126f0, 210f0)
        glassline!(scene, [Point3f(-1420, 1320, z), Point3f(1420, 1320, z + 28f0)];
                   linewidth = 0.26, color = RGBf(0.006f0, 0.052f0, 0.066f0))
    end
end

function add_arena!(scene)
    add_background_bridges!(scene)
    add_city_silhouette!(scene)
end

function route_lengths(route)
    lens = Float32[]
    total = 0.0f0
    for i in 1:(length(route) - 1)
        d = route[i + 1] - route[i]
        l = Float32(hypot(d[1], d[2]))
        push!(lens, l)
        total += l
    end
    return lens, total
end

function sample_route(table, s)
    u = mod(Float32(s), table.total)
    idx = clamp(searchsortedlast(table.cum, u), 1, length(table.cum) - 1)
    a = table.cum[idx]
    b = table.cum[idx + 1]
    f = b == a ? 0f0 : (u - a) / (b - a)
    t = table.ts[idx] + f * (table.ts[idx + 1] - table.ts[idx])
    pos = route_point(t; z = 12f0)
    dx, dy = route_derivative(t)
    yaw = Float32(atan(dy, dx))
    return pos, yaw, u
end

mix_vec3(a::Vec3f, b::Vec3f, t) =
    Vec3f(a[1] + t * (b[1] - a[1]),
          a[2] + t * (b[2] - a[2]),
          a[3] + t * (b[3] - a[3]))

function reset_camera!(state)
    state.camera_seeded[] = false
    return state
end

car_model(pos, yaw) = translationmatrix(Vec3f(pos[1], pos[2], pos[3])) * rotationmatrix_z(yaw)
concept_model(pos, yaw) = car_model(pos, yaw) * CONCEPT_BASE

function add_player_glow!(scene)
    underglow = mesh!(scene, Rect3f(Point3f(-410, -118, 2), Vec3f(525, 236, 4));
                      color = RGBf(0.0f0, 0.22f0, 0.28f0),
                      material = (; emissive = RGBf(0.0f0, 0.070f0, 0.090f0), opacity = 0.08f0))
    rear_trail = mesh!(scene, Rect3f(Point3f(-840, -58, 4), Vec3f(430, 116, 4));
                       color = RGBf(0.0f0, 0.26f0, 0.32f0),
                       material = (; emissive = RGBf(0.0f0, 0.085f0, 0.100f0), opacity = 0.07f0))
    wheel_disks = [
        mesh!(scene, disk_mesh(p, 34f0);
              color = RGBf(0.030f0, 0.48f0, 0.58f0),
              material = (; emissive = RGBf(0.0f0, 0.20f0, 0.25f0), opacity = 0.42f0))
        for p in WHEEL_DISK_POINTS
    ]
    return (underglow, rear_trail, wheel_disks...)
end

function build_scene(car_path)
    sky = scene_sky_image()
    lights = AbstractLight[
        EnvironmentLight(1.0f0, sky),
        AmbientLight(RGBf(0.46f0, 0.46f0, 0.46f0)),
        DirectionalLight(RGBf(1.45f0, 1.45f0, 1.45f0), Vec3f(-0.35, -0.35, -1.0), true),
        RectLight(RGBf(2.75f0, 2.75f0, 2.75f0), Point3f(0, -650, 760),
                  Vec3f(1080, 0, 0), Vec3f(0, 520, 0), Vec3f(0.16, 0.32, -1.0)),
        RectLight(RGBf(1.85f0, 1.85f0, 1.85f0), Point3f(820, -260, 510),
                  Vec3f(0, 760, 0), Vec3f(0, 0, 320), Vec3f(-0.90, -0.22, -0.45)),
        PointLight(RGBf(2.95f0, 2.95f0, 2.95f0), Vec3f(-520, -460, 420)),
        PointLight(RGBf(2.25f0, 2.25f0, 2.25f0), Vec3f(620, -700, 380)),
        PointLight(RGBf(1.15f0, 1.15f0, 1.15f0), Vec3f(140, 150, 520)),
    ]
    scene = Scene(size = (1280, 720); lights = lights)
    cam3d!(scene)

    add_ocean!(scene)
    add_floor!(scene)
    add_arena!(scene)
    player_glow = add_player_glow!(scene)

    car = usdplot!(scene, car_path; up = :z, bbox = CAR_BBOX)
    spins = Dict(w.target => Observable(concept_wheel_mat(w.hub, 0.0f0, 0.0f0, w.localmat)) for w in WHEELS)
    for w in WHEELS
        bind_usd!(car, w.target, spins[w.target])
    end

    return (; scene, car, spins, player_glow,
            route_table = ROUTE_TABLE,
            route_total = ROUTE_TABLE.total,
            camera_eye = Ref(Vec3f(0, 0, 0)),
            camera_look = Ref(Vec3f(0, 0, 0)),
            camera_seeded = Ref(false))
end

function animate_frame!(state, i, nframes)
    elapsed_seconds = Float32(i - 1) / Float32(FPS)
    s = SPEED_MPS * SCENE_UNITS_PER_METER * elapsed_seconds
    pos, yaw, dist = sample_route(state.route_table, s)
    model = car_model(pos, yaw)
    state.car.model[] = concept_model(pos, yaw)
    for p in state.player_glow
        p.model[] = model
    end

    theta = Float32(s) / 38.6f0
    steer = 0.13f0 * sin(Float32(2pi) * dist / 1250f0)
    for w in WHEELS
        state.spins[w.target][] =
            concept_wheel_mat(w.hub, w.front ? steer : 0.0f0,
                              theta * (38.6f0 / w.radius), w.localmat)
    end

    dir = Vec3f(cos(yaw), sin(yaw), 0)
    right = Vec3f(-sin(yaw), cos(yaw), 0)
    target_eye = Vec3f(pos[1], pos[2], 0) - 690f0 * dir + 210f0 * right + Vec3f(0, 0, 172)
    target_look = Vec3f(pos[1], pos[2], 0) + 560f0 * dir + Vec3f(0, 0, 70)
    alpha = state.camera_seeded[] ? 1f0 - exp(-CAMERA_RESPONSE_HZ / Float32(FPS)) : 1f0
    state.camera_eye[] = mix_vec3(state.camera_eye[], target_eye, alpha)
    state.camera_look[] = mix_vec3(state.camera_look[], target_look, alpha)
    state.camera_seeded[] = true
    update_cam!(state.scene, state.camera_eye[], state.camera_look[], Vec3f(0, 0, 1))
end

lum(c) = Float32(red(c)) + Float32(green(c)) + Float32(blue(c))

function main()
    isfile(CAR) || error("NVIDIA ConceptCar asset not found: $CAR (set CONCEPTCAR_USD to ConceptCar01_Adjust.usd).")
    FRAMES > 0 || error("TRON_FRAMES must be positive, got $FRAMES.")
    FPS > 0 || error("TRON_FPS must be positive, got $FPS.")
    SPEED_MPS >= 0 || error("TRON_SPEED_MPS must be non-negative, got $SPEED_MPS.")
    mkpath(dirname(abspath(MP4)))
    OM.activate!(mode = RENDER_MODE,
                 warmup = WARMUP,
                 samples = SAMPLES,
                 background = :domelight,
                 accumulate_across_frames = RENDER_MODE === :rt2,
                 accumulation_preroll = 16)
    state = build_scene(CAR)
    animate_frame!(state, 1, FRAMES)

    screen = OM.Screen(state.scene; background = :domelight)
    animate_frame!(state, 1, FRAMES)
    a = Makie.colorbuffer(screen)
    animate_frame!(state, Int(round(FPS)) + 1, FRAMES)
    b = Makie.colorbuffer(screen)
    close(screen)

    nb = count(c -> lum(c) > 0.05f0, a)
    motion = count(k -> abs(lum(a[k]) - lum(b[k])) > 0.08f0, eachindex(a))

    reset_camera!(state)
    record_seconds = @elapsed Makie.record(state.scene, MP4, 1:FRAMES; framerate = FPS) do i
        animate_frame!(state, i, FRAMES)
    end
    @assert isfile(MP4) "record wrote no mp4"

    sz = filesize(MP4)
    println("SPEED_MPS=", SPEED_MPS, "  FPS=", FPS,
            "  DURATION_SECONDS=", round(FRAMES / FPS; digits = 2))
    println("RENDER_SECONDS=", round(record_seconds; digits = 2),
            "  RENDER_FPS=", round(FRAMES / record_seconds; digits = 2))
    println("FRAME_NONBLACK=", nb, "  MOTION_PIXELS=", motion)
    println("MP4=", MP4, "  BYTES=", sz)
    @assert sz > 20_000 "mp4 suspiciously small ($sz bytes)"
    @assert nb > 20_000 "scene did not render enough non-black pixels (nonblack=$nb)"
    @assert motion > 1_000 "drive animation did not visibly move enough pixels ($motion)"
    println("OK_TRON_CONCEPTCAR_DRIVE")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
