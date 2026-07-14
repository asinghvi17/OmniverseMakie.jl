# examples/common/water_cube.jl — shared water-cube harness: a watertight
# box mesh whose top surface is an N×N height field, a pool scene around it
# (checkerboard floor, reference spheres, sky dome), a camera orbit, and the
# acceptance checks.  Drivers (Gerstner sines, shallow water) fill `h`.

using OmniverseMakie, GeometryBasics, ColorTypes, FixedPointNumbers
import OmniverseMakie as OM

# ---- geometry ----------------------------------------------------------------
# Six separate sheets (top, bottom, 4 walls) so edge vertices are duplicated
# and cube edges stay crisp under per-vertex normals.  Vertex layout:
#   1        … N²        top grid       (z = depth + h)
#   N²+1     … 2N²       bottom grid    (z = 0)
#   2N²+1    … 2N²+8N    walls y-,y+,x-,x+; each wall = N bottom + N top verts
# Vertex COUNT is constant across frames (the live mesh push requires it).

"""
    water_cube_faces(N) -> Vector{GLTriangleFace}

Triangle faces for the six-sheet watertight water cube (built once; only
positions/normals change per frame).
"""
function water_cube_faces(N::Int)
    T(i, j) = (j - 1) * N + i
    B0 = N * N
    faces = GeometryBasics.GLTriangleFace[]
    sizehint!(faces, 4 * (N - 1) * (N - 1) + 8 * (N - 1))
    for j in 1:N-1, i in 1:N-1
        a, b, c, d = T(i, j), T(i + 1, j), T(i + 1, j + 1), T(i, j + 1)
        push!(faces, GeometryBasics.GLTriangleFace(a, b, c))        # top, +z out
        push!(faces, GeometryBasics.GLTriangleFace(a, c, d))
        push!(faces, GeometryBasics.GLTriangleFace(B0 + a, B0 + c, B0 + b))  # bottom, -z out
        push!(faces, GeometryBasics.GLTriangleFace(B0 + a, B0 + d, B0 + c))
    end
    # Walls in order y-, y+, x-, x+; verts per wall: N bottom then N top.
    # Winding gives outward normals (-y, +y, -x, +x respectively).
    for (wall, W) in enumerate((2N^2, 2N^2 + 2N, 2N^2 + 4N, 2N^2 + 6N))
        for k in 1:N-1
            bl, br, tl, tr = W + k, W + k + 1, W + N + k, W + N + k + 1
            if wall == 1 || wall == 4
                push!(faces, GeometryBasics.GLTriangleFace(bl, br, tr))
                push!(faces, GeometryBasics.GLTriangleFace(bl, tr, tl))
            else
                push!(faces, GeometryBasics.GLTriangleFace(bl, tr, br))
                push!(faces, GeometryBasics.GLTriangleFace(bl, tl, tr))
            end
        end
    end
    return faces
end

"""
    water_cube_positions(h; L, depth) -> Vector{Point3f}

Vertex positions for height field `h` (N×N, meters) over footprint
[-L/2, L/2]²; still water sits at z = `depth`, the cube bottom at z = 0.
"""
function water_cube_positions(h::AbstractMatrix{Float32}; L::Float32, depth::Float32)
    N = size(h, 1)
    @assert size(h) == (N, N) "square height field required, got $(size(h))"
    xs = range(-L / 2, L / 2; length = N)
    pos = Vector{Point3f}(undef, 2N^2 + 8N)
    k = 0
    for j in 1:N, i in 1:N
        pos[k += 1] = Point3f(xs[i], xs[j], depth + h[i, j])
    end
    for j in 1:N, i in 1:N
        pos[k += 1] = Point3f(xs[i], xs[j], 0)
    end
    for (j, s) in ((1, -1f0), (N, 1f0))                    # walls y-, y+
        for i in 1:N; pos[k += 1] = Point3f(xs[i], s * L / 2, 0); end
        for i in 1:N; pos[k += 1] = Point3f(xs[i], s * L / 2, depth + h[i, j]); end
    end
    for (i, s) in ((1, -1f0), (N, 1f0))                    # walls x-, x+
        for j in 1:N; pos[k += 1] = Point3f(s * L / 2, xs[j], 0); end
        for j in 1:N; pos[k += 1] = Point3f(s * L / 2, xs[j], depth + h[i, j]); end
    end
    return pos
end

"""
    water_cube_normals(h; L) -> Vector{Vec3f}

Per-vertex normals: central-difference surface normals on the top sheet,
constant outward normals on bottom and walls.
"""
function water_cube_normals(h::AbstractMatrix{Float32}; L::Float32)
    N  = size(h, 1)
    dx = L / (N - 1)
    ns = Vector{Vec3f}(undef, 2N^2 + 8N)
    k = 0
    for j in 1:N, i in 1:N
        i0, i1 = max(i - 1, 1), min(i + 1, N)
        j0, j1 = max(j - 1, 1), min(j + 1, N)
        dhdx = (h[i1, j] - h[i0, j]) / ((i1 - i0) * dx)
        dhdy = (h[i, j1] - h[i, j0]) / ((j1 - j0) * dx)
        inv_len = 1f0 / sqrt(dhdx^2 + dhdy^2 + 1f0)
        ns[k += 1] = Vec3f(-dhdx * inv_len, -dhdy * inv_len, inv_len)
    end
    for _ in 1:N^2
        ns[k += 1] = Vec3f(0, 0, -1)
    end
    for s in (-1f0, 1f0), _ in 1:2N
        ns[k += 1] = Vec3f(0, s, 0)
    end
    for s in (-1f0, 1f0), _ in 1:2N
        ns[k += 1] = Vec3f(s, 0, 0)
    end
    return ns
end

water_cube_mesh(h::AbstractMatrix{Float32}, faces; L::Float32, depth::Float32) =
    GeometryBasics.Mesh(water_cube_positions(h; L, depth), faces;
                        normal = water_cube_normals(h; L))

# ---- scene dressing ------------------------------------------------------------

# Pool-tile checkerboard (light/dark) used as the floor texture.
function pool_checker_image(; tiles::Int = 9, px::Int = 48)
    W   = tiles * px
    img = Matrix{RGBf}(undef, W, W)
    @inbounds for j in 1:W, i in 1:W
        even = iseven((i - 1) ÷ px + (j - 1) ÷ px)
        img[i, j] = even ? RGBf(0.93, 0.94, 0.95) : RGBf(0.13, 0.32, 0.50)
    end
    return img
end

_lerp(a::RGBf, b::RGBf, t) = RGBf((1 - t) * a.r + t * b.r,
                                  (1 - t) * a.g + t * b.g,
                                  (1 - t) * a.b + t * b.b)

# Latlong sky gradient (row 1 = zenith) for the environment dome; LDR (≤ 1)
# so the env-light PNG route does not clamp.
function sky_image(; W::Int = 256, H::Int = 128)
    zen, hor, gnd = RGBf(0.30, 0.50, 0.85), RGBf(0.88, 0.92, 0.97), RGBf(0.28, 0.31, 0.34)
    img = Matrix{RGBf}(undef, H, W)
    for r in 1:H
        t = (r - 1) / (H - 1)
        c = t < 0.5 ? _lerp(zen, hor, Float32(t / 0.5)) :
                      _lerp(hor, gnd, clamp(Float32((t - 0.5) / 0.12), 0f0, 1f0))
        for cc in 1:W
            img[r, cc] = c
        end
    end
    return img
end

# Flat textured floor quad (UVs + normals so the checker maps as `st`).
function floor_mesh(half::Float32; z::Float32 = -0.01f0)
    ps  = Point3f[(-half, -half, z), (half, -half, z), (half, half, z), (-half, half, z)]
    uvs = Vec2f[(0, 0), (1, 0), (1, 1), (0, 1)]
    nsv = Vec3f[(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)]
    fcs = [GeometryBasics.GLTriangleFace(1, 2, 3), GeometryBasics.GLTriangleFace(1, 3, 4)]
    return GeometryBasics.Mesh(ps, fcs; uv = uvs, normal = nsv)
end

sphere_mesh(o, r) = GeometryBasics.uv_normal_mesh(
    GeometryBasics.Tesselation(GeometryBasics.Sphere(Point3f(o...), Float32(r)), 48))

"""
    pool_scene(h0; L, depth, canvas) -> (scene, water_plot, mesh_obs, faces)

Build the pool scene: checker floor, three reference spheres (two submerged,
one piercing the surface), sky dome + sun, and the OmniGlass water cube
(IOR 1.33) as a live mesh Observable.
"""
function pool_scene(h0::AbstractMatrix{Float32}; L::Float32 = 2f0,
                    depth::Float32 = 0.6f0, canvas = (720, 460))
    lights = Makie.AbstractLight[
        AmbientLight(RGBf(0.22, 0.24, 0.27)),
        DirectionalLight(RGBf(2.6, 2.5, 2.3), Vec3f(-0.45, -0.35, -0.85), false),
        EnvironmentLight(1.0, sky_image()),
    ]
    scene = Scene(size = canvas; lights)
    cam3d!(scene)
    mesh!(scene, floor_mesh(1.6f0 * L); color = pool_checker_image())
    mesh!(scene, sphere_mesh((-0.36 * L, -0.22 * L, 0.42 * depth), 0.11 * L);
          color = RGBf(0.90, 0.45, 0.10))
    mesh!(scene, sphere_mesh((0.30 * L, 0.18 * L, 0.35 * depth), 0.09 * L);
          color = RGBf(0.75, 0.10, 0.12))
    mesh!(scene, sphere_mesh((0.02 * L, 0.33 * L, depth + 0.02), 0.12 * L);
          color = RGBf(0.10, 0.55, 0.25))
    faces = water_cube_faces(size(h0, 1))
    mobs  = Observable(water_cube_mesh(Matrix{Float32}(h0), faces; L, depth))
    water = mesh!(scene, mobs; color = RGBf(0.80, 0.93, 0.96),
                  material = (; glass = true, ior = 1.33f0))
    return scene, water, mobs, faces
end

# Slow orbit around the pool; `s` ∈ [0, 1] over the clip.
function orbit_cam!(scene, s::Real)
    θ = -2.0f0 + 0.7f0 * Float32(s)
    update_cam!(scene, Vec3f(2.4cos(θ), 2.4sin(θ), 1.45), Vec3f(0, 0, 0.22),
                Vec3f(0, 0, 1))
    return nothing
end

# ---- acceptance ----------------------------------------------------------------

"""
    water_acceptance!(scene, water, set_time!) -> Matrix{RGBA{N0f8}}

Pixel-oracle checks on an explicit Screen: frame non-black, toggling the
water cube's visibility changes the image (refraction/reflection real), and
the surface moves between two sim times.  Returns the t=0 frame.
"""
function water_acceptance!(scene, water, set_time!)
    lum(c) = Float32(red(c)) + Float32(green(c)) + Float32(blue(c))
    screen = OM.Screen(scene)
    set_time!(0.0)
    a = Makie.colorbuffer(screen)
    water.visible = false
    ctrl = Makie.colorbuffer(screen)
    water.visible = true
    set_time!(1.4)
    b = Makie.colorbuffer(screen)
    Base.close(screen)
    nb     = count(c -> lum(c) > 0.05f0, a)
    effect = count(k -> abs(lum(a[k]) - lum(ctrl[k])) > 0.12f0, eachindex(a))
    motion = count(k -> abs(lum(a[k]) - lum(b[k])) > 0.10f0, eachindex(a))
    println("NONBLACK=", nb, "  WATER_EFFECT=", effect, "  WAVE_MOTION=", motion)
    @assert nb > 20_000 "scene rendered near-black (nonblack=$nb)"
    @assert effect > 3_000 "water on/off changes almost nothing (effect=$effect) — cube not rendering as glass?"
    @assert motion > 800 "surface did not move between sim times (motion=$motion)"
    return a
end
