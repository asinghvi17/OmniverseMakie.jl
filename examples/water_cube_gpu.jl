# Water-cube showcase, GPU-direct — the Gerstner-style wave field is
# computed by a CUDA kernel and pushed into the renderer with
# `gpu_update_mesh!` (kDLCUDA device tensors through the persistent ovrtx
# bindings): the per-frame hot path never touches host memory, and Makie's
# CPU-side mesh Observable is never written after author time.  Scene and
# geometry are examples/common/water_cube.jl, matching the CPU twin
# examples/water_cube_gerstner.jl.
#
# The frame loop drives an explicit `OM.Screen` (recording via colorbuffer +
# FFMPEG_jll rather than `Makie.record`, whose own Screen would not see the
# device writes routed at this one).
#
# Run it (GPU; serialize on the shared lock as every ovrtx job does):
#   flock -w 3600 /tmp/omniversemakie-gpu.lock -c \
#     'OVRTX_LIBRARY_PATH=<…>/libovrtx-dynamic.so JULIA_CUDA_USE_COMPAT=false \
#      julia --project=examples examples/water_cube_gpu.jl'
#
# Env overrides: WATER_MP4 (output path), WATER_FRAMES, WATER_N (grid).

using OmniverseMakie, GeometryBasics, ColorTypes, FixedPointNumbers, FileIO, CUDA
import OmniverseMakie as OM
import FFMPEG_jll
include(joinpath(@__DIR__, "common", "water_cube.jl"))

const MP4    = get(ENV, "WATER_MP4", joinpath(tempdir(), "water_cube_gpu.mp4"))
const FRAMES = parse(Int, get(ENV, "WATER_FRAMES", "96"))
const N      = parse(Int, get(ENV, "WATER_N", "128"))
const L      = 2.0f0    # pool footprint (m)
const DEPTH  = 0.6f0    # still-water depth (m)
const FPS    = 24

OM.activate!(warmup = 48, samples = 256, max_bounces = 6)

# Same wave set as the CPU twin, precomputed to per-wave (A, kx, ky, ω, φ)
# kernel constants (deep-water dispersion ω = k·√(gλ/2π)).
const WAVES = map(((A, λ, dir, φ),) -> (A, 2f0π / λ * cos(dir), 2f0π / λ * sin(dir),
                                        2f0π / λ * sqrt(9.81f0 * λ / (2f0π)), φ),
                  ((0.034f0, 1.10f0, 0.4f0, 0.0f0),
                   (0.021f0, 0.65f0, 2.3f0, 1.7f0),
                   (0.013f0, 0.38f0, 3.9f0, 3.9f0),
                   (0.008f0, 0.22f0, 5.5f0, 2.6f0)))

# One thread per top-grid node: analytic height → top vertex z, plus the
# four wall top rings (vertex layout: see water_cube.jl).  No normals: ovrtx
# shades from geometry-derived normals; normals writes are pixel-inert
# (tripwired in test/live/gpu_direct_test.jl).
function _waves_kernel!(pos, n, l, depth, t, waves)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idx > n * n && return nothing
    i = (idx - 1) % n + 1
    j = (idx - 1) ÷ n + 1
    dxg = l / (n - 1)
    x = -l / 2 + (i - 1) * dxg
    y = -l / 2 + (j - 1) * dxg
    h = 0f0
    for (A, kx, ky, ω, φ) in waves
        h += A * sin(kx * x + ky * y - ω * t + φ)
    end
    z = depth + h
    @inbounds begin
        pos[3 * ((j - 1) * n + i)] = z
        n2 = n * n
        j == 1 && (pos[3 * (2n2 + n + i)]  = z)   # wall y- top ring
        j == n && (pos[3 * (2n2 + 3n + i)] = z)   # wall y+
        i == 1 && (pos[3 * (2n2 + 5n + j)] = z)   # wall x-
        i == n && (pos[3 * (2n2 + 7n + j)] = z)   # wall x+
    end
    return nothing
end

# ---- scene + device state ------------------------------------------------------
h0 = zeros(Float32, N, N)
scene, water, mobs, faces = pool_scene(h0; L, depth = DEPTH)
mesh0 = mobs[]                                   # Makie-side mesh, frozen from here on

screen = OM.Screen(scene)
orbit_cam!(scene, 0.0)
Makie.colorbuffer(screen)   # author the stage — gpu_update_mesh! needs the plot's binding

cupos = CuArray(collect(reinterpret(Float32, water_cube_positions(h0; L, depth = DEPTH))))
const _THREADS = 256

function set_time!(t)
    @cuda threads = _THREADS blocks = cld(N * N, _THREADS) _waves_kernel!(
        cupos, N, L, DEPTH, Float32(t), WAVES)
    OM.gpu_update_mesh!(screen, water; points = cupos)
    return nothing
end

# ---- acceptance (same oracle as the harness, on the explicit screen) -----------
lum(c) = Float32(red(c)) + Float32(green(c)) + Float32(blue(c))
set_time!(0.0)
a = Makie.colorbuffer(screen)
water.visible = false
ctrl = Makie.colorbuffer(screen)
water.visible = true
set_time!(1.4)
b = Makie.colorbuffer(screen)
nb     = count(c -> lum(c) > 0.05f0, a)
effect = count(k -> abs(lum(a[k]) - lum(ctrl[k])) > 0.12f0, eachindex(a))
motion = count(k -> abs(lum(a[k]) - lum(b[k])) > 0.10f0, eachindex(a))
println("NONBLACK=", nb, "  WATER_EFFECT=", effect, "  WAVE_MOTION=", motion)
@assert nb > 20_000 "scene rendered near-black (nonblack=$nb)"
@assert effect > 3_000 "water on/off changes almost nothing (effect=$effect)"
@assert motion > 800 "surface did not move between sim times (motion=$motion)"
@assert mobs[] === mesh0 "Makie-side mesh Observable was touched — GPU path not side-channel"
FileIO.save(joinpath(@__DIR__, "renders", "water_cube_gpu.png"), a)

# ---- record (explicit screen → PNG frames → ffmpeg) ----------------------------
framedir = mktempdir()
t_sim = 0.0; t_render = 0.0
for i in 1:FRAMES
    global t_sim, t_render
    t_sim += @elapsed begin
        set_time!((i - 1) / FPS)
    end
    orbit_cam!(scene, (i - 1) / max(FRAMES - 1, 1))
    t_render += @elapsed begin
        img = Makie.colorbuffer(screen)
        FileIO.save(joinpath(framedir, "frame_$(lpad(i, 4, '0')).png"), img)
    end
end
close(screen)
println("SIM+PUSH_MS_PER_FRAME=", round(1000t_sim / FRAMES; digits = 2),
        "  RENDER_MS_PER_FRAME=", round(1000t_render / FRAMES; digits = 1))
run(`$(FFMPEG_jll.ffmpeg()) -y -loglevel error -framerate $(FPS)
     -i $(joinpath(framedir, "frame_%04d.png")) -pix_fmt yuv420p $(MP4)`)
@assert isfile(MP4) "ffmpeg wrote no mp4"
sz = filesize(MP4)
println("MP4=", MP4, "  BYTES=", sz)
@assert sz > 50_000 "mp4 suspiciously small ($sz bytes)"
println("OK_WATER_CUBE_GPU")
