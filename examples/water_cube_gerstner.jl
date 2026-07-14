# Water-cube showcase, phase 1 — four directional sine waves (deep-water
# dispersion) animate the top surface of a watertight OmniGlass cube of
# water (IOR 1.33) over a checkerboard pool with reference spheres and a sky
# dome; recorded to an .mp4.  Phase 2 swaps the driver for a dam-break
# shallow-water solve on the same harness (see
# docs/superpowers/specs/2026-07-13-water-cube-example-design.md).
#
# Run it (GPU; serialize on the shared lock as every ovrtx job does):
#   flock -w 3600 /tmp/omniversemakie-gpu.lock -c \
#     'OVRTX_LIBRARY_PATH=<…>/libovrtx-dynamic.so JULIA_CUDA_USE_COMPAT=false \
#      julia --project=examples examples/water_cube_gerstner.jl'
#
# Env overrides: WATER_MP4 (output path), WATER_FRAMES, WATER_N (grid).

using OmniverseMakie, GeometryBasics, ColorTypes, FixedPointNumbers, FileIO
import OmniverseMakie as OM
include(joinpath(@__DIR__, "common", "water_cube.jl"))

const MP4    = get(ENV, "WATER_MP4", joinpath(tempdir(), "water_cube_gerstner.mp4"))
const FRAMES = parse(Int, get(ENV, "WATER_FRAMES", "96"))
const N      = parse(Int, get(ENV, "WATER_N", "128"))
const L      = 2.0f0    # pool footprint (m)
const DEPTH  = 0.6f0    # still-water depth (m)
const FPS    = 24

# Default per-frame reconverge → crisp, ghost-free frames of the moving
# surface (accumulate_across_frames would motion-blur it).
OM.activate!(warmup = 48, samples = 256, max_bounces = 6)

# (amplitude m, wavelength m, heading rad, phase rad); each component moves
# at its deep-water phase speed √(gλ/2π).
const WAVES = [
    (0.034f0, 1.10f0, 0.4f0, 0.0f0),
    (0.021f0, 0.65f0, 2.3f0, 1.7f0),
    (0.013f0, 0.38f0, 3.9f0, 3.9f0),
    (0.008f0, 0.22f0, 5.5f0, 2.6f0),
]

function heights!(h::Matrix{Float32}, t::Real)
    n  = size(h, 1)
    xs = range(-L / 2, L / 2; length = n)
    fill!(h, 0f0)
    for (A, λ, dir, φ) in WAVES
        kx, ky = 2f0π / λ * cos(dir), 2f0π / λ * sin(dir)
        ω = 2f0π / λ * sqrt(9.81f0 * λ / (2f0π))
        for j in 1:n, i in 1:n
            h[i, j] += A * sin(kx * xs[i] + ky * xs[j] - ω * Float32(t) + φ)
        end
    end
    return h
end

h = zeros(Float32, N, N)
heights!(h, 0.0)
scene, water, mobs, faces = pool_scene(h; L, depth = DEPTH)
set_time!(t) = (heights!(h, t); mobs[] = water_cube_mesh(h, faces; L, depth = DEPTH); nothing)

# --- acceptance + preview still (before the expensive record: fails fast, and
# --- its Screen is closed again so record's own screen is the only live one) ---
orbit_cam!(scene, 0.0)
still = water_acceptance!(scene, water, set_time!)
png = joinpath(@__DIR__, "renders", "water_cube_gerstner.png")
FileIO.save(png, still)
println("PNG=", png)

# --- record -------------------------------------------------------------------
Makie.record(scene, MP4, 1:FRAMES; framerate = FPS) do i
    set_time!((i - 1) / FPS)
    orbit_cam!(scene, (i - 1) / max(FRAMES - 1, 1))
end
@assert isfile(MP4) "record wrote no mp4"
sz = filesize(MP4)
println("MP4=", MP4, "  BYTES=", sz)
@assert sz > 50_000 "mp4 suspiciously small ($sz bytes)"
println("OK_WATER_CUBE_GERSTNER")
