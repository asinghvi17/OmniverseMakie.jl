# usdplot showcase — a real vendor asset (the NVIDIA Kit "Zeus ZS300" sedan) placed into a Makie
# scene through the ovrtx path tracer, with all four wheels spun live via `bind_usd!`, recorded to
# an .mp4.
#
# This is the acceptance demo for the usdplot recipe (docs/superpowers/specs/2026-07-02-usdplot-
# recipe-design.md).  It exercises the whole composition stack in one asset: a crate `.usd`, a
# payload-composed model, RELATIVE texture paths (resolved from the file's own directory), and
# self-contained MDL materials under /World/Looks — none of which a `bind_usd!` recipe parses.
#
# Run it (GPU; serialize on the shared lock as every ovrtx job does):
#   flock -w 3600 /tmp/omniversemakie-gpu.lock -c \
#     'OVRTX_LIBRARY_PATH=<…>/libovrtx-dynamic.so JULIA_CUDA_USE_COMPAT=false \
#      julia --project=. examples/usdplot_zeus_wheels.jl'
#
# Env overrides: ZEUS_USD (asset path), ZEUS_MP4 (output path), ZEUS_FRAMES (frame count).

using OmniverseMakie, ColorTypes, FixedPointNumbers
const OM = OmniverseMakie
using Makie: Scene, Vec3f, Point3f, Rect3f, RGBf, Observable, AbstractLight, AmbientLight,
             DirectionalLight, cam3d!, update_cam!, translationmatrix, scalematrix,
             rotationmatrix_x, rotationmatrix_z

const CAR = get(ENV, "ZEUS_USD",
    "/home/juliahub/temp/dsx-content/DSX_BP_/DSX_BP/Library/Assets/Collected_assembly_Site/" *
    "art.ov.nvidia.com/Projects/GDC-GTC/2025/GTC25_Aurora/Props/Terraform/VEHICLES/Zeus_zs300/" *
    "sm_ZeusZS300_a1_1.usd")
const MP4    = get(ENV, "ZEUS_MP4", joinpath(tempdir(), "zeus_wheels.mp4"))
const FRAMES = parse(Int, get(ENV, "ZEUS_FRAMES", "72"))
isfile(CAR) || error("Zeus asset not found: $CAR  (set ZEUS_USD to the Kit assets download).")

OM.activate!(warmup = 48, samples = 256)

# --- wheel spin -------------------------------------------------------------------------------
# Each wheel Xform is authored `translate · rotateXYZ(90,0,0) · scale(0.01)` (values dumped offline
# with pxr — the recipe never parses USD).  `bind_usd!` writes `omni:xform`, which REPLACES the
# prim's whole local transform, so to spin the wheel in place we reconstruct the authored linear
# part (Rx90·S) and insert a spin about the mesh axle (mesh Z; the authored rotateXYZ(90) is what
# orients that axle to the car's roll axis):  new_local = T · Rx90 · Rz(θ) · S.
const RX90 = rotationmatrix_x(Float32(pi / 2))
const SCL  = scalematrix(Vec3f(0.01, 0.01, 0.01))
wheel_mat(t, θ) = translationmatrix(t) * RX90 * rotationmatrix_z(Float32(θ)) * SCL

# (bind target relative to the file's defaultPrim /World) => authored translate
const WHEELS = [
    "/SM_ZeusZS300_A1_1/SM_ZeusZS300_WheelFrontRight_A1_1" => Vec3f( 1.5456491, -0.8227462, 0.3527515),
    "/SM_ZeusZS300_A1_1/SM_ZeusZS300_WheelFrontLeft_A1_1"  => Vec3f( 1.5455345,  0.8227460, 0.3528402),
    "/SM_ZeusZS300_A1_1/SM_ZeusZS300_WheelRearLeft_A1_1"   => Vec3f(-1.4424653,  0.8227460, 0.3528400),
    "/SM_ZeusZS300_A1_1/SM_ZeusZS300_WheelRearRight_A1_1"  => Vec3f(-1.4424653, -0.8227462, 0.3528397),
]

# --- scene ------------------------------------------------------------------------------------
# The car is authored Z-up in centimetres (upAxis Z, metersPerUnit 0.01), so the scene's default
# up = :z needs no correction; bbox is the ~5.2 m car's world bounds (cm) for camera framing.
lights = AbstractLight[
    AmbientLight(RGBf(0.65, 0.65, 0.70)),
    DirectionalLight(RGBf(2.6, 2.5, 2.4), Vec3f(-0.3, -0.4, -1.0), true),   # camera-relative key
]
scene = Scene(size = (720, 460); lights = lights); cam3d!(scene)
update_cam!(scene, Vec3f(255, -470, 95), Vec3f(-10, 0, 42), Vec3f(0, 0, 1))  # low side-3/4 view
p = usdplot!(scene, CAR; up = :z, bbox = Rect3f(Point3f(-260, -105, 0), Vec3f(520, 210, 150)))

# One Observable per wheel; bind_usd! probes each target (fail-fast on a wrong path).
spins = Dict(path => Observable(wheel_mat(t, 0.0f0)) for (path, t) in WHEELS)
for (path, t) in WHEELS
    bind_usd!(p, path, spins[path])
end

# --- record -----------------------------------------------------------------------------------
# Default (per-frame reconverge) mode → crisp, ghost-free frames at each wheel angle (a fast spin
# in accumulate_across_frames mode would motion-blur the rim; that mode is for realtime preview).
# Two full wheel rotations over the clip; the callback only sets observables (record grabs each
# frame via colorbuffer internally).
Makie.record(scene, MP4, 1:FRAMES; framerate = 24) do i
    θ = Float32(4π * (i - 1) / FRAMES)
    for (path, t) in WHEELS
        spins[path][] = wheel_mat(t, θ)
    end
end
@assert isfile(MP4) "record wrote no mp4"
sz = filesize(MP4)

# --- acceptance: non-black + frame-over-frame wheel motion (explicit screen, two wheel angles) ---
lum(c) = Float32(red(c)) + Float32(green(c)) + Float32(blue(c))
screen = OM.Screen(scene)
for (path, t) in WHEELS; spins[path][] = wheel_mat(t, 0.0f0); end
a = Makie.colorbuffer(screen)
for (path, t) in WHEELS; spins[path][] = wheel_mat(t, 2.5f0); end
b = Makie.colorbuffer(screen)
close(screen)
nb     = count(c -> lum(c) > 0.05f0, a)
motion = count(k -> abs(lum(a[k]) - lum(b[k])) > 0.08f0, eachindex(a))

println("FRAME_NONBLACK=", nb, "  WHEEL_MOTION_PIXELS=", motion)
println("MP4=", MP4, "  BYTES=", sz)
@assert sz > 20_000 "mp4 suspiciously small ($sz bytes)"
@assert nb > 5_000 "car did not render (nonblack=$nb)"
@assert motion > 500 "wheels did not visibly move between angles ($motion px)"
println("OK_ZEUS_WHEELS")
