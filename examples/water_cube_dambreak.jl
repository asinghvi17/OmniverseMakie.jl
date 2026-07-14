# Water-cube showcase, phase 2 — a dam-break shallow-water simulation
# (h, hu, hv; local Lax–Friedrichs finite volume; reflective walls) drives
# the top surface of the OmniGlass water cube: a raised water column
# collapses and sloshes around the pool.  Same harness + scene as
# examples/water_cube_gerstner.jl (see
# docs/superpowers/specs/2026-07-13-water-cube-example-design.md).
#
# Run it (GPU; serialize on the shared lock as every ovrtx job does):
#   flock -w 3600 /tmp/omniversemakie-gpu.lock -c \
#     'OVRTX_LIBRARY_PATH=<…>/libovrtx-dynamic.so JULIA_CUDA_USE_COMPAT=false \
#      julia --project=examples examples/water_cube_dambreak.jl'
#
# Env overrides: WATER_MP4 (output path), WATER_FRAMES, WATER_N (grid).

using OmniverseMakie, GeometryBasics, ColorTypes, FixedPointNumbers, FileIO
import OmniverseMakie as OM
include(joinpath(@__DIR__, "common", "water_cube.jl"))

const MP4    = get(ENV, "WATER_MP4", joinpath(tempdir(), "water_cube_dambreak.mp4"))
const FRAMES = parse(Int, get(ENV, "WATER_FRAMES", "120"))
const N      = parse(Int, get(ENV, "WATER_N", "128"))
const L      = 2.0f0    # pool footprint (m)
const DEPTH  = 0.6f0    # still-water depth (m)
const FPS    = 24
const GRAV   = 9.81f0
const CFL    = 0.4f0

# Default per-frame reconverge → crisp, ghost-free frames of the moving
# surface (accumulate_across_frames would motion-blur it).
OM.activate!(warmup = 48, samples = 256, max_bounces = 6)

# ---- shallow-water solver ------------------------------------------------------
# State on the N×N mesh nodes (cell size dx = L/(N-1)): total depth `H` and
# momenta `HU`, `HV`.  One first-order LLF step; walls reflect (mirror state,
# negated normal momentum).

@inline _flux_x(h, hu, hv) = (hu, hu * hu / h + 0.5f0 * GRAV * h * h, hu * hv / h)
@inline _flux_y(h, hu, hv) = (hv, hu * hv / h, hv * hv / h + 0.5f0 * GRAV * h * h)

# LLF interface flux from left/right (or bottom/top) states, component `fl`.
@inline function _llf(fL::NTuple{3,Float32}, fR::NTuple{3,Float32},
                      uL::NTuple{3,Float32}, uR::NTuple{3,Float32}, α::Float32)
    return (0.5f0 * (fL[1] + fR[1]) - 0.5f0 * α * (uR[1] - uL[1]),
            0.5f0 * (fL[2] + fR[2]) - 0.5f0 * α * (uR[2] - uL[2]),
            0.5f0 * (fL[3] + fR[3]) - 0.5f0 * α * (uR[3] - uL[3]))
end

# Mirror ghost state for a reflective wall; `nx, ny` picks the normal axis.
@inline _reflect(u::NTuple{3,Float32}, nx::Bool) =
    nx ? (u[1], -u[2], u[3]) : (u[1], u[2], -u[3])

_wavespeed(u::NTuple{3,Float32}, nx::Bool) =
    abs((nx ? u[2] : u[3]) / u[1]) + sqrt(GRAV * u[1])

function swe_step!(H, HU, HV, Hn, HUn, HVn, dt::Float32, dx::Float32)
    n = size(H, 1)
    @inbounds for j in 1:n, i in 1:n
        c = (H[i, j], HU[i, j], HV[i, j])
        acc = (0f0, 0f0, 0f0)
        # x-faces: (i-1|i) and (i|i+1); y-faces: (j-1|j) and (j|j+1)
        for (di, dj, nx, sgn) in ((-1, 0, true, -1f0), (1, 0, true, 1f0),
                                  (0, -1, false, -1f0), (0, 1, false, 1f0))
            ii, jj = i + di, j + dj
            nb = (1 <= ii <= n && 1 <= jj <= n) ?
                 (H[ii, jj], HU[ii, jj], HV[ii, jj]) : _reflect(c, nx)
            uL, uR = sgn < 0 ? (nb, c) : (c, nb)
            α  = max(_wavespeed(uL, nx), _wavespeed(uR, nx))
            fL = nx ? _flux_x(uL...) : _flux_y(uL...)
            fR = nx ? _flux_x(uR...) : _flux_y(uR...)
            f  = _llf(fL, fR, uL, uR, α)
            acc = (acc[1] - sgn * f[1], acc[2] - sgn * f[2], acc[3] - sgn * f[3])
        end
        Hn[i, j]  = H[i, j]  + dt / dx * acc[1]
        HUn[i, j] = HU[i, j] + dt / dx * acc[2]
        HVn[i, j] = HV[i, j] + dt / dx * acc[3]
    end
    return nothing
end

max_wavespeed(H, HU, HV) = maximum(
    max(abs(HU[k] / H[k]), abs(HV[k] / H[k])) + sqrt(GRAV * H[k]) for k in eachindex(H))

"""
    SWEState(N)

Dam-break initial condition: still water of depth `DEPTH` plus a raised
column (`+0.22` m) inside a circle at (-0.22L, -0.18L), its edge smoothed
over ~0.05L (a hard jump renders as a picket-fence cliff on the node grid);
zero momentum.
"""
mutable struct SWEState
    H::Matrix{Float32}
    HU::Matrix{Float32}
    HV::Matrix{Float32}
    Hn::Matrix{Float32}
    HUn::Matrix{Float32}
    HVn::Matrix{Float32}
    t::Float32
end
function SWEState(n::Int)
    xs = range(-L / 2, L / 2; length = n)
    H = Matrix{Float32}(undef, n, n)
    for j in 1:n, i in 1:n
        r = sqrt((xs[i] + 0.22f0 * L)^2 + (xs[j] + 0.18f0 * L)^2)
        H[i, j] = DEPTH + 0.11f0 * (1f0 - tanh((r - 0.28f0 * L) / (0.05f0 * L)))
    end
    return SWEState(H, zeros(Float32, n, n), zeros(Float32, n, n),
                    similar(H), similar(H), similar(H), 0f0)
end

# Advance to absolute sim time `t` with CFL-bounded substeps; re-initialize
# when asked for an earlier time (the sim is stateful).  ALL time arithmetic
# is Float32: a Float64 target that sits strictly above its Float32 rounding
# makes `st.t < t` true forever at dt = 0 (a silent infinite loop).
# NaN/dry-out/no-progress abort loudly.
function advance!(st::SWEState, t::Real)
    target = Float32(t)
    target < st.t && (fresh = SWEState(size(st.H, 1));
                      st.H = fresh.H; st.HU = fresh.HU; st.HV = fresh.HV; st.t = 0f0)
    dx = L / (size(st.H, 1) - 1)
    while st.t < target
        dt = min(CFL * dx / max_wavespeed(st.H, st.HU, st.HV), target - st.t)
        swe_step!(st.H, st.HU, st.HV, st.Hn, st.HUn, st.HVn, dt, dx)
        st.H, st.Hn = st.Hn, st.H
        st.HU, st.HUn = st.HUn, st.HU
        st.HV, st.HVn = st.HVn, st.HV
        new_t = st.t + dt
        new_t > st.t || error("shallow-water solve stopped advancing at t=$(st.t) (dt=$dt)")
        st.t = new_t
        all(isfinite, st.H) || error("shallow-water solve produced NaN/Inf at t=$(st.t)")
        minimum(st.H) > 0f0 || error("shallow-water solve dried out at t=$(st.t) (min H ≤ 0)")
    end
    return st
end

# ---- scene + record ------------------------------------------------------------
state = SWEState(N)
h = Matrix{Float32}(state.H .- DEPTH)
scene, water, mobs, faces = pool_scene(h; L, depth = DEPTH)
function set_time!(t)
    advance!(state, t)
    h .= state.H .- DEPTH
    mobs[] = water_cube_mesh(h, faces; L, depth = DEPTH)
    return nothing
end

# acceptance + preview still first (fails fast; its Screen closes before
# record opens its own)
orbit_cam!(scene, 0.0)
still = water_acceptance!(scene, water, set_time!)
png = joinpath(@__DIR__, "renders", "water_cube_dambreak.png")
FileIO.save(png, still)
println("PNG=", png)

set_time!(0.0)   # restart the collapse for the clip
Makie.record(scene, MP4, 1:FRAMES; framerate = FPS) do i
    set_time!((i - 1) / FPS)
    orbit_cam!(scene, (i - 1) / max(FRAMES - 1, 1))
end
@assert isfile(MP4) "record wrote no mp4"
sz = filesize(MP4)
println("MP4=", MP4, "  BYTES=", sz)
@assert sz > 50_000 "mp4 suspiciously small ($sz bytes)"
println("OK_WATER_CUBE_DAMBREAK")
