# bench/hot_path.jl
#
# M2.6 hot-path throughput benchmark.
#
# Measures write throughput for the two M2 persistent binding tiers:
#   Path A — write_mapped_xform! : N_XFORM=10,000 omni:xform writes per "virtual frame"
#   Path B — write_binding!      : N_POINTS=100,000 Float32×3 point writes per "virtual frame"
#
# Gate: ≥ 30 Hz for N_XFORM = 10,000 (Path A) OR N_POINTS = 100,000 (Path B).
# If below target: documents the shortfall and escalates GPU-resident writes to M3.
#
# Standalone run (requires OVRTX_LIBRARY_PATH set):
#   OVRTX_LIBRARY_PATH=<path> julia --project=. bench/hot_path.jl
#
# Via test harness: test/m2_bench_test.jl wraps this via run_ovrtx_subprocess.
#
# Output format: KEY=VALUE lines; parsed by the test harness.
# Always exits 0 — shortfalls are documented, not fatal.
# Hardware target: NVIDIA A5000.

using OmniverseMakie
using LibOVRTX
const OV = OmniverseMakie.OV
const L  = LibOVRTX

# ── Parameters ────────────────────────────────────────────────────────────────
const N_XFORM    = 10_000   # write_mapped_xform! calls per "virtual frame"
const M_FRAMES_A = 50       # timed frames (after warmup)
const WARMUP_A   = 10       # discarded warmup frames (JIT + ovrtx settle)

const N_POINTS   = 100_000  # point elements per write_binding! call
const M_FRAMES_B = 20       # timed frames (after warmup)
const WARMUP_B   = 5        # discarded warmup frames

const GATE_HZ    = 30.0     # minimum target rate (Hz) for gate pass

println("BENCH_START")
println("BENCH_N_XFORM=$(N_XFORM) N_POINTS=$(N_POINTS) GATE_HZ=$(GATE_HZ)")

# ── Minimal USDA helper ───────────────────────────────────────────────────────
#
# Build a standalone USDA layer containing one Mesh prim with N point3f[] entries.
# The `points` attribute is pre-authored so the array binding (EXISTING_ONLY mode)
# can attach to it at its current size.
# The `omni:xform` attribute (Path A) does NOT need to be pre-authored: M2.4 proved
# that create_binding EXISTING_ONLY targets the PRIM (not the attribute), and
# map_attribute/write_attribute create the attribute on first write.
#
# Convention (M2 constraints): no upAxis in reference layers.
function _make_mesh_usda(N::Int; prim_name::String = "bench")
    buf = IOBuffer()
    println(buf, "#usda 1.0")
    println(buf, "def Mesh \"$(prim_name)\"")
    println(buf, "{")
    # points array: N entries; values (i.0, 0.0, 0.0) for i=1..N keep format
    # compact (no scientific notation, no floating-point rounding artifacts).
    print(buf, "    point3f[] points = [")
    for i in 1:N
        i > 1 && print(buf, ", ")
        print(buf, "($(i).0, 0.0, 0.0)")
    end
    println(buf, "]")
    # Minimal valid face topology (triangle over the first three vertices).
    println(buf, "    int[] faceVertexCounts = [3]")
    println(buf, "    int[] faceVertexIndices = [0, 1, 2]")
    println(buf, "}")
    return String(take!(buf))
end

# ── Path A: write_mapped_xform! ───────────────────────────────────────────────
#
# Each "frame": N_XFORM calls to write_mapped_xform! through a single persistent
# MAPPED omni:xform binding (OVRTX_BINDING_FLAG_OPTIMIZE, fixed-size tier).
# One binding is benchmarked repeatedly; the cost per call is the C FFI overhead
# (map_attribute + 16 unsafe_store! + unmap_attribute).
# Measures the worst-case serial update rate for a single prim — N prims in
# parallel would require N such calls, so updates/sec = N_XFORM / frame_time
# is the direct gate metric.

println()
println("=== PATH A: write_mapped_xform! (N_XFORM=$(N_XFORM), M_FRAMES=$(M_FRAMES_A)) ===")

path_a_hz = 0.0
path_a_ok = false

r_a = OV.Renderer()
try
    # Minimal 3-point mesh so the prim exists (binding EXISTING_ONLY targets the prim).
    OV.open_usd_string!(r_a, _make_mesh_usda(3; prim_name = "bench_xform"))

    xb = OV.create_binding(
        r_a, "/bench_xform", "omni:xform",
        L.DLDataType(UInt8(L.kDLFloat), UInt8(64), UInt16(16));
        array    = false,
        semantic = L.OVRTX_SEMANTIC_XFORM_MAT4x4,
        optimize = true)

    # Identity matrix reused every call (translation in last row — USD row-vector form).
    mat = Float64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]

    # Warmup: let JIT compile write_mapped_xform! + ovrtx internal paths settle.
    for _ in 1:WARMUP_A
        for _ in 1:N_XFORM
            OV.write_mapped_xform!(xb, mat)
        end
    end

    # Timed measurement.
    t_total = 0.0
    for _ in 1:M_FRAMES_A
        t_total += @elapsed begin
            for _ in 1:N_XFORM
                OV.write_mapped_xform!(xb, mat)
            end
        end
    end

    OV.destroy!(xb)

    t_mean           = t_total / M_FRAMES_A
    global path_a_hz = 1.0 / t_mean
    ups_a            = N_XFORM / t_mean
    global path_a_ok = true

    gate_a_str = path_a_hz >= GATE_HZ ? "PASS" : "BELOW"
    println("PATH_A_FRAME_MS=$(round(t_mean * 1e3, sigdigits = 4))")
    println("PATH_A_UPS=$(round(ups_a, sigdigits = 4))")
    println("PATH_A_HZ=$(round(path_a_hz, sigdigits = 4))")
    println("PATH_A_GATE=$(gate_a_str)")
catch e
    println("PATH_A_ERROR=$(typeof(e)): $(sprint(showerror, e))")
    println("PATH_A_GATE=ERROR")
finally
    close(r_a)
end

# ── Path B: write_binding! (points array) ────────────────────────────────────
#
# Each "frame": one write_binding! call pushing N_POINTS = 100,000 Float32×3
# elements through a persistent array binding (EXISTING_ONLY, no OPTIMIZE flag).
# Exercises the array-tier write path (bind+write), closing the M2.4 coverage gap
# (the array tier was spike-proven but not in the committed benchmark suite).
# Measures frame time and effective update rate (points/second).

println()
println("=== PATH B: write_binding! points (N_POINTS=$(N_POINTS), M_FRAMES=$(M_FRAMES_B)) ===")

path_b_hz = 0.0
path_b_ok = false

r_b = OV.Renderer()
try
    # Generate and open a stage whose `points` attribute has N_POINTS elements.
    # The binding attaches to this pre-existing attribute (EXISTING_ONLY is satisfied).
    t_gen  = @elapsed usda_b = _make_mesh_usda(N_POINTS; prim_name = "bench_pts")
    println("  USDA $(round(length(usda_b) / 1024, sigdigits = 3)) KiB generated in $(round(t_gen, sigdigits = 3)) s")

    t_open = @elapsed OV.open_usd_string!(r_b, usda_b)
    println("  stage opened in $(round(t_open, sigdigits = 3)) s")
    usda_b = nothing; GC.gc()   # free the large string before benchmarking

    pb = OV.create_binding(
        r_b, "/bench_pts", "points",
        L.DLDataType(UInt8(L.kDLFloat), UInt8(32), UInt16(3));
        array    = true,
        semantic = L.OVRTX_SEMANTIC_NONE,
        optimize = false)

    # Pre-allocated flat Float32 buffer: 3 lanes (x,y,z) × N_POINTS elements.
    data_b  = zeros(Float32, 3 * N_POINTS)
    shape_b = Int64[N_POINTS]

    # Warmup.
    for _ in 1:WARMUP_B
        OV.write_binding!(pb, data_b, shape_b)
    end

    # Timed measurement.
    t_total = 0.0
    for i in 1:M_FRAMES_B
        data_b[1] = Float32(i) * 1.0f-6   # minimal perturbation per frame
        t_total += @elapsed OV.write_binding!(pb, data_b, shape_b)
    end

    OV.destroy!(pb)

    t_mean           = t_total / M_FRAMES_B
    global path_b_hz = 1.0 / t_mean
    ups_b            = Float64(N_POINTS) / t_mean
    global path_b_ok = true

    gate_b_str = path_b_hz >= GATE_HZ ? "PASS" : "BELOW"
    println("PATH_B_FRAME_MS=$(round(t_mean * 1e3, sigdigits = 4))")
    println("PATH_B_UPS=$(round(ups_b, sigdigits = 4))")
    println("PATH_B_HZ=$(round(path_b_hz, sigdigits = 4))")
    println("PATH_B_GATE=$(gate_b_str)")
catch e
    println("PATH_B_ERROR=$(typeof(e)): $(sprint(showerror, e))")
    println("PATH_B_GATE=ERROR")
finally
    close(r_b)
end

# ── Gate verdict ──────────────────────────────────────────────────────────────
println()
gate_pass = (path_a_ok && path_a_hz >= GATE_HZ) || (path_b_ok && path_b_hz >= GATE_HZ)
println("GATE_PASS=$(gate_pass)")

if !gate_pass
    println("GATE_BELOW_TARGET=true")
    println("# --- SHORTFALL DOCUMENTED (M2.6) ---")
    println("# Measured rates: Path A $(round(path_a_hz, sigdigits=3)) Hz, Path B $(round(path_b_hz, sigdigits=3)) Hz")
    println("# Target: $(GATE_HZ) Hz for $(N_XFORM) transforms OR $(N_POINTS) points")
    println("# Escalation: GPU-resident DLPack writes → M3")
    println("# Rationale: per-frame CPU→GPU data copy inside write_mapped_xform!/write_binding!")
    println("# is the bottleneck at this scale.  M3 will pin GPU-resident write buffers")
    println("# (DLPack kDLCUDA device type) to eliminate the host→device transfer overhead.")
end

# ── Scatter/MeshScatter gap note ─────────────────────────────────────────────
println()
println("SCATTER_ROUTE_DECISION=b_documented_carry")
println("# Scatter/MeshScatter per-instance 'positions' via UsdGeomPointInstancer is")
println("# a no-op today: push_to_ovrtx! routes :positions_transformed_f32c to the")
println("# 'points' attribute, but the instancer's per-instance attribute is 'positions'.")
println("# Decision (b): benchmark the WORKING paths (Mesh xform + Mesh points), which")
println("# independently cover the gate; the scatter route fix is a documented M3 carry.")
println("# (The whole-scatter omni:xform write still works — only per-instance position")
println("# updates are a no-op.)")

println()
println("BENCH_OK")
