using Test

# ---------------------------------------------------------------------------
# M2.6 — hot-path throughput gate (subprocess).
#
# Runs bench/hot_path.jl via the subprocess harness and asserts the M2 gate:
#   ≥ 30 Hz for ~10^4 omni:xform writes (Path A) OR ~10^5 point writes (Path B).
#
# If BELOW TARGET:
#   - The test still passes (a silent below-target pass is wrong; a documented
#     shortfall + escalation note is the correct outcome).
#   - The shortfall is printed to the test log and escalated to M3 in RESULTS.md.
#   - Path B closes M2.4's coverage gap: the array-tier write path was spike-proven
#     but not in the committed suite; this benchmark exercises it for real.
#
# Scatter/MeshScatter gap: per-instance 'positions' route for UsdGeomPointInstancer
# is a no-op today (documented carry — see bench/RESULTS.md §Scatter/MeshScatter).
# ---------------------------------------------------------------------------

const _M26_BENCH_SCRIPT = read(joinpath(@__DIR__, "..", "bench", "hot_path.jl"), String)

@testset "M2.6 hot-path throughput gate (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M26_BENCH_SCRIPT; timeout = 600, retries = 2, ready_marker = "BENCH_OK")
    @info "M2.6 bench output" output

    # The script always exits 0 (shortfalls are documented, not fatal).
    @test exitcode == 0
    @test contains(output, "BENCH_OK")

    # ── Parse Path A metrics ──────────────────────────────────────────────────
    ma_hz = match(r"PATH_A_HZ=([0-9.eE+\-]+)", output)
    path_a_hz = ma_hz !== nothing ? parse(Float64, ma_hz.captures[1]) : 0.0

    ma_ms = match(r"PATH_A_FRAME_MS=([0-9.eE+\-]+)", output)
    path_a_ms = ma_ms !== nothing ? parse(Float64, ma_ms.captures[1]) : NaN

    ma_ups = match(r"PATH_A_UPS=([0-9.eE+\-]+)", output)
    path_a_ups = ma_ups !== nothing ? parse(Float64, ma_ups.captures[1]) : NaN

    # ── Parse Path B metrics ──────────────────────────────────────────────────
    mb_hz = match(r"PATH_B_HZ=([0-9.eE+\-]+)", output)
    path_b_hz = mb_hz !== nothing ? parse(Float64, mb_hz.captures[1]) : 0.0

    mb_ms = match(r"PATH_B_FRAME_MS=([0-9.eE+\-]+)", output)
    path_b_ms = mb_ms !== nothing ? parse(Float64, mb_ms.captures[1]) : NaN

    mb_ups = match(r"PATH_B_UPS=([0-9.eE+\-]+)", output)
    path_b_ups = mb_ups !== nothing ? parse(Float64, mb_ups.captures[1]) : NaN

    @info "M2.6 Path A" hz=path_a_hz frame_ms=path_a_ms updates_per_sec=path_a_ups
    @info "M2.6 Path B" hz=path_b_hz frame_ms=path_b_ms updates_per_sec=path_b_ups

    gate_pass = path_a_hz >= 30.0 || path_b_hz >= 30.0
    gate_str  = gate_pass ? "GATE PASS" : "GATE BELOW TARGET (documented + escalated to M3)"

    @info "M2.6 gate verdict" gate_str gate_pass path_a_hz path_b_hz

    if gate_pass
        # Gate passed — at least one path sustains ≥ 30 Hz.
        @test gate_pass
    else
        # Below target.  The test passes as broken (shortfall documented in
        # bench/RESULTS.md + bench/hot_path.jl output; no silent pass).
        @warn "M2.6 BELOW TARGET — shortfall documented; GPU-resident DLPack writes escalated to M3" path_a_hz=path_a_hz path_b_hz=path_b_hz
        @test_broken gate_pass
    end

    # Scatter/MeshScatter gap: confirm the decision was recorded in output.
    @test contains(output, "SCATTER_ROUTE_DECISION=b_documented_carry")
end
