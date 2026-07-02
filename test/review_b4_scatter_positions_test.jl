using Test

# ---------------------------------------------------------------------------
# Review Track B / Task B4 — Scatter/MeshScatter live positions (subprocess, skip-if-absent).
#
# A NON-materialized Scatter/MeshScatter authors a UsdGeomPointInstancer, whose per-instance
# attribute is `positions` — NOT `points`.  `push_to_ovrtx!` used to route
# `:positions_transformed_f32c` to a `points` write on EVERY plot type, which ovrtx SILENTLY
# DROPS on an instancer (writes to a nonexistent attr → num_error_ops=0): a live per-instance
# move was an invisible no-op that still burned an accumulation reset (bench/RESULTS.md carry).
#
# B4 (spike-verified — pixel-centroid oracle; absence of error proves nothing on this ovrtx
# build): a one-shot AND a persistent-binding `positions` write BOTH move an authored instancer,
# so non-materialized Scatter/MeshScatter now route to `positions` (zero-copy binding when the
# instance count is unchanged, one-shot resize otherwise — the B2 frozen-size gate).  A
# MATERIALIZED scatter/meshscatter is a merged tessellated UsdGeomMesh (n×~150 verts): a
# positions-sized write there is size-mismatched corruption, so a live position edit is
# `@warn maxlog=1` + `routed=false` (NO write, NO reset burn) — it needs a re-author.
#
# TEST 1 (non-materialized, same prog for scatter + meshscatter): a LEFT→RIGHT position edit
#   MOVES the lit centroid (RED pre-B4: the `points` write is dropped → centroid unmoved), the
#   push ROUTED, and the stage was NOT re-authored (in-place `positions` write).
# TEST 2 (materialized): a live position edit leaves the image UNCHANGED, warns exactly once,
#   the observer shows the push was NOT routed, and no RT2 reset is burned (`_sync_and_needs_reset!`
#   returns false for the skip pull).
# ---------------------------------------------------------------------------

const _B4_MOVE_PROG = raw"""
using OmniverseMakie, ColorTypes
import OmniverseMakie as OM

OM.activate!(warmup = 40)

function stats(img)
    H, W = size(img); lit = 0; sr = 0.0; sc = 0.0
    for h in 1:H, w in 1:W
        c = img[h, w]
        if Float32(red(c)) + Float32(green(c)) + Float32(blue(c)) > 0.10
            lit += 1; sr += h; sc += w
        end
    end
    return (lit = lit, row = lit > 0 ? sr / lit : -1.0, col = lit > 0 ? sc / lit : -1.0)
end

# Author a LEFT cluster on an instancer, then live-edit positions to a RIGHT cluster: the lit
# centroid column must MOVE right.  Assert the push ROUTED (observer) and the stage was NOT
# re-authored (a live in-place `positions` write, _ROOT_OPEN_COUNT unchanged over the edit).
function run_move(label, mk!)
    scene = Scene(size = (240, 240)); cam3d!(scene)
    update_cam!(scene, Vec3f(0, 0, 14), Vec3f(0, 0, 0), Vec3f(0, 1, 0))
    p0 = [Point3f(-3, -1.5, 0), Point3f(-3, 1.5, 0), Point3f(-3.6, 0, 0)]     # LEFT
    plt = mk!(scene, p0)
    screen = OM.Screen(scene)
    s0 = stats(Makie.colorbuffer(screen))
    println(label, "_S0=", s0)

    pushed = Symbol[]
    OM._PUSH_OBSERVER[] = name -> push!(pushed, name)
    opens_before = OM._ROOT_OPEN_COUNT[]
    p1 = [Point3f(3, -1.5, 0), Point3f(3, 1.5, 0), Point3f(3.6, 0, 0)]        # RIGHT
    plt[1][] = p1
    s1 = stats(Makie.colorbuffer(screen))
    OM._PUSH_OBSERVER[] = nothing
    reopened = OM._ROOT_OPEN_COUNT[] - opens_before
    delta = s1.col - s0.col

    println(label, "_S1=", s1)
    println(label, "_PUSHED=", pushed)
    println(label, "_DELTA=", round(delta; digits = 2))
    println(label, "_REOPENED=", reopened)
    println(label, "_MOVED=", delta > 20 && s0.lit > 150 && s1.lit > 150)
    println(label, "_ROUTED=", :positions_transformed_f32c in pushed)
    close(screen)
end

run_move("SCATTER",     (sc, p) -> scatter!(sc, p; markersize = 1.2, color = :red))
run_move("MESHSCATTER", (sc, p) -> meshscatter!(sc, p; markersize = 1.0, color = :orange))
println("OK_SCATTER_MOVE")
"""

const _B4_MATERIALIZED_PROG = raw"""
using OmniverseMakie, ColorTypes
import OmniverseMakie as OM
import Logging

OM.activate!(warmup = 40)

function changed(x, y; thr = 0.15)
    n = 0
    @inbounds for i in eachindex(x, y)
        d = abs(Float32(red(x[i]))   - Float32(red(y[i]))) +
            abs(Float32(green(x[i])) - Float32(green(y[i]))) +
            abs(Float32(blue(x[i]))  - Float32(blue(y[i])))
        d > thr && (n += 1)
    end
    n
end
nonblack(img) = count(c -> (Float32(red(c)) + Float32(green(c)) + Float32(blue(c))) > 0.05, img)

scene = Scene(size = (240, 240)); cam3d!(scene)
update_cam!(scene, Vec3f(0, 0, 14), Vec3f(0, 0, 0), Vec3f(0, 1, 0))
p0 = [Point3f(-3, -1.5, 0), Point3f(-3, 1.5, 0), Point3f(-3.6, 0, 0)]
# MATERIALIZED scatter → merged tessellated UsdGeomMesh (NOT a PointInstancer).
sp = scatter!(scene, p0; markersize = 1.2, color = :red, material = (; metallic = 0.0, roughness = 0.6))
screen = OM.Screen(scene)

imgA = Makie.colorbuffer(screen)
robj = screen.plot2robj[objectid(sp)]
println("MATERIAL_SHADER_SET=", robj.material_shader !== nothing)   # authored as the merged materialized mesh
println("NONBLACK_A=", nonblack(imgA))

cam_scene = something(OM._scene_for_camera(scene), scene)
pushed = Symbol[]
OM._PUSH_OBSERVER[] = name -> push!(pushed, name)

# Two position edits under ONE logger (maxlog state is per-logger: a fresh SimpleLogger per edit
# would reset it), each pulled via _sync_and_needs_reset! → two skip attempts → maxlog=1 logs
# exactly ONE warn.  A skip writes nothing → _sync_and_needs_reset! must return FALSE (no reset burn).
buf = IOBuffer()
p1 = [Point3f(3, -1.5, 0), Point3f(3, 1.5, 0), Point3f(3.6, 0, 0)]
p2 = [Point3f(0, -4, 0), Point3f(0, 4, 0), Point3f(0.5, 0, 0)]
need1, need2 = Logging.with_logger(Logging.SimpleLogger(buf, Logging.Debug)) do
    sp[1][] = p1
    n1 = OM._sync_and_needs_reset!(screen, cam_scene)
    sp[1][] = p2
    n2 = OM._sync_and_needs_reset!(screen, cam_scene)
    (n1, n2)
end
logtxt = String(take!(buf))
OM._PUSH_OBSERVER[] = nothing

imgB = Makie.colorbuffer(screen)
diffAB = changed(imgA, imgB)

println("PUSHED=", pushed)
println("POS_NOT_ROUTED=", !(:positions_transformed_f32c in pushed))
println("WARN_COUNT=", count(r"live position edits on a materialized scatter", logtxt))
println("NEED1=", need1)
println("NEED2=", need2)
println("CHANGED_A_vs_B=", diffAB)
println("IMAGE_UNCHANGED=", diffAB < 100)
println("ROOT_OPENS=", OM._ROOT_OPEN_COUNT[])
close(screen)
println("OK_MATERIALIZED_SKIP")
"""

include("helpers.jl")

@testset "B4 non-materialized scatter/meshscatter live positions MOVE (subprocess)" begin
    # Retry past the known intermittent GeometryGroup::attachToContext startup crash (house pattern).
    ec = -1; out = ""
    for _ in 1:4
        ec, out = run_ovrtx_subprocess(_B4_MOVE_PROG; timeout = 900)
        contains(out, "SCATTER_S0=") && break
    end
    contains(out, "OK_SCATTER_MOVE") || @info "B4 scatter/meshscatter move output" out
    @test ec == 0 && contains(out, "OK_SCATTER_MOVE")     # subprocess completed (no crash)
    for label in ("SCATTER", "MESHSCATTER")
        @testset "$label" begin
            @test contains(out, "$(label)_MOVED=true")    # RED pre-B4: `points` write dropped → centroid unmoved
            @test contains(out, "$(label)_ROUTED=true")   # the :positions_transformed_f32c push routed a write
            @test contains(out, "$(label)_REOPENED=0")    # live in-place `positions` write — NO re-author
            m = match(Regex("$(label)_DELTA=(-?[0-9.]+)"), out)
            @test m !== nothing && parse(Float64, m.captures[1]) > 20   # centroid shifted right ≥ threshold
        end
    end
end

@testset "B4 materialized scatter live positions edit is a NO-OP (subprocess)" begin
    ec = -1; out = ""
    for _ in 1:4
        ec, out = run_ovrtx_subprocess(_B4_MATERIALIZED_PROG; timeout = 900)
        contains(out, "NONBLACK_A=") && break
    end
    contains(out, "OK_MATERIALIZED_SKIP") || @info "B4 materialized-skip output" out
    @test ec == 0 && contains(out, "OK_MATERIALIZED_SKIP")
    @test contains(out, "MATERIAL_SHADER_SET=true")       # authored as a merged materialized mesh (not an instancer)
    @test contains(out, "IMAGE_UNCHANGED=true")           # pixel-compare before/after: render did NOT change
    @test contains(out, "POS_NOT_ROUTED=true")            # observer: the :positions push was NOT routed (skipped)
    @test contains(out, "WARN_COUNT=1")                   # exactly one @warn across two edits (maxlog=1)
    @test contains(out, "NEED1=false")                    # skip burns NO RT2 accumulation reset
    @test contains(out, "NEED2=false")
    m = match(r"CHANGED_A_vs_B=(\d+)", out)
    @test m !== nothing && parse(Int, m.captures[1]) < 100 # near-zero pixel delta (only convergence noise)
end
