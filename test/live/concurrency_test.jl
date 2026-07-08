using Test
include(joinpath(@__DIR__, "..", "helpers.jl"))
using OmniverseMakie
import OmniverseMakie as OM

# ===========================================================================
# Two latent-concurrency guards on Screen:
#
#  1. requires_update / structural_dirty (Bool flags) + pending_usd_writes
#     (Dict) are SET by user-task Observable listeners and READ-then-CLEARED
#     by the render tick.  A `edit_lock` serializes every SET site and every
#     render-tick read-and-clear window so a threaded renderloop cannot lose an
#     edit landing between a read and its clear, nor mutate the Dict mid-flush.
#  2. A plot's compute graph carries ONE `:ovrtx_screen`, so two screens
#     displaying the same figure concurrently misroute diffs to the
#     last-registered screen.  `register_ovrtx_robj!` @warns (maxlog=1) when it
#     re-points away from a still-open screen; the earlier screen then shows a
#     frozen stage (full multi-screen routing is out of scope).
# ===========================================================================

# --- (1) pure: the lock field + the locked flag/queue helpers ---------------

# Minimal duck-typed stand-in exposing exactly the fields the helpers touch, so
# the real helpers can be hammered without a GPU-backed Screen (must be a
# top-level struct — Julia forbids struct defs inside a testset function body).
mutable struct _FakeLockScreen
    edit_lock::ReentrantLock
    requires_update::Bool
    structural_dirty::Bool
    pending_usd_writes::Dict{Any,Any}
    path_resolver::Any
end
_FakeLockScreen() = _FakeLockScreen(ReentrantLock(), false, false, Dict{Any,Any}(), nothing)

@testset "edit_lock field + locked flag/queue helpers (pure)" begin
    # The lock exists, is a ReentrantLock, and is LAST so the positional core
    # constructor (Screen(...)) appends ReentrantLock() correctly.
    @test :edit_lock in fieldnames(OM.Screen)
    @test fieldtype(OM.Screen, :edit_lock) === ReentrantLock
    @test fieldnames(OM.Screen)[end] === :edit_lock

    f = _FakeLockScreen()
    # requires_update: SET raises it; TAKE is an atomic read-and-clear.
    @test OM._take_requires_update!(f) === false
    OM._set_requires_update!(f);  @test f.requires_update === true
    @test OM._take_requires_update!(f) === true     # read-and-clear
    @test f.requires_update === false
    @test OM._take_requires_update!(f) === false     # stays cleared

    # _note_composition_change! flags structural_dirty + drops the resolver.
    f.path_resolver = :cached
    OM._note_composition_change!(f)
    @test f.structural_dirty === true
    @test f.path_resolver === nothing
    @test OM._take_structural_dirty!(f) === true
    @test OM._take_structural_dirty!(f) === false

    # _clear_requires_update! is the ext consume path.
    OM._set_requires_update!(f);  OM._clear_requires_update!(f)
    @test f.requires_update === false

    # pending_usd_writes: enqueue also raises the flag; swap snapshots the Dict
    # and installs a fresh empty one.
    @test OM._swap_pending_usd_writes!(f) === nothing         # empty → nothing
    OM._enqueue_usd_write!(f, (:p, "a"), 1)
    @test f.requires_update === true
    d = OM._swap_pending_usd_writes!(f)
    @test d !== nothing && d[(:p, "a")] == 1
    @test isempty(f.pending_usd_writes)                       # swapped out
    @test OM._swap_pending_usd_writes!(f) === nothing
end

@testset "locked queue is race-free under Threads.@spawn hammering (pure)" begin
    # Many writers enqueue unique keys while a drainer snapshot-swaps; nothing
    # is lost or duplicated and the Dict is never corrupted mid-iteration.
    # Meaningful at any nthreads() (genuine contention only when >1).
    f         = _FakeLockScreen()
    nwriters  = 8
    per       = 200
    collected = Dict{Any,Any}()
    done      = Threads.Atomic{Int}(0)
    drainer = Threads.@spawn begin
        while true
            d = OM._swap_pending_usd_writes!(f)
            d === nothing || merge!(collected, d)
            # Exit only once every writer has finished AND a swap came back
            # empty — no producer remains to refill the queue.
            (done[] >= nwriters && d === nothing) && break
            yield()
        end
    end
    writers = map(1:nwriters) do w
        Threads.@spawn begin
            for k in 1:per
                OM._enqueue_usd_write!(f, (w, k), w * 1000 + k)
            end
            Threads.atomic_add!(done, 1)
        end
    end
    foreach(wait, writers)
    wait(drainer)
    d = OM._swap_pending_usd_writes!(f)   # belt-and-suspenders final drain
    d === nothing || merge!(collected, d)
    @test length(collected) == nwriters * per   # no lost / duplicated keys
    @test all(collected[(w, k)] == w * 1000 + k for w in 1:nwriters for k in 1:per)
end

# --- (2) GPU subprocess: two concurrently-live screens ----------------------

const _MULTISCREEN_PROG = """
using OmniverseMakie, ColorTypes, FixedPointNumbers
using Logging
import OmniverseMakie as OM
import Test

OM.activate!(warmup = 32)

scene = Scene(size = (300, 300))
cam3d!(scene)
update_cam!(scene, Vec3d(6, 6, 4), Vec3d(0, 0, 0), Vec3d(0, 0, 1))
m = mesh!(scene, Rect3f(Point3f(-1, -1, -1), Vec3f(2)); color = :red)

nonblack(img) = count(c -> (Float32(red(c)) + Float32(green(c)) + Float32(blue(c))) > 0.05f0, img)
warned(lg) = any(r -> occursin("most recent Screen", r.message), lg.logs)

# ---- Screen A alone: authoring/registering the FIRST screen never warns
#      (the plot has no prior :ovrtx_screen — the add_input! branch). ----
lgA = Test.TestLogger(min_level = Logging.Warn)
screenA = OM.Screen(scene)
imgA = with_logger(lgA) do
    Makie.colorbuffer(screenA)
end
println("NONBLACK_A=\$(nonblack(imgA))")
println("WARN_SINGLE=\$(warned(lgA))")
@assert nonblack(imgA) > 300 "screen A (near) black"
@assert !warned(lgA) "single-screen authoring wrongly warned"

# ---- Screen B WITHOUT closing A: its first colorbuffer re-points m's
#      :ovrtx_screen from A (still open) to B → the second-live-screen warn
#      fires once, and B renders the geometry on its own fresh stage. ----
lg1 = Test.TestLogger(min_level = Logging.Warn)
screenB = OM.Screen(scene)
imgB = with_logger(lg1) do
    Makie.colorbuffer(screenB)
end
println("A_STILL_OPEN=\$(Base.isopen(screenA))")
println("WARN_ON_B=\$(warned(lg1))")
println("NONBLACK_B=\$(nonblack(imgB))")
@assert Base.isopen(screenA) "screen A closed before the two-live-screen check"
@assert warned(lg1) "second live screen did not warn"
@assert nonblack(imgB) > 300 "screen B (near) black"

# ---- Close A; an ongoing live edit on B re-renders via the diff path (no
#      re-register) and must NOT spam the warn (maxlog=1). ----
close(screenA)
lg2 = Test.TestLogger(min_level = Logging.Warn)
imgC = with_logger(lg2) do
    m.color = :blue
    Makie.colorbuffer(screenB)
end
println("WARN_REPEAT=\$(warned(lg2))")
println("NONBLACK_C=\$(nonblack(imgC))")
@assert !warned(lg2) "multi-screen warn repeated on a later live edit"
@assert nonblack(imgC) > 300 "screen B post-edit (near) black"

close(screenB)
println("OK_MULTISCREEN")
"""

@testset "two concurrently-live screens: warn + B renders (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_MULTISCREEN_PROG; timeout = 900,
                                            retries = 2, ready_marker = "NONBLACK_A=")
    @info "multiscreen subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_MULTISCREEN")

    # Guard behaviour: single screen silent; second live screen warns once;
    # A stays open; no repeat spam on a later edit.
    @test contains(output, "WARN_SINGLE=false")
    @test contains(output, "A_STILL_OPEN=true")
    @test contains(output, "WARN_ON_B=true")
    @test contains(output, "WARN_REPEAT=false")

    # The newer screen renders the figure on its own stage (not black).
    mb = match(r"NONBLACK_B=(\d+)", output)
    @test mb !== nothing && parse(Int, mb.captures[1]) > 300
end
