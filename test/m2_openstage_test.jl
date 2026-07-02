using Test

# ---------------------------------------------------------------------------
# M2.1 Step 1 — Open-stage Screen: author ONCE, reuse across colorbuffer calls.
#
# RED (M1 re-author-per-call): every colorbuffer re-opened the stage, so two
#   colorbuffers on one Screen → _ROOT_OPEN_COUNT == 2 (and the plot's USD handle
#   was wiped + re-added → not stable).  The ROOT_OPENS==1 assert fails.
#
# GREEN (M2.1 open-once): the root is authored on the FIRST colorbuffer only; with
#   the camera UNCHANGED the second call pushes no writes and re-renders the same
#   open stage → _ROOT_OPEN_COUNT == 1, screen.authored == true, the plot's
#   OvrtxRObj handle is identical across both calls, both frames non-black.
# ---------------------------------------------------------------------------

const _M21_OPENSTAGE_PROG = """
using OmniverseMakie, ColorTypes, FixedPointNumbers
const OM = OmniverseMakie

OM.activate!(warmup = 32)

fig = Figure()
ax  = LScene(fig[1, 1])
m   = mesh!(ax, Rect3f(Point3f(0), Vec3f(1)); color = :red)
scene = ax.scene

screen = OM.Screen(scene)
mid    = objectid(m)

function nonblack(img)
    n = 0
    for c in img
        (Float32(red(c)) + Float32(green(c)) + Float32(blue(c))) > 0.05f0 && (n += 1)
    end
    return n
end
handle_of(scr, id) = haskey(scr.plot2robj, id) ? scr.plot2robj[id].usd_handle : UInt64(0)

# ---- Render twice with the camera UNCHANGED ----
imgA = Makie.colorbuffer(screen)
nbA  = nonblack(imgA)
hA   = handle_of(screen, mid)
println("NONBLACK_A=\$(nbA)")

imgB = Makie.colorbuffer(screen)
nbB  = nonblack(imgB)
hB   = handle_of(screen, mid)
println("NONBLACK_B=\$(nbB)")

opens = OM._ROOT_OPEN_COUNT[]
println("ROOT_OPENS=\$(opens)")
println("AUTHORED=\$(screen.authored)")
println("PLOT2ROBJ_N=\$(length(screen.plot2robj))")
println("HANDLE_A=\$(hA) HANDLE_B=\$(hB)")

@assert nbA > 500 "frame A rendered (near) black: nonblack=\$(nbA)"
@assert nbB > 500 "frame B rendered (near) black: nonblack=\$(nbB)"
@assert opens == 1 "stage authored \$(opens)× across two colorbuffers (expected 1): open-stage regressed"
@assert screen.authored "screen.authored is false after colorbuffer"
@assert length(screen.plot2robj) >= 1 "mesh not recorded in plot2robj"
@assert hA != 0 "mesh OvrtxRObj handle is zero (not authored)"
@assert hA == hB "plot handle not stable across calls (A=\$(hA) B=\$(hB)): stage was re-authored"

close(screen)
println("OK_OPENSTAGE")
"""

@testset "M2.1 open-stage Screen: authored once (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M21_OPENSTAGE_PROG; timeout = 900, retries = 2, ready_marker = "OK_OPENSTAGE")
    @info "M2.1 open-stage subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_OPENSTAGE")

    # Stage opened exactly once across the two colorbuffer calls.
    mo = match(r"ROOT_OPENS=(\d+)", output)
    if mo !== nothing
        @test parse(Int, mo.captures[1]) == 1
    else
        @test false   # ROOT_OPENS line missing
    end

    @test contains(output, "AUTHORED=true")

    # Both frames non-black.
    for tag in ("NONBLACK_A", "NONBLACK_B")
        mnb = match(Regex("$(tag)=(\\d+)"), output)
        if mnb !== nothing
            @test parse(Int, mnb.captures[1]) > 500
        else
            @test false
        end
    end

    # Plot handle stable across both calls (not re-authored).
    mh = match(r"HANDLE_A=(\d+) HANDLE_B=(\d+)", output)
    if mh !== nothing
        @test mh.captures[1] == mh.captures[2]
        @test parse(UInt64, mh.captures[1]) != 0
    else
        @test false
    end
end
