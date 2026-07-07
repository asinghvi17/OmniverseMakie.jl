using Test
include(joinpath(@__DIR__, "..", "helpers.jl"))

# ---------------------------------------------------------------------------
# Imperative open-stage insert!:
#   Scene + Makie.push_screen!(scene, screen); author the open stage; then a
#   LIVE scatter!(scene, …) after attach → Makie calls insert!(screen, scene,
#   plot) → plot2robj grows by one and the stage re-renders WITHOUT a re-open.
#   A poly! recipe (composite) registers its atomic children.  Stage authored
#   ONCE; a no-op re-render (unchanged camera) reuses the SAME usd_handle.
# ---------------------------------------------------------------------------

const _M21_INSERT_PROG = """
using OmniverseMakie, ColorTypes, FixedPointNumbers
const OM = OmniverseMakie

OM.activate!(warmup = 32)

scene = Scene(size = (400, 400))
cam3d!(scene)
update_cam!(scene, Vec3d(6, 6, 4), Vec3d(0, 0, 0), Vec3d(0, 0, 1))
m = mesh!(scene, Rect3f(Point3f(-1, -1, -1), Vec3f(2)); color = :red)

screen = OM.Screen(scene)
# attach: now push!(scene, plot) → insert!(screen, …)
Makie.push_screen!(scene, screen)

function nonblack(img)
    n = 0
    for c in img
        (Float32(red(c)) + Float32(green(c)) + Float32(blue(c))) > 0.05f0 && (n += 1)
    end
    return n
end

# ---- author the open stage (adds the pre-existing mesh via insertplots!) ----
img0 = Makie.colorbuffer(screen)
n0   = length(screen.plot2robj)
println("ROBJ_AFTER_AUTHOR=\$(n0)")
println("NONBLACK_AUTHOR=\$(nonblack(img0))")
@assert haskey(screen.plot2robj, objectid(m)) "mesh not registered at author time"
@assert nonblack(img0) > 500 "author frame (near) black"

# ---- degenerate no-op path: a second colorbuffer with NOTHING changed
#      does not re-author and reuses the SAME robj handle ----
handle_a = screen.plot2robj[objectid(m)].usd_handle
imgN = Makie.colorbuffer(screen)
handle_b = screen.plot2robj[objectid(m)].usd_handle
println("HANDLE_STABLE=\$(handle_a == handle_b && handle_a != 0)")
@assert handle_a == handle_b && handle_a != 0 "no-op re-render changed the usd_handle (a=\$(handle_a) b=\$(handle_b))"
@assert nonblack(imgN) > 500 "no-op re-render frame (near) black"

# ---- LIVE scatter! after attach → insert! fires → plot2robj grows by one ----
s  = scatter!(scene, [Point3f(2cos(t), 2sin(t), 0) for t in range(0, 2pi, length = 10)];
              markersize = 0.3, color = :cyan)
n1 = length(screen.plot2robj)
println("ROBJ_AFTER_SCATTER=\$(n1)")
@assert haskey(screen.plot2robj, objectid(s)) "scatter not registered after live insert!"
@assert n1 == n0 + 1 "plot2robj did not grow by one after scatter (n0=\$(n0) n1=\$(n1))"

# ---- LIVE poly! recipe (composite) → its atomic children get registered ----
# (3-D points so the recipe's child UsdGeomMesh has Point3f vertices.)
p  = poly!(scene, Point3f[(2, 2, 0), (3, 2, 0), (3, 3, 0), (2, 3, 0)]; color = :yellow)
n2 = length(screen.plot2robj)
println("ROBJ_AFTER_POLY=\$(n2)")
@assert n2 > n1 "poly recipe registered no atomic children (n1=\$(n1) n2=\$(n2))"

# ---- re-render the live-updated stage ----
img1 = Makie.colorbuffer(screen)
println("NONBLACK_FINAL=\$(nonblack(img1))")
@assert nonblack(img1) > 500 "final frame (near) black"

opens = OM._ROOT_OPEN_COUNT[]
println("ROOT_OPENS=\$(opens)")
@assert opens == 1 "stage re-opened during live inserts (opens=\$(opens)); insert! must NOT re-author"

close(screen)
println("OK_INSERT")
"""

@testset "imperative insert! on open stage (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M21_INSERT_PROG; timeout = 900, retries = 2, ready_marker = "ROBJ_AFTER_AUTHOR=")
    @info "insert subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_INSERT")
    @test contains(output, "HANDLE_STABLE=true")

    # plot2robj grew by exactly one after the live scatter!
    n0 = match(r"ROBJ_AFTER_AUTHOR=(\d+)", output)
    n1 = match(r"ROBJ_AFTER_SCATTER=(\d+)", output)
    n2 = match(r"ROBJ_AFTER_POLY=(\d+)", output)
    if n0 !== nothing && n1 !== nothing && n2 !== nothing
        a = parse(Int, n0.captures[1]); b = parse(Int, n1.captures[1]); c = parse(Int, n2.captures[1])
        @test b == a + 1          # scatter added exactly one robj
        @test c > b               # poly recipe registered atomic children
    else
        @test false               # robj-count lines missing
    end

    # Stage authored exactly once across all live inserts.
    mo = match(r"ROOT_OPENS=(\d+)", output)
    @test mo !== nothing && parse(Int, mo.captures[1]) == 1
end
