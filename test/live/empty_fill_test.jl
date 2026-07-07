using Test

# Empty→fill rebuild is UNIVERSAL (subprocess).
#
# A plot authored EMPTY (author_usd_prim! returns nothing on an empty guard —
# no points, <2 finite curve points, all-zero volume) has no robj; a later
# live FILL must take the BUILD path in the diff-node callback and register
# the pick maps, so the filled plot renders and picks like a first-built one.
#
# The prog authors an EMPTY scatter + an EMPTY lines (blank colorbuffer,
# neither in `plot2robj`), FILLs both live (both render + register), then
# PICKs the filled single-point marker (plot-level hit, index 1).
# markersize is WORLD-scale in this backend.  An explicit camera pins the
# framing (the plots start empty, so cam3d auto-framing has no data to fit).

const _B3_EMPTY_FILL_PROG = """
using OmniverseMakie
import OmniverseMakie as OM
using OmniverseMakie: OV
OM.activate!(warmup = 24)
lit(img) = count(c -> (Float32(c.r) + Float32(c.g) + Float32(c.b)) > 0.04, img)
scene = Scene(size = (200, 200)); cam3d!(scene)
update_cam!(scene, Vec3f(0, 0, 10), Vec3f(0, 0, 0), Vec3f(0, 1, 0))
mk = Point3f(2.5, 2.5, 1.5)  # off-centre, isolated → clean pick
# EMPTY scatter → authors nothing
sp = scatter!(scene, Point3f[]; markersize = 0.5, color = :red)
# EMPTY lines → authors nothing
lp = lines!(scene, Point3f[]; color = :green, linewidth = 10)
screen = OM.Screen(scene)
lit0   = lit(Makie.colorbuffer(screen))  # both empty → blank frame
s_has0 = haskey(screen.plot2robj, objectid(sp))
l_has0 = haskey(screen.plot2robj, objectid(lp))
# LIVE fill scatter → late build
sp[1][] = [mk]
# LIVE fill lines (≥2 pts) → late build
lp[1][] = [Point3f(-3.5, -3, 0), Point3f(3.5, -3, 0)]
# the late build now renders both
lit1   = lit(Makie.colorbuffer(screen))
s_has1 = haskey(screen.plot2robj, objectid(sp))
l_has1 = haskey(screen.plot2robj, objectid(lp))
for _ in 1:8; sr0 = OV.step!(screen.renderer, screen.product; timeout_ns = UInt64(60_000_000_000)); close(sr0); end
sxy = Makie.project(scene, mk)  # marker's Makie (bottom-left) pixel
plt, idx = Makie.pick(scene, screen, Makie.Vec{2,Float64}(sxy[1], sxy[2]))
close(screen)
println("EMPTY_LIT=", lit0)
println("EMPTY_SCATTER_ROBJ=", s_has0)
println("EMPTY_LINES_ROBJ=", l_has0)
println("FILLED_LIT=", lit1)
println("FILLED_SCATTER_ROBJ=", s_has1)
println("FILLED_LINES_ROBJ=", l_has1)
println("PICK_IS_SCATTER=", plt === sp)
println("PICK_INDEX=", idx)
println("OK_EMPTY_FILL")
"""

include(joinpath(@__DIR__, "..", "helpers.jl"))

@testset "B3 empty→fill scatter+lines late-build renders + pickable (subprocess)" begin
    # Retry past ovrtx's intermittent pre-render startup crash.
    ec, out = run_ovrtx_subprocess(_B3_EMPTY_FILL_PROG; timeout = 600, retries = 4,
                                   ready_marker = "EMPTY_LIT=")
    contains(out, "OK_EMPTY_FILL") || @info "B3 empty→fill output" out
    # subprocess completed (no mid-run death)
    @test ec == 0 && contains(out, "OK_EMPTY_FILL")
    # Empty author: blank frame, NEITHER plot registered (author_usd_prim!
    # returned nothing).
    m0 = match(r"EMPTY_LIT=(\d+)", out)
    # nothing authored → (near-)black
    @test m0 !== nothing && parse(Int, m0.captures[1]) < 100
    @test contains(out, "EMPTY_SCATTER_ROBJ=false")
    @test contains(out, "EMPTY_LINES_ROBJ=false")
    # Live fill: the late build renders BOTH (>150 lit px) and registers BOTH.
    m1 = match(r"FILLED_LIT=(\d+)", out)
    @test m1 !== nothing && parse(Int, m1.captures[1]) > 150
    # late build registered the scatter…
    @test contains(out, "FILLED_SCATTER_ROBJ=true")
    # …and the lines
    @test contains(out, "FILLED_LINES_ROBJ=true")
    # Pick maps registered on the late build → the filled marker is pickable
    # (single point → index 1).
    @test contains(out, "PICK_IS_SCATTER=true")
    @test contains(out, "PICK_INDEX=1")
end
