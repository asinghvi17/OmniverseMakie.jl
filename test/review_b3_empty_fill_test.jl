using Test

# Review Track B / Task B3 — empty→fill rebuild is UNIVERSAL (subprocess, skip-if-absent).
#
# A plot authored EMPTY (author_usd_prim! returns nothing on an empty guard — no points, <2 finite
# curve points, all-zero volume) used to keep `robj === nothing` FOREVER on that screen: the
# diff-node callback rebuilt only on first-resolve / new-screen, so a later live FILL fired the
# tracked input but the push found no robj and was silently dropped ("start empty, fill in the
# animation loop" never rendered).  B3 makes the callback's else-branch take the BUILD path when
# `robj === nothing` and a tracked input changed, and registers the pick maps on that late build so
# the filled plot is pickable identically to a first-built one.
#
# This subprocess authors an EMPTY scatter (`Point3f[]`) + an EMPTY lines (`Point3f[]`) → colorbuffer
# is blank and NEITHER is in `plot2robj` → FILLs both live → colorbuffer now renders (lit) AND both
# are registered → the filled single-point scatter marker is PICKABLE (plot-level hit, index 1 — the
# pick shape mirrors test/m6b_pick_test.jl).  markersize is WORLD-scale in this backend.  An explicit
# camera pins the framing (the plots start empty, so cam3d auto-framing has no data to fit).

const _B3_EMPTY_FILL_PROG = """
using OmniverseMakie
import OmniverseMakie as OM
using OmniverseMakie: OV
OM.activate!(warmup = 24)
lit(img) = count(c -> (Float32(c.r) + Float32(c.g) + Float32(c.b)) > 0.04, img)
scene = Scene(size = (200, 200)); cam3d!(scene)
update_cam!(scene, Vec3f(0, 0, 10), Vec3f(0, 0, 0), Vec3f(0, 1, 0))
mk = Point3f(2.5, 2.5, 1.5)                                              # off-centre, isolated → clean pick
sp = scatter!(scene, Point3f[]; markersize = 0.5, color = :red)         # EMPTY scatter → authors nothing
lp = lines!(scene, Point3f[]; color = :green, linewidth = 10)           # EMPTY lines   → authors nothing
screen = OM.Screen(scene)
lit0   = lit(Makie.colorbuffer(screen))                                 # both empty → blank frame
s_has0 = haskey(screen.plot2robj, objectid(sp))
l_has0 = haskey(screen.plot2robj, objectid(lp))
sp[1][] = [mk]                                                          # LIVE fill scatter → late build
lp[1][] = [Point3f(-3.5, -3, 0), Point3f(3.5, -3, 0)]                   # LIVE fill lines (≥2 pts) → late build
lit1   = lit(Makie.colorbuffer(screen))                                 # now renders both (the B3 fix)
s_has1 = haskey(screen.plot2robj, objectid(sp))
l_has1 = haskey(screen.plot2robj, objectid(lp))
for _ in 1:8; sr0 = OV.step!(screen.renderer, screen.product; timeout_ns = UInt64(60_000_000_000)); close(sr0); end
sxy = Makie.project(scene, mk)                                          # marker's Makie (bottom-left) pixel
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

include("helpers.jl")

@testset "B3 empty→fill scatter+lines late-build renders + pickable (subprocess)" begin
    # Retry past the known intermittent GeometryGroup::attachToContext startup crash (house pattern).
    ec = -1; out = ""
    for _ in 1:4
        ec, out = run_ovrtx_subprocess(_B3_EMPTY_FILL_PROG; timeout = 600)
        contains(out, "EMPTY_LIT=") && break
    end
    contains(out, "OK_EMPTY_FILL") || @info "B3 empty→fill output" out
    @test ec == 0 && contains(out, "OK_EMPTY_FILL")            # subprocess completed (no mid-run death)
    # Empty author: blank frame, NEITHER plot registered (author_usd_prim! returned nothing).
    m0 = match(r"EMPTY_LIT=(\d+)", out)
    @test m0 !== nothing && parse(Int, m0.captures[1]) < 100   # nothing authored → (near-)black
    @test contains(out, "EMPTY_SCATTER_ROBJ=false")
    @test contains(out, "EMPTY_LINES_ROBJ=false")
    # Live fill: the late build renders BOTH and registers BOTH (the B3 fix — previously dropped).
    m1 = match(r"FILLED_LIT=(\d+)", out)
    @test m1 !== nothing && parse(Int, m1.captures[1]) > 150   # LIT_PX_MIN: the fill actually renders
    @test contains(out, "FILLED_SCATTER_ROBJ=true")            # late build registered the scatter
    @test contains(out, "FILLED_LINES_ROBJ=true")              # late build registered the lines
    # Pick maps registered on the late build → the filled marker is pickable (single point → index 1).
    @test contains(out, "PICK_IS_SCATTER=true")
    @test contains(out, "PICK_INDEX=1")
end
