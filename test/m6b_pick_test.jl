using Test

# M6.B Task 2 — Screen.path2plot reverse map (prim_path => objectid(plot)) kept in
# strict lockstep with the forward plot2robj map.  Authors the screen via the shared
# `_author_screen!` helper (which calls Makie.insertplots! to author plot geometry —
# `author_root_from_scene!` alone leaves plot2robj EMPTY, per Task 1 forward-notes),
# then checks: the forward map has the plot, the reverse map resolves the plot's prim
# path back to objectid(plot), and a typed `delete!` clears the reverse entry too.
const _M6B_PATH2PLOT_PROG = """
using OmniverseMakie
OM = OmniverseMakie; OM.activate!(warmup = 8)
scene = Scene(size=(96,96)); cam3d!(scene)
p = mesh!(scene, Rect3f(Point3f(0), Vec3f(1)); color = :red)
screen = OM.Screen(scene)
OM._author_screen!(screen, scene, scene)
prim = screen.plot2robj[objectid(p)].prim_path
println("FWD_OK=", haskey(screen.plot2robj, objectid(p)))
println("REV_OK=", get(screen.path2plot, prim, UInt64(0)) == objectid(p))
delete!(screen, scene, p)
println("REV_CLEARED=", !haskey(screen.path2plot, prim))
close(screen)
println("OK_PATH2PLOT")
"""

include("helpers.jl")
@testset "M6.B path2plot reverse map (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M6B_PATH2PLOT_PROG; timeout = 300, retries = 2, ready_marker = "OK_PATH2PLOT")
    @info "M6.B path2plot output" output
    @test exitcode == 0
    @test contains(output, "FWD_OK=true")
    @test contains(output, "REV_OK=true")
    @test contains(output, "REV_CLEARED=true")
end

# ---------------------------------------------------------------------------
# M6.B Task 3 — Makie.pick / pick_closest / pick_sorted over the native AOV pick.
#
# Authors a scatter marker placed OFF the vertical center (high world-Z → projects to the
# TOP half) plus a mesh low in world-Z, then drives the STANDARD Makie pick protocol:
#   - pick at the scatter's projected Makie pixel → (sp, 1).  Off-center is what locks the
#     empirically-verified y-flip (`py = H - y`): a no-flip regression maps the marker's
#     Makie position to the opposite RenderProduct row and would MISS — so HIT_IS_SCATTER
#     alone proves the flip, and MIRROR_NOT_SCATTER witnesses it from the other side.
#   - pick the mesh → plot-level index 0 (mesh/lines/surface are plot-level by design).
#   - a far corner → (nothing, 0) background.
#   - pick_closest / pick_sorted compose on top of pick (so DataInspector works).
# Plots are authored via the shared `_author_screen!` helper (NOT `author_root_from_scene!`,
# which leaves plot2robj/path2plot EMPTY — Task 1 forward-note).  `markersize` is WORLD-scale
# in this backend (≈ sphere radius), so 0.3 yields a ~50px marker.
#
# NOTE (verified): ovrtx's PointInstancer pick reports geometryInstanceId == 0 for every
# instance, so a multi-point scatter index is not recoverable today — index is exact only
# for a single-point marker (this test).  See `_element_index` in src/screen.jl.
const _M6B_PICK_PROG = """
using OmniverseMakie
OM = OmniverseMakie; const OV = OM.OV
OM.activate!(warmup = 24)
scene = Scene(size=(200,200)); cam3d!(scene)
mk = Point3f(0, 0, 1.1)
sp = scatter!(scene, [mk]; markersize = 0.3, color = :red)
mh = mesh!(scene, Rect3f(Point3f(-0.35,-0.35,-1.5), Vec3f(0.7,0.7,0.3)); color = :blue)
screen = OM.Screen(scene)
OM._author_screen!(screen, scene, scene)
for _ in 1:8; sr0 = OV.step!(screen.renderer, screen.product; timeout_ns=UInt64(60_000_000_000)); close(sr0); end
W, H = screen.fb_size
sxy = Makie.project(scene, mk)                     # scatter's Makie (bottom-left) pixel
mxy = Makie.project(scene, Point3f(0,0,-1.35))     # mesh-center Makie pixel
son = Makie.Vec{2,Float64}(sxy[1], sxy[2])
plt, idx = Makie.pick(scene, screen, son)
println("HIT_IS_SCATTER=", plt === sp)
println("HIT_INDEX=", idx)
mplt, midx = Makie.pick(scene, screen, Makie.Vec{2,Float64}(mxy[1], mxy[2]))
println("HIT_IS_MESH=", mplt === mh)
println("MESH_INDEX=", midx)
mir = Makie.pick(scene, screen, Makie.Vec{2,Float64}(sxy[1], H - sxy[2]))   # no-flip row → not the scatter
println("MIRROR_NOT_SCATTER=", mir[1] !== sp)
miss = Makie.pick(scene, screen, Makie.Vec{2,Float64}(2.0, 2.0))            # corner → background
println("CORNER_MISS=", miss == (nothing, 0))
pc = Makie.pick_closest(scene, screen, son, 10)
println("CLOSEST_OK=", pc == (sp, idx))
ps = Makie.pick_sorted(scene, screen, son, 10)
println("SORTED_OK=", length(ps) == 1 && ps[1] == (sp, idx))
close(screen)
println("OK_PICK")
"""

@testset "M6.B Makie.pick / pick_closest / pick_sorted (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M6B_PICK_PROG; timeout = 400, retries = 2, ready_marker = "OK_PICK")
    @info "M6.B pick output" output
    @test exitcode == 0
    @test contains(output, "HIT_IS_SCATTER=true")
    @test contains(output, "HIT_INDEX=1")
    @test contains(output, "HIT_IS_MESH=true")
    @test contains(output, "MESH_INDEX=0")
    @test contains(output, "MIRROR_NOT_SCATTER=true")
    @test contains(output, "CORNER_MISS=true")
    @test contains(output, "CLOSEST_OK=true")
    @test contains(output, "SORTED_OK=true")
    @test contains(output, "OK_PICK")
end
