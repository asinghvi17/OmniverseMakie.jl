using Test

# Screen.path2plot reverse map (prim_path => objectid(plot)) stays in strict
# lockstep with the forward plot2robj map. Authoring uses _author_screen!
# (author_root_from_scene! alone leaves plot2robj empty); checks the forward
# map, reverse resolution, and that a typed delete! clears the reverse entry.
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

include(joinpath(@__DIR__, "..", "helpers.jl"))
@testset "M6.B path2plot reverse map (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M6B_PATH2PLOT_PROG; timeout = 600, retries = 2, ready_marker = "OK_PATH2PLOT")
    @info "M6.B path2plot output" output
    @test exitcode == 0
    @test contains(output, "FWD_OK=true")
    @test contains(output, "REV_OK=true")
    @test contains(output, "REV_CLEARED=true")
end

# ---------------------------------------------------------------------------
# Makie.pick / pick_closest / pick_sorted over the native AOV pick. The
# scatter marker sits off the vertical center to lock the y-flip (py = H - y):
# a no-flip regression maps it to the opposite RenderProduct row and misses.
# Mesh picks are plot-level (index 0); markersize is world-scale here.
# PointInstancer picks report geometryInstanceId == 0 for every instance, so
# a multi-point scatter index is not recoverable — index is exact only for a
# single-point marker (see `_element_index` in src/screen.jl).
# ---------------------------------------------------------------------------
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
sxy = Makie.project(scene, mk)   # scatter's Makie (bottom-left) pixel
mxy = Makie.project(scene, Point3f(0,0,-1.35))     # mesh-center Makie pixel
son = Makie.Vec{2,Float64}(sxy[1], sxy[2])
plt, idx = Makie.pick(scene, screen, son)
println("HIT_IS_SCATTER=", plt === sp)
println("HIT_INDEX=", idx)
mplt, midx = Makie.pick(scene, screen, Makie.Vec{2,Float64}(mxy[1], mxy[2]))
println("HIT_IS_MESH=", mplt === mh)
println("MESH_INDEX=", midx)
# no-flip row → not the scatter
mir = Makie.pick(scene, screen, Makie.Vec{2,Float64}(sxy[1], H - sxy[2]))
println("MIRROR_NOT_SCATTER=", mir[1] !== sp)
# corner → background
miss = Makie.pick(scene, screen, Makie.Vec{2,Float64}(2.0, 2.0))
println("CORNER_MISS=", miss == (nothing, 0))
pc = Makie.pick_closest(scene, screen, son, 10)
println("CLOSEST_OK=", pc == (sp, idx))
ps = Makie.pick_sorted(scene, screen, son, 10)
println("SORTED_OK=", length(ps) == 1 && ps[1] == (sp, idx))
# pick_hit decodes the worldPositionM/worldNormal tensors — both must be
# finite on a real hit.
hit = OM.pick_hit(screen, son)
println("WORLDPOS_FINITE=", hit !== nothing && all(isfinite, hit.world_position) && all(isfinite, hit.normal))
close(screen)
println("OK_PICK")
"""

@testset "M6.B Makie.pick / pick_closest / pick_sorted (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M6B_PICK_PROG; timeout = 600, retries = 2, ready_marker = "OK_PICK")
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
    @test contains(output, "WORLDPOS_FINITE=true")
    @test contains(output, "OK_PICK")
end
