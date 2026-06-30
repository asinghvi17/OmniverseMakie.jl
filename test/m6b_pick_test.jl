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
    exitcode, output = run_ovrtx_subprocess(_M6B_PATH2PLOT_PROG; timeout = 300)
    @info "M6.B path2plot output" output
    @test exitcode == 0
    @test contains(output, "FWD_OK=true")
    @test contains(output, "REV_OK=true")
    @test contains(output, "REV_CLEARED=true")
end
