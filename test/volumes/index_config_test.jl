# IndeX enablement (_ensure_index OncePerProcess + carb-config injection) — formerly
# volumes_index_config_test.jl, plus the disabled-author error path folded in from the
# retired volumes_usda test.
#
# Enabled path (env-gated, skip-if-libs-absent): with OMNIVERSEMAKIE_INDEX_LIBS forwarded,
# OV._ensure_index() enables NVIDIA IndeX (idempotent) and OV._index_enabled() flips true.
# Disabled path (ALWAYS runs, no libs needed): no volume env → _ensure_index() is a no-op
# returning false, _index_enabled() stays false, a normal offscreen render still works
# (the zero-overhead / no-regression guard), and author_vdb_volume! throws a clear
# "IndeX"-naming error rather than silently authoring a black volume.

using Test

const _IDX_ON_PROG = """
using OmniverseMakie: OV
println("ENABLED=", OV._ensure_index())
println("ENABLED2=", OV._ensure_index())     # idempotent (memoized OncePerProcess)
println("QUERY=", OV._index_enabled())
"""

const _IDX_OFF_PROG = """
using OmniverseMakie
import OmniverseMakie as OM
using OmniverseMakie: OV
delete!(ENV, "OMNIVERSEMAKIE_INDEX_LIBS"); delete!(ENV, "OMNIVERSEMAKIE_OVRTX_CONFIG")
println("ENABLED=", OV._ensure_index())
println("QUERY=", OV._index_enabled())
# non-volume render still works (mirror the existing mesh render tests: size ≥ 64², and
# `_author_screen!` — NOT `author_root_from_scene!` — since the latter bakes only camera/lights
# while the former ALSO insertplots! the mesh, as colorbuffer does; a lit red cube is non-black):
scene = Scene(size=(128,128)); cam3d!(scene)
mesh!(scene, Rect3f(Point3f(-1), Vec3f(2)); color=:red)
screen = OM.Screen(scene)
OM._author_screen!(screen, scene, scene)
img = OV.render_to_matrix(screen.renderer, screen.product; warmup = 48)
println("NONVOL_OK=", any(c -> (Float32(c.r)+Float32(c.g)+Float32(c.b)) > 0.02, img))
# …and author_vdb_volume! on the disabled Screen errors CLEARLY, naming IndeX:
try
    OM.author_vdb_volume!(screen, scene, "/nonexistent/torus.vdb")
    println("RESULT=NO_ERROR")
catch e
    println("RESULT=ERROR")
    println("ERRMSG_HAS_INDEX=", occursin("IndeX", sprint(showerror, e)))
end
close(screen)
println("OK_IDX_OFF")
"""

include(joinpath(@__DIR__, "..", "helpers.jl"))

@testset "Volumes: IndeX enablement (subprocess)" begin
    # The IndeX-enabled assertions need the Kit libs present; skip cleanly if absent so CI
    # without them stays green.  The disabled-path guard below ALWAYS runs (needs no libs).
    libs = get(ENV, "OMNIVERSEMAKIE_INDEX_LIBS", "")
    if isempty(libs) || !isdir(libs)
        @test_skip "OMNIVERSEMAKIE_INDEX_LIBS unset/absent — IndeX-enabled check skipped"
    else
        ec_on, out_on = run_ovrtx_subprocess(_IDX_ON_PROG; timeout = 600,
            env = ("OMNIVERSEMAKIE_INDEX_LIBS" => libs))
        @test ec_on == 0
        @test contains(out_on, "ENABLED=true")
        @test contains(out_on, "ENABLED2=true")
        @test contains(out_on, "QUERY=true")
    end

    ec_off, out_off = run_ovrtx_subprocess(_IDX_OFF_PROG; timeout = 600, retries = 2,
                                           ready_marker = "OK_IDX_OFF")
    @test ec_off == 0
    @test contains(out_off, "ENABLED=false")
    @test contains(out_off, "QUERY=false")
    @test contains(out_off, "NONVOL_OK=true")
    # author_vdb_volume! fails loud when IndeX is disabled (folded from volumes_usda).
    @test contains(out_off, "RESULT=ERROR")
    @test contains(out_off, "ERRMSG_HAS_INDEX=true")
end
