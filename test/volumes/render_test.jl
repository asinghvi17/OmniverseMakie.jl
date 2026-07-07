# End-to-end VDB render (subprocess, env-gated). With IndeX enabled
# (OMNIVERSEMAKIE_INDEX_LIBS), author the on-disk torus.vdb into a Screen's
# stage via author_vdb_volume! and render through RT2 → IndeX Direct; assert
# a non-black volume and IndeX enabled. Skips cleanly when the Kit IndeX
# libs dir or the sample VDB is absent.

using Test

include(joinpath(@__DIR__, "..", "helpers.jl"))

const _VDB  = "/home/juliahub/.local/share/ov/data/exts/v2/omni.rtx.index_composite-718bb6a388c21baf/data/tests/volumes/torus.vdb"
const _LIBS = _HELPER_INDEX_LIBS

const _VOL_RENDER_PROG = """
using OmniverseMakie
import OmniverseMakie as OM
using OmniverseMakie: OV
OM.activate!(warmup = 48)
scene = Scene(size=(256,256)); cam3d!(scene)
# Frame the torus_fog grid (near origin).
update_cam!(scene, Vec3f(38,38,22), Vec3f(0,0,0), Vec3f(0,0,1))
screen = OM.Screen(scene)   # creating the Screen enables IndeX (env is set)
OM.author_root_from_scene!(screen, scene; resolution = screen.fb_size)
OM.author_vdb_volume!(screen, scene, "$(_VDB)"; field="torus_fog", colormap=:viridis)
img = OV.render_to_matrix(screen.renderer, screen.product; warmup = 48)
nonblack = count(c -> (Float32(c.r)+Float32(c.g)+Float32(c.b)) > 0.04, img)
println("INDEX_ENABLED=", OV._index_enabled())
println("VOL_NONBLACK=", nonblack)
close(screen)
println("OK_VOL_RENDER")
"""

@testset "Volumes: end-to-end render (subprocess)" begin
    if !isdir(_LIBS) || !isfile(_VDB)
        @test_skip "IndeX libs ($_LIBS) or sample VDB ($_VDB) absent — volume render test skipped"
    else
        # retries=4 absorbs the intermittent ovrtx startup crash
        # (GeometryGroup::attachToContext).
        ec, out = run_ovrtx_subprocess(_VOL_RENDER_PROG; timeout = 600,
            retries = 4, ready_marker = "VOL_NONBLACK=",
            env = ("OMNIVERSEMAKIE_INDEX_LIBS" => _LIBS))
        # Dump the full subprocess log only on failure (it is thousands of
        # ovrtx lines).
        contains(out, "OK_VOL_RENDER") || @info "Volume render output (failure)" out
        @test ec == 0
        @test contains(out, "INDEX_ENABLED=true")
        @test contains(out, "OK_VOL_RENDER")
        m = match(r"VOL_NONBLACK=(\d+)", out)
        # a volume appeared
        @test m !== nothing && parse(Int, m.captures[1]) > 500
    end
end
