# Volumes M1 Task 2 — author_vdb_volume! (UsdVol + colormap material) authoring.
#
# Part 1 (PURE, no renderer): _vdb_volume_usda emits a well-formed Volume + OpenVDBAsset + IndeX
#   Colormap material with the correct (VERIFIED) shape.  NOTE the internal rel is LAYER-RELATIVE
#   (`</Volume/density>`), NOT the composed `</World/Volume/density>`: USD remaps it under the
#   reference target on composition, and an absolute layer path renders BLACK (Task 2 Step 1).
# Part 2 (subprocess): author_vdb_volume! errors CLEARLY when IndeX is not enabled (no volume env
#   → OV._index_enabled() false), rather than authoring a prim that would silently render black.

using Test
using OmniverseMakie: _vdb_volume_usda

@testset "Volumes: _vdb_volume_usda snippet (pure)" begin
    s = _vdb_volume_usda("/data/torus.vdb"; prim_path = "/World/Volume",
                         field = "density", field_dtype = "float",
                         colormap = :viridis, colorrange = (0.0, 1.0))
    @test occursin("def Volume \"Volume\"", s)
    # layer-relative (USD remaps under prim_path on reference; absolute renders black — verified)
    @test occursin("rel field:density = </Volume/density>", s)
    @test occursin("def OpenVDBAsset \"density\"", s)
    @test occursin("token fieldName = \"density\"", s)
    @test occursin("asset filePath = @/data/torus.vdb@", s)
    @test occursin("Colormap", s) || occursin("nvindex", s)   # colormap material present
    # newline-separated metadata (the RenderProduct-prim gotcha): no two ' = ' on a metadata '(' line
    @test !occursin(r"\(\s*[^\n)]*=[^\n)]*=[^\n)]*\)", s)
end

# Subprocess: no volume env forwarded → IndeX disabled → author_vdb_volume! errors clearly.
const _VOL_DISABLED_PROG = """
using OmniverseMakie
import OmniverseMakie as OM
scene = Scene(size=(64,64)); cam3d!(scene)
screen = OM.Screen(scene)                       # no volume env → OV._ensure_index() = false
println("INDEX_ENABLED=", OM.OV._index_enabled())
try
    OM.author_vdb_volume!(screen, scene, "/nonexistent/torus.vdb")
    println("RESULT=NO_ERROR")
catch e
    println("RESULT=ERROR")
    println("ERRMSG_HAS_INDEX=", occursin("IndeX", sprint(showerror, e)))
end
close(screen)
println("DONE_DISABLED")
"""

include("helpers.jl")

@testset "Volumes: author_vdb_volume! errors clearly when IndeX disabled (subprocess)" begin
    ec, out = run_ovrtx_subprocess(_VOL_DISABLED_PROG; timeout = 300)
    @test ec == 0
    @test contains(out, "INDEX_ENABLED=false")
    @test contains(out, "RESULT=ERROR")
    @test contains(out, "ERRMSG_HAS_INDEX=true")
end
