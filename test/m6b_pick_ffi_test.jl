# M6.B Task 1 — ovrtx pick-query FFI chain (subprocess).
#
# Authors a known single-mesh scene, picks the CENTER pixel (which lands on the
# mesh), reads the synthetic `ovrtx_pick_hit` render var, and asserts the hit's
# primPath resolves — via the path-dictionary vtable — to that plot's authored
# prim path, with a finite world position.  CPU-only map; no GLMakie/CUDA.
using Test

const _M6B_FFI_PROG = """
using OmniverseMakie
const OV = OmniverseMakie.OV
OM = OmniverseMakie
OM.activate!(warmup = 16)
scene = Scene(size=(128,128)); cam3d!(scene)
p = mesh!(scene, Rect3f(Point3f(-1), Vec3f(2)); color = :red)
screen = OM.Screen(scene)
# Author root + plots: `author_root_from_scene!` alone bakes only camera/lights/scopes;
# `_author_screen!` is the shared helper colorbuffer/M5 use that ALSO sets authored=true
# and `insertplots!`es each plot (its USD reference + `plot2robj` entry).
OM._author_screen!(screen, scene, scene)
# Warm a few frames so geometry is resident, then pick the center pixel.
for _ in 1:8; sr0 = OV.step!(screen.renderer, screen.product; timeout_ns = UInt64(60_000_000_000)); close(sr0); end
OV.enqueue_pick_query(screen.renderer, screen.product, (64, 64, 65, 65))
sr = OV.step!(screen.renderer, screen.product; timeout_ns = UInt64(60_000_000_000))
hits = OV.read_pick_hit(sr); close(sr)
println("HITCOUNT=", length(hits))
if !isempty(hits)
    pr = OV.path_resolver(screen.renderer)
    path = OV.resolve_prim_path(pr, hits[1].primpath_id)
    expected = screen.plot2robj[objectid(p)].prim_path
    println("PICK_PATH=", path)
    println("EXPECTED=", expected)
    println("PATH_MATCH=", path == expected)
    wp = hits[1].world_position
    println("WORLDPOS_FINITE=", all(isfinite, wp))
end
close(screen)
println("OK_PICK_FFI")
"""

include("helpers.jl")
@testset "M6.B pick FFI chain (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M6B_FFI_PROG; timeout = 400, retries = 2, ready_marker = "OK_PICK_FFI")
    @info "M6.B pick FFI output" output
    @test exitcode == 0
    @test contains(output, "OK_PICK_FFI")
    m = match(r"HITCOUNT=(\d+)", output)
    @test m !== nothing && parse(Int, m.captures[1]) >= 1
    @test contains(output, "PATH_MATCH=true")
    @test contains(output, "WORLDPOS_FINITE=true")
end
