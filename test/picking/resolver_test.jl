using Test

# ---------------------------------------------------------------------------
# Task A3 — PathResolver lifetime + Screen path-resolver cache invalidation.
#
# The cached `screen.path_resolver` captures the ovrtx path dictionary as of the CURRENT
# stage composition, so it MUST be dropped whenever the composition changes
# (add_usd_reference! / remove_usd!): plot delete, plot insert, empty!, volume reload.
# Before the fix the cache was reused for the renderer's whole life, so a pick after a
# delete+re-add could resolve against a stale dictionary.
#
# Subprocess (needs the GPU, but no IndeX/GLMakie/CUDA):
#   1. author a mesh A, pick it → builds + caches the resolver.
#   2. delete! A                        → RESOLVER_INVALIDATED_ON_DELETE  (RED pre-fix)
#   3. insert! a NEW mesh B (same spot) → pick B resolves to B, not A     (RED pre-fix if
#                                          the stale dictionary mis-resolves)
#   4. with the resolver re-cached, insert! a mesh C
#                                       → RESOLVER_INVALIDATED_ON_INSERT   (RED pre-fix)
#
# The two RESOLVER_INVALIDATED_* checks are the deterministic RED discriminators (they hold
# regardless of whether the GPU dictionary handle happens to be live); PICK_B_IS_B is the
# user-visible correctness the fix guarantees.  Built on the picking/pick_test.jl pick shape.
# ---------------------------------------------------------------------------
const _A3_RESOLVER_PROG = """
using OmniverseMakie
import OmniverseMakie as OM
using OmniverseMakie: OV
OM.activate!(warmup = 24)

scene = Scene(size = (200, 200))
cam3d!(scene)
update_cam!(scene, Vec3d(6, 6, 4), Vec3d(0, 0, 0), Vec3d(0, 0, 1))
box = Rect3f(Point3f(-1, -1, -1), Vec3f(2))
a = mesh!(scene, box; color = :red)

screen = OM.Screen(scene)
OM._author_screen!(screen, scene, scene)
for _ in 1:8
    sr = OV.step!(screen.renderer, screen.product; timeout_ns = UInt64(60_000_000_000)); close(sr)
end

pxy = Makie.Vec{2,Float64}(Makie.project(scene, Point3f(0, 0, 0))...)
pa, _ = Makie.pick(scene, screen, pxy)
println("PICK_A_IS_A=", pa === a)
println("RESOLVER_CACHED_AFTER_PICK_A=", screen.path_resolver !== nothing)

# --- composition change 1: DELETE A (remove_usd! → invalidate) ---
delete!(screen, scene, a)
println("RESOLVER_INVALIDATED_ON_DELETE=", screen.path_resolver === nothing)

# --- composition change 2: ADD a NEW mesh B at the same spot (register → add_usd_reference!) ---
b = mesh!(scene, box; color = :blue)
insert!(screen, scene, b)
println("B_REGISTERED=", haskey(screen.plot2robj, objectid(b)))
println("A_UNREGISTERED=", !haskey(screen.plot2robj, objectid(a)))
OV.reset!(screen.renderer)
for _ in 1:8
    sr = OV.step!(screen.renderer, screen.product; timeout_ns = UInt64(60_000_000_000)); close(sr)
end
pb, _ = Makie.pick(scene, screen, pxy)
println("PICK_B_IS_B=", pb === b)
println("PICK_B_NOT_A=", pb !== a)
println("RESOLVER_RECACHED_AFTER_PICK_B=", screen.path_resolver !== nothing)

# --- composition change 3: isolate the insert seam — add C while the resolver is cached ---
cplot = mesh!(scene, Rect3f(Point3f(3, 3, 3), Vec3f(0.5)); color = :green)
insert!(screen, scene, cplot)
println("RESOLVER_INVALIDATED_ON_INSERT=", screen.path_resolver === nothing)

close(screen)
println("OK_STALE_RESOLVER")
"""

include(joinpath(@__DIR__, "..", "helpers.jl"))
@testset "A3 PathResolver cache invalidation on composition change (subprocess)" begin
    # Retry the known intermittent ovrtx GeometryGroup::attachToContext startup crash:
    # re-run until the child reaches its first pick.
    _, out = run_ovrtx_subprocess(_A3_RESOLVER_PROG; timeout = 600, retries = 4,
                                  ready_marker = "PICK_A_IS_A=")
    contains(out, "OK_STALE_RESOLVER") || @info "A3 resolver output" out
    @test contains(out, "OK_STALE_RESOLVER")                # subprocess completed all work
    @test contains(out, "PICK_A_IS_A=true")                 # baseline pick resolves the first plot
    @test contains(out, "RESOLVER_CACHED_AFTER_PICK_A=true")
    # The deterministic fix assertions (RED before the invalidation is wired):
    @test contains(out, "RESOLVER_INVALIDATED_ON_DELETE=true")
    @test contains(out, "RESOLVER_INVALIDATED_ON_INSERT=true")
    # The user-visible correctness: a pick after delete+re-add resolves the NEW plot.
    @test contains(out, "B_REGISTERED=true")
    @test contains(out, "A_UNREGISTERED=true")
    @test contains(out, "PICK_B_IS_B=true")
    @test contains(out, "PICK_B_NOT_A=true")
    @test contains(out, "RESOLVER_RECACHED_AFTER_PICK_B=true")
end
