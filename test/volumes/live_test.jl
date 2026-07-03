# Volumes M2 Task 5 — LIVE volume DATA edits (subprocess, env-gated, skip-if-absent).
#
# `plot[4][] = new_array` re-renders the NEW density field.  The diff node tracks the `:volume`
# compute output (the converted scalar-data output — Task-5 spike-verified to RESOLVE and FIRE on
# `plot[4][] =`); `push_to_ovrtx!`'s `:volume` branch re-writes a FRESH temp `.nvdb` and RELOADS it
# via `remove_usd!` + `add_usd_reference!` (the Task-5 spike proved a `filePath` write does NOT update
# — IndeX loads the grid into its own memory and evicts the file, so an on-disk change is never
# re-read; a fresh layer + fresh temp path is the mechanism that reliably shows the new data).
#
# Colors are OFF for M2 (IndeX Direct shades scalar grids grayscale — Task 3 degrade), so this tests
# live DATA only, which is fully meaningful in grayscale.  Two GRADED blobs in DIAGONALLY-OPPOSITE
# octants (a UNIFORM fill would render transparent under IndeX Direct — Task 1 carry-forward): the
# baseline blob renders lower-half, the edited blob upper-area, so BOTH render non-black AND the
# lit-pixel centroid MOVES detectably (spike: MOVED ~26 px).  Also asserts the temp `.nvdb` count
# stays BOUNDED (the prior temp is deleted each edit; the last one is cleaned on close).
#
# A second testset covers the empty→fill self-heal (Task B3): a `volume!` of all zeros authors nothing
# (near-black, no render object), then a live FILL with GRADED data re-renders via the diff-node
# late-build path (author_usd_prim! re-runs once `:volume` changes) and registers the render object.
# Skips cleanly when the Kit IndeX libs dir is absent (CI without them stays green).

using Test
const _LIBS = get(ENV, "OMNIVERSEMAKIE_INDEX_LIBS", "/home/juliahub/.local/share/ov/data/exts/v2/omni.index.libs-1287db94366cf6fe")

const _LIVEDATA_PROG = """
using OmniverseMakie
import OmniverseMakie as OM
using OmniverseMakie: OV
# Loops wrapped in functions to dodge Julia top-level soft-scope on the accumulators (as in
# test/volumes_plot_test.jl).
function blob(n, cx, cy, cz; R = 0.25)
    v = zeros(Float32, n, n, n)
    CX = cx*n; CY = cy*n; CZ = cz*n; RR = R*n
    for k in 1:n, j in 1:n, i in 1:n
        d = sqrt((i-CX)^2 + (j-CY)^2 + (k-CZ)^2)
        v[i,j,k] = d < RR ? Float32(3*(1 - d/RR)) : 0.0f0    # SPATIALLY-VARYING (uniform → IndeX-transparent)
    end
    return v
end
function centroid(img)
    H, W = size(img); sr = 0.0; sc = 0.0; nb = 0
    for h in 1:H, w in 1:W
        cc = img[h, w]
        if (Float32(cc.r) + Float32(cc.g) + Float32(cc.b)) > 0.04
            sr += h; sc += w; nb += 1
        end
    end
    return (nb = nb, row = nb > 0 ? sr/nb : -1.0, col = nb > 0 ? sc/nb : -1.0)
end
n = 40
lowoct  = blob(n, 0.75, 0.75, 0.25)    # HIGH-x HIGH-y LOW-z  → renders lower-half
highoct = blob(n, 0.25, 0.25, 0.75)    # LOW-x  LOW-y  HIGH-z → renders upper-area (diagonally opposite)
scene = Scene(size=(256,256)); cam3d!(scene)
update_cam!(scene, Vec3f(38,38,22), Vec3f(0,0,0), Vec3f(0,0,1))
p = volume!(scene, -10..10, -10..10, -10..10, lowoct; colormap=:viridis)
screen = OM.Screen(scene)
c1 = centroid(Makie.colorbuffer(screen))          # author + build + render (octant A)
robj = screen.plot2robj[objectid(p)]
tmp1 = get(robj.meta, :vdb_tmp, "")
p[4][] = highoct                                  # LIVE data edit → :volume fires
c2 = centroid(Makie.colorbuffer(screen))          # re-write fresh .nvdb + reload + render (octant B)
tmp2 = get(robj.meta, :vdb_tmp, "")
old_gone = !isfile(tmp1)                           # prior temp deleted each edit (bounded)
new_here = isfile(tmp2)
close(screen)
both_gone = !isfile(tmp1) && !isfile(tmp2)         # last temp cleaned on close
moved = abs(c1.row - c2.row) + abs(c1.col - c2.col)
println("INDEX_ENABLED=", OV._index_enabled())
println("CENTROID1=", c1)
println("CENTROID2=", c2)
println("MOVED=", moved)
println("DATA_MOVED=", moved > 5.0 && c1.nb > 100 && c2.nb > 100)
println("TMP_ROTATED=", (tmp1 != tmp2) && old_gone && new_here)
println("TMP_CLEANED_ON_CLOSE=", both_gone)
println("OK_LIVE_DATA")
"""

const _EMPTYVOL_PROG = """
using OmniverseMakie
import OmniverseMakie as OM
using OmniverseMakie: OV
# GRADED blob (a UNIFORM fill renders transparent under IndeX Direct — must be spatially varying).
function blob(n, cx, cy, cz; R = 0.3)
    v = zeros(Float32, n, n, n); CX = cx*n; CY = cy*n; CZ = cz*n; RR = R*n
    for k in 1:n, j in 1:n, i in 1:n
        d = sqrt((i-CX)^2 + (j-CY)^2 + (k-CZ)^2); v[i,j,k] = d < RR ? Float32(3*(1 - d/RR)) : 0.0f0
    end
    return v
end
litpx(img) = count(c -> (Float32(c.r) + Float32(c.g) + Float32(c.b)) > 0.04, img)
n = 40; z = zeros(Float32, n, n, n)                # start ALL-ZERO → author nothing, no render object
scene = Scene(size=(128,128)); cam3d!(scene)
update_cam!(scene, Vec3f(38,38,22), Vec3f(0,0,0), Vec3f(0,0,1))
p = volume!(scene, -10..10, -10..10, -10..10, z; colormap=:viridis)
screen = OM.Screen(scene)
lit0 = litpx(Makie.colorbuffer(screen))            # empty field → (near-)black, must NOT crash
has0 = haskey(screen.plot2robj, objectid(p))       # false: author returned nothing → not registered
p[4][] = blob(n, 0.5, 0.5, 0.5)                    # LIVE FILL with graded data → :volume fires → late build
lit1 = litpx(Makie.colorbuffer(screen))            # now RENDERS the filled grid (the B3 self-heal)
has1 = haskey(screen.plot2robj, objectid(p))       # true: the late build registered the render object
close(screen)
println("INDEX_ENABLED=", OV._index_enabled())
println("EMPTY_LIT=", lit0)                         # ~0: an all-zero field renders nothing
println("EMPTY_HAS_ROBJ=", has0)                    # false before the fill
println("FILLED_LIT=", lit1)                         # >0: the graded fill renders (late build)
println("FILLED_HAS_ROBJ=", has1)                   # true: late build registered the plot
println("OK_EMPTY_VOL")
"""

# ─────────────────────────────────────────────────────────────────────────────────────────────
# Task-5 reviewer finding — reload FAILURE path (the catch branch had zero coverage).
#
# Forces `add_usd_reference!` to throw on a reload (after `remove_usd!` already dropped the old
# layer — the exact "remove succeeds, add throws" split the finding describes) via a Ref-gated
# re-definition (body mirrors OV.add_usd_reference!) installed AFTER the real build, so the build
# uses the real add and every subsequent reload throws while the flag is set.  Asserts:
#   (a) the failed edit does NOT escape (push_to_ovrtx! swallows) — colorbuffer keeps rendering;
#   (b) the last good frame is kept (a bare remove doesn't clear IndeX's loaded grid → still lit);
#   (c) the temp `.nvdb` count stays BOUNDED across several repeated failed edits (orphan rm'd);
#   (d) a later SUCCESSFUL edit RECOVERS the plot (renders the new grid — not wedged);
#   (e) delete!/close after a failed reload does NOT throw (guarded stale handle) and leaves 0 temps.
# TMPDIR is redirected to a fresh dir so the `.nvdb` count is this process's only (tempname honors it).
const _FAILPATH_PROG = """
using OmniverseMakie
import OmniverseMakie as OM
using OmniverseMakie: OV
import OmniverseMakie.LibOVRTX
const TDIR = mktempdir(); ENV["TMPDIR"] = TDIR          # isolate temps → count this process's .nvdb only
nvdb_count() = count(f -> endswith(f, ".nvdb"), readdir(TDIR))
function blob(n, cx, cy, cz; R = 0.25)
    v = zeros(Float32, n, n, n); CX = cx*n; CY = cy*n; CZ = cz*n; RR = R*n
    for k in 1:n, j in 1:n, i in 1:n
        d = sqrt((i-CX)^2 + (j-CY)^2 + (k-CZ)^2); v[i,j,k] = d < RR ? Float32(3*(1 - d/RR)) : 0.0f0
    end
    return v
end
function centroid(img)
    H, W = size(img); sr = 0.0; sc = 0.0; nb = 0
    for h in 1:H, w in 1:W
        cc = img[h, w]
        (Float32(cc.r)+Float32(cc.g)+Float32(cc.b)) > 0.04 && (sr += h; sc += w; nb += 1)
    end
    return (nb = nb, row = nb > 0 ? sr/nb : -1.0, col = nb > 0 ? sc/nb : -1.0)
end
# Edit + re-render inside a function so the centroid isn't lost to top-level soft-scope; returns
# whether an exception ESCAPED push_to_ovrtx!'s catch (it must not) plus the resulting centroid.
function edit_render(screen, p, data)
    esc = false; c = (nb = -1, row = -1.0, col = -1.0)
    try; p[4][] = data; c = centroid(Makie.colorbuffer(screen)); catch; esc = true; end
    return (esc, c)
end
threw(f) = try; f(); false; catch; true; end            # did calling f() throw? (function → no soft-scope)
n = 40
A = blob(n, 0.75, 0.75, 0.25)                            # renders lower-half
B = blob(n, 0.25, 0.25, 0.75)                            # renders upper-area (diagonally opposite)
scene = Scene(size=(256,256)); cam3d!(scene)
update_cam!(scene, Vec3f(38,38,22), Vec3f(0,0,0), Vec3f(0,0,1))
p = volume!(scene, -10..10, -10..10, -10..10, A; colormap=:viridis)
screen = OM.Screen(scene)
c0 = centroid(Makie.colorbuffer(screen))                 # BUILD (real add_usd_reference!)
robj = screen.plot2robj[objectid(p)]
println("BUILD_C=", c0)

# ── Fault injection: Ref-gated re-definition of add_usd_reference! (mirrors the OV body). ──
# flag=false → the real add; flag=true → throw an OVRTXError WITHOUT touching the renderer, i.e.
# remove_usd! already ran but the add throws.  Installed here (after BUILD) at top level, so the
# later top-level colorbuffer sees it (world age) and the build above used the real add.
const INJECT = Ref(false)
@eval OV function add_usd_reference!(r::Renderer, usda::AbstractString, prim_path::AbstractString)
    r.alive || error("add_usd_reference! on a closed Renderer")
    \$(INJECT)[] && throw(LibOVRTX.OVRTXError("add_usd_reference", "injected reload failure (Task-5 test)"))
    layer_s = String(usda); path_s = String(prim_path)
    h = Ref{LibOVRTX.ovrtx_usd_handle_t}(0)
    GC.@preserve layer_s path_s begin
        enqueue_wait(r, "add_usd_reference") do
            LibOVRTX.ovrtx_add_usd_reference_from_string(
                r.ptr, LibOVRTX.ovx_string(layer_s), LibOVRTX.ovx_string(path_s), h)
        end
    end
    return h[]
end

INJECT[] = true
esc1, c1 = edit_render(screen, p, B)                     # failed reload (remove ok, add throws)
for _ in 1:4; edit_render(screen, p, B); end            # repeat → temps must stay bounded
tmps_failed = nvdb_count()
INJECT[] = false
escR, cR = edit_render(screen, p, B)                     # RECOVERY to octant B (!= build A)
moved = cR.nb > 0 ? abs(c0.row - cR.row) + abs(c0.col - cR.col) : -1.0

# ── Teardown SITE 1 — _delete_atomic_plot! (screen.jl:211) with a stale handle. ──
INJECT[] = true
edit_render(screen, p, A)                                # fail again → p's handle is invalid
td_delete = threw(() -> delete!(screen, scene, p))       # guarded remove_usd! must NOT throw out

# ── Teardown SITE 2 — the empty! loop (screen.jl:~299) with a stale handle. ──
INJECT[] = false
p2 = volume!(scene, -10..10, -10..10, -10..10, B; colormap=:viridis)
centroid(Makie.colorbuffer(screen))                      # BUILD p2 on the still-open stage (real add)
INJECT[] = true
edit_render(screen, p2, A)                               # fail → p2's handle is invalid
td_empty = threw(() -> empty!(screen))                   # guarded remove_usd! must NOT throw out

close(screen)
final_tmps = nvdb_count()
println("FAIL_NO_ESCAPE=", !esc1)                        # (a) failed edit did not escape push_to_ovrtx!
println("LAST_GOOD=", c1.nb > 100)                       # (b) last good frame kept (still lit)
println("TMPS_BOUNDED=", tmps_failed <= 2)               # (c) no unbounded leak over 5 failed edits
println("RECOVERED=", !escR && cR.nb > 100 && moved > 5.0)  # (d) later success re-renders the new grid
println("TEARDOWN_DELETE_OK=", !td_delete)               # (e1) delete! (site 211) stale handle → no throw
println("TEARDOWN_EMPTY_OK=", !td_empty)                 # (e2) empty! (site 299) stale handle → no throw
println("FINAL_TMPS_ZERO=", final_tmps == 0)             #      no orphan temp survives teardown
println("FAILEDTMPS=", tmps_failed)
println("OK_FAIL_PATH")
"""

include(joinpath(@__DIR__, "..", "helpers.jl"))

@testset "Volumes: live data edit (subprocess)" begin
    if !isdir(_LIBS)
        @test_skip "IndeX libs absent — live data test skipped"
    else
        # ovrtx has a known INTERMITTENT startup crash (GeometryGroup::attachToContext) that can kill
        # the child before it renders; retry until it reports a render result so the hard @tests
        # are not flaky on that crash.
        ec, out = run_ovrtx_subprocess(_LIVEDATA_PROG; timeout=700, retries=4, ready_marker="MOVED=",
                                       env=("OMNIVERSEMAKIE_INDEX_LIBS"=>_LIBS))
        contains(out, "OK_LIVE_DATA") || @info "live data output" out
        @test ec == 0 && contains(out, "OK_LIVE_DATA")     # subprocess completed all work (no mid-run death)
        @test contains(out, "INDEX_ENABLED=true")
        @test contains(out, "DATA_MOVED=true")             # the new density re-rendered (centroid moved)
        @test contains(out, "TMP_ROTATED=true")            # a FRESH temp each edit + prior deleted (bounded)
        @test contains(out, "TMP_CLEANED_ON_CLOSE=true")   # the last temp is cleaned on close (no leak)
    end
end

@testset "Volumes: empty→fill volume late-builds and renders (subprocess)" begin
    if !isdir(_LIBS)
        @test_skip "IndeX libs absent — empty→fill volume test skipped"
    else
        ec, out = run_ovrtx_subprocess(_EMPTYVOL_PROG; timeout=600, retries=4, ready_marker="EMPTY_LIT=",
                                       env=("OMNIVERSEMAKIE_INDEX_LIBS"=>_LIBS))
        contains(out, "OK_EMPTY_VOL") || @info "empty→fill volume output" out
        @test ec == 0 && contains(out, "OK_EMPTY_VOL")     # no crash on the empty field or the fill
        # Empty author: all-zero field renders (near-)black and registers no render object.
        @test contains(out, "EMPTY_HAS_ROBJ=false")        # author returned nothing → not registered
        m0 = match(r"EMPTY_LIT=(\d+)", out)
        @test m0 !== nothing && parse(Int, m0.captures[1]) < 300   # (near-)black (nothing authored)
        # Live FILL with graded data: the late build renders the grid AND registers the plot (B3 self-heal).
        m1 = match(r"FILLED_LIT=(\d+)", out)
        @test m1 !== nothing && parse(Int, m1.captures[1]) > 100   # the filled grid renders non-black
        @test contains(out, "FILLED_HAS_ROBJ=true")        # late build registered the render object
    end
end

@testset "Volumes: reload FAILURE path is bounded + recoverable + teardown-safe (subprocess)" begin
    if !isdir(_LIBS)
        @test_skip "IndeX libs absent — reload-failure test skipped"
    else
        ec, out = run_ovrtx_subprocess(_FAILPATH_PROG; timeout=700, retries=4, ready_marker="BUILD_C=",
                                       env=("OMNIVERSEMAKIE_INDEX_LIBS"=>_LIBS))
        contains(out, "OK_FAIL_PATH") || @info "reload-failure output" out
        @test ec == 0 && contains(out, "OK_FAIL_PATH")   # subprocess completed (no mid-run death)
        @test contains(out, "FAIL_NO_ESCAPE=true")       # (a) the injected add-failure did not escape the edit
        @test contains(out, "LAST_GOOD=true")            # (b) the last good frame is kept on failure
        @test contains(out, "TMPS_BOUNDED=true")           # (c) temp .nvdb count bounded across 5 failed edits
        @test contains(out, "RECOVERED=true")              # (d) a later successful edit recovers the plot
        @test contains(out, "TEARDOWN_DELETE_OK=true")     # (e1) delete! (site 211) with a stale handle: no throw
        @test contains(out, "TEARDOWN_EMPTY_OK=true")      # (e2) empty! (site 299) with a stale handle: no throw
        @test contains(out, "FINAL_TMPS_ZERO=true")        #      no orphan temp survives teardown
    end
end
