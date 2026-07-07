# Live volume DATA edits (subprocess, env-gated, skips if IndeX libs absent).
# `plot[4][] = new_array` re-renders the new density: a filePath write is
# never re-read (IndeX loads the grid into its own memory), so the :volume
# branch writes a fresh temp .nvdb and reloads via remove_usd! +
# add_usd_reference!. Blobs are graded in diagonally-opposite octants — a
# uniform fill renders transparent under IndeX Direct. Also: temps stay
# bounded, and an empty→fill edit late-builds and registers the plot.

using Test
# _HELPER_INDEX_LIBS is the single env-overridable IndeX-libs default.
include(joinpath(@__DIR__, "..", "helpers.jl"))
const _LIBS = _HELPER_INDEX_LIBS

const _LIVEDATA_PROG = """
using OmniverseMakie
import OmniverseMakie as OM
using OmniverseMakie: OV
# Loops wrapped in functions to dodge top-level soft-scope on accumulators.
function blob(n, cx, cy, cz; R = 0.25)
    v = zeros(Float32, n, n, n)
    CX = cx*n; CY = cy*n; CZ = cz*n; RR = R*n
    for k in 1:n, j in 1:n, i in 1:n
        d = sqrt((i-CX)^2 + (j-CY)^2 + (k-CZ)^2)
        # SPATIALLY-VARYING (uniform → IndeX-transparent)
        v[i,j,k] = d < RR ? Float32(3*(1 - d/RR)) : 0.0f0
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
# lowoct: HIGH-x HIGH-y LOW-z → renders lower-half; highoct: diagonally
# opposite (LOW-x LOW-y HIGH-z) → renders upper-area
lowoct  = blob(n, 0.75, 0.75, 0.25)
highoct = blob(n, 0.25, 0.25, 0.75)
scene = Scene(size=(256,256)); cam3d!(scene)
update_cam!(scene, Vec3f(38,38,22), Vec3f(0,0,0), Vec3f(0,0,1))
p = volume!(scene, -10..10, -10..10, -10..10, lowoct; colormap=:viridis)
screen = OM.Screen(scene)
c1 = centroid(Makie.colorbuffer(screen))   # author+build+render (octant A)
robj = screen.plot2robj[objectid(p)]
tmp1 = get(robj.meta, :vdb_tmp, "")
p[4][] = highoct                           # LIVE data edit → :volume fires
c2 = centroid(Makie.colorbuffer(screen))   # fresh .nvdb + reload + render (B)
tmp2 = get(robj.meta, :vdb_tmp, "")
old_gone = !isfile(tmp1)   # prior temp deleted each edit (bounded)
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
# GRADED blob (a UNIFORM fill renders transparent under IndeX Direct — must
# be spatially varying).
function blob(n, cx, cy, cz; R = 0.3)
    v = zeros(Float32, n, n, n); CX = cx*n; CY = cy*n; CZ = cz*n; RR = R*n
    for k in 1:n, j in 1:n, i in 1:n
        d = sqrt((i-CX)^2 + (j-CY)^2 + (k-CZ)^2); v[i,j,k] = d < RR ? Float32(3*(1 - d/RR)) : 0.0f0
    end
    return v
end
litpx(img) = count(c -> (Float32(c.r) + Float32(c.g) + Float32(c.b)) > 0.04, img)
# start ALL-ZERO → author nothing, no render object
n = 40; z = zeros(Float32, n, n, n)
scene = Scene(size=(128,128)); cam3d!(scene)
update_cam!(scene, Vec3f(38,38,22), Vec3f(0,0,0), Vec3f(0,0,1))
p = volume!(scene, -10..10, -10..10, -10..10, z; colormap=:viridis)
screen = OM.Screen(scene)
lit0 = litpx(Makie.colorbuffer(screen))   # empty → (near-)black, NOT a crash
# false: author returned nothing → not registered
has0 = haskey(screen.plot2robj, objectid(p))
# LIVE FILL with graded data → :volume fires → late build
p[4][] = blob(n, 0.5, 0.5, 0.5)
lit1 = litpx(Makie.colorbuffer(screen))   # renders the filled grid
# true: the late build registered the render object
has1 = haskey(screen.plot2robj, objectid(p))
close(screen)
println("INDEX_ENABLED=", OV._index_enabled())
println("EMPTY_LIT=", lit0)         # ~0: an all-zero field renders nothing
println("EMPTY_HAS_ROBJ=", has0)    # false before the fill
println("FILLED_LIT=", lit1)        # >0: the graded fill renders (late build)
println("FILLED_HAS_ROBJ=", has1)   # true: late build registered the plot
println("OK_EMPTY_VOL")
"""

# ---------------------------------------------------------------------------
# Reload FAILURE path: force `add_usd_reference!` to throw on reload (after
# `remove_usd!` already dropped the old layer) via a Ref-gated re-definition
# installed after the real build. Asserts:
#   (a) the failed edit does not escape (push_to_ovrtx! swallows);
#   (b) the last good frame is kept (a bare remove leaves IndeX's grid lit);
#   (c) the temp `.nvdb` count stays bounded across repeated failed edits;
#   (d) a later successful edit recovers the plot (not wedged);
#   (e) delete!/close after a failed reload does not throw, leaves 0 temps.
# TMPDIR is redirected to a fresh dir so the `.nvdb` count is this process's.
const _FAILPATH_PROG = """
using OmniverseMakie
import OmniverseMakie as OM
using OmniverseMakie: OV
import OmniverseMakie.LibOVRTX
# isolate temps → count this process's .nvdb only
const TDIR = mktempdir(); ENV["TMPDIR"] = TDIR
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
# Edit + re-render inside a function so the centroid isn't lost to top-level
# soft-scope; returns whether an exception ESCAPED push_to_ovrtx!'s catch
# (it must not) plus the resulting centroid.
function edit_render(screen, p, data)
    esc = false; c = (nb = -1, row = -1.0, col = -1.0)
    try; p[4][] = data; c = centroid(Makie.colorbuffer(screen)); catch; esc = true; end
    return (esc, c)
end
# did calling f() throw? (function → no soft-scope)
threw(f) = try; f(); false; catch; true; end
n = 40
A = blob(n, 0.75, 0.75, 0.25)   # renders lower-half
B = blob(n, 0.25, 0.25, 0.75)   # renders upper-area (diagonally opposite)
scene = Scene(size=(256,256)); cam3d!(scene)
update_cam!(scene, Vec3f(38,38,22), Vec3f(0,0,0), Vec3f(0,0,1))
p = volume!(scene, -10..10, -10..10, -10..10, A; colormap=:viridis)
screen = OM.Screen(scene)
c0 = centroid(Makie.colorbuffer(screen))   # BUILD (real add_usd_reference!)
robj = screen.plot2robj[objectid(p)]
println("BUILD_C=", c0)

# ── Fault injection: regenerate OV.add_usd_reference! FROM ITS OWN SOURCE with
# one INJECT gate spliced into the body, so this fault path can never silently
# drift from the production body — it IS that body plus the gate. INJECT[]=true
# throws before touching the renderer, so remove_usd! has run but the add
# fails. Installed after BUILD at top level: world age means the build used the
# real add and later colorbuffers see this one.
const INJECT = Ref(false)
let m = which(OV.add_usd_reference!, (OV.Renderer, AbstractString, AbstractString))
    isfile(String(m.file)) || error("add_usd_reference! source not on disk: \$(m.file)")
    src = readlines(String(m.file))
    iend = findnext(==("end"), src, m.line)
    iend === nothing && error("could not find end of add_usd_reference! at line \$(m.line)")
    fn = Meta.parse(join(src[m.line:iend], "\\n"))   # Expr(:function, sig, body)
    gate = :(Main.INJECT[] && throw(LibOVRTX.OVRTXError("add_usd_reference", "injected reload failure (test)")))
    pushfirst!(fn.args[2].args, gate)                # gate runs first in the body
    Base.eval(OV, fn)
end
println("INJECT_INSTALLED=true")

INJECT[] = true
esc1, c1 = edit_render(screen, p, B)   # failed reload (remove ok, add throws)
for _ in 1:4; edit_render(screen, p, B); end   # repeat → temps stay bounded
tmps_failed = nvdb_count()
INJECT[] = false
escR, cR = edit_render(screen, p, B)   # RECOVERY to octant B (!= build A)
moved = cR.nb > 0 ? abs(c0.row - cR.row) + abs(c0.col - cR.col) : -1.0

# ── Teardown SITE 1 — _delete_atomic_plot! (screen.jl:211), stale handle. ──
INJECT[] = true
edit_render(screen, p, A)   # fail again → p's handle is invalid
# guarded remove_usd! must NOT throw out
td_delete = threw(() -> delete!(screen, scene, p))

# ── Teardown SITE 2 — the empty! loop (screen.jl:~299) with a stale handle. ──
INJECT[] = false
p2 = volume!(scene, -10..10, -10..10, -10..10, B; colormap=:viridis)
centroid(Makie.colorbuffer(screen))   # BUILD p2 on still-open stage (real add)
INJECT[] = true
edit_render(screen, p2, A)   # fail → p2's handle is invalid
# guarded remove_usd! must NOT throw out
td_empty = threw(() -> empty!(screen))

close(screen)
final_tmps = nvdb_count()
# (a) failed edit did not escape push_to_ovrtx!
println("FAIL_NO_ESCAPE=", !esc1)
println("LAST_GOOD=", c1.nb > 100)   # (b) last good frame kept (still lit)
println("TMPS_BOUNDED=", tmps_failed <= 2)   # (c) no leak over 5 failed edits
# (d) later success re-renders the new grid
println("RECOVERED=", !escR && cR.nb > 100 && moved > 5.0)
# (e1) delete! (site 211) stale handle → no throw
println("TEARDOWN_DELETE_OK=", !td_delete)
# (e2) empty! (site 299) stale handle → no throw
println("TEARDOWN_EMPTY_OK=", !td_empty)
println("FINAL_TMPS_ZERO=", final_tmps == 0)  # no orphan temp after teardown
println("FAILEDTMPS=", tmps_failed)
println("OK_FAIL_PATH")
"""

@testset "Volumes: live data edit (subprocess)" begin
    if !isdir(_LIBS)
        @test_skip "IndeX libs absent — live data test skipped"
    else
        # An intermittent ovrtx startup crash (GeometryGroup::attachToContext)
        # can kill the child before it renders; retry until it reports a
        # render result so the hard @tests are not flaky on that crash.
        ec, out = run_ovrtx_subprocess(_LIVEDATA_PROG; timeout=700, retries=4, ready_marker="MOVED=",
                                       env=("OMNIVERSEMAKIE_INDEX_LIBS"=>_LIBS))
        contains(out, "OK_LIVE_DATA") || @info "live data output" out
        # subprocess completed all work (no mid-run death)
        @test ec == 0 && contains(out, "OK_LIVE_DATA")
        @test contains(out, "INDEX_ENABLED=true")
        # the new density re-rendered (centroid moved)
        @test contains(out, "DATA_MOVED=true")
        # a FRESH temp each edit + prior deleted (bounded)
        @test contains(out, "TMP_ROTATED=true")
        # the last temp is cleaned on close (no leak)
        @test contains(out, "TMP_CLEANED_ON_CLOSE=true")
    end
end

@testset "Volumes: empty→fill volume late-builds and renders (subprocess)" begin
    if !isdir(_LIBS)
        @test_skip "IndeX libs absent — empty→fill volume test skipped"
    else
        ec, out = run_ovrtx_subprocess(_EMPTYVOL_PROG; timeout=600, retries=4, ready_marker="EMPTY_LIT=",
                                       env=("OMNIVERSEMAKIE_INDEX_LIBS"=>_LIBS))
        contains(out, "OK_EMPTY_VOL") || @info "empty→fill volume output" out
        # no crash on the empty field or the fill
        @test ec == 0 && contains(out, "OK_EMPTY_VOL")
        # Empty author: all-zero field renders (near-)black and registers
        # no render object (author returned nothing → not registered).
        @test contains(out, "EMPTY_HAS_ROBJ=false")
        m0 = match(r"EMPTY_LIT=(\d+)", out)
        # (near-)black (nothing authored)
        @test m0 !== nothing && parse(Int, m0.captures[1]) < 300
        # Live fill with graded data: the late build renders the grid and
        # registers the plot.
        m1 = match(r"FILLED_LIT=(\d+)", out)
        # the filled grid renders non-black
        @test m1 !== nothing && parse(Int, m1.captures[1]) > 100
        # late build registered the render object
        @test contains(out, "FILLED_HAS_ROBJ=true")
    end
end

@testset "Volumes: reload FAILURE path is bounded + recoverable + teardown-safe (subprocess)" begin
    if !isdir(_LIBS)
        @test_skip "IndeX libs absent — reload-failure test skipped"
    else
        ec, out = run_ovrtx_subprocess(_FAILPATH_PROG; timeout=700, retries=4, ready_marker="BUILD_C=",
                                       env=("OMNIVERSEMAKIE_INDEX_LIBS"=>_LIBS))
        contains(out, "OK_FAIL_PATH") || @info "reload-failure output" out
        # subprocess completed (no mid-run death)
        @test ec == 0 && contains(out, "OK_FAIL_PATH")
        # the source-regenerated add_usd_reference! (with the INJECT gate)
        # installed cleanly — the gated fault path is a copy-free mirror.
        @test contains(out, "INJECT_INSTALLED=true")
        # (a) the injected add-failure did not escape the edit
        @test contains(out, "FAIL_NO_ESCAPE=true")
        # (b) the last good frame is kept on failure
        @test contains(out, "LAST_GOOD=true")
        # (c) temp .nvdb count bounded across 5 failed edits
        @test contains(out, "TMPS_BOUNDED=true")
        # (d) a later successful edit recovers the plot
        @test contains(out, "RECOVERED=true")
        # (e1) delete! (site 211) with a stale handle: no throw
        @test contains(out, "TEARDOWN_DELETE_OK=true")
        # (e2) empty! (site 299) with a stale handle: no throw
        @test contains(out, "TEARDOWN_EMPTY_OK=true")
        # no orphan temp survives teardown
        @test contains(out, "FINAL_TMPS_ZERO=true")
    end
end
