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
# A second testset covers the all-zero (empty-field) no-op: a `volume!` of all zeros authors nothing,
# renders (near-)black, and registers no render object — it must not crash.
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
n = 20; z = zeros(Float32, n, n, n)                # all-zero field → author nothing, render nothing
scene = Scene(size=(128,128)); cam3d!(scene)
update_cam!(scene, Vec3f(38,38,22), Vec3f(0,0,0), Vec3f(0,0,1))
p = volume!(scene, -10..10, -10..10, -10..10, z; colormap=:viridis)
screen = OM.Screen(scene)
img = Makie.colorbuffer(screen)                    # must NOT crash on the empty field
lit = count(c -> (Float32(c.r) + Float32(c.g) + Float32(c.b)) > 0.04, img)
has_robj = haskey(screen.plot2robj, objectid(p))
close(screen)
println("INDEX_ENABLED=", OV._index_enabled())
println("EMPTY_LIT=", lit)                          # ~0: an all-zero field renders nothing
println("EMPTY_HAS_ROBJ=", has_robj)                # false: author returned nothing → not registered
println("OK_EMPTY_VOL")
"""

include("helpers.jl")

@testset "Volumes: live data edit (subprocess)" begin
    if !isdir(_LIBS)
        @test_skip "IndeX libs absent — live data test skipped"
    else
        # ovrtx has a known INTERMITTENT startup crash (GeometryGroup::attachToContext) that can kill
        # the child before it renders; retry until it reports a render result so the hard @tests aren't
        # flaky on that crash (mirrors volumes_plot_test.jl).
        ec = -1; out = ""
        for _ in 1:4
            ec, out = run_ovrtx_subprocess(_LIVEDATA_PROG; timeout=700, env=("OMNIVERSEMAKIE_INDEX_LIBS"=>_LIBS))
            contains(out, "MOVED=") && break
        end
        contains(out, "OK_LIVE_DATA") || @info "live data output" out
        @test ec == 0 && contains(out, "OK_LIVE_DATA")     # subprocess completed all work (no mid-run death)
        @test contains(out, "INDEX_ENABLED=true")
        @test contains(out, "DATA_MOVED=true")             # the new density re-rendered (centroid moved)
        @test contains(out, "TMP_ROTATED=true")            # a FRESH temp each edit + prior deleted (bounded)
        @test contains(out, "TMP_CLEANED_ON_CLOSE=true")   # the last temp is cleaned on close (no leak)
    end
end

@testset "Volumes: all-zero volume! no-ops cleanly (subprocess)" begin
    if !isdir(_LIBS)
        @test_skip "IndeX libs absent — empty-volume test skipped"
    else
        ec = -1; out = ""
        for _ in 1:4
            ec, out = run_ovrtx_subprocess(_EMPTYVOL_PROG; timeout=600, env=("OMNIVERSEMAKIE_INDEX_LIBS"=>_LIBS))
            contains(out, "EMPTY_LIT=") && break
        end
        contains(out, "OK_EMPTY_VOL") || @info "empty volume output" out
        @test ec == 0 && contains(out, "OK_EMPTY_VOL")     # no crash on the empty field
        @test contains(out, "EMPTY_HAS_ROBJ=false")        # author returned nothing → no render object
        m = match(r"EMPTY_LIT=(\d+)", out)
        @test m !== nothing && parse(Int, m.captures[1]) < 300   # renders (near-)black (nothing authored)
    end
end
