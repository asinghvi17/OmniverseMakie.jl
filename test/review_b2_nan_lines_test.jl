using Test

# ---------------------------------------------------------------------------
# Review Task B2 — NaN-separated lines.
#
# Makie's standard broken-line idiom (NaN separators — contour output etc.) must NOT
# poison the USDA.  A NaN point previously flowed through: `_bbox_diag` propagated NaN
# (its `d < 1e-8` guard is false for NaN) → `_curve_width` emitted `widths = [NaN]`, and
# the Lines author always emitted ONE polyline `curveVertexCounts=[n]` INCLUDING the NaN
# points.
#
# Fix (pure surface, tested here):
#   `_split_nan_runs(pts) -> (finite_pts, counts, keep)` — contiguous finite runs of
#     length ≥ 2 become curve entries (BasisCurves is natively multi-curve); runs of 1 are
#     DROPPED; `keep` is the per-input-vertex survivor mask so a parallel per-vertex colour
#     array is filtered IDENTICALLY (index-aligned).  No runs → `(empty, empty, …)`.
#   `_finite_segments(pts)` — LineSegments variant: drop any 2-pt segment with a non-finite
#     endpoint.
#   `_bbox_diag` skips non-finite points → `_curve_width` stays finite.
#
# GOLDEN (captured from the PRE-change code on FINITE data): a no-NaN Lines / LineSegments
# emit must stay BYTE-identical.  The golden embeds the width (from `_bbox_diag`), so it
# also locks the `_bbox_diag` hardening as byte-preserving on finite input.
#
# PURE (no GPU): split + USDA string emission only.  The subprocess render test (a
# NaN-separated `lines!` renders two clusters with a GAP; a live NaN move re-renders) is
# review_b2_nan_render_test.jl.
# ---------------------------------------------------------------------------

import OmniverseMakie as OM

# Identity model (matches usda_matrix4d(Mat4f(I)) used to capture the golden).
const _B2_I4 = [1.0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
# The golden reference points (all finite).
_b2_gpts() = Point3f[(0, 0, 0), (1, 2, 0), (3, 1, 0), (4, 4, 0)]

# Byte-for-byte no-NaN Lines USDA (ONE curve of 4 points), captured from the pre-change
# `_usda_basiscurves(pts, [4], _curve_width(pts, 2f0), (1,0,0), "constant")`.
const _GOLDEN_LINES_USDA = """#usda 1.0
( defaultPrim = "curve" )
def BasisCurves "curve"
{
    uniform token type = "linear"
    int[] curveVertexCounts = [4]
    point3f[] points = [(0.0, 0.0, 0.0), (1.0, 2.0, 0.0), (3.0, 1.0, 0.0), (4.0, 4.0, 0.0)]
    float[] widths = [0.11313708] (
        interpolation = "constant"
    )
    color3f[] primvars:displayColor = [(1.0, 0.0, 0.0)] (
        interpolation = "constant"
    )
    matrix4d xformOp:transform = ( (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0) )
    uniform token[] xformOpOrder = ["xformOp:transform"]
}
"""

# Byte-for-byte no-NaN LineSegments USDA (2 segments), same points.
const _GOLDEN_SEGS_USDA = """#usda 1.0
( defaultPrim = "curve" )
def BasisCurves "curve"
{
    uniform token type = "linear"
    int[] curveVertexCounts = [2, 2]
    point3f[] points = [(0.0, 0.0, 0.0), (1.0, 2.0, 0.0), (3.0, 1.0, 0.0), (4.0, 4.0, 0.0)]
    float[] widths = [0.11313708] (
        interpolation = "constant"
    )
    color3f[] primvars:displayColor = [(1.0, 0.0, 0.0)] (
        interpolation = "constant"
    )
    matrix4d xformOp:transform = ( (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0) )
    uniform token[] xformOpOrder = ["xformOp:transform"]
}
"""

@testset "B2 _split_nan_runs: finite runs ≥2 → curves, runs of 1 dropped, mask aligned" begin
    N = Point3f(NaN, NaN, NaN)
    a = Point3f(0, 0, 0); b = Point3f(1, 0, 0); c = Point3f(2, 0, 0); d = Point3f(3, 0, 0)

    # no-NaN: unchanged, one curve, full keep, SAME eltype (byte-identity precondition).
    f, cnt, keep = OM._split_nan_runs([a, b, c])
    @test f == [a, b, c] && cnt == [3] && keep == Bool[1, 1, 1]
    @test eltype(f) === Point3f

    # leading / trailing / middle / consecutive NaN.
    @test OM._split_nan_runs([N, a, b])[1:2]       == ([a, b], [2])
    @test OM._split_nan_runs([a, b, N])[1:2]       == ([a, b], [2])
    @test OM._split_nan_runs([a, b, N, c, d])      == ([a, b, c, d], [2, 2], Bool[1, 1, 0, 1, 1])
    @test OM._split_nan_runs([a, b, N, N, c, d])[1:2] == ([a, b, c, d], [2, 2])

    # isolated finite point (run length 1) DROPPED (leading and trailing).
    @test OM._split_nan_runs([a, N, b, c]) == ([b, c], [2], Bool[0, 0, 1, 1])
    @test OM._split_nan_runs([a, b, N, c]) == ([a, b], [2], Bool[1, 1, 0, 0])

    # all-NaN / empty / single point → no runs.
    @test OM._split_nan_runs([N, N])  == (Point3f[], Int[], Bool[0, 0])
    @test OM._split_nan_runs(Point3f[]) == (Point3f[], Int[], Bool[])
    @test OM._split_nan_runs([a])     == (Point3f[], Int[], Bool[0])

    # Inf and a PARTIAL-NaN point are separators too (any non-finite coord).
    @test OM._split_nan_runs([a, b, Point3f(Inf, 0, 0), c, d])[1:2] == ([a, b, c, d], [2, 2])
    @test OM._split_nan_runs([a, b, Point3f(1, NaN, 0), c, d])[1:2] == ([a, b, c, d], [2, 2])
end

@testset "B2 _finite_segments: drop any segment with a non-finite endpoint" begin
    N = Point3f(NaN, NaN, NaN)
    a = Point3f(0, 0, 0); b = Point3f(1, 0, 0); c = Point3f(2, 0, 0); d = Point3f(3, 0, 0)

    @test OM._finite_segments([a, b, c, d]) == ([a, b, c, d], [2, 2], Bool[1, 1, 1, 1])
    @test OM._finite_segments([a, b, N, d])  == ([a, b], [2], Bool[1, 1, 0, 0])   # seg 2 dropped
    @test OM._finite_segments([a, N, c, d])  == ([c, d], [2], Bool[0, 0, 1, 1])   # seg 1 dropped
    @test OM._finite_segments([N, a, b, N])  == (Point3f[], Int[], Bool[0, 0, 0, 0])
    @test OM._finite_segments([a, b, c])     == ([a, b], [2], Bool[1, 1, 0])       # odd trailing pt ignored
    @test OM._finite_segments(Point3f[])     == (Point3f[], Int[], Bool[])
end

@testset "B2 golden: no-NaN Lines emit byte-identical (split is a no-op on finite data)" begin
    pts = _b2_gpts()
    f, cnt, keep = OM._split_nan_runs(pts)
    @test f == pts && cnt == [length(pts)] && all(keep)
    usda = OM._usda_basiscurves(f, cnt, OM._curve_width(f, 2.0f0),
                                (1.0f0, 0.0f0, 0.0f0), "constant"; model = _B2_I4)
    @test usda == _GOLDEN_LINES_USDA
end

@testset "B2 golden: no-NaN LineSegments emit byte-identical" begin
    pts = _b2_gpts()
    f, cnt, keep = OM._finite_segments(pts)
    @test f == pts && cnt == [2, 2] && all(keep)
    usda = OM._usda_basiscurves(f, cnt, OM._curve_width(f, 2.0f0),
                                (1.0f0, 0.0f0, 0.0f0), "constant"; model = _B2_I4)
    @test usda == _GOLDEN_SEGS_USDA
end

@testset "B2 _bbox_diag skips non-finite → finite, byte-preserving on finite input" begin
    a = Point3f(0, 0, 0); b = Point3f(1, 0, 0); c = Point3f(2, 0, 0)
    N = Point3f(NaN, NaN, NaN)
    # Finite bbox unchanged (locks the golden width above): diag of the golden pts.
    @test OM._bbox_diag(_b2_gpts()) ≈ sqrt(32.0)
    # A NaN point is SKIPPED: same diag as without it (was NaN before the fix).
    @test OM._bbox_diag([a, b, N, c]) == OM._bbox_diag([a, b, c])
    @test isfinite(OM._bbox_diag([a, b, N, c]))
    # No finite point at all → the empty fallback (1.0), not NaN.
    @test OM._bbox_diag([N, N]) == 1.0
end

@testset "B2 _curve_width stays finite through a NaN-bearing point cloud" begin
    a = Point3f(0, 0, 0); b = Point3f(4, 4, 0); N = Point3f(NaN, NaN, NaN)
    w = OM._curve_width([a, N, b], 2.0f0)
    @test isfinite(w) && w > 0
    @test w == OM._curve_width([a, b], 2.0f0)   # the NaN contributes nothing to the extent
end

@testset "B2 per-vertex colours filtered by the SAME mask stay index-aligned" begin
    a = Point3f(0, 0, 0); b = Point3f(1, 0, 0); c = Point3f(2, 0, 0); d = Point3f(3, 0, 0)
    N = Point3f(NaN, NaN, NaN)
    cols = NTuple{3,Float32}[(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1)]  # 5, one per pt
    f, cnt, keep = OM._split_nan_runs([a, b, N, c, d])
    filtered = cols[keep]                                # the author's colour-filter step
    @test length(filtered) == length(f) == sum(cnt)      # aligned: one colour per surviving vertex
    @test filtered == NTuple{3,Float32}[(1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 1, 1)]  # blue (NaN pt) dropped

    # End-to-end at the emitter: the split counts + filtered colours author a clean 2-curve prim.
    usda = OM._usda_basiscurves(f, cnt, 0.5f0, filtered, "vertex"; model = _B2_I4)
    @test occursin("int[] curveVertexCounts = [2, 2]", usda)
    @test occursin("primvars:displayColor = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1.0)]", usda)
    @test !occursin("nan", lowercase(usda))              # no NaN leaked into the USDA
end

# ---------------------------------------------------------------------------
# Subprocess render: a NaN-separated `lines!` RENDERS as two clusters with a visible GAP
# (the multi-curve split — no connecting segment across the NaN), and a live edit that MOVES
# the NaN re-renders without error.
#
# Two short vertical bars far apart (x = ±8), NaN-separated → BasisCurves curveVertexCounts=[2,2].
# Before the fix a NaN point poisoned the layer (widths=[NaN] → nothing rendered).  The GAP
# assertion (both thirds lit, centre band DARK) guards the split: were the NaN merely dropped into
# ONE curve, the two clusters would be JOINED by a segment across the middle (no gap).
# ---------------------------------------------------------------------------

const _B2_RENDER_PROG = raw"""
using OmniverseMakie, ColorTypes
import OmniverseMakie as OM

OM.activate!(warmup = 40)

NANP = Point3f(NaN, NaN, NaN)
p1 = [Point3f(-8, -3, 0), Point3f(-8, 3, 0), NANP, Point3f(8, -3, 0), Point3f(8, 3, 0)]  # [2,2]
scene = Scene(size = (320, 320)); cam3d!(scene)
update_cam!(scene, Vec3f(0, 0, 40), Vec3f(0, 0, 0), Vec3f(0, 1, 0))
l = lines!(scene, p1; color = :red, linewidth = 8)
screen = OM.Screen(scene)

function stats(img)
    H, W = size(img); lit = 0; sr = 0.0; sc = 0.0; left = 0; right = 0; mid = 0
    for h in 1:H, w in 1:W
        c = img[h, w]
        if Float32(red(c)) + Float32(green(c)) + Float32(blue(c)) > 0.05
            lit += 1; sr += h; sc += w
            w <= W ÷ 3        && (left += 1)
            w >= 2W ÷ 3       && (right += 1)
            (w > 2W ÷ 5 && w < 3W ÷ 5) && (mid += 1)   # central band, between the two bars
        end
    end
    return (lit = lit, row = lit > 0 ? sr / lit : -1.0, col = lit > 0 ? sc / lit : -1.0,
            left = left, right = right, mid = mid)
end

s1 = stats(Makie.colorbuffer(screen))
println("S1=", s1)

# Live edit: MOVE the NaN + change the finite point count (4→5 → the one-shot resize path) and
# relocate the second cluster to the centre.  Must re-render without error.
p2 = [Point3f(-8, -3, 0), Point3f(-8, 0, 0), Point3f(-8, 3, 0), NANP, Point3f(0, -3, 0), Point3f(0, 3, 0)]  # [3,2]
l[1][] = p2
s2 = stats(Makie.colorbuffer(screen))
println("S2=", s2)
moved = abs(s1.col - s2.col) + abs(s1.row - s2.row)
close(screen)

println("RENDERS=", s1.lit > 300)
println("GAP=", s1.left > 50 && s1.right > 50 && s1.mid < 80)
println("LIVE_OK=", s2.lit > 300 && moved > 3.0)
println("OK_NAN_LINES")
"""

@testset "B2 NaN-separated lines render with a GAP + live NaN move (subprocess)" begin
    # Retry past the known intermittent GeometryGroup::attachToContext startup crash (house pattern).
    ec = -1; out = ""
    for _ in 1:4
        ec, out = run_ovrtx_subprocess(_B2_RENDER_PROG; timeout = 900)
        contains(out, "S1=") && break
    end
    contains(out, "OK_NAN_LINES") || @info "B2 nan-lines render output" out
    @test ec == 0 && contains(out, "OK_NAN_LINES")   # subprocess completed (no NaN poison, no crash)
    @test contains(out, "RENDERS=true")              # the NaN-separated line rendered non-black
    @test contains(out, "GAP=true")                  # two clusters, dark centre band → multi-curve split
    @test contains(out, "LIVE_OK=true")              # the live NaN move re-rendered (centroid moved)
end

# ---------------------------------------------------------------------------
# Regression: `robj.meta[:curve_npoints]` must be FROZEN at author time — the live-push gate
# (`_push_curve_positions!`) writes through the persistent `points` binding ONLY when the new
# split count equals the AUTHOR-time bound size, else one-shot.  The pre-fix code re-assigned
# the meta on EVERY push, so it tracked the LAST edit, not the bound size.  That made this
# sequence corrupt-through-binding:
#   author N=4 (binding buffer = 4)
#   → edit M=6  (6≠4 → one-shot; pre-fix meta becomes 6)
#   → edit M=6  (6==meta(6) pre-fix → BINDING write of shape [6] on the 4-sized buffer)
# The last write is a resize-through-binding on the author-sized buffer — the unverified path
# ovrtx silently ignores (invisible corruption).  Frozen meta stays 4, so BOTH edits take the
# proven one-shot path and render correctly, and the 2nd (same-count) edit visibly MOVES.
#
# Finite-only polylines (no NaN) keep the split counts exactly equal to the input counts, so the
# gate math is unambiguous: author 4 pts, both edits 6 pts (different locations).
# ---------------------------------------------------------------------------

const _B2_FREEZE_PROG = raw"""
using OmniverseMakie, ColorTypes
import OmniverseMakie as OM

OM.activate!(warmup = 40)

# author N=4 → points binding sized 4, meta[:curve_npoints] frozen at 4 (centred zigzag).
p0 = [Point3f(-6, -4, 0), Point3f(-2, 4, 0), Point3f(2, -4, 0), Point3f(6, 4, 0)]                 # 4 pts
# edit1: M=6 (≠4 → one-shot resize), zigzag on the LEFT.
p1 = [Point3f(-11, -4, 0), Point3f(-9, 4, 0), Point3f(-7, -4, 0), Point3f(-5, 4, 0), Point3f(-3, -4, 0), Point3f(-1, 4, 0)]  # 6 pts
# edit2: SAME count M=6, zigzag moved to the RIGHT.  Pre-fix: 6==meta(6) → resize-through-binding
# on the 4-sized buffer (silently ignored/corrupt).  Fixed: 6≠4 (frozen) → one-shot (correct move).
p2 = [Point3f(1, -4, 0), Point3f(3, 4, 0), Point3f(5, -4, 0), Point3f(7, 4, 0), Point3f(9, -4, 0), Point3f(11, 4, 0)]        # 6 pts

scene = Scene(size = (320, 320)); cam3d!(scene)
update_cam!(scene, Vec3f(0, 0, 40), Vec3f(0, 0, 0), Vec3f(0, 1, 0))
l = lines!(scene, p0; color = :red, linewidth = 8)
screen = OM.Screen(scene)

function stats(img)
    H, W = size(img); lit = 0; sr = 0.0; sc = 0.0
    for h in 1:H, w in 1:W
        c = img[h, w]
        if Float32(red(c)) + Float32(green(c)) + Float32(blue(c)) > 0.05
            lit += 1; sr += h; sc += w
        end
    end
    return (lit = lit, row = lit > 0 ? sr / lit : -1.0, col = lit > 0 ? sc / lit : -1.0)
end

s0 = stats(Makie.colorbuffer(screen))
println("S0=", s0)
robj = screen.plot2robj[objectid(l)]
author_np = robj.meta[:curve_npoints]
println("AUTHOR_NP=", author_np)

l[1][] = p1                                    # edit1: M=6, LEFT
s1 = stats(Makie.colorbuffer(screen))
np1 = robj.meta[:curve_npoints]
println("S1=", s1); println("NP_AFTER_EDIT1=", np1)

l[1][] = p2                                    # edit2: SAME M=6, RIGHT
s2 = stats(Makie.colorbuffer(screen))
np2 = robj.meta[:curve_npoints]
println("S2=", s2); println("NP_AFTER_EDIT2=", np2)
close(screen)

W = 320
println("AUTHOR_RENDERS=", s0.lit > 200)
println("EDIT1_RENDERS=", s1.lit > 200)
println("EDIT2_RENDERS=", s2.lit > 200)
println("EDIT1_LEFT=", s1.col < W / 2)              # edit1 sits left of centre
println("EDIT2_RIGHT=", s2.col > W / 2)             # edit2 sits right of centre → not a stale frame
println("MOVED=", (s2.col - s1.col) > 20.0)         # the 2nd same-count edit visibly moved right
# THE regression guard: curve_npoints is the FROZEN author-time size after BOTH edits.
println("FROZEN=", np1 == author_np && np2 == author_np)
println("OK_FREEZE")
"""

@testset "B2 curve_npoints FROZEN at author time — same-count re-edit stays one-shot (subprocess)" begin
    # Retry past the known intermittent GeometryGroup::attachToContext startup crash (house pattern).
    ec = -1; out = ""
    for _ in 1:4
        ec, out = run_ovrtx_subprocess(_B2_FREEZE_PROG; timeout = 900)
        contains(out, "S0=") && break
    end
    contains(out, "OK_FREEZE") || @info "B2 freeze render output" out
    @test ec == 0 && contains(out, "OK_FREEZE")   # subprocess completed (no crash/corruption)
    @test contains(out, "AUTHOR_RENDERS=true")    # author frame non-black
    @test contains(out, "EDIT1_RENDERS=true")     # 1st edit (one-shot resize 4→6) rendered
    @test contains(out, "EDIT2_RENDERS=true")     # 2nd edit (same count 6) rendered
    @test contains(out, "EDIT1_LEFT=true")        # 1st edit landed on the left
    @test contains(out, "EDIT2_RIGHT=true")       # 2nd edit landed on the right (not stale)
    @test contains(out, "MOVED=true")             # 2nd same-count edit MOVED → not a silently-dropped
                                                  #   resize-through-binding (proven one-shot path)
    @test contains(out, "FROZEN=true")            # meta == author count after BOTH edits

    # Belt-and-suspenders: the printed metas — author is 4, and it stays 4 after EACH edit.
    # Pre-fix (per-push update) NP_AFTER_EDIT1 would be 6 (RED); frozen it is 4 (GREEN).
    ma = match(r"AUTHOR_NP=(\d+)", out)
    m1 = match(r"NP_AFTER_EDIT1=(\d+)", out)
    m2 = match(r"NP_AFTER_EDIT2=(\d+)", out)
    @test ma !== nothing && m1 !== nothing && m2 !== nothing
    @test parse(Int, ma.captures[1]) == 4
    @test parse(Int, m1.captures[1]) == 4         # would be 6 under the pre-fix per-push meta update
    @test parse(Int, m2.captures[1]) == 4
end
