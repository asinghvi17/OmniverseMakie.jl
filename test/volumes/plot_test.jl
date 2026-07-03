# Volumes M2 Task 2 — Makie `volume!(x, y, z, ::Array{Float32,3})` renders (subprocess,
# env-gated, skip-if-absent).
#
# Puts a GRADED (spatially-varying) density blob in ONE octant of the array — the HIGH-x, HIGH-y,
# LOW-z octant (the LOW-z placement is the orientation signal) — and renders it through the full pipeline:
# `volume!` → Screen → `Makie.colorbuffer` (author `.nvdb` via NanoVDBWriter.save_nanovdb +
# author UsdVol via M1's `_vdb_volume_usda` → RT2 → NVIDIA IndeX Direct).  Asserts:
#   1. non-black (>300 lit px) — the render lands (a UNIFORM fill would render transparent under
#      IndeX Direct's default TF, so the blob MUST be graded — a spike-verified IndeX constraint),
#   2. ORIENTATION — the lit-pixel centroid is in the LOWER half of the image (row > H/2): the mass
#      is in the −z octant and world −Z projects DOWNWARD (larger row) in the top-left-origin buffer
#      (same camera+convention as test/offscreen/orientation_test.jl), i.e. the volume lands where the
#      `volume!(x,y,z,array)` axes + camera place it,
#   3. the screen-owned temp `.nvdb` is CLEANED on `close(screen)` (no leak).
# Skips cleanly when the Kit IndeX libs dir is absent (CI without them stays green).

using Test
include(joinpath(@__DIR__, "..", "helpers.jl"))   # _HELPER_INDEX_LIBS + PROG_PIXEL_HELPERS (the latter spliced into the prog below)
const _VOLPLOT_PROG = """
using OmniverseMakie
import OmniverseMakie as OM
using OmniverseMakie: OV
$(PROG_PIXEL_HELPERS)
# make_vol's loop is wrapped in a FUNCTION to avoid Julia top-level soft-scope errors on the
# accumulator (as test/offscreen/orientation_test.jl wraps its centroid loop); lit_centroid now comes from
# the shared PROG_PIXEL_HELPERS prelude spliced above (identical (H, nb, crow, ccol); LUM_MIN=0.04f0).
function make_vol(n)
    vol = zeros(Float32, n, n, n)
    # Graded blob in ONE octant: HIGH-x, HIGH-y, LOW-z (indices ~[n/2..n, n/2..n, 1..n/2]).  The
    # LOW-z placement is the orientation signal: world −Z projects toward the image BOTTOM (larger
    # row); the +x/+y are chosen so their screen projections cancel horizontally (this camera) rather
    # than fighting the −z, giving a strong, unambiguous DOWN offset.
    cx = 0.75 * n; cy = 0.75 * n; cz = 0.25 * n; R = 0.25 * n
    for k in 1:n, j in 1:n, i in 1:n
        d = sqrt((i - cx)^2 + (j - cy)^2 + (k - cz)^2)
        vol[i, j, k] = d < R ? Float32(3 * (1 - d / R)) : 0.0f0   # SPATIALLY-VARYING (uniform → IndeX-transparent)
    end
    return vol
end
scene = Scene(size=(256,256)); cam3d!(scene)
update_cam!(scene, Vec3f(38,38,22), Vec3f(0,0,0), Vec3f(0,0,1))
p = volume!(scene, -10..10, -10..10, -10..10, make_vol(40); colormap=:viridis)
screen = OM.Screen(scene)
img = Makie.colorbuffer(screen)                        # full author+insert+render pipeline
st = lit_centroid(img)
# GRAYSCALE tripwire (folded in from the retired volumes-color test): IndeX Direct ignores
# the authored Colormap transfer function — the viridis volume renders R≈G≈B.  If a future
# ovrtx/IndeX build starts honoring TF colors, COLORED flips true and this trips so the
# grayscale degrade (docs + colormap plumbing) gets revisited.
function mean_rgb_lit(img)
    tr = 0.0; tg = 0.0; tb = 0.0; n = 0
    for c in img
        if lum(c) > LUM_MIN
            tr += Float32(c.r); tg += Float32(c.g); tb += Float32(c.b); n += 1
        end
    end
    n == 0 ? (0.0, 0.0, 0.0) : (tr / n, tg / n, tb / n)
end
mr, mg, mb = mean_rgb_lit(img)
colored = abs(mr - mg) > 0.02 || abs(mg - mb) > 0.02 || abs(mr - mb) > 0.02
println("COLORED=", colored)
# temp-.nvdb cleanup: the Volume robj records the screen-owned temp path; close(screen) → destroy_bindings! removes it.
robj = screen.plot2robj[objectid(p)]
tmpf = get(robj.meta, :vdb_tmp, "")
existed = isfile(tmpf)
close(screen)
gone = !isfile(tmpf)
println("INDEX_ENABLED=", OV._index_enabled())
println("VOLPLOT_NONBLACK=", st.nb)
println("VOLPLOT_H=", st.H)
println("VOLPLOT_CENTROID_ROW=", st.crow)
println("VOLPLOT_CENTROID_COL=", st.ccol)
println("VOLTMP_EXISTED=", existed)
println("VOLTMP_GONE=", gone)
println("OK_VOLPLOT")
"""
@testset "Volumes: volume! renders (subprocess)" begin
    if !isdir(_HELPER_INDEX_LIBS)
        @test_skip "IndeX libs absent — volume! render test skipped"
    else
        # ovrtx has a known INTERMITTENT startup crash (GeometryGroup::attachToContext) that can kill
        # the child before it renders; `retries`/`ready_marker` re-run until it reports a render result
        # so the hard @tests aren't flaky on that crash (the house retry pattern).
        _, out = run_ovrtx_subprocess(_VOLPLOT_PROG; timeout=600,
            env=("OMNIVERSEMAKIE_INDEX_LIBS"=>_HELPER_INDEX_LIBS), retries=4, ready_marker="VOLPLOT_NONBLACK=")
        contains(out, "OK_VOLPLOT") || @info "volume! output" out
        @test contains(out, "OK_VOLPLOT")                                  # subprocess completed all work (no mid-run death)
        @test contains(out, "INDEX_ENABLED=true")

        m_nb = match(r"VOLPLOT_NONBLACK=(\d+)", out)
        @test m_nb !== nothing && parse(Int, m_nb.captures[1]) > 300        # renders non-black

        # Orientation: mass in the −z octant → lit centroid in the LOWER half (row > H/2).
        m_h   = match(r"VOLPLOT_H=(\d+)", out)
        m_row = match(r"VOLPLOT_CENTROID_ROW=([-\d.]+)", out)
        @test m_h !== nothing && m_row !== nothing
        if m_h !== nothing && m_row !== nothing
            H    = parse(Int, m_h.captures[1])
            crow = parse(Float64, m_row.captures[1])
            @test crow > H / 2                                             # blob renders below centre
        end

        # Temp `.nvdb` is authored, then cleaned on close (no leak).
        @test contains(out, "VOLTMP_EXISTED=true")
        @test contains(out, "VOLTMP_GONE=true")

        # GRAYSCALE tripwire: colors are DOCUMENTED-IMPOSSIBLE in standalone ovrtx (IndeX
        # Direct = scalar-only default TF; the composite compositor is Kit-only).  If this
        # flips true, a future build started honoring TF colors — revisit the degrade.
        @test contains(out, "COLORED=false")
    end
end
