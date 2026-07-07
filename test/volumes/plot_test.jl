# Makie `volume!(x, y, z, ::Array{Float32,3})` renders (subprocess,
# env-gated, skips if IndeX libs absent). A graded density blob in one octant
# (a uniform fill renders transparent under IndeX Direct) goes through
# volume! → Screen → colorbuffer (.nvdb + UsdVol → RT2 → IndeX Direct).
# Asserts non-black, orientation (mass in the −z octant → lit centroid in
# the lower image half), and the screen-owned temp .nvdb cleaned on close.

using Test
# _HELPER_INDEX_LIBS + PROG_PIXEL_HELPERS (latter spliced into prog below)
include(joinpath(@__DIR__, "..", "helpers.jl"))
const _VOLPLOT_PROG = """
using OmniverseMakie
import OmniverseMakie as OM
using OmniverseMakie: OV
$(PROG_PIXEL_HELPERS)
# make_vol's loop is wrapped in a function to avoid top-level soft-scope on
# the accumulator; lit_centroid comes from the PROG_PIXEL_HELPERS prelude.
function make_vol(n)
    vol = zeros(Float32, n, n, n)
    # Graded blob in ONE octant: HIGH-x, HIGH-y, LOW-z (indices ~[n/2..n,
    # n/2..n, 1..n/2]).  The LOW-z placement is the orientation signal:
    # world −Z projects toward the image BOTTOM (larger row); the +x/+y are
    # chosen so their screen projections cancel horizontally (this camera)
    # rather than fighting the −z, giving a strong, unambiguous DOWN offset.
    cx = 0.75 * n; cy = 0.75 * n; cz = 0.25 * n; R = 0.25 * n
    for k in 1:n, j in 1:n, i in 1:n
        d = sqrt((i - cx)^2 + (j - cy)^2 + (k - cz)^2)
        # SPATIALLY-VARYING (uniform → IndeX-transparent)
        vol[i, j, k] = d < R ? Float32(3 * (1 - d / R)) : 0.0f0
    end
    return vol
end
scene = Scene(size=(256,256)); cam3d!(scene)
update_cam!(scene, Vec3f(38,38,22), Vec3f(0,0,0), Vec3f(0,0,1))
p = volume!(scene, -10..10, -10..10, -10..10, make_vol(40); colormap=:viridis)
screen = OM.Screen(scene)
img = Makie.colorbuffer(screen)   # full author+insert+render pipeline
st = lit_centroid(img)
# Grayscale tripwire: IndeX Direct ignores the authored Colormap transfer
# function — the viridis volume renders R≈G≈B. If a future ovrtx/IndeX build
# honors TF colors, COLORED flips true so the grayscale degrade is revisited.
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
# temp-.nvdb cleanup: the Volume robj records the screen-owned temp path;
# close(screen) → destroy_bindings! removes it.
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
        # An intermittent ovrtx startup crash (GeometryGroup::attachToContext)
        # can kill the child before it renders; retries/ready_marker re-run
        # until it reports a render result so the hard @tests are not flaky.
        ec, out = run_ovrtx_subprocess(_VOLPLOT_PROG; timeout=600,
            env=("OMNIVERSEMAKIE_INDEX_LIBS"=>_HELPER_INDEX_LIBS), retries=4, ready_marker="VOLPLOT_NONBLACK=")
        contains(out, "OK_VOLPLOT") || @info "volume! output" out
        # subprocess completed all work (no mid-run death), truthful exit code
        @test ec == 0 && contains(out, "OK_VOLPLOT")
        @test contains(out, "INDEX_ENABLED=true")

        m_nb = match(r"VOLPLOT_NONBLACK=(\d+)", out)
        # renders non-black
        @test m_nb !== nothing && parse(Int, m_nb.captures[1]) > 300

        # Orientation: mass in the −z octant → lit centroid in the LOWER
        # half (row > H/2).
        m_h   = match(r"VOLPLOT_H=(\d+)", out)
        m_row = match(r"VOLPLOT_CENTROID_ROW=([-\d.]+)", out)
        @test m_h !== nothing && m_row !== nothing
        if m_h !== nothing && m_row !== nothing
            H    = parse(Int, m_h.captures[1])
            crow = parse(Float64, m_row.captures[1])
            @test crow > H / 2   # blob renders below centre
        end

        # Temp `.nvdb` is authored, then cleaned on close (no leak).
        @test contains(out, "VOLTMP_EXISTED=true")
        @test contains(out, "VOLTMP_GONE=true")

        # Grayscale tripwire: colors are impossible in standalone ovrtx (IndeX
        # Direct = scalar-only default TF; the composite compositor is
        # Kit-only). If this flips true, a future build started honoring TF
        # colors — revisit the degrade.
        @test contains(out, "COLORED=false")
    end
end
