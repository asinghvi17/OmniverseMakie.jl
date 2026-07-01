# Volumes M2 Task 2 — Makie `volume!(x, y, z, ::Array{Float32,3})` renders (subprocess,
# env-gated, skip-if-absent).
#
# Puts a GRADED (spatially-varying) density blob in ONE octant of the array — the HIGH-x, HIGH-y,
# LOW-z octant (the LOW-z placement is the orientation signal) — and renders it through the full pipeline:
# `volume!` → Screen → `Makie.colorbuffer` (author `.nvdb` via NanoVDBWriter.save_nanovdb +
# author UsdVol via M1's `_vdb_volume_usda` → RT2 → NVIDIA IndeX Direct).  Asserts:
#   1. non-black (>300 lit px) — the render lands (a UNIFORM fill would render transparent under
#      IndeX Direct's default TF, so the blob MUST be graded — see volumes_writer_test.jl / M2 Task 1),
#   2. ORIENTATION — the lit-pixel centroid is in the LOWER half of the image (row > H/2): the mass
#      is in the −z octant and world −Z projects DOWNWARD (larger row) in the top-left-origin buffer
#      (same camera+convention as test/m1_orientation_test.jl), i.e. the volume lands where the
#      `volume!(x,y,z,array)` axes + camera place it,
#   3. the screen-owned temp `.nvdb` is CLEANED on `close(screen)` (no leak).
# Skips cleanly when the Kit IndeX libs dir is absent (CI without them stays green).

using Test
const _LIBS = get(ENV, "OMNIVERSEMAKIE_INDEX_LIBS", "/home/juliahub/.local/share/ov/data/exts/v2/omni.index.libs-1287db94366cf6fe")
const _VOLPLOT_PROG = """
using OmniverseMakie
import OmniverseMakie as OM
using OmniverseMakie: OV
# Loops are wrapped in FUNCTIONS to avoid Julia top-level soft-scope errors on the accumulators
# (the same reason test/m1_orientation_test.jl wraps its centroid loop).
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
function lit_centroid(img)
    H, W = size(img)                                   # row INCREASES DOWNWARD (top-left origin); world −Z → larger row
    sr = 0.0; sc = 0.0; nb = 0
    for h in 1:H, w in 1:W
        cc = img[h, w]
        if (Float32(cc.r) + Float32(cc.g) + Float32(cc.b)) > 0.04
            sr += h; sc += w; nb += 1
        end
    end
    return (H = H, nb = nb, crow = nb > 0 ? sr / nb : -1.0, ccol = nb > 0 ? sc / nb : -1.0)
end
scene = Scene(size=(256,256)); cam3d!(scene)
update_cam!(scene, Vec3f(38,38,22), Vec3f(0,0,0), Vec3f(0,0,1))
p = volume!(scene, -10..10, -10..10, -10..10, make_vol(40); colormap=:viridis)
screen = OM.Screen(scene)
img = Makie.colorbuffer(screen)                        # full author+insert+render pipeline
st = lit_centroid(img)
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
include("helpers.jl")
@testset "Volumes: volume! renders (subprocess)" begin
    if !isdir(_LIBS)
        @test_skip "IndeX libs absent — volume! render test skipped"
    else
        # ovrtx has a known INTERMITTENT startup crash (GeometryGroup::attachToContext) that can kill
        # the child before it renders; retry until it reports a render result so the hard @tests aren't
        # flaky on that crash (mirrors volumes_writer_test.jl).
        out = ""
        for _ in 1:4
            _, out = run_ovrtx_subprocess(_VOLPLOT_PROG; timeout=600, env=("OMNIVERSEMAKIE_INDEX_LIBS"=>_LIBS))
            contains(out, "VOLPLOT_NONBLACK=") && break
        end
        contains(out, "OK_VOLPLOT") || @info "volume! output" out
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
    end
end
