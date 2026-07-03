using Test
include(joinpath(@__DIR__, "..", "helpers.jl"))

# ---------------------------------------------------------------------------
# M1.7 — the remaining static 3-D primitives render through ovrtx (the M1 GATE).
#
# One subprocess renders each primitive in its OWN Figure+LScene via
# Makie.colorbuffer (lazy USD setup → insert! → to_ovrtx_object → RT2), plus a
# COMBINED mesh!+scatter!+lines! scene (proves several to_ovrtx_object methods
# compose through insert!).  Each render is wrapped so one failure still reports
# the rest; the parent test parses the per-primitive lines and asserts:
#   - image ≥ 300²
#   - non-black pixels > a primitive-specific floor (NOT empty)
#   - lit fraction < 0.95           (NOT the whole frame)
#
# Schemas validated here: UsdGeomPointInstancer (scatter/meshscatter — assumption 4),
# UsdGeomBasisCurves (lines), UsdGeomMesh grid (surface).  If a primary schema
# renders black the implementation switches to a merged-mesh fallback (see
# src/translation/primitives.jl); this test asserts the OUTCOME (non-black) either way.
#
# Fixtures match m1.7-context.md.  Surface uses update_cam! to frame the ±3 grid
# (the offscreen LScene camera is a fixed eye=(3,3,3); it does NOT auto-fit).
# warmup=40 (RT2 convergence); ~5 renderers in one process (proven by M1.6).
# ---------------------------------------------------------------------------

const _M17_PRIM_PROG = """
using OmniverseMakie, ColorTypes, FixedPointNumbers

OmniverseMakie.activate!()

lum(c) = Float32(red(c)) + Float32(green(c)) + Float32(blue(c))
function analyze(img)
    H, W = size(img)
    nonblack = 0
    for h in 1:H, w in 1:W
        lum(img[h, w]) > 0.05f0 && (nonblack += 1)
    end
    total = H * W
    return (H = H, W = W, nonblack = nonblack, total = total,
            frac = total == 0 ? 0.0 : nonblack / total)
end

# Render a freshly-built LScene and print one parseable result line.
function run_prim(name, build!)
    try
        fig = Figure(); ax = LScene(fig[1, 1])
        build!(ax)
        img = Makie.colorbuffer(ax.scene; warmup = 40)
        @assert eltype(img) == RGBA{N0f8} "eltype is \$(eltype(img))"
        st = analyze(img)
        println("PRIM=", name, " STATUS=ok NONBLACK=", st.nonblack,
                " TOTAL=", st.total, " FRAC=", round(st.frac, digits = 5),
                " H=", st.H, " W=", st.W, " ERR=")
    catch e
        println("PRIM=", name, " STATUS=error NONBLACK=0 TOTAL=0 FRAC=0.0 H=0 W=0 ERR=",
                sprint(showerror, e))
    end
end

# --- the four primitives (fixtures from m1.7-context.md) ---
run_prim("scatter", ax -> scatter!(ax, rand(Point3f, 50) .* 2; markersize = 0.15, color = :cyan))
run_prim("meshscatter", ax -> meshscatter!(ax, [Point3f(i, 0, 0) for i in -2:2]; markersize = 0.3, color = :orange))
run_prim("lines", ax -> lines!(ax, [Point3f(cos(t), sin(t), t/3) for t in range(0, 4pi, length = 60)]; linewidth = 4, color = :magenta))
run_prim("surface", function (ax)
    surface!(ax, -3:0.3:3, -3:0.3:3, (x, y) -> sin(x) * cos(y))
    Makie.update_cam!(ax.scene, Vec3f(9, 9, 9), Vec3f(0, 0, 0))   # frame the ±3 grid
end)

# --- combined: mesh + scatter + lines compose through insert! ---
run_prim("combined", function (ax)
    mesh!(ax, Rect3f(Point3f(-0.5), Vec3f(1)); color = :red)
    scatter!(ax, [Point3f(2cos(t), 2sin(t), 0) for t in range(0, 2pi, length = 12)]; markersize = 0.2, color = :cyan)
    lines!(ax, [Point3f(t - 2, sin(t), 0.5) for t in range(0, 4, length = 30)]; linewidth = 4, color = :yellow)
end)

println("OK_PRIMITIVES")
"""

# Per-primitive non-black floor (pixels) — conservative: thin primitives light few px.
const _M17_FLOORS = Dict(
    "scatter"     => 200,
    "meshscatter" => 300,
    "lines"       => 100,
    "surface"     => 2000,
    "combined"    => 1000,
)

@testset "M1.7 primitives → ovrtx (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M17_PRIM_PROG; timeout = 1800, retries = 2, ready_marker = "OK_PRIMITIVES")
    @info "M1.7 primitives subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_PRIMITIVES")

    for name in ("scatter", "meshscatter", "lines", "surface", "combined")
        m = match(Regex("PRIM=$(name) STATUS=(\\w+) NONBLACK=(\\d+) TOTAL=\\d+ FRAC=([0-9.]+) H=(\\d+) W=(\\d+)"), output)
        @testset "$(name)" begin
            if m === nothing
                @test false   # primitive result line missing entirely
                continue
            end
            status   = m.captures[1]
            nonblack = parse(Int, m.captures[2])
            frac     = parse(Float64, m.captures[3])
            H        = parse(Int, m.captures[4])
            W        = parse(Int, m.captures[5])
            @test status == "ok"
            @test H >= 300 && W >= 300
            @test nonblack > _M17_FLOORS[name]      # NOT empty (renders non-black)
            @test frac < 0.95                       # NOT the whole frame
        end
    end
end
