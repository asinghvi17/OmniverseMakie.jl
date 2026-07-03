using Test
include(joinpath(@__DIR__, "..", "helpers.jl"))

# ---------------------------------------------------------------------------
# M4 follow-up — numeric `color` + `colormap` on meshscatter!/lines!/linesegments!
# (and per-vertex `mesh!`).
#
# Before this fix, a colour-mapped scatter/line resolved `:scaled_color` to a
# `Vector{Float32}` that `_displaycolor_from_scaled` could not handle — it fell through to
# `_rgb(to_color(::Vector))` → `MethodError: no method matching red(::Vector{Float32})`,
# failing the whole render (the M4 examples worked around it with an in-scene `cmap_colors`).
#
# Unit (parent process, NO render): `_scaled_to_display` maps a numeric `scaled_color` vector
# through the plot's colormap → a per-VERTEX colour list; a Colorant/scalar `scaled_color`
# still takes the byte-unchanged constant path.
#
# Integration (subprocess, ★): a colormapped meshscatter + linesegments render non-black and
# show a VARIED colormap gradient through the full Screen/colorbuffer pipeline.  Body is
# `test/m4_colormap_prog.jl`.
# ---------------------------------------------------------------------------

@testset "M4 _scaled_to_display maps numeric color via colormap (unit)" begin
    fig  = Figure()
    ax   = LScene(fig[1, 1])
    pts  = [Point3f(cos(t), sin(t), 0) for t in range(0, 2π; length = 10)]
    vals = collect(Float32.(1:10))
    p    = meshscatter!(ax, pts; color = vals, colormap = :plasma)

    values, interp = OmniverseMakie._scaled_to_display(p, vals, length(vals))
    @test interp == "vertex"
    @test length(values) == 10
    @test all(v -> v isa NTuple{3,Float32}, values)
    # the colormap gradient produced VARIED colours (not one flat colour)
    @test length(unique(values)) >= 8
    # endpoints differ (low end of :plasma is dark purple, high end is bright yellow)
    @test values[1] != values[end]

    # a Colorant `scaled_color` still resolves to ONE constant colour (byte-unchanged path)
    cvals, cinterp = OmniverseMakie._scaled_to_display(p, RGBf(1, 0, 0), length(vals))
    @test cinterp == "constant"
    @test cvals == (1.0f0, 0.0f0, 0.0f0)
end

const _M4_COLORMAP_PROG = read(joinpath(@__DIR__, "colormap_prog.jl"), String)

@testset "M4 colormapped scatter+lines render via colorbuffer (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M4_COLORMAP_PROG; timeout = 600, retries = 2, ready_marker = "OK_COLORMAP")
    @info "M4 colormap subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_COLORMAP")

    mnb = match(r"NONBLACK=(\d+)", output)
    @test mnb !== nothing && parse(Int, mnb.captures[1]) > 1000

    mbk = match(r"COLOR_BUCKETS=(\d+)", output)
    @test mbk !== nothing && parse(Int, mbk.captures[1]) >= 6
end

const _M4_COLORMAP_LIVE_PROG = read(joinpath(@__DIR__, "colormap_live_prog.jl"), String)

@testset "M4 LIVE numeric-color edit re-maps via colormap (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M4_COLORMAP_LIVE_PROG; timeout = 600, retries = 2, ready_marker = "OK_COLORMAP_LIVE")
    @info "M4 colormap-live subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_COLORMAP_LIVE")
    mmad = match(r"MEANABSDIFF=([0-9.eE+\-]+)", output)
    @test mmad !== nothing && parse(Float64, mmad.captures[1]) > 0.004
end
