using Test

# select! / clear_selection! outline render test: render a baseline, select!
# a mesh, assert strongly-orange pixels appeared that the baseline lacked
# (new outline pixels, not the plot's own color); clear_selection! removes
# most. Authoring uses _author_screen! — author_root_from_scene! leaves
# plot2robj empty, so select! would find no robj. CPU-only; no GLMakie/CUDA.
const _M6B_OUTLINE_PROG = """
using OmniverseMakie
OM = OmniverseMakie; OM.activate!(warmup = 24, selection_outline = true)
scene = Scene(size=(160,160)); cam3d!(scene)
p = mesh!(scene, Rect3f(Point3f(-1), Vec3f(2)); color = :gray)
screen = OM.Screen(scene)
OM._author_screen!(screen, scene, scene)
base = OM.OV.render_to_matrix(screen.renderer, screen.product; warmup = 24)
OM.select!(screen, p)                       # orange outline, group 1
OM.OV.reset!(screen.renderer)
sel  = OM.OV.render_to_matrix(screen.renderer, screen.product; warmup = 24)
# Count strongly-orange pixels (R high, G mid, B low) gained vs baseline.
isorange(c) = Float32(c.r) > 0.6 && 0.2 < Float32(c.g) < 0.75 && Float32(c.b) < 0.3
gained = count(i -> isorange(sel[i]) && !isorange(base[i]), eachindex(sel))
println("OUTLINE_GAINED=", gained)
# Diagnostic: the LDR color of the most orange-leaning pixel that was not
# orange in the baseline — documents the outline's drawn color.
bi = argmax(i -> (isorange(base[i]) ? -9f0 : Float32(sel[i].r) - Float32(sel[i].b)), eachindex(sel))
sc = sel[bi]
println("OUTLINE_SAMPLE_RGB=", (round(Float32(sc.r); digits=3), round(Float32(sc.g); digits=3), round(Float32(sc.b); digits=3)))
OM.clear_selection!(screen, p)
OM.OV.reset!(screen.renderer)
cleared = OM.OV.render_to_matrix(screen.renderer, screen.product; warmup = 24)
println("OUTLINE_AFTER_CLEAR=", count(i -> isorange(cleared[i]) && !isorange(base[i]), eachindex(cleared)))
close(screen)
println("OK_OUTLINE")
"""

include(joinpath(@__DIR__, "..", "helpers.jl"))
@testset "selection outline (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M6B_OUTLINE_PROG; timeout = 600, retries = 2, ready_marker = "OK_OUTLINE")
    @info "M6.B outline output" output
    @test exitcode == 0
    @test contains(output, "OK_OUTLINE")
    g = match(r"OUTLINE_GAINED=(\d+)", output)
    @test g !== nothing && parse(Int, g.captures[1]) > 50   # outline appeared
    c = match(r"OUTLINE_AFTER_CLEAR=(\d+)", output)
    # clearing removed most of it
    @test c !== nothing && parse(Int, c.captures[1]) < parse(Int, g.captures[1]) ÷ 2
end

# select! on a Screen built WITHOUT selection_outline=true must @warn once
# (maxlog=1) and no-op — the drawn highlight needs the creation-time flag,
# but pick DATA still works.  This needs no render: the gate
# (`_ensure_outline_style!`) short-circuits before any draw, so the style is
# never installed and select! returns nothing.  Three calls ⇒ exactly ONE
# warning.
const _M6B_NOOUTLINE_PROG = """
using OmniverseMakie
import Logging
# default: selection_outline = false
OM = OmniverseMakie; OM.activate!(warmup = 8)
scene = Scene(size=(64,64)); cam3d!(scene)
p = mesh!(scene, Rect3f(Point3f(0), Vec3f(1)); color = :gray)
screen = OM.Screen(scene)
println("OUTLINE_FLAG=", screen.config.selection_outline)   # false (default)
OM._author_screen!(screen, scene, scene)
buf = IOBuffer()
ret = Logging.with_logger(Logging.SimpleLogger(buf, Logging.Debug)) do
    # 3 calls
    r1 = OM.select!(screen, p); OM.select!(screen, p); OM.select!(screen, p); r1
end
logtxt = String(take!(buf))
println("NOOUTLINE_RET_NOTHING=", ret === nothing)
println("NOOUTLINE_NOT_STYLED=", screen._outline_styled == false)
# exactly 1 (maxlog=1)
println("NOOUTLINE_WARN_COUNT=", count(r"no highlight drawn", logtxt))
close(screen)
println("OK_NOOUTLINE")
"""

@testset "select! no-op without outline flag (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M6B_NOOUTLINE_PROG; timeout = 600, retries = 2, ready_marker = "OK_NOOUTLINE")
    @info "M6.B no-outline output" output
    @test exitcode == 0
    @test contains(output, "OUTLINE_FLAG=false")
    @test contains(output, "NOOUTLINE_RET_NOTHING=true")
    @test contains(output, "NOOUTLINE_NOT_STYLED=true")
    @test contains(output, "NOOUTLINE_WARN_COUNT=1")
    @test contains(output, "OK_NOOUTLINE")
end
