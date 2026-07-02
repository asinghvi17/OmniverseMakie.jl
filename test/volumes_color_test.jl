# Volumes M2 Task 3 — composite colormap COLORS.  SPIKE OUTCOME = BLOCKED-DEGRADE (grayscale).
#
# The GOAL was: a `volume!` with a colormap renders the transfer-function COLORS via NVIDIA IndeX's
# COMPOSITE path (mean R≠G≠B).  The Task-3 verify-or-degrade spike (.superpowers/sdd/task-3-report.md)
# proved the composite path is ARCHITECTURALLY ABSENT from this standalone ovrtx runtime:
#   • The `nvindex:composite` / `omni:rtx:skip` prim flags and the `rtx:index:compositeEnabled` /
#     `rtx:index:compositeDepthMode` render settings are consumed ONLY by the Kit
#     `omni.rtx.index_composite` extension (pure-Python; ships NO `.so`) + `omni.hydra.rtx`.  NO
#     bundled ovrtx binary (`librtx.indexlib.plugin.so`, `libcarb.scenerenderer-index.plugin.so`,
#     `libovrtx-dynamic.so`) references any of those strings.
#   • Enabling them via a carb setting (mechanism a) OR a root-layer `customLayerData.renderSettings`
#     (mechanism b) had ZERO effect on color; and turning on `omni:rtx:skip` merely removed the
#     volume from the ONLY working volume path (IndeX Direct) → BLACK (0 lit px).
#   • IndeX DIRECT (the sole bundled volume renderer) draws a graded volume but IGNORES the authored
#     Colormap → GRAYSCALE (spike: a viridis gradient → mean R≈G≈B, COLORED=false).
# So COLORS DEFER to a composite-capable ovrtx build (a Kit runtime carrying omni.rtx.index_composite).
#
# This test is therefore INVERTED from the Task-3 brief: it asserts the `volume!` path renders
# NON-BLACK grayscale (the degrade M2 ships), and documents `COLORED=false` as the current reality.
# `COLORED=false` doubles as a TRIPWIRE: if a future ovrtx build gains the composite extension and
# this flips to `COLORED=true`, delete the degrade note and assert `COLORED=true` (the real goal).
# Subprocess + env-gated; skips cleanly when the Kit IndeX libs dir is absent (CI without them stays green).

using Test
const _COLOR_PROG = """
using OmniverseMakie
import OmniverseMakie as OM
using OmniverseMakie: OV
n = 40; vol = zeros(Float32, n, n, n)
for k in 1:n, j in 1:n, i in 1:n
    vol[i,j,k] = Float32((i+j+k)/(3n))           # a gradient so the TF spans low→high
end
scene = Scene(size=(256,256)); cam3d!(scene)
update_cam!(scene, Vec3f(38,38,22), Vec3f(0,0,0), Vec3f(0,0,1))
volume!(scene, -10..10, -10..10, -10..10, vol; colormap=:viridis)
screen = OM.Screen(scene); img = Makie.colorbuffer(screen)
lit = [c for c in img if (Float32(c.r)+Float32(c.g)+Float32(c.b)) > 0.04]
mr = sum(c->Float32(c.r), lit)/max(length(lit),1); mg = sum(c->Float32(c.g), lit)/max(length(lit),1); mb = sum(c->Float32(c.b), lit)/max(length(lit),1)
close(screen)
println("INDEX_ENABLED=", OV._index_enabled()); println("LIT=", length(lit))
println("MEANRGB=", mr, ",", mg, ",", mb)
println("COLORED=", (abs(mr-mg) > 0.02 || abs(mg-mb) > 0.02 || abs(mr-mb) > 0.02))
println("OK_COLOR")
"""
include("helpers.jl")
@testset "Volumes: colormap colors — BLOCKED-DEGRADE grayscale (subprocess)" begin
    if !isdir(_HELPER_INDEX_LIBS)
        @test_skip "IndeX libs absent — volume color test skipped"
    else
        # ovrtx has a known INTERMITTENT startup crash (GeometryGroup::attachToContext) that can kill
        # the child before it renders; `retries`/`ready_marker` re-run until it reports a render result
        # (mirrors volumes_plot_test.jl).
        _, out = run_ovrtx_subprocess(_COLOR_PROG; timeout=600,
            env=("OMNIVERSEMAKIE_INDEX_LIBS"=>_HELPER_INDEX_LIBS), retries=4, ready_marker="LIT=")
        contains(out, "OK_COLOR") || @info "volume color output" out
        @test contains(out, "OK_COLOR")                      # subprocess completed all work
        @test contains(out, "INDEX_ENABLED=true")
        # DEGRADE gate: the volume! path renders NON-BLACK (grayscale IndeX Direct); spike saw ~9.7k lit px.
        m = match(r"LIT=(\d+)", out)
        @test m !== nothing && parse(Int, m.captures[1]) > 300
        # DEGRADE tripwire: colors are unavailable in standalone ovrtx → grayscale (mean R≈G≈B).  When a
        # composite-capable ovrtx build lands and this flips to COLORED=true, assert COLORED=true instead.
        @test contains(out, "COLORED=false")
    end
end
