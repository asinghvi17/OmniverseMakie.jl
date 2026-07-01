# Volumes M2 Task 1 — a WRITTEN .nvdb renders in ovrtx (the go/no-go spike; subprocess,
# env-gated, skip-if-absent).
#
# Writes a dense Float32 ball with a radial (spatially-varying) density to a temp .nvdb via
# NanoVDBWriter.save_nanovdb (Codec::NONE, major-32), authors it with M1's author_vdb_volume!
# (field="density"), renders through the RT2 → NVIDIA IndeX Direct path, and asserts a non-black
# volume appeared.  The density MUST vary in space: IndeX Direct's default volume shading renders
# a uniform-density field fully transparent, so a solid ball → 0 px while this graded one → ~1.9k.
# Skips cleanly when the Kit IndeX libs dir is absent (CI without them stays green).

using Test
const _LIBS = get(ENV, "OMNIVERSEMAKIE_INDEX_LIBS", "/home/juliahub/.local/share/ov/data/exts/v2/omni.index.libs-1287db94366cf6fe")
const _WRITER_PROG = """
using OmniverseMakie
import OmniverseMakie as OM
using OmniverseMakie: OV
using NanoVDBWriter: save_nanovdb
using OmniverseMakie: Point3f, Vec3f
n = 40; data = zeros(Float32, n, n, n); R = n/3
for k in 1:n, j in 1:n, i in 1:n
    d = sqrt((i-n/2)^2 + (j-n/2)^2 + (k-n/2)^2)
    data[i,j,k] = d < R ? Float32(3 * (1 - d/R)) : 0.0f0   # radial density falloff (SPATIALLY-VARYING)
end
tmp = tempname()*".nvdb"; save_nanovdb(tmp, data, Point3f(-10,-10,-10), Vec3f(20,20,20))
scene = Scene(size=(256,256)); cam3d!(scene)
update_cam!(scene, Vec3f(38,38,22), Vec3f(0,0,0), Vec3f(0,0,1))
screen = OM.Screen(scene); OM.author_root_from_scene!(screen, scene; resolution=screen.fb_size)
OM.author_vdb_volume!(screen, scene, tmp; field="density", colormap=:viridis)
img = OV.render_to_matrix(screen.renderer, screen.product; warmup=48)
nb = count(c -> (Float32(c.r)+Float32(c.g)+Float32(c.b)) > 0.04, img); close(screen); rm(tmp)
println("INDEX_ENABLED=", OV._index_enabled()); println("WRITER_NONBLACK=", nb); println("OK_WRITER_RENDER")
"""
include("helpers.jl")
@testset "Volumes: written .nvdb renders (subprocess)" begin
    if !isdir(_LIBS)
        @test_skip "IndeX libs absent — writer render spike skipped"
    else
        # ovrtx has a known INTERMITTENT startup assertion crash (GeometryGroup::attachToContext,
        # seen on unrelated runs too) that can kill the child before it renders.  Retry a few times
        # until the child reports a render result, so this hard @test is not flaky on that crash.
        out = ""
        for _ in 1:4
            _, out = run_ovrtx_subprocess(_WRITER_PROG; timeout=600, env=("OMNIVERSEMAKIE_INDEX_LIBS"=>_LIBS))
            contains(out, "WRITER_NONBLACK=") && break
        end
        contains(out, "OK_WRITER_RENDER") || @info "writer render output" out
        @test contains(out, "INDEX_ENABLED=true")          # IndeX enables + the render completes
        m = match(r"WRITER_NONBLACK=(\d+)", out)
        @test m !== nothing
        nb = m === nothing ? -1 : parse(Int, m.captures[1])
        # ── Volumes M2 Task 1 go/no-go — the written .nvdb RENDERS (see task-1-debug-report.md) ─────
        # NVIDIA IndeX successfully LOADS the written grid ("load and upload of NanoVDB volume done…
        # success loading" in its verbose log; PNanoVDB + the round-trip test also read it correctly),
        # then renders the SPATIALLY-VARYING density above → ~1.9k non-black px.  The earlier BLOCK was
        # a red herring: the prior spike wrote a UNIFORM solid ball, and IndeX Direct's default volume
        # shading renders a uniform-density field FULLY TRANSPARENT (verified: solid balls at density
        # 0.5/1/3/10 all render exactly 0 px, while a graded ball renders ~1.9k) — it needs varying
        # density with contrast, exactly like the torus_fog / bunny_cloud references.  The writer was
        # format-correct all along; the fix was realistic test data (+ a NanoVDB mask-disjointness
        # correctness fix in NanoVDBWriter.jl).
        @test nb > 300
    end
end
