# Volumes M2 Task 1 — a WRITTEN .nvdb renders in ovrtx (the go/no-go spike; subprocess,
# env-gated, skip-if-absent).
#
# Writes a dense Float32 ball to a temp .nvdb via NanoVDBWriter.save_nanovdb (Codec::NONE,
# major-32), authors it with M1's author_vdb_volume! (field="density"), renders through the
# RT2 → NVIDIA IndeX Direct path, and asserts a non-black volume appeared.  If IndeX loads the
# file at all it renders (torus.vdb saw ~9k px @ 256²); WRITER_NONBLACK==0 would mean IndeX
# rejected the written file (wrong codec/checksum/version) — the go/no-go for the whole writer.
# Skips cleanly when the Kit IndeX libs dir is absent (CI without them stays green).

using Test
const _LIBS = get(ENV, "OMNIVERSEMAKIE_INDEX_LIBS", "/home/juliahub/.local/share/ov/data/exts/v2/omni.index.libs-1287db94366cf6fe")
const _WRITER_PROG = """
using OmniverseMakie
import OmniverseMakie as OM
using OmniverseMakie: OV
using NanoVDBWriter: save_nanovdb
using OmniverseMakie: Point3f, Vec3f
n = 40; data = zeros(Float32, n, n, n)
for k in 1:n, j in 1:n, i in 1:n
    c = ((i-n/2)^2 + (j-n/2)^2 + (k-n/2)^2)
    data[i,j,k] = c < (n/3)^2 ? 1.0f0 : 0.0f0        # a solid ball
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
        ec, out = run_ovrtx_subprocess(_WRITER_PROG; timeout=600, env=("OMNIVERSEMAKIE_INDEX_LIBS"=>_LIBS))
        contains(out, "OK_WRITER_RENDER") || @info "writer render output" out
        @test contains(out, "INDEX_ENABLED=true")          # IndeX enables + the render completes
        m = match(r"WRITER_NONBLACK=(\d+)", out)
        @test m !== nothing
        nb = m === nothing ? -1 : parse(Int, m.captures[1])
        # ── BLOCKED — Volumes M2 Task 1 go/no-go (see .superpowers/sdd/task-1-report.md) ──────────
        # The written .nvdb is a SPEC-VALID NanoVDB grid: the pure round-trip test (green) parses its
        # header, and PNanoVDB (NanoVDB's own reference C reader) reads correct values / bbox / voxel
        # count / valueMask straight out of it.  Yet NVIDIA IndeX Direct renders it BLACK
        # (WRITER_NONBLACK == 0), while a real-library-produced .nvdb (bunny_cloud.nvdb) renders
        # NON-BLACK through the IDENTICAL author_vdb_volume! path — even after bunny is stripped to this
        # writer's exact framing (Codec::NONE, disabled checksum, cleared grid/node/leaf flags).
        # Exhaustive single-variable bisection against that known-good grid (codec, checksum, flags,
        # node min/max/avg/std, tight node bboxes, worldBBox↔Map consistency, transform translation,
        # world scale, camera, density, thin-vs-bushy tree topology) did NOT identify the field IndeX
        # requires that the hand-built (Hikari-lifted) grid lacks.  So this stays @test_broken: it
        # documents the blocker and will flag "Unexpectedly Pass" the moment a written grid renders.
        @test_broken nb > 300
    end
end
