using Test

# ---------------------------------------------------------------------------
# M4 follow-up — image textures on `surface!`.
#
# A materialized surface (image `color`) is a UsdGeomMesh bound to an OmniPBR `diffuse_texture`
# material, but the grid mesh was emitted WITHOUT `st` UVs, so the texture sampled nothing and
# the surface rendered white.  `_surface_texcoords` now emits per-vertex parametric `st`.
#
# Unit (parent, NO render): `_surface_texcoords(nx, ny)` lays out u=(j-1)/(ny-1) (2nd grid axis),
# v=(nx-i)/(nx-1) (1st axis, FLIPPED) — GLMakie's surface-texture convention — in the same i-major
# order as `_surface_mesh`.  (A `u←i, v←j` layout rotates textures 90°: the submarineCables bug.)
#
# Integration (subprocess, ★): a checker-textured flat surface renders BOTH colours sampled
# (not white) through the full Screen/colorbuffer pipeline.  Body `test/m4_surface_texture_prog.jl`.
# ---------------------------------------------------------------------------

@testset "M4 _surface_texcoords parametric UVs (unit)" begin
    # Convention (matches GLMakie; VERIFIED by comparing a GLMakie equirectangular-earth
    # render against ours): u = (j-1)/(ny-1) along the 2nd grid axis; v = (nx-i)/(nx-1)
    # along the 1st, FLIPPED (i=1 → v=1 = the image's top row).  A `u←i, v←j` layout
    # rotates every textured surface 90° — the submarineCables earth-on-its-side bug.
    st = OmniverseMakie._surface_texcoords(3, 5)
    @test length(st) == 15
    @test st[1] isa Vec2f
    # i-major order: index (i-1)*ny + j
    @test st[(1 - 1) * 5 + 1] == Vec2f(0, 1)        # (i=1,j=1) → u=0, v=1 (image top-left)
    @test st[(3 - 1) * 5 + 5] == Vec2f(1, 0)        # (i=3,j=5) → u=1, v=0 (image bottom-right)
    @test st[(3 - 1) * 5 + 1] == Vec2f(0, 0)        # (i=3,j=1) → u=0, v=0
    @test st[(1 - 1) * 5 + 5] == Vec2f(1, 1)        # (i=1,j=5) → u=1, v=1
    # degenerate single-row/col → 0 (no division by zero)
    @test OmniverseMakie._surface_texcoords(1, 4)[1] == Vec2f(0, 0)
end

const _M4_SURF_TEX_PROG = read(joinpath(@__DIR__, "m4_surface_texture_prog.jl"), String)

@testset "M4 textured surface! samples the texture via colorbuffer (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M4_SURF_TEX_PROG; timeout = 600)
    @info "M4 surface-texture subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_SURFACE_TEXTURE")

    mr = match(r"RED_DOMINANT=(\d+)", output)
    mb = match(r"BLUE_DOMINANT=(\d+)", output)
    @test mr !== nothing && parse(Int, mr.captures[1]) > 200
    @test mb !== nothing && parse(Int, mb.captures[1]) > 200
end
