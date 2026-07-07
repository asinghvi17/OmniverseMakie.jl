using Test

# ---------------------------------------------------------------------------
# One colormap mapper (`_map_through_colormap`) + one colorrange resolver
# (`_resolve_colorrange`), NaN-safe — shared by materials `_displaycolor`,
# primitives `_surface_colors`, compute `_scaled_to_display`, and the
# `_colorrange` / `_volume_colorrange` resolvers.
#
# GOLDEN regression anchors: the surface displayColor USDA, the
# numeric-scatter colours, and the volume colorrange stay byte/value-identical.
# NaN safety: a NaN-masked surface colour path must NOT throw (the mesh path
# deliberately drops non-finite cells) and must yield finite RGB for every
# vertex — a non-finite value maps to `plot.nan_color`, not an error.
#
# PURE (no GPU): plot construction + USDA string emission + resolver calls.
# ---------------------------------------------------------------------------

import OmniverseMakie as OM
# Bring Makie's re-exports used bare below into scope so this file runs
# standalone too (runtests.jl otherwise `using`s OmniverseMakie globally).
using OmniverseMakie: Figure, LScene, surface!, meshscatter!, volume!, Point3f, RGBf, (..)

# Byte-for-byte USDA of a 2×2 finite surface (z = [0 .5; .5 1], :viridis,
# explicit colorrange = (0, 1)) via `_surface_colors` + `usda_mesh`.
# Regression anchor for the surface displayColor path.
const _GOLDEN_SURFACE_USDA = """#usda 1.0
( defaultPrim = "mesh" )
def Mesh "mesh"
{
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 2, 3, 1]
    normal3f[] normals = [(-0.40824828, -0.40824828, 0.81649655), (-0.40824828, -0.40824828, 0.81649655), (-0.40824828, -0.40824828, 0.81649655), (-0.40824828, -0.40824828, 0.81649655)] (
        interpolation = "vertex"
    )
    point3f[] points = [(0.0, 0.0, 0.0), (0.0, 1.0, 0.5), (1.0, 0.0, 0.5), (1.0, 1.0, 1.0)]
    color3f[] primvars:displayColor = [(0.267004, 0.004874, 0.329415), (0.1281485, 0.565107, 0.5508925), (0.1281485, 0.565107, 0.5508925), (0.993248, 0.906157, 0.143936)] (
        interpolation = "vertex"
    )
    uniform token subdivisionScheme = "none"
    matrix4d xformOp:transform = ( (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0) )
    uniform token[] xformOpOrder = ["xformOp:transform"]
}
"""

# Per-vertex colours for a numeric meshscatter (color = 1:5, :viridis,
# colorrange = (1, 5)) via `_scaled_to_display`.
const _GOLDEN_SCATTER_VALS = NTuple{3,Float32}[
    (0.267004f0, 0.004874f0, 0.329415f0),
    (0.23022275f0, 0.32129723f0, 0.545488f0),
    (0.1281485f0, 0.565107f0, 0.5508925f0),
    (0.36285925f0, 0.786695f0, 0.386589f0),
    (0.993248f0, 0.906157f0, 0.143936f0),
]

@testset "B1 regression: surface displayColor USDA byte-identical (finite, explicit)" begin
    xs = Float32[0, 1]; ys = Float32[0, 1]; zs = Float32[0.0 0.5; 0.5 1.0]
    fig = Figure(); ls = LScene(fig[1, 1])
    p = surface!(ls, xs, ys, zs; colormap = :viridis, colorrange = (0.0, 1.0))
    pts, faces0, nrms = OM._surface_mesh(xs, ys, zs)
    vals, interp = OM._surface_colors(p, zs)
    usda = OM.usda_mesh(pts, OM._flat_faces(faces0)..., nrms, vals;
                        model = p.model[], normal_interpolation = "vertex",
                        color_interpolation = interp)
    @test interp == "vertex"
    @test usda == _GOLDEN_SURFACE_USDA
end

@testset "B1 NaN-safe: surface colours finite for every vertex, no throw" begin
    xs = Float32[0, 1]; ys = Float32[0, 1]; zs = Float32[0.0 NaN; 0.5 1.0]
    fig = Figure(); ls = LScene(fig[1, 1])

    # automatic colorrange (extrema now over FINITE zs only)
    pa = surface!(ls, xs, ys, zs; colormap = :viridis)
    vals, interp = OM._surface_colors(pa, zs)             # must NOT throw
    @test interp == "vertex"
    @test length(vals) == 4
    @test all(v -> v isa NTuple{3,Float32}, vals)
    @test all(v -> all(isfinite, v), vals)

    # explicit colorrange + a NaN VALUE is fine too; the NaN vertex takes
    # nan_color.  zvec is i-major → [zs[1,1], zs[1,2], zs[2,1], zs[2,2]] so
    # the NaN lands at index 2.
    pe = surface!(ls, xs, ys, zs; colormap = :viridis, colorrange = (0.0, 1.0), nan_color = :red)
    valse, _ = OM._surface_colors(pe, zs)
    @test all(v -> all(isfinite, v), valse)
    @test valse[2] == (1.0f0, 0.0f0, 0.0f0)               # nan_color = :red
    @test valse[1] != valse[2] && valse[3] != valse[2]  # finite vertices mapped
end

@testset "B1 _resolve_colorrange: explicit verbatim / finite extrema / empty fallback" begin
    xs = Float32[0, 1]; ys = Float32[0, 1]; zs = Float32[0.0 0.5; 0.5 1.0]
    fig = Figure(); ls = LScene(fig[1, 1])
    pe = surface!(ls, xs, ys, zs; colormap = :viridis, colorrange = (0.0, 1.0))
    pa = surface!(ls, xs, ys, zs; colormap = :viridis)

    # explicit honored
    @test OM._resolve_colorrange(pe, Float32[0, 0.5, 1]) === (0.0f0, 1.0f0)
    # automatic extrema
    @test OM._resolve_colorrange(pa, Float32[2, 5, 9]) === (2.0f0, 9.0f0)
    # finite-only extrema
    @test OM._resolve_colorrange(pa, Float32[2, NaN, 9]) === (2.0f0, 9.0f0)
    # no finite → (0,1)
    @test OM._resolve_colorrange(pa, Float32[NaN, Inf, -Inf]) === (0.0f0, 1.0f0)
    @test eltype(OM._resolve_colorrange(pe, Float32[0, 1])) === Float32
end

@testset "B1 _map_through_colormap: typed, NaN → nan_color, finite → lookup" begin
    xs = Float32[0, 1]; ys = Float32[0, 1]; zs = Float32[0.0 0.5; 0.5 1.0]
    fig = Figure(); ls = LScene(fig[1, 1])
    p = surface!(ls, xs, ys, zs; colormap = :viridis, colorrange = (0.0, 1.0), nan_color = :red)

    out = OM._map_through_colormap(p, Float32[0.0, NaN, 1.0])
    @test out isa Vector{NTuple{3,Float32}}
    @test length(out) == 3
    @test out[2] == (1.0f0, 0.0f0, 0.0f0)  # NaN → nan_color = :red
    @test all(isfinite, out[1]) && all(isfinite, out[3])
    @test out[1] != out[3]  # real colormap lookup at the ends
end

@testset "B1 regression: numeric-scatter colours unchanged" begin
    fig = Figure(); ls = LScene(fig[1, 1])
    spts = [Point3f(i, 0, 0) for i in 1:5]
    svals = Float32[1, 2, 3, 4, 5]
    sp = meshscatter!(ls, spts; color = svals, colormap = :viridis, colorrange = (1.0, 5.0))

    sd, sdi = OM._scaled_to_display(sp, svals, length(svals))
    @test sdi == "vertex"
    @test sd == _GOLDEN_SCATTER_VALS

    # a Colorant `scaled_color` still resolves to ONE constant colour
    # (unchanged path)
    cv, ci = OM._scaled_to_display(sp, RGBf(1, 0, 0), length(svals))
    @test ci == "constant"
    @test cv == (1.0f0, 0.0f0, 0.0f0)
end

@testset "B1 regression: volume colorrange unchanged (Float64, explicit + automatic)" begin
    fig = Figure(); ls = LScene(fig[1, 1])
    vol = reshape(Float32.(0:7), 2, 2, 2)
    vp_e = volume!(ls, 0 .. 1, 0 .. 1, 0 .. 1, vol; colorrange = (0.0, 2.0))
    vp_a = volume!(ls, 0 .. 1, 0 .. 1, 0 .. 1, vol)

    @test OM._volume_colorrange(vp_e, vol) === (0.0, 2.0)   # explicit, Float64
    @test OM._volume_colorrange(vp_a, vol) === (0.0, 7.0)   # automatic Float64
    @test eltype(OM._volume_colorrange(vp_e, vol)) === Float64
end
