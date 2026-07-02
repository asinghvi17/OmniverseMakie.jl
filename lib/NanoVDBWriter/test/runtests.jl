using Test, NanoVDBWriter
using GeometryBasics: Point3f, Vec3f
using SHA: sha256

# ---------------------------------------------------------------------------
# Frozen fixtures + golden byte anchors.
#
# The golden SHA-256s below hash the exact bytes save_nanovdb writes today; a
# later task refactors the writer's memory behavior against precisely these
# hashes.  DO NOT edit the fixtures, their origin/extent, or the goldens.
# ---------------------------------------------------------------------------

# Single 8³ leaf: a graded ramp; voxel (1,1,1) is background (value 0).
function fixture_single_leaf()
    d = zeros(Float32, 8, 8, 8)
    for k in 1:8, j in 1:8, i in 1:8
        d[i, j, k] = Float32((i - 1) + 8 * (j - 1) + 64 * (k - 1)) / 512f0
    end
    return d
end

# 136×64×64 graded field that crosses the 128-voxel lower-node boundary in x →
# TWO lower nodes (a dense ≤128³ array collapses to a single lower node).  The
# k ≤ 32 background slab leaves whole leaf blocks empty (exercises leaf-skip).
function fixture_multi_lower()
    nx, ny, nz = 136, 64, 64
    d = zeros(Float32, nx, ny, nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        k <= 32 && continue
        d[i, j, k] = Float32(i + j + k) / Float32(nx + ny + nz)
    end
    return d
end

# SHA-256 (hex) of the .nvdb bytes save_nanovdb writes for `data`.
function nvdb_sha256(data, origin, extent)
    path = tempname() * ".nvdb"
    save_nanovdb(path, data, origin, extent)
    h = bytes2hex(sha256(read(path)))
    rm(path)
    return h
end

# Frozen origin/extent per fixture (unit voxel size = extent ./ size).
const SINGLE_LEAF_ORIGIN, SINGLE_LEAF_EXTENT = Point3f(0, 0, 0), Vec3f(8, 8, 8)
const MULTI_LOWER_ORIGIN, MULTI_LOWER_EXTENT = Point3f(0, 0, 0), Vec3f(136, 64, 64)

# Golden SHA-256 of the current (pre-refactor) writer output.
const GOLDEN_SINGLE_LEAF = "911c21fb5f7993acf97703ca2a1d1922ac8b2a08aedb02bec1845e3aa91f0aed"
const GOLDEN_MULTI_LOWER = "fe52a8ae08a7941e382dc5f1bcc6f2b8bb863fead21a55b0cbcf72a84ba7f8ae"

@testset "NanoVDBWriter round-trip" begin
    data = zeros(Float32, 8, 8, 8); data[2,3,4] = 1.5f0; data[7,7,7] = 0.25f0   # sparse, asymmetric
    path = tempname() * ".nvdb"
    save_nanovdb(path, data, Point3f(0,0,0), Vec3f(1,1,1))
    @test isfile(path) && filesize(path) > 0
    h = NanoVDBWriter.parse_nanovdb_header(path)
    @test h.magic == "NanoVDB0"
    @test h.version_major == 32
    @test h.voxel_count == 2                      # two non-background voxels
    rm(path)
end

@testset "golden byte anchor (frozen for the D2 memory refactor)" begin
    h1 = nvdb_sha256(fixture_single_leaf(), SINGLE_LEAF_ORIGIN, SINGLE_LEAF_EXTENT)
    h2 = nvdb_sha256(fixture_multi_lower(), MULTI_LOWER_ORIGIN, MULTI_LOWER_EXTENT)
    # deterministic: the same fixture written twice produces identical bytes.
    @test h1 == nvdb_sha256(fixture_single_leaf(), SINGLE_LEAF_ORIGIN, SINGLE_LEAF_EXTENT)
    @test h2 == nvdb_sha256(fixture_multi_lower(), MULTI_LOWER_ORIGIN, MULTI_LOWER_EXTENT)
    # golden anchors.
    @test h1 == GOLDEN_SINGLE_LEAF
    @test h2 == GOLDEN_MULTI_LOWER
end
