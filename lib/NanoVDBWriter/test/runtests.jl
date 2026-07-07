using Test, NanoVDBWriter
using GeometryBasics: Point3f, Vec3f
using SHA: sha256

# ---------------------------------------------------------------------------
# Frozen fixtures + golden byte anchors.
#
# The golden SHA-256s pin the exact bytes save_nanovdb writes.  Do not edit
# the fixtures, their origin/extent, or the goldens unless the file format
# intentionally changes.
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
# two lower nodes (a dense ≤128³ array collapses to a single lower node).  The
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

# Golden SHA-256 of the writer output per fixture.  Re-anchored when the format
# was corrected (inclusive index bbox, Float64 Map, HasBBox|HasMinMax flags,
# per-node min/max over active voxels, name-hash nameKey); the structural
# testset below pins those semantics independently of the hash.
const GOLDEN_SINGLE_LEAF = "9515dffe015a2bc53d5cb272c03ae424a4ffbaffeb957fa3202f254f05a9f3c8"
const GOLDEN_MULTI_LOWER = "7158e4b63846f84249eb31d65b272479d2b373536f539f09f283c7f73d19c964"

@testset "NanoVDBWriter round-trip" begin
    # sparse, asymmetric input
    data = zeros(Float32, 8, 8, 8); data[2,3,4] = 1.5f0; data[7,7,7] = 0.25f0
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
    @test h1 == GOLDEN_SINGLE_LEAF
    @test h2 == GOLDEN_MULTI_LOWER
end

@testset "fixed format semantics (findings 1, 3, 5), independent of the hash" begin
    # GridFlags bits we set; reference stringHash("density").
    HAS_BBOX        = UInt32(1) << 1
    HAS_MINMAX      = UInt32(1) << 2
    DENSITY_NAMEKEY = UInt64(9184452543000)

    # string_hash reproduces the reference nameKey (finding 5).
    @test NanoVDBWriter.string_hash("density") == DENSITY_NAMEKEY

    function parse_fixture(data, origin, extent)
        p = tempname() * ".nvdb"
        save_nanovdb(p, data, origin, extent)
        h = NanoVDBWriter.parse_nanovdb_header(p)
        rm(p)
        return h
    end

    # (data, origin, extent, expected index min/max (INCLUSIVE), world min/max).
    # world bbox spans the full input domain — the active-only mapped box (the
    # more NanoVDB-idiomatic form) renders black in IndeX (see the writer note).
    cases = (
        (fixture_single_leaf(), SINGLE_LEAF_ORIGIN, SINGLE_LEAF_EXTENT,
         (Int32(0), Int32(0), Int32(0)), (Int32(7), Int32(7), Int32(7)),
         (0.0, 0.0, 0.0), (8.0, 8.0, 8.0)),
        # MULTI_LOWER: the k ≤ 32 background slab makes the z index-bbox start at
        # 32 (not 0) — the discriminating case for the inclusive-max fix.
        (fixture_multi_lower(), MULTI_LOWER_ORIGIN, MULTI_LOWER_EXTENT,
         (Int32(0), Int32(0), Int32(32)), (Int32(135), Int32(63), Int32(63)),
         (0.0, 0.0, 0.0), (136.0, 64.0, 64.0)),
    )
    for (data, o, e, imin, imax, wmin, wmax) in cases
        h = parse_fixture(data, o, e)
        # finding 1: index bbox max is INCLUSIVE (coord + LEAF_DIM - 1); z-min of
        # MULTI_LOWER is 32, so this is not a symmetric all-zeros case.
        @test h.index_bbox_min == imin
        @test h.index_bbox_max == imax
        # worldBBox spans the full authored domain (IndeX-compat constraint).
        @test h.world_bbox_min == wmin
        @test h.world_bbox_max == wmax
        # finding 3: mFlags advertises exactly the fields we populate.
        @test h.grid_flags & HAS_BBOX   != 0
        @test h.grid_flags & HAS_MINMAX != 0
        # finding 5: nameKey = stringHash("density").
        @test h.name_key == DENSITY_NAMEKEY
    end
end

@testset "double-precision Map/voxelSize computed in Float64 (finding 4)" begin
    # dx = 1/3 is not exactly Float32-representable, so a Float64 division and a
    # reconstruction from the Float32 Map field diverge — this pins the double
    # Map/voxelSize to the Float64 path.
    d = zeros(Float32, 3, 3, 3); d[2, 2, 2] = 1f0
    p = tempname() * ".nvdb"
    save_nanovdb(p, d, Point3f(0, 0, 0), Vec3f(1, 1, 1))
    raw = read(p); rm(p)
    # GridData begins at file byte 200 (0-indexed); Map at GridData +296, so
    # mMatD[0] at +384, mVecD[0] at +528, and mVoxelSize[0] at +608.
    f64(off0) = only(reinterpret(Float64, raw[200 + off0 + 1 : 200 + off0 + 8]))
    dx_f64 = 1.0 / 3
    # old (buggy) path: reconstruct dx from the Float32 inverse-matrix field.
    dx_f32path  = Float64(1f0 / Float32(1 / dx_f64))
    vec_f64     = dx_f64 / 2                   # origin 0 + voxel/2
    vec_f32path = Float64(Float32(Float32(1f0 / 3) / 2))
    @test f64(384) == dx_f64          # mMatD[0]
    @test f64(384) != dx_f32path
    @test f64(608) == dx_f64          # mVoxelSize[0]
    @test f64(528) == vec_f64         # mVecD[0]
    @test f64(528) != vec_f32path
end

@testset "input validation" begin
    good = fixture_single_leaf()
    o, e = Point3f(0, 0, 0), Vec3f(1, 1, 1)
    # NaN voxel → rejected (would count as active and poison min/max).
    nan_data = copy(good); nan_data[1, 1, 1] = NaN32
    @test_throws ArgumentError save_nanovdb(tempname() * ".nvdb", nan_data, o, e)
    # Inf voxel → rejected (same non-finite guard).
    inf_data = copy(good); inf_data[2, 2, 2] = Inf32
    @test_throws ArgumentError save_nanovdb(tempname() * ".nvdb", inf_data, o, e)
    # zero extent → rejected (would silently write an Inf voxel size).
    @test_throws ArgumentError save_nanovdb(tempname() * ".nvdb", good, o, Vec3f(0, 0, 0))
    # zero-dim array → rejected up front, not via the all-background error.
    @test_throws ArgumentError save_nanovdb(tempname() * ".nvdb", zeros(Float32, 0, 0, 0), o, e)
    # all-background input → ErrorException.
    @test_throws ErrorException save_nanovdb(tempname() * ".nvdb", zeros(Float32, 8, 8, 8), o, e)
end

@testset "multi-node header invariants + node-offset bias" begin
    d = fixture_multi_lower()
    path = tempname() * ".nvdb"
    save_nanovdb(path, d, MULTI_LOWER_ORIGIN, MULTI_LOWER_EXTENT)
    h = NanoVDBWriter.parse_nanovdb_header(path)
    @test h.magic == "NanoVDB0"
    @test h.version_major == 32
    @test h.voxel_count == count(!=(0f0), d)

    # Node-offset sanity via the named bias.  TreeData holds mNodeOffset[4] =
    # (leaf, lower, upper, root) Int64s at its start; the root node is first in
    # the node buffer (position 1), so its offset from TreeData = 1 + BIAS.
    io_header_size = 200   # FileHeader(16) + FileMetaData(176) + name(8)
    # 0-indexed TreeData start
    td = io_header_size + NanoVDBWriter.NANOVDB_GRIDDATA_SIZE
    bytes = read(path)
    offs = [only(reinterpret(Int64, bytes[td + (i - 1) * 8 + 1 : td + i * 8])) for i in 1:4]
    @test offs[4] == NanoVDBWriter.TREEDATA_NODE_OFFSET_BIAS + 1  # root == 64
    @test offs[1] > offs[2] > offs[3] > offs[4]  # leaf … root buffer order
    rm(path)
end

@testset "allocation regression (D2 memory refactor)" begin
    # save_nanovdb runs on every live volume edit, so transient allocation is
    # a hot path.  The returned node buffer (~= payload) is the only
    # unavoidable large allocation; guard @allocated below 1.5× the payload.
    # Warm up first so the measured call excludes compilation.
    d = fixture_multi_lower()
    path = tempname() * ".nvdb"
    save_nanovdb(path, d, MULTI_LOWER_ORIGIN, MULTI_LOWER_EXTENT)  # warmup
    payload = filesize(path)
    allocated = @allocated save_nanovdb(path, d, MULTI_LOWER_ORIGIN, MULTI_LOWER_EXTENT)
    rm(path)
    @test 2 * allocated < 3 * payload   # < 1.5× payload
end
