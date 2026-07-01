# NanoVDBWriter.jl — dense `Array{Float32,3}` → NanoVDB (`.nvdb`) file writer.
#
# Writes a dense Julia 3-D Float32 array to a standard, UNCOMPRESSED (`Codec::NONE`),
# major-version-32 NanoVDB file that NVIDIA IndeX (via ovrtx) can load and render.
#
# ── ATTRIBUTION ────────────────────────────────────────────────────────────────────────
# The byte-level NanoVDB tree serialisation — `build_nanovdb_from_dense`, the `save_nanovdb`
# grid-buffer authoring, the coordinate hashers, and the `write_buf!`/`bitmask_set!` helpers —
# is LIFTED from `JuliaGraphics/Hikari.jl` (branch `sd/vk-hw-accel`),
# `src/integrators/volpath/nanovdb.jl`, by Simon Danisch and Anton Smirnov.  The lift was done
# with the author's verbal permission.  Hikari.jl carries no license file; this vendored subset
# is used here by that permission.
#
# The NanoVDB binary format itself (struct offsets; the Root → Upper 32³ → Lower 16³ → Leaf 8³
# tree layout; the `io::writeUncompressedGrid` file framing this writer targets) is defined by
# NanoVDB (AcademySoftwareFoundation/openvdb), MPL-2.0.
#
# ── CHANGES vs the lifted Hikari writer (all deliberate, for ovrtx / IndeX compatibility) ──
#   1. `Codec::NONE` — the grid payload is written UNCOMPRESSED (Hikari used ZIP=1); the
#      FileHeader/FileMetaData codec fields are 0 and there is NO 8-byte size prefix.  This is
#      exactly NanoVDB's dependency-free `io::writeUncompressedGrid` framing.
#   2. `GridData::mChecksum` is written as the disabled sentinel `~UInt64(0)` (Hikari omitted it;
#      a disabled checksum is NanoVDB's own default state, so IndeX accepts it).
#   3. The DOUBLE-precision affine `Map` (`mMatD`/`mInvMatD`/`mVecD`) is populated.  Hikari wrote
#      only the single-precision `mMatF`/`mInvMatF`/`mVecF`, leaving the double map zero — a null
#      world→index transform for any double-precision reader (IndeX very likely uses it).
#   4. `TreeData::mVoxelCount` and `FileMetaData::voxelCount` carry the active-voxel count.
#   5. A clear error is raised for all-background input (no active voxels) instead of crashing.

module NanoVDBWriter

using GeometryBasics: Point3f, Vec3f
import Zlib_jll

export save_nanovdb

# ============================================================================
# NanoVDB format constants (pinned to file-format MAJOR version 32 — IndeX's
# Version::isCompatible() requires an equal major, and this box ships v32.8)
# ============================================================================

# magic "NanoVDB0" as a little-endian UInt64 (NANOVDB_MAGIC_NUMB)
const NANOVDB_MAGIC   = UInt64(0x304244566f6e614e)
# version = major<<21 | minor<<10 | patch  →  v32.3.3 (major 32)
const NANOVDB_VERSION = UInt32((32 << 21) | (3 << 10) | 3)
# GridData::mChecksum disabled sentinel — "all 64 bits ON means checksum is disabled"
const CHECKSUM_DISABLED = typemax(UInt64)
const GRIDTYPE_FLOAT  = UInt32(1)   # GridType::Float
const GRIDCLASS_FOG   = UInt32(2)   # GridClass::FogVolume (density)
const GRID_NAME       = "density"

# GridData structure size (NanoVDB.h: sizeof(GridData) == 672B)
const NANOVDB_GRIDDATA_SIZE = 672

# TreeData (64B, immediately after GridData at byte 672):
#   mNodeOffset[4]: 4×Int64 = 32B (offsets to leaf, lower, upper, root)
#   mNodeCount[3]:  3×UInt32 = 12B (counts of leaf, lower, upper)
#   mTileCount[3]:  3×UInt32 = 12B
#   mVoxelCount:    UInt64 = 8B
const TREEDATA_SIZE               = 64
const TREEDATA_NODE_OFFSET_START  = NANOVDB_GRIDDATA_SIZE + 1        # 673 (1-indexed)
const TREEDATA_NODE_COUNT_START   = NANOVDB_GRIDDATA_SIZE + 32 + 1   # 705
const TREEDATA_VOXELCOUNT_OFFSET  = NANOVDB_GRIDDATA_SIZE + 56 + 1   # 729

# Map (264B, at GridData byte 296).  NanoVDB.h Map layout:
#   mMatF[9] 36B | mInvMatF[9] 36B | mVecF[3] 12B | mTaperF 4B
#   mMatD[9] 72B | mInvMatD[9] 72B | mVecD[3] 24B | mTaperD 8B
const MAP_OFFSET          = 296 + 1          # 297 (1-indexed) — mMatF   (index→world, single)
const MAP_INVMATF_OFFSET  = MAP_OFFSET + 36  # 333 — mInvMatF (world→index, single)
const MAP_VECF_OFFSET     = MAP_OFFSET + 72  # 369 — mVecF    (translation, single)
const MAP_TAPERF_OFFSET   = MAP_OFFSET + 84  # 381 — mTaperF
const MAP_MATD_OFFSET     = MAP_OFFSET + 88  # 385 — mMatD    (index→world, double)
const MAP_INVMATD_OFFSET  = MAP_OFFSET + 160 # 457 — mInvMatD (world→index, double)
const MAP_VECD_OFFSET     = MAP_OFFSET + 232 # 529 — mVecD    (translation, double)
const MAP_TAPERD_OFFSET   = MAP_OFFSET + 256 # 553 — mTaperD

# worldBBox (6×Float64) at GridData byte 560; voxelSize (3×Float64) at byte 608
const WORLDBBOX_OFFSET = 560 + 1  # 561
const VOXELSIZE_OFFSET = 608 + 1  # 609

# GridData scalar-field byte offsets (1-indexed), per NanoVDB.h GridData:
#   mMagic(0) mChecksum(8) mVersion(16) mFlags(20) mGridIndex(24) mGridCount(28)
#   mGridSize(32) mGridName[256](40) … mGridClass(632) mGridType(636)
const GRIDDATA_MAGIC_OFFSET     = 1
const GRIDDATA_CHECKSUM_OFFSET  = 9
const GRIDDATA_VERSION_OFFSET   = 17
const GRIDDATA_GRIDCOUNT_OFFSET = 29
const GRIDDATA_GRIDSIZE_OFFSET  = 33
const GRIDDATA_GRIDNAME_OFFSET  = 40   # name bytes go at OFFSET + i (i = 1…)
const GRIDDATA_GRIDCLASS_OFFSET = 633
const GRIDDATA_GRIDTYPE_OFFSET  = 637

# VDB tree configuration for Float grids:
# Root → Upper (32³, LOG2DIM=5) → Lower (16³, LOG2DIM=4) → Leaf (8³, LOG2DIM=3)
const LEAF_LOG2DIM = 3
const LEAF_DIM     = 1 << LEAF_LOG2DIM        # 8
const LEAF_MASK    = (1 << LEAF_LOG2DIM) - 1  # 7

const LOWER_LOG2DIM = 4
const LOWER_DIM     = 1 << LOWER_LOG2DIM       # 16
const LOWER_SIZE    = 1 << (3 * LOWER_LOG2DIM) # 4096
const LOWER_TOTAL   = LEAF_LOG2DIM + LOWER_LOG2DIM  # 7
const LOWER_MASK    = (1 << LOWER_TOTAL) - 1   # 127

const UPPER_LOG2DIM = 5
const UPPER_DIM     = 1 << UPPER_LOG2DIM       # 32
const UPPER_SIZE    = 1 << (3 * UPPER_LOG2DIM) # 32768
const UPPER_TOTAL   = LOWER_TOTAL + UPPER_LOG2DIM  # 12
const UPPER_MASK    = (1 << UPPER_TOTAL) - 1   # 4095

# LeafData<float> (2144B): coords(12) bboxDif(3) flags(1) valueMask(64) min/max/avg/dev(16) values[512](2048)
const LEAFDATA_BBOXMIN_OFFSET = 0
const LEAFDATA_MASK_OFFSET    = 16   # after coords(12) + bboxDif(3) + flags(1)
const LEAFDATA_MIN_OFFSET     = 80   # after valueMask(64)
const LEAFDATA_VALUES_OFFSET  = 96   # min/max/avg/dev(16), aligned to 32
const LEAFDATA_SIZE           = 2144

# InternalData (Upper 32³): header 8240B → aligned 8256; Tile[32768]×8
const UPPER_BBOX_OFFSET      = 0
const UPPER_VALUEMASK_OFFSET = 32
const UPPER_CHILDMASK_OFFSET = 32 + 4096   # 4128
const UPPER_TABLE_OFFSET     = 8256

# InternalData (Lower 16³): header 1072B → aligned 1088; Tile[4096]×8
const LOWER_BBOX_OFFSET      = 0
const LOWER_VALUEMASK_OFFSET = 32
const LOWER_CHILDMASK_OFFSET = 32 + 512    # 544
const LOWER_TABLE_OFFSET     = 1088

# RootData: header 48B → aligned 64; Tile[mTableSize]×32
const ROOTDATA_BBOX_OFFSET       = 0
const ROOTDATA_TABLESIZE_OFFSET  = 24
const ROOTDATA_BACKGROUND_OFFSET = 28
const ROOTDATA_HEADER_SIZE       = 64

# Root Tile (32B): key(8) child(8) state(4) value(4) padding(8)
const ROOTTILE_SIZE         = 32
const ROOTTILE_KEY_OFFSET   = 0
const ROOTTILE_CHILD_OFFSET = 8
const ROOTTILE_STATE_OFFSET = 16
const ROOTTILE_VALUE_OFFSET = 20

# Node sizes (bytes)
const UPPER_NODE_SIZE = UPPER_TABLE_OFFSET + UPPER_SIZE * 8  # 270400
const LOWER_NODE_SIZE = LOWER_TABLE_OFFSET + LOWER_SIZE * 8  # 33856

# ============================================================================
# Coordinate → key / offset hashers (match NanoVDB CoordToKey / CoordToOffset)
# ============================================================================

# USE_SINGLE_ROOT_KEY 64-bit key: shift by UPPER_TOTAL (12) to root-tile coords
@inline function coord_to_root_key(ijk::NTuple{3, Int32})::UInt64
    x, y, z = ijk
    xu, yu, zu = reinterpret(UInt32, x), reinterpret(UInt32, y), reinterpret(UInt32, z)
    zk = UInt64((zu >> UPPER_TOTAL) & 0x1fffff)         # 21 bits
    yk = UInt64((yu >> UPPER_TOTAL) & 0x1fffff) << 21
    xk = UInt64((xu >> UPPER_TOTAL) & 0x1fffff) << 42
    return zk | yk | xk
end

@inline function upper_coord_to_offset(ijk::NTuple{3, Int32})::Int32
    x, y, z = ijk
    xu, yu, zu = reinterpret(UInt32, x), reinterpret(UInt32, y), reinterpret(UInt32, z)
    ox = Int32((xu >> LOWER_TOTAL) & (UPPER_DIM - 1)) << (2 * UPPER_LOG2DIM)
    oy = Int32((yu >> LOWER_TOTAL) & (UPPER_DIM - 1)) << UPPER_LOG2DIM
    oz = Int32((zu >> LOWER_TOTAL) & (UPPER_DIM - 1))
    return ox | oy | oz
end

@inline function lower_coord_to_offset(ijk::NTuple{3, Int32})::Int32
    x, y, z = ijk
    xu, yu, zu = reinterpret(UInt32, x), reinterpret(UInt32, y), reinterpret(UInt32, z)
    ox = Int32((xu >> LEAF_LOG2DIM) & (LOWER_DIM - 1)) << (2 * LOWER_LOG2DIM)
    oy = Int32((yu >> LEAF_LOG2DIM) & (LOWER_DIM - 1)) << LOWER_LOG2DIM
    oz = Int32((zu >> LEAF_LOG2DIM) & (LOWER_DIM - 1))
    return ox | oy | oz
end

@inline function leaf_coord_to_offset(ijk::NTuple{3, Int32})::Int32
    x, y, z = ijk
    ox = (x & LEAF_MASK) << (2 * LEAF_LOG2DIM)
    oy = (y & LEAF_MASK) << LEAF_LOG2DIM
    oz = z & LEAF_MASK
    return ox | oy | oz
end

# ============================================================================
# Buffer write helpers (1-indexed byte offsets)
# ============================================================================

@inline function write_buf!(buffer::Vector{UInt8}, offset::Integer, value::Float32)
    GC.@preserve buffer unsafe_store!(reinterpret(Ptr{Float32}, pointer(buffer, offset)), value)
end
@inline function write_buf!(buffer::Vector{UInt8}, offset::Integer, value::Int32)
    GC.@preserve buffer unsafe_store!(reinterpret(Ptr{Int32}, pointer(buffer, offset)), value)
end
@inline function write_buf!(buffer::Vector{UInt8}, offset::Integer, value::UInt32)
    GC.@preserve buffer unsafe_store!(reinterpret(Ptr{UInt32}, pointer(buffer, offset)), value)
end
@inline function write_buf!(buffer::Vector{UInt8}, offset::Integer, value::Int64)
    GC.@preserve buffer unsafe_store!(reinterpret(Ptr{Int64}, pointer(buffer, offset)), value)
end
@inline function write_buf!(buffer::Vector{UInt8}, offset::Integer, value::UInt64)
    GC.@preserve buffer unsafe_store!(reinterpret(Ptr{UInt64}, pointer(buffer, offset)), value)
end
@inline function write_buf!(buffer::Vector{UInt8}, offset::Integer, value::Float64)
    GC.@preserve buffer unsafe_store!(reinterpret(Ptr{Float64}, pointer(buffer, offset)), value)
end
@inline function write_buf!(buffer::Vector{UInt8}, offset::Integer, value::UInt16)
    GC.@preserve buffer unsafe_store!(reinterpret(Ptr{UInt16}, pointer(buffer, offset)), value)
end

# Set bit n (0-indexed) in a bitmask at mask_offset (1-indexed)
@inline function bitmask_set!(buffer::Vector{UInt8}, mask_offset::Integer, n::Integer)
    byte_idx = n >> 3
    bit_idx = n & 7
    buffer[mask_offset + byte_idx] |= UInt8(1) << bit_idx
end

# ============================================================================
# Dense Array{Float32,3} → in-memory NanoVDB tree buffer
# ============================================================================

"""
    build_nanovdb_from_dense(data, origin, extent; background=0f0) -> (buffer, metadata)

Convert a dense 3-D `Float32` array to a NanoVDB node buffer (`Root | Upper | Lower | Leaf`),
storing only 8³ leaf blocks that contain non-`background` voxels.  `origin` is the world origin,
`extent` the world-space size (voxel size = `extent ./ size(data)`).  Returns the node `buffer`
(`Vector{UInt8}`, WITHOUT the 736-byte GridData+TreeData header) and a `metadata` NamedTuple.
"""
function build_nanovdb_from_dense(
    data::Array{Float32, 3},
    origin::Point3f,
    extent::Vec3f;
    background::Float32 = 0f0,
)
    nx, ny, nz = size(data)
    dx, dy, dz = extent[1] / nx, extent[2] / ny, extent[3] / nz

    # ---- Phase 1: collect active leaf blocks (8³) + count active voxels ----
    n_bx = cld(nx, LEAF_DIM)
    n_by = cld(ny, LEAF_DIM)
    n_bz = cld(nz, LEAF_DIM)

    leaf_coords = NTuple{3, Int32}[]
    leaf_values = Vector{Vector{Float32}}()
    scratch = Vector{Float32}(undef, 512)
    n_active = 0

    for bz in 0:n_bz-1, by in 0:n_by-1, bx in 0:n_bx-1
        has_active = false
        fill!(scratch, background)

        for lz in 0:LEAF_DIM-1, ly in 0:LEAF_DIM-1, lx in 0:LEAF_DIM-1
            ix = bx * LEAF_DIM + lx + 1
            iy = by * LEAF_DIM + ly + 1
            iz = bz * LEAF_DIM + lz + 1
            v = (ix <= nx && iy <= ny && iz <= nz) ? data[ix, iy, iz] : background
            leaf_idx = (lx << (2*LEAF_LOG2DIM)) | (ly << LEAF_LOG2DIM) | lz
            scratch[leaf_idx + 1] = v
            if v != background
                has_active = true
                n_active += 1
            end
        end

        if has_active
            base = (Int32(bx * LEAF_DIM), Int32(by * LEAF_DIM), Int32(bz * LEAF_DIM))
            push!(leaf_coords, base)
            push!(leaf_values, copy(scratch))
        end
    end

    n_leaves = length(leaf_coords)
    n_leaves == 0 && error(
        "build_nanovdb_from_dense: input has no active (non-background) voxels — " *
        "nothing to write (background = $background).")

    # ---- Phase 2: group leaves into lower nodes (each covers 128³) ----
    lower_to_leaves = Dict{NTuple{3,Int32}, Vector{Int}}()
    for (li, coord) in enumerate(leaf_coords)
        lb = (coord[1] & ~Int32(LOWER_MASK),
              coord[2] & ~Int32(LOWER_MASK),
              coord[3] & ~Int32(LOWER_MASK))
        push!(get!(lower_to_leaves, lb, Int[]), li)
    end
    lower_bases = sort!(collect(keys(lower_to_leaves)))
    n_lowers = length(lower_bases)

    # ---- Phase 3: group lower nodes into upper nodes (each covers 4096³) ----
    upper_to_lowers = Dict{NTuple{3,Int32}, Vector{Int}}()
    for (low_i, lb) in enumerate(lower_bases)
        ub = (lb[1] & ~Int32(UPPER_MASK),
              lb[2] & ~Int32(UPPER_MASK),
              lb[3] & ~Int32(UPPER_MASK))
        push!(get!(upper_to_lowers, ub, Int[]), low_i)
    end
    upper_bases = sort!(collect(keys(upper_to_lowers)))
    n_uppers = length(upper_bases)

    # ---- Phase 4: buffer layout ----
    root_n_tiles = n_uppers
    root_size = ROOTDATA_HEADER_SIZE + root_n_tiles * ROOTTILE_SIZE
    upper_section = n_uppers * UPPER_NODE_SIZE
    lower_section = n_lowers * LOWER_NODE_SIZE
    leaf_section = n_leaves * LEAFDATA_SIZE
    total_size = root_size + upper_section + lower_section + leaf_section

    buffer = zeros(UInt8, total_size)

    root_pos = 1
    upper_pos(i) = root_pos + root_size + (i - 1) * UPPER_NODE_SIZE
    lower_pos(i) = root_pos + root_size + upper_section + (i - 1) * LOWER_NODE_SIZE

    leaf_sorted_order = sortperm(leaf_coords)
    leaf_buf_pos = Vector{Int64}(undef, n_leaves)
    for (slot, li) in enumerate(leaf_sorted_order)
        leaf_buf_pos[li] = root_pos + root_size + upper_section + lower_section + (slot - 1) * LEAFDATA_SIZE
    end

    # ---- Phase 5: leaf nodes ----
    for li in 1:n_leaves
        coord = leaf_coords[li]
        values = leaf_values[li]
        off = leaf_buf_pos[li]

        write_buf!(buffer, off + LEAFDATA_BBOXMIN_OFFSET, coord[1])
        write_buf!(buffer, off + LEAFDATA_BBOXMIN_OFFSET + 4, coord[2])
        write_buf!(buffer, off + LEAFDATA_BBOXMIN_OFFSET + 8, coord[3])
        buffer[off + 12] = 0x07
        buffer[off + 13] = 0x07
        buffer[off + 14] = 0x07

        vmin = typemax(Float32)
        vmax = typemin(Float32)
        for i in 0:511
            v = values[i + 1]
            if v != background
                bitmask_set!(buffer, off + LEAFDATA_MASK_OFFSET, i)
            end
            vmin = min(vmin, v)
            vmax = max(vmax, v)
        end
        write_buf!(buffer, off + LEAFDATA_MIN_OFFSET, vmin)
        write_buf!(buffer, off + LEAFDATA_MIN_OFFSET + 4, vmax)

        for i in 0:511
            write_buf!(buffer, off + LEAFDATA_VALUES_OFFSET + i * 4, values[i + 1])
        end
    end

    # ---- Phase 6: lower nodes ----
    for (low_i, lb) in enumerate(lower_bases)
        off = lower_pos(low_i)
        write_buf!(buffer, off + LOWER_BBOX_OFFSET, lb[1])
        write_buf!(buffer, off + LOWER_BBOX_OFFSET + 4, lb[2])
        write_buf!(buffer, off + LOWER_BBOX_OFFSET + 8, lb[3])
        write_buf!(buffer, off + LOWER_BBOX_OFFSET + 12, lb[1] + Int32(LEAF_DIM * LOWER_DIM - 1))
        write_buf!(buffer, off + LOWER_BBOX_OFFSET + 16, lb[2] + Int32(LEAF_DIM * LOWER_DIM - 1))
        write_buf!(buffer, off + LOWER_BBOX_OFFSET + 20, lb[3] + Int32(LEAF_DIM * LOWER_DIM - 1))

        for li in lower_to_leaves[lb]
            coord = leaf_coords[li]
            n = lower_coord_to_offset(coord)
            bitmask_set!(buffer, off + LOWER_CHILDMASK_OFFSET, n)
            bitmask_set!(buffer, off + LOWER_VALUEMASK_OFFSET, n)
            child_off = Int64(leaf_buf_pos[li] - off)
            write_buf!(buffer, off + LOWER_TABLE_OFFSET + n * 8, child_off)
        end
    end

    # ---- Phase 7: upper nodes ----
    for (up_i, ub) in enumerate(upper_bases)
        off = upper_pos(up_i)
        write_buf!(buffer, off + UPPER_BBOX_OFFSET, ub[1])
        write_buf!(buffer, off + UPPER_BBOX_OFFSET + 4, ub[2])
        write_buf!(buffer, off + UPPER_BBOX_OFFSET + 8, ub[3])
        write_buf!(buffer, off + UPPER_BBOX_OFFSET + 12, ub[1] + Int32(LEAF_DIM * LOWER_DIM * UPPER_DIM - 1))
        write_buf!(buffer, off + UPPER_BBOX_OFFSET + 16, ub[2] + Int32(LEAF_DIM * LOWER_DIM * UPPER_DIM - 1))
        write_buf!(buffer, off + UPPER_BBOX_OFFSET + 20, ub[3] + Int32(LEAF_DIM * LOWER_DIM * UPPER_DIM - 1))

        for low_i in upper_to_lowers[ub]
            lb = lower_bases[low_i]
            n = upper_coord_to_offset(lb)
            bitmask_set!(buffer, off + UPPER_CHILDMASK_OFFSET, n)
            bitmask_set!(buffer, off + UPPER_VALUEMASK_OFFSET, n)
            child_off = Int64(lower_pos(low_i) - off)
            write_buf!(buffer, off + UPPER_TABLE_OFFSET + n * 8, child_off)
        end
    end

    # ---- Phase 8: root node ----
    idx_min = (typemax(Int32), typemax(Int32), typemax(Int32))
    idx_max = (typemin(Int32), typemin(Int32), typemin(Int32))
    for coord in leaf_coords
        idx_min = (min(idx_min[1], coord[1]), min(idx_min[2], coord[2]), min(idx_min[3], coord[3]))
        idx_max = (max(idx_max[1], coord[1] + Int32(LEAF_DIM)),
                   max(idx_max[2], coord[2] + Int32(LEAF_DIM)),
                   max(idx_max[3], coord[3] + Int32(LEAF_DIM)))
    end

    write_buf!(buffer, root_pos + ROOTDATA_BBOX_OFFSET, idx_min[1])
    write_buf!(buffer, root_pos + ROOTDATA_BBOX_OFFSET + 4, idx_min[2])
    write_buf!(buffer, root_pos + ROOTDATA_BBOX_OFFSET + 8, idx_min[3])
    write_buf!(buffer, root_pos + ROOTDATA_BBOX_OFFSET + 12, idx_max[1])
    write_buf!(buffer, root_pos + ROOTDATA_BBOX_OFFSET + 16, idx_max[2])
    write_buf!(buffer, root_pos + ROOTDATA_BBOX_OFFSET + 20, idx_max[3])
    write_buf!(buffer, root_pos + ROOTDATA_TABLESIZE_OFFSET, UInt32(root_n_tiles))
    write_buf!(buffer, root_pos + ROOTDATA_BACKGROUND_OFFSET, background)

    tile_base = root_pos + ROOTDATA_HEADER_SIZE
    for (ti, ub) in enumerate(upper_bases)
        t_off = tile_base + (ti - 1) * ROOTTILE_SIZE
        write_buf!(buffer, t_off + ROOTTILE_KEY_OFFSET, coord_to_root_key(ub))
        write_buf!(buffer, t_off + ROOTTILE_CHILD_OFFSET, Int64(upper_pos(ti) - root_pos))
        write_buf!(buffer, t_off + ROOTTILE_STATE_OFFSET, UInt32(1))
        write_buf!(buffer, t_off + ROOTTILE_VALUE_OFFSET, background)
    end

    # ---- metadata (world↔index transform: p_index = inv_mat * (p_world - vec)) ----
    vec = (Float32(origin[1] + dx/2), Float32(origin[2] + dy/2), Float32(origin[3] + dz/2))
    inv_mat = (Float32(1/dx), 0f0, 0f0,
               0f0, Float32(1/dy), 0f0,
               0f0, 0f0, Float32(1/dz))
    world_min = (Float32(origin[1]), Float32(origin[2]), Float32(origin[3]))
    world_max = (Float32(origin[1] + extent[1]), Float32(origin[2] + extent[2]), Float32(origin[3] + extent[3]))

    metadata = (
        world_min = world_min,
        world_max = world_max,
        inv_mat = inv_mat,
        vec = vec,
        root_offset = Int64(root_pos),
        upper_offset = Int64(upper_pos(1)),
        lower_offset = Int64(lower_pos(1)),
        leaf_offset = Int64(leaf_buf_pos[leaf_sorted_order[1]]),
        leaf_count = Int32(n_leaves),
        lower_count = Int32(n_lowers),
        upper_count = Int32(n_uppers),
        root_table_size = Int32(root_n_tiles),
        index_min = idx_min,
        index_max = idx_max,
        voxel_count = n_active,
    )
    return buffer, metadata
end

# ============================================================================
# Grid buffer + metadata → uncompressed (Codec::NONE) NanoVDB file
# ============================================================================

"""
    save_nanovdb(filepath, buffer::Vector{UInt8}, metadata::NamedTuple) -> String

Author the 736-byte `GridData`+`TreeData` header in front of the node `buffer` and write a
standard, UNCOMPRESSED (`Codec::NONE`) NanoVDB file (NanoVDB's `io::writeUncompressedGrid`
framing): `[16B FileHeader][176B FileMetaData][nameSize B grid name][raw grid]`.  Returns `filepath`.
"""
function save_nanovdb(filepath::AbstractString, buffer::Vector{UInt8}, metadata::NamedTuple)
    header_size = NANOVDB_GRIDDATA_SIZE + TREEDATA_SIZE   # 736
    full_buffer = zeros(UInt8, header_size + length(buffer))
    copyto!(full_buffer, header_size + 1, buffer, 1, length(buffer))
    grid_size = UInt64(length(full_buffer))

    # ---- GridData scalar fields ----
    write_buf!(full_buffer, GRIDDATA_MAGIC_OFFSET,     NANOVDB_MAGIC)
    write_buf!(full_buffer, GRIDDATA_CHECKSUM_OFFSET,  CHECKSUM_DISABLED)   # (change #2)
    write_buf!(full_buffer, GRIDDATA_VERSION_OFFSET,   NANOVDB_VERSION)
    write_buf!(full_buffer, GRIDDATA_GRIDCOUNT_OFFSET, UInt32(1))
    write_buf!(full_buffer, GRIDDATA_GRIDSIZE_OFFSET,  grid_size)
    for (i, c) in enumerate(codeunits(GRID_NAME * "\0"))
        full_buffer[GRIDDATA_GRIDNAME_OFFSET + i] = c
    end
    write_buf!(full_buffer, GRIDDATA_GRIDCLASS_OFFSET, GRIDCLASS_FOG)
    write_buf!(full_buffer, GRIDDATA_GRIDTYPE_OFFSET,  GRIDTYPE_FLOAT)

    # ---- Map: single- AND double-precision (change #3) ----
    inv_mat = metadata.inv_mat                       # world→index (mInvMat)
    m = inv_mat                                       # index→world (mMat) = inverse of inv_mat
    cof = (m[5]*m[9] - m[6]*m[8], m[3]*m[8] - m[2]*m[9], m[2]*m[6] - m[3]*m[5],
           m[6]*m[7] - m[4]*m[9], m[1]*m[9] - m[3]*m[7], m[3]*m[4] - m[1]*m[6],
           m[4]*m[8] - m[5]*m[7], m[2]*m[7] - m[1]*m[8], m[1]*m[5] - m[2]*m[4])
    det = m[1]*cof[1] + m[2]*cof[4] + m[3]*cof[7]
    mat = ntuple(i -> cof[i] / det, 9)                # mMat (index→world)
    vec = metadata.vec

    for i in 1:9;  write_buf!(full_buffer, MAP_OFFSET        + (i-1)*4, Float32(mat[i]));     end
    for i in 1:9;  write_buf!(full_buffer, MAP_INVMATF_OFFSET + (i-1)*4, Float32(inv_mat[i])); end
    for i in 1:3;  write_buf!(full_buffer, MAP_VECF_OFFSET    + (i-1)*4, Float32(vec[i]));     end
    write_buf!(full_buffer, MAP_TAPERF_OFFSET, 1.0f0)
    for i in 1:9;  write_buf!(full_buffer, MAP_MATD_OFFSET    + (i-1)*8, Float64(mat[i]));     end
    for i in 1:9;  write_buf!(full_buffer, MAP_INVMATD_OFFSET + (i-1)*8, Float64(inv_mat[i])); end
    for i in 1:3;  write_buf!(full_buffer, MAP_VECD_OFFSET    + (i-1)*8, Float64(vec[i]));     end
    write_buf!(full_buffer, MAP_TAPERD_OFFSET, 1.0)

    # ---- worldBBox (6×Float64) + voxelSize (3×Float64) ----
    wmin, wmax = metadata.world_min, metadata.world_max
    for (i, v) in enumerate((wmin[1], wmin[2], wmin[3], wmax[1], wmax[2], wmax[3]))
        write_buf!(full_buffer, WORLDBBOX_OFFSET + (i-1)*8, Float64(v))
    end
    voxel_size = (Float64(1f0 / inv_mat[1]), Float64(1f0 / inv_mat[5]), Float64(1f0 / inv_mat[9]))
    for (i, v) in enumerate(voxel_size)
        write_buf!(full_buffer, VOXELSIZE_OFFSET + (i-1)*8, Float64(v))
    end

    # ---- TreeData: node offsets, counts, voxel count ----
    for (i, off) in enumerate((metadata.leaf_offset + 63, metadata.lower_offset + 63,
                                metadata.upper_offset + 63, metadata.root_offset + 63))
        write_buf!(full_buffer, TREEDATA_NODE_OFFSET_START + (i-1)*8, UInt64(off))
    end
    for (i, n) in enumerate((metadata.leaf_count, metadata.lower_count, metadata.upper_count))
        write_buf!(full_buffer, TREEDATA_NODE_COUNT_START + (i-1)*4, UInt32(n))
    end
    write_buf!(full_buffer, TREEDATA_VOXELCOUNT_OFFSET, UInt64(metadata.voxel_count))  # (change #4)

    # ---- FileHeader(16) + FileMetaData(176) + grid name(8) — Codec::NONE (change #1) ----
    io_hdr = zeros(UInt8, 200)
    # FileHeader: magic(0) version(8) gridCount(12) codec(14)
    write_buf!(io_hdr,  1, NANOVDB_MAGIC)
    write_buf!(io_hdr,  9, NANOVDB_VERSION)
    write_buf!(io_hdr, 13, UInt16(1))       # gridCount
    write_buf!(io_hdr, 15, UInt16(0))       # codec = NONE

    m0 = 17  # FileMetaData base (1-indexed); field offsets relative to file byte 16
    write_buf!(io_hdr, m0 +   0, grid_size)                        # gridSize
    write_buf!(io_hdr, m0 +   8, grid_size)                        # fileSize == gridSize (NONE)
    write_buf!(io_hdr, m0 +  16, UInt64(0))                        # nameKey (0 for uncompressed)
    write_buf!(io_hdr, m0 +  24, UInt64(metadata.voxel_count))     # voxelCount
    write_buf!(io_hdr, m0 +  32, GRIDTYPE_FLOAT)                   # gridType
    write_buf!(io_hdr, m0 +  36, GRIDCLASS_FOG)                    # gridClass
    for (i, v) in enumerate((wmin[1], wmin[2], wmin[3], wmax[1], wmax[2], wmax[3]))
        write_buf!(io_hdr, m0 + 40 + (i-1)*8, Float64(v))          # worldBBox
    end
    for (i, v) in enumerate((metadata.index_min[1], metadata.index_min[2], metadata.index_min[3],
                              metadata.index_max[1], metadata.index_max[2], metadata.index_max[3]))
        write_buf!(io_hdr, m0 + 88 + (i-1)*4, Int32(v))            # indexBBox
    end
    for (i, v) in enumerate(voxel_size)
        write_buf!(io_hdr, m0 + 112 + (i-1)*8, Float64(v))         # voxelSize
    end
    write_buf!(io_hdr, m0 + 136, UInt32(length(GRID_NAME) + 1))    # nameSize ("density\0" = 8)
    for (i, n) in enumerate((metadata.leaf_count, metadata.lower_count, metadata.upper_count, 1))
        write_buf!(io_hdr, m0 + 140 + (i-1)*4, UInt32(n))          # nodeCount[4]
    end
    # tileCount[3] stays zero (offset 156)
    write_buf!(io_hdr, m0 + 168, UInt16(0))                        # codec = NONE
    write_buf!(io_hdr, m0 + 172, NANOVDB_VERSION)                  # version

    for (i, c) in enumerate(codeunits(GRID_NAME * "\0"))           # grid name at file byte 192
        io_hdr[192 + i] = c
    end

    open(filepath, "w") do io
        write(io, io_hdr)
        write(io, full_buffer)   # raw grid, uncompressed, no size prefix
    end
    return String(filepath)
end

"""
    save_nanovdb(path::AbstractString, data::Array{Float32,3}, origin, extent) -> String

Build a NanoVDB tree from a dense 3-D array and write it to `path` as a standard, uncompressed
(`Codec::NONE`), major-32 NanoVDB file (grid name `"density"`, `GridClass::FogVolume`,
background `0f0`).  `origin`/`extent` are `GeometryBasics.Point3f`/`Vec3f` (voxel size =
`extent ./ size(data)`).  Returns `path`.
"""
function save_nanovdb(path::AbstractString, data::Array{Float32,3}, origin, extent)
    buffer, metadata = build_nanovdb_from_dense(data, Point3f(origin...), Vec3f(extent...))
    return save_nanovdb(path, buffer, metadata)
end

# ============================================================================
# zlib FFI — LIFTED from Hikari (kept available for a future ZIP-codec path; the
# default `save_nanovdb` above writes Codec::NONE and does NOT call this).
# ============================================================================

mutable struct ZStream
    next_in::Ptr{UInt8};  avail_in::Cuint;  total_in::Culong
    next_out::Ptr{UInt8}; avail_out::Cuint; total_out::Culong
    msg::Ptr{Cchar};      state::Ptr{Cvoid}
    zalloc::Ptr{Cvoid};   zfree::Ptr{Cvoid}; opaque::Ptr{Cvoid}
    data_type::Cint;      adler::Culong;     reserved::Culong
end
ZStream() = ZStream(C_NULL, 0, 0, C_NULL, 0, 0, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, 0, 0, 0)

function compress_zlib(data::Vector{UInt8})
    out_buf = Vector{UInt8}(undef, length(data) + div(length(data), 100) + 1024)
    z = Ref(ZStream())
    z[].next_in = pointer(data);     z[].avail_in = length(data)
    z[].next_out = pointer(out_buf); z[].avail_out = length(out_buf)
    ret = ccall((:deflateInit_, Zlib_jll.libz), Cint,
        (Ref{ZStream}, Cint, Cstring, Cint), z, 6, "1.2.11", sizeof(ZStream))
    ret != 0 && error("deflateInit failed: $ret")
    GC.@preserve data out_buf begin
        ccall((:deflate, Zlib_jll.libz), Cint, (Ref{ZStream}, Cint), z, 4)  # Z_FINISH
    end
    compressed_size = z[].total_out
    ccall((:deflateEnd, Zlib_jll.libz), Cint, (Ref{ZStream},), z)
    return out_buf[1:compressed_size]
end

# ============================================================================
# Minimal reader — reads only the file-IO header (FileHeader + FileMetaData),
# used by the round-trip test.  Works for any codec (the header is uncompressed).
# ============================================================================

"""
    parse_nanovdb_header(path) -> NamedTuple

Read the NanoVDB file-IO header of `path` and return
`(; magic::String, version_major::Int, grid_type::UInt32, voxel_count::Int)`.
"""
function parse_nanovdb_header(path::AbstractString)
    bytes = open(path, "r") do io
        read(io, 52)
    end
    length(bytes) >= 52 || error("parse_nanovdb_header: file too short to be a NanoVDB file: $path")
    magic = String(bytes[1:8])
    version = only(reinterpret(UInt32, bytes[9:12]))
    version_major = Int(version >> 21)
    # FileMetaData starts at file byte 16: voxelCount at +24 (byte 40), gridType at +32 (byte 48)
    voxel_count = Int(only(reinterpret(UInt64, bytes[41:48])))
    grid_type = only(reinterpret(UInt32, bytes[49:52]))
    return (; magic, version_major, grid_type, voxel_count)
end

end # module NanoVDBWriter
