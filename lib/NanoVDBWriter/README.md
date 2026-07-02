# NanoVDBWriter

A tiny, standalone, pure-Julia writer that serialises a dense `Array{Float32,3}` to a standard
**NanoVDB** (`.nvdb`) file — the format NVIDIA IndeX (via `ovrtx`) loads for volume rendering.

It is a sub-package of [OmniverseMakie](../../); the Makie `volume!` plot path calls
`save_nanovdb` to materialise the on-disk file that `author_vdb_volume!` then references.

## API

```julia
using NanoVDBWriter
using GeometryBasics: Point3f, Vec3f

data = rand(Float32, 64, 64, 64)                       # dense density field
path = save_nanovdb(tempname() * ".nvdb", data,
                    Point3f(-10, -10, -10),            # world origin
                    Vec3f(20, 20, 20))                 # world extent (voxel size = extent ./ size(data))
```

- `save_nanovdb(path, data::Array{Float32,3}, origin, extent) -> String`
  writes a major-32, `Codec::NONE` (uncompressed) NanoVDB grid named `"density"`
  (`GridClass::FogVolume`, background `0f0`) and returns `path`.
- `NanoVDBWriter.parse_nanovdb_header(path) -> (; magic, version_major, grid_type, voxel_count)`
  reads back the file-IO header (used by the round-trip test).

## Format specifics (why this file loads in IndeX)

- **Major version 32.** IndeX's `Version::isCompatible()` requires an equal major version; this box
  ships NanoVDB v32.8.
- **`Codec::NONE`.** The grid is written uncompressed, following NanoVDB's dependency-free
  `io::writeUncompressedGrid` framing: `[16B FileHeader][176B FileMetaData][8B name][raw grid]`.
- **Disabled checksum.** `GridData::mChecksum` is the "disabled" sentinel (`~UInt64(0)`), which is
  NanoVDB's own default state for a freshly built grid.
- **Full affine `Map`.** Both the single- and double-precision index↔world transforms are written.

## Attribution / license

The byte-level NanoVDB tree serialisation is **lifted from
[`JuliaGraphics/Hikari.jl`](https://github.com/JuliaGraphics/Hikari.jl)** (branch `sd/vk-hw-accel`,
`src/integrators/volpath/nanovdb.jl`) by **Simon Danisch** and **Anton Smirnov**, used here with the
author's verbal permission (Hikari.jl carries no license file). The NanoVDB binary format itself is
defined by NanoVDB (AcademySoftwareFoundation/openvdb), **MPL-2.0**. See the header of
`src/NanoVDBWriter.jl` for the exact list of changes made to the lifted code for ovrtx/IndeX
compatibility.

## Dependencies

`GeometryBasics` (for `Point3f`/`Vec3f`) is the only runtime dependency. (A lifted `compress_zlib`
zlib FFI for an optional ZIP-codec path was removed as dead code — the writer is `Codec::NONE` only;
it lives in git history.)
