# dlpack.jl — single-pass repack of a [C=4,W,H] channel-fastest UInt8 buffer
# (map_cpu's still-mapped LdrColor view) into a Matrix{RGBA{N0f8}} (H,W), no
# intermediate copy.  ovrtx LdrColor is top-left origin (row 1 = top); no
# y-flip here or in colorbuffer.

using ColorTypes, FixedPointNumbers

"""
    cwh_to_matrix(pixels::AbstractArray{UInt8,3}) -> Matrix{RGBA{N0f8}}

Convert a [C=4, W, H] UInt8 array (from `map_cpu`) to a `Matrix{RGBA{N0f8}}`
of size (H, W).  `RGBA{N0f8}` is layout-identical to the 4-byte
channel-fastest texel, so this is one pass: `reinterpret` collapses the C
axis (→ [W,H] RGBA) and `permutedims` transposes to the (H, W) display
matrix, copying into an owned `Matrix` (safe to keep after the source mapping
is released).  Top-left origin, no y-flip.
"""
function cwh_to_matrix(pixels::AbstractArray{UInt8,3})::Matrix{RGBA{N0f8}}
    @assert size(pixels, 1) == 4 "expected 4 channels, got $(size(pixels, 1))"
    return permutedims(reinterpret(reshape, RGBA{N0f8}, pixels))
end
