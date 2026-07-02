# dlpack.jl — single-pass repack of a [C=4,W,H] channel-fastest UInt8 buffer
# (map_cpu's still-mapped LdrColor view; tensor [H,W,4] read as (C,W,H)) into a
# Matrix{RGBA{N0f8}} (H,W), no intermediate copy.
# ORIENTATION: ovrtx LdrColor is TOP-LEFT origin (row 1 = top); NO y-flip here
# or in colorbuffer.  Verified: test/m1_orientation_test.jl.

using ColorTypes, FixedPointNumbers

"""
    cwh_to_matrix(pixels::AbstractArray{UInt8,3}) -> Matrix{RGBA{N0f8}}

Convert a [C=4, W, H] UInt8 array (from `map_cpu`) to a `Matrix{RGBA{N0f8}}`
of size (H, W).  `RGBA{N0f8}` is layout-identical to the 4-byte channel-fastest
texel, so this is ONE pass: `reinterpret` collapses the C axis (→ [W,H] RGBA),
`permutedims` transposes to the (H, W) display matrix and copies into an owned
`Matrix` (safe to keep after the source mapping is released).  Top-left origin,
NO y-flip (verified: m1_orientation_test.jl).
"""
function cwh_to_matrix(pixels::AbstractArray{UInt8,3})::Matrix{RGBA{N0f8}}
    @assert size(pixels, 1) == 4 "expected 4 channels, got $(size(pixels, 1))"
    return permutedims(reinterpret(reshape, RGBA{N0f8}, pixels))
end
