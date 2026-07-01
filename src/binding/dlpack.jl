# dlpack.jl — reshape a [C=4,W,H] channel-fastest UInt8 buffer (from map_cpu;
# LdrColor tensor [H,W,4] read as (C,W,H)) into a Matrix{RGBA{N0f8}} (H,W).
# ORIENTATION: ovrtx LdrColor is TOP-LEFT origin (row 1 = top); NO y-flip here
# or in colorbuffer.  Verified: test/m1_orientation_test.jl.

using ColorTypes, FixedPointNumbers

"""
    cwh_to_matrix(pixels::Array{UInt8,3}) -> Matrix{RGBA{N0f8}}

Convert a [C=4, W, H] UInt8 array (from `map_cpu`) to a `Matrix{RGBA{N0f8}}`
of size (H, W).  Top-left origin, NO y-flip (verified: m1_orientation_test.jl).
"""
function cwh_to_matrix(pixels::Array{UInt8,3})::Matrix{RGBA{N0f8}}
    C, W, H = size(pixels)
    @assert C == 4 "expected 4 channels, got $C"
    img = Matrix{RGBA{N0f8}}(undef, H, W)
    @inbounds for h in 1:H, w in 1:W
        img[h, w] = RGBA{N0f8}(
            reinterpret(N0f8, pixels[1, w, h]),
            reinterpret(N0f8, pixels[2, w, h]),
            reinterpret(N0f8, pixels[3, w, h]),
            reinterpret(N0f8, pixels[4, w, h]),
        )
    end
    return img
end
