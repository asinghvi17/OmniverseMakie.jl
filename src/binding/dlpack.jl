# dlpack.jl — reshape a [C,W,H] UInt8 pixel buffer into Matrix{RGBA{N0f8}}.
#
# The pixels array is [C=4, W, H] (channel-fastest, as returned by map_cpu /
# the DLPack LdrColor tensor with shape [H,W,4] read back in (C,W,H) order).
# M0 only asserts non-black — image orientation (y-flip) is deferred to M3.

using ColorTypes, FixedPointNumbers

"""
    cwh_to_matrix(pixels::Array{UInt8,3}) -> Matrix{RGBA{N0f8}}

Convert a [C=4, W, H] UInt8 array (as returned by `map_cpu`) to a
`Matrix{RGBA{N0f8}}` of size (H, W).  Orientation (y-flip) is NOT applied here
— that is deferred to M3.  M0 only needs to assert the image is non-black.
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
