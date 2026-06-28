# dlpack.jl — reshape a [C,W,H] UInt8 pixel buffer into Matrix{RGBA{N0f8}}.
#
# The pixels array is [C=4, W, H] (channel-fastest, as returned by map_cpu /
# the DLPack LdrColor tensor with shape [H,W,4] read back in (C,W,H) order).
#
# ORIENTATION (verified by test/m1_orientation_test.jl):
# ovrtx LdrColor is TOP-LEFT origin — row 1 = top of the rendered scene.
# NO vertical flip is applied here or in colorbuffer.  The M1.5 asymmetric
# fixture (red box at world +Z, blue box at world -Z) measured red_row=103,
# blue_row=306 (red ABOVE blue), confirming right-side-up readback.  The
# earlier "y-flip deferred to M3" note is now superseded.

using ColorTypes, FixedPointNumbers

"""
    cwh_to_matrix(pixels::Array{UInt8,3}) -> Matrix{RGBA{N0f8}}

Convert a [C=4, W, H] UInt8 array (as returned by `map_cpu`) to a
`Matrix{RGBA{N0f8}}` of size (H, W).

Orientation: ovrtx LdrColor is top-left origin (right-side-up); NO y-flip is
applied.  This is verified empirically by `test/m1_orientation_test.jl`: a
red box at world +Z renders at centroid row ≈103 and a blue box at world −Z at
row ≈306 (red above blue), confirming the buffer is not inverted.
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
