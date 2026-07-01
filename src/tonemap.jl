# Shared HDR → display tonemap.  Pure Float32 scalar math so BOTH the CPU host path
# (Task 2) and the CUDA kernel (Task 4) call the identical functions.
@inline _aces(x::Float32) = clamp((x*(2.51f0x+0.03f0)) / (x*(2.43f0x+0.59f0)+0.14f0), 0f0, 1f0)
@inline _srgb(c::Float32) = c <= 0.0031308f0 ? 12.92f0c : 1.055f0*c^(1f0/2.4f0) - 0.055f0
# Non-throwing N0f8 quantization: clamp guarantees [0,1], so build the raw byte directly (round-to-
# nearest-even, == N0f8(clamp(c,0,1)), byte-identical) + `reinterpret` instead of the checked
# `N0f8(...)` constructor — whose bounds-throw is dead code here and whose error path (Ryu/show) is
# NOT GPU-compilable, so this lets the SAME `tonemap` run unchanged in the M6 CUDA kernel.
@inline _u8(c::Float32)   = reinterpret(N0f8, round(UInt8, clamp(c, 0f0, 1f0) * 255f0))

"""
    tonemap(rgb::NTuple{3,Float32}, exposure::Float32) -> RGBA{N0f8}

`sRGB( ACES( 2^exposure · rgb ) )`.  `exposure` is in stops (EV); 0 = no change.
"""
@inline function tonemap(rgb::NTuple{3,Float32}, exposure::Float32)
    scale = exp2(exposure)
    RGBA{N0f8}(_u8(_srgb(_aces(scale*rgb[1]))), _u8(_srgb(_aces(scale*rgb[2]))), _u8(_srgb(_aces(scale*rgb[3]))), N0f8(1))
end

# Broadcast a [C,W,H] float HDR buffer (channel-fastest, as map_cpu/map_cuda return)
# into an [H,W] RGBA{N0f8} display matrix, top-left origin (matches render_to_matrix).
function tonemap_frame(hdr::AbstractArray{Float32,3}, exposure::Float32)
    C, W, H = size(hdr)
    out = Matrix{RGBA{N0f8}}(undef, H, W)
    @inbounds for j in 1:W, i in 1:H
        out[i, j] = tonemap((hdr[1,j,i], hdr[2,j,i], hdr[3,j,i]), exposure)
    end
    return out
end
