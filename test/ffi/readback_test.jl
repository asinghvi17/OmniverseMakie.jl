using Test
using OmniverseMakie: OV
using ColorTypes, FixedPointNumbers

# ---------------------------------------------------------------------------
# Task A2 — leak-proof readback + with_mapped_hdr.
#
# 1. Pure: the single-pass OV.cwh_to_matrix must equal the OLD two-pass loop
#    (inlined below as an independent oracle) on synthetic [C,W,H] bytes —
#    proving the reinterpret/permute collapse preserves the top-left-origin,
#    no-y-flip, R,G,B,A channel order (dlpack.jl header, offscreen/orientation_test).
# 2. Subprocess: with_mapped_hdr passes f's result through AND unmaps in a
#    `finally` when f throws — asserted by a *subsequent* map succeeding on the
#    same render var (a leaked mapping would block it).  CPU map, no CUDA/GL.
# ---------------------------------------------------------------------------

# The pre-A2 dlpack.jl algorithm, verbatim, as the equivalence oracle: the
# single-pass path is checked against independent code, never against itself.
function _a2_old_cwh(pixels::Array{UInt8,3})
    C, W, H = size(pixels)
    @assert C == 4
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

@testset "A2 single-pass cwh_to_matrix == old two-pass (pure)" begin
    # Asymmetric, non-square [C=4, W, H], every channel distinct: a transpose,
    # y-flip, W/H swap, or channel swap would diverge from the oracle.
    C, W, H = 4, 7, 5
    pixels = Array{UInt8,3}(undef, C, W, H)
    for h in 1:H, w in 1:W, c in 1:C
        pixels[c, w, h] = UInt8((37c + 7w + 101h) % 256)
    end

    got  = OV.cwh_to_matrix(pixels)
    want = _a2_old_cwh(pixels)

    @test got isa Matrix{RGBA{N0f8}}
    @test size(got) == (H, W)            # top-left origin, no flip: rows = H
    @test got == want                    # byte-identical to the two-pass oracle
    # one texel, spelled out: channel c → RGBA component in R,G,B,A order (no swap)
    @test got[2, 3] == RGBA{N0f8}(
        reinterpret(N0f8, pixels[1, 3, 2]), reinterpret(N0f8, pixels[2, 3, 2]),
        reinterpret(N0f8, pixels[3, 3, 2]), reinterpret(N0f8, pixels[4, 3, 2]))
end

# HDR path: render a tiny scene, map HdrColor on CPU (no CUDA/GL) and exercise
# with_mapped_hdr's result-passthrough + unmap-in-finally, plus the map_cpu_f32 wrapper.
const _A2_HDR_PROG = """
using OmniverseMakie
using OmniverseMakie: OV
import OmniverseMakie as OM
OM.activate!(warmup = 8)
scene = Scene(size = (96, 96)); cam3d!(scene)
mesh!(scene, Rect3f(Point3f(0), Vec3f(1)); color = :red)
screen = OM.Screen(scene)
OM.author_root_from_scene!(screen, scene; resolution = screen.fb_size)
for _ in 1:8   # warm frames so HdrColor is resident
    sr0 = OV.step!(screen.renderer, screen.product; timeout_ns = UInt64(60_000_000_000)); close(sr0)
end
sr = OV.step!(screen.renderer, screen.product; timeout_ns = UInt64(60_000_000_000))

# 1. passthrough: with_mapped_hdr returns f's value verbatim; f sees a [C=4,W,H] Float16 view.
res = OV.with_mapped_hdr(sr, "HdrColor") do raw16, W, H
    (size(raw16), W, H, eltype(raw16))
end
println("PASSTHROUGH_SHAPE=", res[1], " W=", res[2], " H=", res[3], " ET=", res[4])
println("PASSTHROUGH_OK=", res[1] == (4, res[2], res[3]) && res[4] === Float16)

# 2. unmap-on-throw: a throwing f must propagate the error...
threw = false; expected = false
try
    OV.with_mapped_hdr(sr, "HdrColor") do raw16, W, H
        error("boom_a2_leakcheck")
    end
catch e
    global threw = true
    global expected = occursin("boom_a2_leakcheck", sprint(showerror, e))
end
println("THREW=", threw)
println("THREW_EXPECTED=", expected)

# 3. ...and must have unmapped in `finally`: a subsequent map on the SAME var succeeds
#    (a leaked mapping would block/error this).
val = OV.with_mapped_hdr(sr, "HdrColor") do raw16, W, H
    Float32(raw16[1, 1, 1])
end
println("AFTER_MAP_OK=", isfinite(val))

# 4. map_cpu_f32 (thin wrapper over with_mapped_hdr) still yields a Float32 [C,W,H].
px, W2, H2 = OV.map_cpu_f32(sr, "HdrColor")
println("F32_WRAPPER_OK=", px isa Array{Float32,3} && size(px) == (4, W2, H2))

close(sr); close(screen)
println("OK_A2_HDR")
"""

include(joinpath(@__DIR__, "..", "helpers.jl"))

@testset "A2 with_mapped_hdr passthrough + unmap-on-throw (subprocess)" begin
    # Retry the known intermittent ovrtx GeometryGroup::attachToContext startup crash:
    # re-run until the child reaches its first map.
    _, out = run_ovrtx_subprocess(_A2_HDR_PROG; timeout = 400, retries = 4,
                                  ready_marker = "PASSTHROUGH_OK=")
    contains(out, "OK_A2_HDR") || @info "A2 with_mapped_hdr output" out
    @test contains(out, "OK_A2_HDR")           # subprocess completed all work
    @test contains(out, "PASSTHROUGH_OK=true") # f's result passed through; view is [4,W,H] Float16
    @test contains(out, "THREW=true")          # throwing f propagates
    @test contains(out, "THREW_EXPECTED=true")
    @test contains(out, "AFTER_MAP_OK=true")   # mapping released in finally → re-map works
    @test contains(out, "F32_WRAPPER_OK=true") # map_cpu_f32 wrapper intact
end
