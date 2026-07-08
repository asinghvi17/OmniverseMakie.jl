using Test
# CPU present!(::Val{:cpu}) is ONE fused pass with zero steady-state display
# garbage.  Proves (subprocess, offscreen GL + ovrtx):
#   1. PIXEL EQUALITY — the fused `_tonemap_orient!` loop reproduces the
#      reference reverse(permutedims(tonemap_frame(hdr, ev)), dims=2) chain
#      byte-for-byte.
#   2. IN PLACE — the image! data array IS session.present_buf (written in
#      place + notify), so a tick allocates no new display buffer.
#   3. ALLOCATION — a warmed-up present! tonemaps STRAIGHT from the
#      still-mapped Float16 HdrColor (no Float32 HDR transient, no display
#      buffers), so it allocates only small per-step + map-handle
#      bookkeeping; measured against a reproduction of the unfused 4-buffer
#      body, which adds the HDR transient + ≥ 2 full display frames per tick.
const _C2_PRESENT_PROG = """
using OmniverseMakie, GLMakie, ColorTypes, FixedPointNumbers
import OmniverseMakie as OM
using OmniverseMakie: OV, tonemap_frame

OM.activate!(warmup = 24)
fig = Figure(); ax = LScene(fig[1,1]; show_axis = false)
mesh!(ax, Rect3f(Point3f(-1), Vec3f(2)); color = :orange)
session = OM.interactive_display(fig; size = (400, 300), gpu_direct = false)

Ext = Base.get_extension(OmniverseMakie, :OmniverseMakieGLMakieExt)
println("EXT_LOADED=", Ext !== nothing)
println("BLITTER=", session.blitter)

# Manual present! is the sole ovrtx driver: detach the auto render-tick AND
# stop the render loop so no render-task allocations pollute @allocated.
off(session.tick_listener)
GLMakie.stop_renderloop!(session.glscreen)

ev   = session.exposure
W, H = session.screen.fb_size
frame_bytes = W * H * sizeof(RGBA{N0f8})
hdr_bytes   = 4 * W * H * sizeof(Float32)
println("FRAME_BYTES=", frame_bytes)
println("HDR_BYTES=", hdr_bytes)

# 1. PIXEL EQUALITY: the fused loop vs the reference
#    reverse(permutedims(tonemap_frame)) chain.
hdr0 = OV.render_hdr_to_array(session.screen.renderer, session.screen.product; warmup = 4)
buf0 = Matrix{RGBA{N0f8}}(undef, W, H)
Ext._tonemap_orient!(buf0, hdr0, ev)
old0 = reverse(permutedims(tonemap_frame(hdr0, ev)), dims = 2)
mism = count(k -> buf0[k] != old0[k], eachindex(buf0))
println("PIXEL_MISMATCH=", mism)
@assert mism == 0 "fused tonemap+orient differs from the old reverse(permutedims(tonemap_frame)) chain"

# 2. IN PLACE: after a present!, the image! data array IS the cached session
#    buffer.
# compile + steady
OM.present!(session, Val(:cpu)); OM.present!(session, Val(:cpu))
println("INPLACE_OK=", session.image_plot[3][] === session.present_buf)

# The RTX frame actually reached the cached buffer (non-black).
pb = session.present_buf
nb = count(c -> (Float32(red(c)) + Float32(green(c)) + Float32(blue(c))) > 0.1, pb)
println("PRESENT_NONBLACK=", nb)

# 3. ALLOCATION: the fused present! (measured with the invariant intact,
#    BEFORE any old_present! swaps the array) vs the unfused 4-buffer body.
OM.present!(session, Val(:cpu))                      # extra warm
a_new = @allocated OM.present!(session, Val(:cpu))
println("A_NEW=", a_new)

# The unfused 4-buffer body: HDR copy + tonemap_frame out + permutedims +
# reverse + set-Observable.
function old_present!(s)
    hdr = OV.render_hdr_to_array(s.screen.renderer, s.screen.product; warmup = s.steps_per_tick)
    fr  = tonemap_frame(hdr, s.exposure)
    s.image_plot[3][] = reverse(permutedims(fr), dims = 2)
    return nothing
end
old_present!(session)                                # compile
a_old = @allocated old_present!(session)
println("A_OLD=", a_old)

println("OK_C2_PRESENT")
"""

include(joinpath(@__DIR__, "..", "helpers.jl"))
@testset "CPU present! — one fused pass, zero steady-state garbage (subprocess)" begin
    # Retry past ovrtx's intermittent pre-render startup crash, up to the
    # allocation-measurement marker.
    _, out = run_ovrtx_subprocess(_C2_PRESENT_PROG; timeout = 600, retries = 4,
                                  ready_marker = "BLITTER=")
    contains(out, "OK_C2_PRESENT") || @info "C2 present output" out
    @test contains(out, "EXT_LOADED=true")
    @test contains(out, "OK_C2_PRESENT")  # completed (no mid-run death)

    # 1. Fused loop == old chain, pixel for pixel.
    @test contains(out, "PIXEL_MISMATCH=0")

    # 2. The image! data array IS the cached buffer (written in place + notify).
    @test contains(out, "INPLACE_OK=true")

    m_nb = match(r"PRESENT_NONBLACK=(\d+)", out)
    # RTX frame reached the buffer
    @test m_nb !== nothing && parse(Int, m_nb.captures[1]) > 500

    # 3. Allocation gate.  Parse the measured bytes + the frame/HDR sizes.
    m_new = match(r"A_NEW=(\d+)", out)
    m_old = match(r"A_OLD=(\d+)", out)
    m_fb  = match(r"FRAME_BYTES=(\d+)", out)
    m_hb  = match(r"HDR_BYTES=(\d+)", out)
    @test m_new !== nothing && m_old !== nothing && m_fb !== nothing && m_hb !== nothing
    if m_new !== nothing && m_old !== nothing && m_fb !== nothing && m_hb !== nothing
        a_new = parse(Int, m_new.captures[1])
        a_old = parse(Int, m_old.captures[1])
        frame_bytes = parse(Int, m_fb.captures[1])
        hdr_bytes   = parse(Int, m_hb.captures[1])
        # surface before/after numbers
        @info "C2 alloc (INT-2)" a_new a_old frame_bytes hdr_bytes
        # The zero-copy CPU present tonemaps STRAIGHT from the still-mapped
        # Float16 HdrColor view, so the Float32 [C,W,H] HDR transient
        # (hdr_bytes = 4·frame_bytes) never appears in a steady tick.  What
        # remains is small, RESOLUTION-INDEPENDENT per-step + map-handle
        # bookkeeping (~1 KB), so the ceiling is an ABSOLUTE 32 KiB — far
        # above bookkeeping drift yet far below hdr_bytes, which a present
        # with the HDR transient would allocate every tick.
        @test a_new < 32 * 1024
        # Same-session, same-step comparison (ovrtx-step overhead cancels):
        # the unfused 4-buffer body allocates the HDR transient + ≥ 2 extra
        # full display frames per tick that the zero-copy present does not.
        @test a_old - a_new > 2 * frame_bytes
        @test a_old > a_new
    end
end
