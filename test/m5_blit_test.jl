using Test
const _M5_BLIT_PROG = """
using OmniverseMakie, GLMakie, ColorTypes, FixedPointNumbers
OM = OmniverseMakie
# A 2-colour host frame: top half red, bottom half blue (row 1 = top, our convention).
H, W = 80, 120
frame = Matrix{RGBA{N0f8}}(undef, H, W)
frame[1:H÷2, :]   .= RGBA{N0f8}(1,0,0,1)
frame[H÷2+1:H, :] .= RGBA{N0f8}(0,0,1,1)

# Build the campixel! display path exactly as interactive_display does.
# This tests cpu_blit! against the ACTUAL display model (M5 campixel! scene),
# not just a bare Axis.
GLMakie.activate!()
scene = Makie.Scene(size = (W, H))
Makie.campixel!(scene)
img = image!(scene, 0 .. W, 0 .. H, reverse(permutedims(frame), dims = 2); interpolate = false)
glscr = GLMakie.Screen(visible = false)
display(glscr, scene)

# Second frame: swap the colours (top blue, bottom red) to prove blit actually updates.
frame2 = Matrix{RGBA{N0f8}}(undef, H, W)
frame2[1:H÷2, :]   .= RGBA{N0f8}(0,0,1,1)
frame2[H÷2+1:H, :] .= RGBA{N0f8}(1,0,0,1)
OM.cpu_blit!(img, frame2)         # update the texture from the second host frame

buf = GLMakie.colorbuffer(glscr)  # read the rendered window back
println("BUF_SIZE=", size(buf))

# After blit with frame2: top half should be BLUE, bottom half RED.
# colorbuffer returns [rows, cols] with row 1 = top of screen (JuliaNative format).
topc = buf[round(Int, 0.25 * size(buf, 1)), round(Int, 0.5 * size(buf, 2))]
botc = buf[round(Int, 0.75 * size(buf, 1)), round(Int, 0.5 * size(buf, 2))]
println("TOP=", (Float32(red(topc)), Float32(green(topc)), Float32(blue(topc))))
println("BOT=", (Float32(red(botc)), Float32(green(botc)), Float32(blue(botc))))
@assert blue(topc) > red(topc)  "top not blue after blit — blit flipped/failed (frame2 top=blue)"
@assert red(botc)  > blue(botc) "bottom not red after blit — blit flipped/failed (frame2 bot=red)"
println("OK_BLIT")
"""
include("helpers.jl")
@testset "M5 cpu_blit! campixel! orientation (subprocess, offscreen GL)" begin
    exitcode, output = run_ovrtx_subprocess(_M5_BLIT_PROG; timeout = 300)
    @info "M5 blit output" output
    @test exitcode == 0
    @test contains(output, "OK_BLIT")
end
