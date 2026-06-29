using Test
const _M5_BLIT_PROG = """
using OmniverseMakie, GLMakie, ColorTypes, FixedPointNumbers
OM = OmniverseMakie
# A 2-colour host frame: top half red, bottom half blue (row 1 = top, our convention).
H, W = 80, 120
frame = Matrix{RGBA{N0f8}}(undef, H, W)
frame[1:H÷2, :]   .= RGBA{N0f8}(1,0,0,1)
frame[H÷2+1:H, :] .= RGBA{N0f8}(0,0,1,1)
GLMakie.activate!()
fig = Figure(); ax = Makie.Axis(fig[1,1]); ax.aspect = Makie.DataAspect()
img = image!(ax, frame)
glscr = GLMakie.Screen(visible = false); display(glscr, fig)
OM.cpu_blit!(img, frame)         # update the texture from the host frame
buf = GLMakie.colorbuffer(glscr) # read the rendered window back
println("BUF_SIZE=", size(buf))
# top region should read red-dominant, bottom blue-dominant (orientation preserved, no flip)
topc = buf[round(Int,0.25*size(buf,1)), round(Int,0.5*size(buf,2))]
botc = buf[round(Int,0.75*size(buf,1)), round(Int,0.5*size(buf,2))]
println("TOP=", (Float32(red(topc)),Float32(green(topc)),Float32(blue(topc))))
println("BOT=", (Float32(red(botc)),Float32(green(botc)),Float32(blue(botc))))
@assert red(topc) > blue(topc) "top not red — blit flipped/failed"
@assert blue(botc) > red(botc) "bottom not blue — blit flipped/failed"
println("OK_BLIT")
"""
include("helpers.jl")
@testset "M5 cpu_blit! (subprocess, offscreen GL)" begin
    exitcode, output = run_ovrtx_subprocess(_M5_BLIT_PROG; timeout = 300)
    @info "M5 blit output" output
    @test exitcode == 0
    @test contains(output, "OK_BLIT")
end
