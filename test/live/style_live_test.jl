using Test
include(joinpath(@__DIR__, "..", "helpers.jl"))

# Live style attributes that are baked into USD arrays at author time must still
# route through the diff node. This guards the common "plot exists, but changing
# marker size / rotation / line width does nothing until recreating the screen"
# first-user failure mode.
const _STYLE_LIVE_PROG = raw"""
using OmniverseMakie, GeometryBasics, ColorTypes
import OmniverseMakie as OM

OM.activate!(warmup = 24)

scene = Scene(size = (260, 260))
cam3d!(scene)
update_cam!(scene, Vec3f(0, 0, 16), Vec3f(0, 0, 0), Vec3f(0, 1, 0))

sp = scatter!(scene, [Point3f(-4, -1, 0), Point3f(-4, 1, 0)]; markersize = 0.7, color = :red)
mp = meshscatter!(scene, [Point3f(0, -1, 0), Point3f(0, 1, 0)]; markersize = 0.7, color = :orange)
ln = lines!(scene, [Point3f(3, -2, 0), Point3f(3, 2, 0)]; linewidth = 2, color = :cyan)

screen = OM.Screen(scene)
imgA = Makie.colorbuffer(screen)
println("STYLE_READY")

counter = Dict{Symbol,Int}()
OM._PUSH_OBSERVER[] = name -> (counter[name] = get(counter, name, 0) + 1)

sp.markersize = 1.2
mp.markersize = 1.1
mp.rotation = Vec4f(0, 0, 0.70710677, 0.70710677)
ln.linewidth = 7
Makie.colorbuffer(screen)

println("COUNTER_STYLE=", counter)
@assert get(counter, :markersize, 0) == 2 "expected two :markersize pushes, got $(counter)"
@assert get(counter, :rotation, 0) == 1 "expected one :rotation push, got $(counter)"
@assert get(counter, :linewidth, 0) == 1 "expected one :linewidth push, got $(counter)"

# Count-changing instancer positions must resize the coupled per-instance arrays
# as well, then later style writes must use the current instance count.
empty!(counter)
sp[1][] = [Point3f(-4, -1.5, 0), Point3f(-4, 0, 0), Point3f(-4, 1.5, 0)]
Makie.colorbuffer(screen)
@assert :positions_transformed_f32c in keys(counter) "scatter resize did not push positions"

empty!(counter)
sp.markersize = 0.9
Makie.colorbuffer(screen)
println("COUNTER_RESIZED_STYLE=", counter)
@assert counter == Dict(:markersize => 1) "resized scatter markersize push fired $(counter)"

OM._PUSH_OBSERVER[] = nothing
close(screen)
println("OK_STYLE_LIVE")
"""

@testset "live style attributes route through diff node (subprocess)" begin
    ec, out = run_ovrtx_subprocess(_STYLE_LIVE_PROG; timeout = 900, retries = 3,
                                   ready_marker = "STYLE_READY")
    contains(out, "OK_STYLE_LIVE") || @info "style live subprocess output" out
    @test ec == 0
    @test contains(out, "OK_STYLE_LIVE")
    @test occursin(":markersize => 2", out)
    @test occursin(":rotation => 1", out)
    @test occursin(":linewidth => 1", out)
    @test contains(out, "COUNTER_RESIZED_STYLE=Dict(:markersize => 1)")
end
