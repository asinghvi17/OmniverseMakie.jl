using Test

# OV.map_cuda: map a render var as LINEAR CUDA device memory (mode 2) and
# return RAW handles (no CUDA.jl in the main module).  HdrColor is
# kDLFloat/16, mapped LINEAR (not CUDA_ARRAY): the GPU present wraps `data`
# as a CuArray{Float16} and tonemaps on-device, which an opaque CUarray
# (mode 3) could not support.
#
# `using CUDA` in the subprocess loads libcuda + a functional GPU context so
# ovrtx's CUDA map succeeds; the helper sets JULIA_CUDA_USE_COMPAT=false so
# CUDA.jl and ovrtx share the system driver (else ovrtx createDevices fails,
# driver result 3).
const _M6_MAPCUDA_PROG = """
using OmniverseMakie, CUDA
const OV = OmniverseMakie.OV
OM = OmniverseMakie
println("CUDA_FUNCTIONAL=", CUDA.functional())
OM.activate!(warmup = 8)
scene = Scene(size=(96,96)); cam3d!(scene)
mesh!(scene, Rect3f(Point3f(0), Vec3f(1)); color=:red)
screen = OM.Screen(scene)
OM.author_root_from_scene!(screen, scene; resolution = screen.fb_size)
sr = OV.step!(screen.renderer, screen.product; timeout_ns = UInt64(60_000_000_000))
data, W, H, C, mh, wait_event = OV.map_cuda(sr, "HdrColor")
println("MAP_CUDA_OK=", data != C_NULL, " W=", W, " H=", H, " C=", C)
println("WAIT_EVENT_NONZERO=", wait_event != Csize_t(0))
OV.unmap_cuda(sr, mh)
close(sr); close(screen)
println("OK_MAP_CUDA")
"""

include(joinpath(@__DIR__, "..", "helpers.jl"))

@testset "M6 OV.map_cuda (subprocess, CUDA linear)" begin
    exitcode, output = run_ovrtx_subprocess(_M6_MAPCUDA_PROG; timeout = 600, retries = 2, ready_marker = "OK_MAP_CUDA")
    @info "M6 map_cuda output" output
    @test exitcode == 0
    @test contains(output, "CUDA_FUNCTIONAL=true")
    @test contains(output, "MAP_CUDA_OK=true")
    @test contains(output, "W=96 H=96 C=4")
    @test contains(output, "WAIT_EVENT_NONZERO=true")
    @test contains(output, "OK_MAP_CUDA")
end
