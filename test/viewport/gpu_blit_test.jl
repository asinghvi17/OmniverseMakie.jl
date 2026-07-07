using Test

# GPU-direct blit: interactive_display(...; gpu_direct=true) shows a
# non-black RTX frame through the on-device CUDA→GL path (no CPU roundtrip).
#
# CUDA↔GL interop for a resource must run on ONE task/context: the GLMakie
# render task, where the auto `render_tick` listener fires `present!` with
# the GL context current.  So the frame is driven via `colorbuffer` (which
# runs the render loop / fires the tick on the render task) — NOT a manual
# `on_render_tick!` from the main task (that would register/map in a
# different CUDA context → segfault).
#
# To guarantee the GPU path GENUINELY ran (no silent CPU fallback), assert
# (a) the frame is non-black AND (b) the session is still on the :gpu blitter
# with a registered gpu_state — a CPU fallback flips session.blitter to :cpu.
const _M6_GPUBLIT_PROG = """
using OmniverseMakie, GLMakie, CUDA, ColorTypes
OM = OmniverseMakie
println("CUDA_FUNCTIONAL=", CUDA.functional())
OM.activate!(warmup = 24)
fig = Figure(); ax = LScene(fig[1,1]; show_axis=false)
mesh!(ax, Rect3f(Point3f(-1), Vec3f(2)); color = :orange)
# force GPU-direct
session = OM.interactive_display(fig; size=(320,240), gpu_direct=true)
println("BLITTER_INIT=", session.blitter)
@assert session.blitter == :gpu "gpu_direct=true did not select the :gpu blitter"

# Drive the GPU present via the render loop (render task, GL current); two
# frames so the lazy register + a steady GPU blit have both run.
_ = GLMakie.colorbuffer(session.glscreen)
buf = GLMakie.colorbuffer(session.glscreen)
nb = count(c -> (Float32(red(c))+Float32(green(c))+Float32(blue(c)))>0.1, buf)
println("GPU_NONBLACK=", nb)
println("GPU_BLITTER=", session.blitter)
println("GPU_STATE_SET=", session.gpu_state !== nothing)
@assert nb > 1000 "GPU-direct viewport black"
@assert session.blitter == :gpu "GPU-direct present! fell back to CPU (genuine GPU path failed)"
@assert session.gpu_state !== nothing "GPU-direct present! did not register the GL texture"
OM.close(session)
println("OK_GPU_BLIT")
"""

include(joinpath(@__DIR__, "..", "helpers.jl"))
@testset "M6 GPU-direct blit (subprocess, CUDA+GL)" begin
    # EARLY marker (per helpers.jl): BLITTER_INIT= is the first line printed
    # after interactive_display's successful build (the startup-crash window),
    # BEFORE any in-child @assert — so retries absorb only startup deaths, never
    # a genuine black-frame / CPU-fallback assert failure.
    exitcode, output = run_ovrtx_subprocess(_M6_GPUBLIT_PROG; timeout = 600, retries = 2, ready_marker = "BLITTER_INIT=")
    @info "M6 gpu blit output" output
    @test exitcode == 0
    @test contains(output, "CUDA_FUNCTIONAL=true")
    @test contains(output, "OK_GPU_BLIT")
    @test contains(output, "GPU_BLITTER=gpu")  # genuine GPU path (no fallback)
    @test contains(output, "GPU_STATE_SET=true")
    m = match(r"GPU_NONBLACK=(\d+)", output)
    @test m !== nothing && parse(Int, m.captures[1]) > 1000
end
