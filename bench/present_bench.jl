# Manual GPU-vs-CPU present! benchmark — NOT part of the default test suite
# (timing-relative gate, machine-dependent).  Run directly:
#   julia --project bench/present_bench.jl
# (needs the test env on JULIA_LOAD_PATH for GLMakie/CUDA — see
# test/helpers.jl).
using Test

# GPU-direct vs CPU blit benchmark + gate.
#
# Times `present!` (the per-frame blit) GPU-direct vs CPU at 800×600 and 4K
# (3840×2160), and gates GPU-direct strictly < CPU at 4K.  `present!` runs
# the same shared ovrtx RT2 step in both paths (it cancels in the
# comparison); the measured difference is the blit — at 4K the CPU path's
# device↔host HdrColor roundtrip + 8.3M-pixel host tonemap loop is far
# slower than the on-device tonemap kernel + CUDA→GL copy.
#
# THREADING (load-bearing): the bench times `present!` in a tight loop on
# the test's main task, but the live viewport also auto-ticks `present!` on
# GLMakie's render task.  Two tasks driving present! (CUDA-GL interop /
# ovrtx step) concurrently = a race → segfault.  So after
# `interactive_display`, before the warm/timing loop, the manual present!
# becomes the sole driver on one task/context:
#   - `off(s.tick_listener)`              — detach the auto render-tick present!
#   - `s.glscreen.close_after_renderloop = false` — keep the window OPEN
#     across the stop (the default `stop_renderloop!` CLOSES the window —
#     GLMakie's `render_asap` pattern, screen.jl:1180 — which would clear
#     the robj cache and break `present!`)
#   - `GLMakie.stop_renderloop!(s.glscreen)` — halt + JOIN the render task
#   - `s.gpu_state === nothing || OM.gpu_unregister!(s)` — drop any
#     registration the render task may have made before we stopped it (a
#     race), so the first manual present! re-registers on THIS (main) task;
#     register + map + copy are then all consistent on one task (present!
#     does `gl_switch_context!` internally → GL current on the calling task).
const _M6_BENCH_PROG = """
using OmniverseMakie, GLMakie, CUDA
OM = OmniverseMakie
OM.activate!(warmup = 16)
function blit_latency(sz, mode)
    fig = Figure(); ax = LScene(fig[1,1]; show_axis=false)
    mesh!(ax, Rect3f(Point3f(-1), Vec3f(2)); color = :orange)
    s = OM.interactive_display(fig; size = sz, gpu_direct = mode)
    # --- single task/context: manual present! is the SOLE driver (header) ---
    off(s.tick_listener)
    s.glscreen.close_after_renderloop = false
    GLMakie.stop_renderloop!(s.glscreen)
    yield()
    s.gpu_state === nothing || OM.gpu_unregister!(s)
    # warm: JIT kernel / first map+register
    for _ in 1:3; OM.present!(s, Val(s.blitter)); end
    @assert s.blitter == (mode === true ? :gpu : :cpu) "present! fell back: blitter=\$(s.blitter)"
    t = @elapsed (for _ in 1:20; OM.present!(s, Val(s.blitter)); end)
    OM.close(s); return t / 20
end
for sz in ((800,600),(3840,2160))
    cpu = blit_latency(sz, false); gpu = blit_latency(sz, true)
    println("BENCH sz=", sz, " cpu_ms=", round(cpu*1e3,digits=3), " gpu_ms=", round(gpu*1e3,digits=3))
    flush(stdout)
end
println("OK_BENCH")
"""

include(joinpath(@__DIR__, "..", "test", "helpers.jl"))
@testset "GPU-direct vs CPU blit benchmark (subprocess, CUDA+GL)" begin
    exitcode, output = run_ovrtx_subprocess(_M6_BENCH_PROG; timeout=900)
    @info "M6 bench output" output
    @test exitcode == 0
    @test contains(output, "OK_BENCH")
    m4 = match(r"BENCH sz=\(3840, 2160\) cpu_ms=([0-9.]+) gpu_ms=([0-9.]+)", output)
    @test m4 !== nothing
    if m4 !== nothing
        cpu4k = parse(Float64, m4.captures[1]); gpu4k = parse(Float64, m4.captures[2])
        @test gpu4k < cpu4k   # GPU-direct strictly faster at 4K (the gate)
        @info "4K blit latency" cpu_ms=cpu4k gpu_ms=gpu4k speedup=cpu4k/gpu4k
    end
end
