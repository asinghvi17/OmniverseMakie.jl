using Test

const _SIGNALS_JL = joinpath(@__DIR__, "..", "src", "binding", "signals.jl")

# Subprocess program: load LibOVRTX, include signals.jl, snapshot → create → restore →
# destroy a renderer, print "OK".  Runs as a script FILE (via the shared harness, not
# `julia -e`) — `-e` mode crashes inside ovrtx_create_renderer (different module-init /
# linker-namespace context); a file runs clean.  Empty config (C_NULL entries, length 0).
# "OK" guards against a silent early-exit-0.
const _SIGNALS_PROG = """
using LibOVRTX
include($(repr(_SIGNALS_JL)))
save = SignalGuard.snapshot()
cfg  = Ref(LibOVRTX.ovrtx_config_t(Ptr{LibOVRTX.ovrtx_config_entry_t}(C_NULL), Csize_t(0)))
rref = Ref{Ptr{LibOVRTX.ovrtx_renderer_t}}(C_NULL)
LibOVRTX.check(LibOVRTX.ovrtx_create_renderer(cfg, rref), "create")
SignalGuard.restore(save)
LibOVRTX.ovrtx_destroy_renderer(rref[])
println("OK")
"""

@testset "M0.4 renderer process exits cleanly" begin
    exitcode, output = run_ovrtx_subprocess(_SIGNALS_PROG)
    @test exitcode == 0          # without restore() the child dies on SIGSEGV (→ -11 via the watchdog)
    @test contains(output, "OK") # guard against a silent early-exit-0
end
