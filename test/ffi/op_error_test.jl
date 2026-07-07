using Test

# ---------------------------------------------------------------------------
# Op-error propagation + alive-check ordering in OV.enqueue_wait.
# open_usd! on a missing file must throw OVRTXError: ovrtx reports the
# failure ONLY via ovrtx_op_wait_result_t.error_op_ids — both the enqueue
# and the wait return OVRTX_API_SUCCESS. A closed Renderer must error
# cleanly on step!/reset!: the alive check runs before the enqueue thunk,
# so no ccall passes C_NULL into ovrtx. Subprocess; flush after each block
# so a later crash cannot swallow earlier evidence.
# ---------------------------------------------------------------------------

const _A1_PROG = """
using OmniverseMakie
const OV = OmniverseMakie.OV
const OVRTXError = OmniverseMakie.LibOVRTX.OVRTXError

r = OV.Renderer()

# 1. open_usd! on a missing file -> OVRTXError (propagated from error_op_ids).
open_threw = false; open_is_ovrtx = false; open_msg = ""
try
    OV.open_usd!(r, "/nonexistent_a1_test.usda")
catch e
    global open_threw = true
    global open_is_ovrtx = e isa OVRTXError
    global open_msg = sprint(showerror, e)
end
println("OPEN_MISSING_THREW=", open_threw)
println("OPEN_MISSING_IS_OVRTX=", open_is_ovrtx)
println("OPEN_MISSING_MSG=", open_msg)
flush(stdout)

# 2. closed Renderer: step!/reset! must error cleanly BEFORE any ccall
#    (no crash).
close(r)

step_threw = false
try
    OV.step!(r, "/Render/Product")
catch e
    global step_threw = true
end
println("STEP_CLOSED_THREW=", step_threw); flush(stdout)

reset_threw = false
try
    OV.reset!(r)
catch e
    global reset_threw = true
end
println("RESET_CLOSED_THREW=", reset_threw); flush(stdout)

println("OK_A1"); flush(stdout)
"""

include(joinpath(@__DIR__, "..", "helpers.jl"))

@testset "A1 op-error propagation + closed-Renderer alive check (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_A1_PROG; timeout = 600, retries = 2, ready_marker = "OK_A1")
    contains(output, "OK_A1") || @info "A1 subprocess output" output
    # Clean teardown (no segfault from a C_NULL ccall on the closed Renderer).
    @test exitcode == 0
    @test contains(output, "OK_A1")
    # open_usd! on a missing file throws OVRTXError, carrying the resolved
    # op-error string.
    @test contains(output, "OPEN_MISSING_THREW=true")
    @test contains(output, "OPEN_MISSING_IS_OVRTX=true")
    @test occursin("Failed to open USD file", output)
    # closed Renderer: step!/reset! error cleanly via the alive check
    # (thunk-deferred ccall).
    @test contains(output, "STEP_CLOSED_THREW=true")
    @test contains(output, "RESET_CLOSED_THREW=true")
end
