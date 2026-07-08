using Test

# PURE unit test of run_ovrtx_subprocess's ready_marker retry loop
# (helpers.jl): a flag-file child emits the marker only on its SECOND run, so
# the harness must re-run the prog.  Models ovrtx's intermittent pre-render
# startup crash with cheap `julia` children — no ovrtx/GPU needed.

# A child that prints the marker "READY" only once `flag` exists: the FIRST
# run creates the flag and prints "NOT_YET" (marker absent → a retry); the
# SECOND run sees the flag and prints "READY" (marker present → stop).  Runs
# are sequential, so the flag is reliable state.
_e2_flagprog(flag) = """
if isfile($(repr(flag)))
    println("READY")
else
    touch($(repr(flag)))
    println("NOT_YET")
end
"""

@testset "ready_marker retry" begin
    @testset "retries until the marker appears; returns the last attempt" begin
        flag = tempname()
        try
            _, out = run_ovrtx_subprocess(_e2_flagprog(flag); retries = 2, ready_marker = "READY")
            @test contains(out, "READY")        # 2nd attempt emitted the marker
            @test !contains(out, "NOT_YET")     # only the LAST attempt's output
        finally
            isfile(flag) && rm(flag)
        end
    end

    @testset "retries=1 does not retry (single attempt)" begin
        flag = tempname()
        try
            _, out = run_ovrtx_subprocess(_e2_flagprog(flag); retries = 1, ready_marker = "READY")
            @test contains(out, "NOT_YET")      # one attempt; marker not chased
            @test !contains(out, "READY")
        finally
            isfile(flag) && rm(flag)
        end
    end

    @testset "empty ready_marker short-circuits (default = one attempt)" begin
        # The default empty marker means "nothing to wait for": exactly one
        # run even with retries>1.
        flag = tempname()
        try
            _, out = run_ovrtx_subprocess(_e2_flagprog(flag); retries = 5)
            @test contains(out, "NOT_YET")
        finally
            isfile(flag) && rm(flag)
        end
    end
end
