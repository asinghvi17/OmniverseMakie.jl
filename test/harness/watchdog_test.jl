using Test

# Review E1 — truthful watchdog.  PURE unit tests of the run_ovrtx_subprocess watchdog
# (helpers.jl): they spawn cheap single-process `julia` children (exit / sleep / crash) —
# NO ovrtx, NO GPU, no GPU lock needed — and assert the returned exit code is TRUTHFUL:
#   * 0    only on a real clean exit,
#   * -124 when the WATCHDOG times out and kills the child (negated GNU-timeout convention
#          — distinct from -9, which is reserved for an EXTERNAL SIGKILL such as the OOM
#          killer, so triage can tell "it hung" from "something killed it"),
#   * -N   when the child dies from uncaught signal N (a crash),
#   * else the child's own nonzero exit code.
# The pre-fix helper read `p.exitcode` without `wait(p)`, so a timed-out child returned
# `typemin(Int64)` (or 0 once reaped) and a crashed child returned 0 — both silently
# passing `@test exitcode == 0`.  Children `redirect_stderr(devnull)` where a signal would
# otherwise dump a runtime backtrace, so the helper's @info stays quiet.

@testset "review E1: truthful watchdog" begin
    @testset "clean exit (0) => 0" begin
        exitcode, _ = run_ovrtx_subprocess("exit(0)"; timeout = 30)
        @test exitcode == 0
    end

    @testset "nonzero exit (3) passes through" begin
        exitcode, _ = run_ovrtx_subprocess("exit(3)"; timeout = 30)
        @test exitcode == 3
    end

    @testset "timeout => -124, stdout captured" begin
        # A plain sleeper that outlives the 2 s timeout; SIGTERM reaps it within the grace.
        prog = "redirect_stderr(devnull); println(\"STARTED\"); flush(stdout); sleep(60)"
        exitcode, output = run_ovrtx_subprocess(prog; timeout = 2)
        @test exitcode == -124
        @test contains(output, "STARTED")   # output collected AFTER the child is reaped
    end

    @testset "SIGTERM-ignoring child escalates to SIGKILL" begin
        # Julia routes SIGTERM to a dedicated sigwait thread, so libc signal() alone can't
        # trap it.  Unblocking SIGTERM on this thread + SIG_IGN makes the kernel deliver-and-
        # discard it here, so this single process ignores SIGTERM — the watchdog must SIGKILL.
        prog = raw"""
        mask = zeros(UInt64, 16); mask[1] = UInt64(1) << (15 - 1)                          # sigset_t bit for SIGTERM(15)
        ccall(:pthread_sigmask, Cint, (Cint, Ptr{UInt64}, Ptr{Cvoid}), 1, mask, C_NULL)    # SIG_UNBLOCK
        ccall(:signal, Ptr{Cvoid}, (Cint, Ptr{Cvoid}), 15, Ptr{Cvoid}(1))                  # SIG_IGN
        println("STARTED"); flush(stdout)
        sleep(60)
        """
        local exitcode, output
        elapsed = @elapsed ((exitcode, output) =
            run_ovrtx_subprocess(prog; timeout = 2, kill_grace = 1))
        @test exitcode == -124
        @test contains(output, "STARTED")
        # Without SIGKILL escalation the child ignores SIGTERM and sleeps ~60 s;
        # a bounded elapsed proves the watchdog forced it down.
        @test elapsed < 30
    end

    @testset "uncaught signal (crash) => -signal, not 0" begin
        # A child that dies from SIGSEGV on its own — the real ovrtx-crash shape.
        # The pre-fix helper returned 0 here (a false pass); the fix returns -11.
        prog = "redirect_stderr(devnull); println(\"STARTED\"); flush(stdout); unsafe_store!(Ptr{Int}(0), 1)"
        exitcode, _ = run_ovrtx_subprocess(prog; timeout = 30)
        @test exitcode == -11
    end
end
