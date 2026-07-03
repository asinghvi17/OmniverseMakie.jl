module SignalGuard
# Julia installs SIG{ILL,ABRT,BUS,FPE,SEGV} handlers with SA_SIGINFO|SA_ONSTACK.
# ovrtx_create_renderer loads carb's breakpad crash reporter, which installs its own
# handlers then chains back to Julia's — killing the process (termsignal=11) even on
# crashes internal to ovrtx/breakpad init (Python's simpler handlers avoid this).
# Fix: ATOMICALLY save Julia's handlers and set SIG_DFL BEFORE create_renderer (breakpad
# then inits cleanly over SIG_DFL); restore Julia's handlers after.
#
# ★ ROOT CAUSE of the suite's historical "intermittent startup crash" (the retry-harness
# mnemonic `GeometryGroup::attachToContext` — a legacy label; the string exists in no
# binary): Julia's GC stop-the-world safepoint works by PROTECTING a page and having every
# thread take a *recoverable* SIGSEGV when it polls.  If a GC fires during the multi-second
# create window, breakpad's SIGSEGV handler (or, under this guard, SIG_DFL) intercepts that
# safepoint fault and kills the process — 4/4 reproducible under thread+GC pressure,
# rare single-threaded.  So the guard alone is NOT enough: `with_restored_signals` ALSO
# disables the GC for the window (no GC → the safepoint page is never armed → no SIGSEGV
# to mis-handle), which measured 0/6 under the stress that crashes 4/4 unguarded.  The
# create ccall is bounded and the calling thread allocates nothing while blocked, so the
# GC pause is harmless.  DEFENSE IN DEPTH: index_config.jl additionally routes carb to a
# config with `/crashreporter/enabled = false` (probe-proven to stop the interception even
# on an UNGUARDED create), which also covers any handler re-arming after create that this
# guard's one-shot restore would miss.
#
# struct sigaction is platform-specific (Linux/glibc x86-64 sizeof==152); treat it as an
# opaque fixed-size blob — _SA_SIZE=256 gives headroom.
const _SIGS    = (4, 6, 7, 8, 11)   # SIGILL, SIGABRT, SIGBUS, SIGFPE, SIGSEGV
const _SA_SIZE = 256                 # >= sizeof(struct sigaction) on linux-x86_64 (152)

"""
    snapshot() -> Dict{Int,Vector{UInt8}}

Atomically set the handlers for SIGILL(4)/SIGABRT(6)/SIGBUS(7)/SIGFPE(8)/SIGSEGV(11)
to SIG_DFL, returning the previous handler blobs.  Call before ovrtx_create_renderer
(breakpad then chains to the harmless SIG_DFL, not Julia's SA_ONSTACK handlers); pass
the result to `restore` afterward.
"""
function snapshot()
    saved = Dict{Int,Vector{UInt8}}()
    try
        for sig in _SIGS
            saved[sig] = _swap_default(sig)
        end
    catch
        restore(saved)   # roll back what we already swapped
        rethrow()
    end
    return saved
end

"""
    restore(saved)

Reinstall the signal handlers returned by a previous snapshot() call.
"""
function restore(saved::AbstractDict)
    for (sig, blob) in saved
        _setaction(sig, blob)
    end
    return nothing
end

"""
    with_restored_signals(f)

Run `f()` (typically `ovrtx_create_renderer`) with the crash-reporter signals reset to
SIG_DFL AND the Julia GC disabled, restoring both afterward.  Neutralises the carb
breakpad crash reporter ovrtx installs at renderer-creation time; the GC pause keeps
Julia's safepoint page from ever being armed during the window, so there is no
recoverable safepoint SIGSEGV for breakpad/SIG_DFL to mis-handle (the root cause of the
historical intermittent startup crash — see the module header).
"""
function with_restored_signals(f)
    saved  = snapshot()
    gc_was = GC.enable(false)          # returns the PREVIOUS state — restore that, not `true`
    try
        return f()
    finally
        restore(saved)                 # evict breakpad FIRST (it persists past create)…
        GC.enable(gc_was)              # …then let the GC run again under Julia's handlers
    end
end

# -- internals ------------------------------------------------------------------

# Atomically set SIG_DFL for `sig`, returning the old handler blob.  A zeroed struct
# sigaction ≡ {sa_handler=SIG_DFL (=ptr 0), sa_mask=0, sa_flags=0}.
function _swap_default(sig::Integer)
    old     = zeros(UInt8, _SA_SIZE)
    new_dfl = zeros(UInt8, _SA_SIZE)   # all-zero ≡ {.sa_handler = SIG_DFL}
    r = @ccall sigaction(sig::Cint, new_dfl::Ptr{UInt8}, old::Ptr{UInt8})::Cint
    r == 0 || error("sigaction(swap_default) failed for signal $sig")
    return old
end

function _setaction(sig::Integer, blob::Vector{UInt8})
    r = @ccall sigaction(sig::Cint, blob::Ptr{UInt8}, C_NULL::Ptr{Cvoid})::Cint
    r == 0 || error("sigaction(set) failed for signal $sig")
    return nothing
end

end # module SignalGuard
