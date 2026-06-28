module SignalGuard
# Julia installs SIGSEGV/SIGABRT/etc. with SA_SIGINFO | SA_ONSTACK and a complex recovery
# path.  When ovrtx loads the carb breakpad crash reporter inside create_renderer, breakpad
# installs its own handlers on top of Julia's.  Breakpad then chains back to whatever was
# installed before it (Julia's handlers).  That chain causes the process to be killed by
# SIGSEGV (termsignal=11) even when the crash was internal to ovrtx/breakpad's own
# initialization.  Python avoids this because it has simpler (or no) signal handlers.
#
# The fix: ATOMICALLY save Julia's handlers and replace them with SIG_DFL *before*
# calling create_renderer.  During renderer initialization, breakpad installs on top of
# SIG_DFL and can do its own initialization cleanly.  After create_renderer, we restore
# Julia's handlers via restore(saved).
#
# struct sigaction is platform-specific; on Linux/glibc x86-64 sizeof == 152.
# We treat it as an opaque fixed-size blob; _SA_SIZE = 256 gives comfortable headroom.
const _SIGS    = (4, 6, 7, 8, 11)   # SIGILL, SIGABRT, SIGBUS, SIGFPE, SIGSEGV
const _SA_SIZE = 256                 # >= sizeof(struct sigaction) on linux-x86_64 (152)

"""
    snapshot() -> Dict{Int,Vector{UInt8}}

Atomically replace the POSIX handlers for SIGILL(4), SIGABRT(6), SIGBUS(7), SIGFPE(8),
SIGSEGV(11) with SIG_DFL, and return the previous handler blobs.

Calling snapshot() before ovrtx_create_renderer means that during renderer
initialization, breakpad chains to SIG_DFL (harmless) rather than Julia's complex
SA_ONSTACK handlers.  Call restore(saved) after create_renderer to put Julia's
handlers back.
"""
snapshot() = Dict(sig => _swap_default(sig) for sig in _SIGS)

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

Run `f()` (which typically calls `ovrtx_create_renderer`) with all crash-reporter
signals reset to SIG_DFL, then restore Julia's original signal handlers afterward.
This neutralises the carb breakpad crash reporter that ovrtx installs at
renderer-creation time.
"""
function with_restored_signals(f)
    saved = snapshot()
    try
        return f()
    finally
        restore(saved)
    end
end

# -- internals ------------------------------------------------------------------

# Atomically install SIG_DFL for `sig` and return the old handler blob.
# SIG_DFL = Ptr 0; a zeroed-out struct sigaction has sa_handler=SIG_DFL,
# sa_mask=0, sa_flags=0 — exactly what we want.
function _swap_default(sig::Integer)
    old     = zeros(UInt8, _SA_SIZE)
    new_dfl = zeros(UInt8, _SA_SIZE)   # all-zero ≡ { .sa_handler = SIG_DFL }
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
