module LibKitJL
# Raw Julia bindings over `libkitjl.so` — the flat `extern "C"` shim that hosts
# the NVIDIA Kit runtime in-process (see src/kitjl_shim.cpp).  Mirrors LibOVRTX:
# a compiled native artifact located against an env-provided runtime (Kit, via
# KIT_RELEASE_DIR) that is NEVER vendored, an RTLD_GLOBAL load-order __init__,
# `@ccall` wrappers, and a `check`/last-error exception idiom.
#
# The in-process transport is OPT-IN.  When Kit is absent at build time, this
# module still loads: `deps/build.jl` writes an "unavailable" marker and every
# entry point throws a clear `KitJlError`; the subprocess transport is
# unaffected.

import Libdl
if Sys.islinux()
    import Libglvnd_jll
end

# deps.jl (from deps/build.jl) defines LIBKITJL_AVAILABLE / LIBKITJL_PATH /
# LIBKITJL_KIT_RELEASE_DIR / LIBKITJL_UNAVAILABLE_REASON.  Missing (never
# built) counts as unavailable.
const _DEPS_JL = joinpath(@__DIR__, "..", "deps", "deps.jl")
if isfile(_DEPS_JL)
    include(_DEPS_JL)
else
    const LIBKITJL_AVAILABLE = false
    const LIBKITJL_PATH = ""
    const LIBKITJL_KIT_RELEASE_DIR = ""
    const LIBKITJL_UNAVAILABLE_REASON = "deps/deps.jl missing (run Pkg.build(\"LibKitJL\") with KIT_RELEASE_DIR set)"
end

# Resolved at runtime in __init__; @ccall uses this global binding.
global libkitjl::String = LIBKITJL_PATH

const _KIT_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
const _PRELOAD_HANDLES = Ptr{Cvoid}[]

struct KitJlError <: Exception
    op::String
    msg::String
end
Base.showerror(io::IO, e::KitJlError) = print(io, "KitJlError during ", e.op, ": ", e.msg)

available() = LIBKITJL_AVAILABLE && _KIT_HANDLE[] != C_NULL

function _ensure_available(op::AbstractString)
    available() && return nothing
    reason = LIBKITJL_AVAILABLE ? "libkitjl.so failed to load in __init__" : LIBKITJL_UNAVAILABLE_REASON
    throw(KitJlError(op, "in-process Kit transport unavailable ($reason)"))
end

# ---------------------------------------------------------------------------
# __init__ load order (load-bearing; ported from kit/kit_app.py).  All preloads
# RTLD_GLOBAL so Kit's later by-soname dlopens (python bindings, MDL SDK, etc.)
# resolve symbols against these images.  Order:
#   CARB_APP_PATH + OMNI_KIT_ACCEPT_EULA env, then libcarb, libre2, libcares,
#   libpython3.12 (Kit-bundled — the system Python may differ), GLVND, then
#   libkitjl.  libGLU is ensured later (at renderer start) by the Julia
#   transport, since setting LD_LIBRARY_PATH mid-process is unreliable.
# ---------------------------------------------------------------------------
function _dlopen_global(path::AbstractString; required::Bool = true, what::AbstractString = path)
    h = try
        Libdl.dlopen(path, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
    catch e
        required && @warn "LibKitJL: failed to RTLD_GLOBAL dlopen $what; in-process Kit may fail to start" path exception = e
        return C_NULL
    end
    push!(_PRELOAD_HANDLES, h)
    return h
end

function __init__()
    LIBKITJL_AVAILABLE || return  # unavailable build: module loads, ops throw

    kit = LIBKITJL_KIT_RELEASE_DIR
    kitdir = joinpath(kit, "kit")

    # Framework config root + EULA (kit_app.py sets both before startup).
    get(ENV, "CARB_APP_PATH", "") == "" && (ENV["CARB_APP_PATH"] = kitdir)
    get(ENV, "OMNI_KIT_ACCEPT_EULA", "") == "" && (ENV["OMNI_KIT_ACCEPT_EULA"] = "YES")

    # Core carb libs, RTLD_GLOBAL, in order (kit_app.py: libcarb, libre2, libcares).
    for lib in ("libcarb.so", "libre2.so", "libcares.so")
        p = joinpath(kitdir, lib)
        isfile(p) && _dlopen_global(p; what = lib)
    end

    # Kit-bundled libpython (the system Python version may not match — this Kit
    # ships 3.12).  Prefer a system libpython3.12, else the bundled one.
    pybundled = joinpath(kitdir, "kernel", "plugins", "libpython3.12.so")
    _dlopen_global("libpython3.12.so"; required = false, what = "system libpython3.12.so") == C_NULL &&
        isfile(pybundled) && _dlopen_global(pybundled; what = "Kit-bundled libpython3.12.so")

    # GLVND (libOpenGL) RTLD_GLOBAL — same discipline as LibOVRTX; non-fatal.
    if Sys.islinux()
        try
            _dlopen_global(Libglvnd_jll.libOpenGL_path; required = false, what = "libOpenGL (GLVND)")
        catch e
            @warn "LibKitJL: could not preload GLVND libOpenGL" exception = e
        end
    end

    # Finally the shim.
    global libkitjl = LIBKITJL_PATH
    _KIT_HANDLE[] = _dlopen_global(LIBKITJL_PATH; what = "libkitjl.so")
    return nothing
end

# ---------------------------------------------------------------------------
# @ccall wrappers over the flat C ABI.  Opaque handle = Ptr{Cvoid}.
# ---------------------------------------------------------------------------

"Thread-local last-error string from the shim (copy immediately on read)."
last_error() = available() ?
    unsafe_string(@ccall libkitjl.kitjl_last_error()::Cstring) : "(libkitjl not loaded)"

"carb SDK version string (no framework start, no GPU — the pure smoke check)."
function sdk_version()
    _ensure_available("sdk_version")
    return unsafe_string(@ccall libkitjl.kitjl_sdk_version()::Cstring)
end

"""
    startup(argv::Vector{String}) -> Ptr{Cvoid}

Start the in-process Kit app with command-line `argv` (argv[1] should be the
program name, e.g. "kit").  Returns an opaque non-null handle, or throws
`KitJlError` on failure.  MUST be wrapped by the caller in the crashreporter/GC
signal guard (see OmniverseKitMakie's InProcessTransport).
"""
function startup(argv::Vector{String})
    _ensure_available("startup")
    # Build a NUL-terminated char* array; GC.@preserve the backing Strings.
    cargv = [Base.unsafe_convert(Cstring, s) for s in argv]
    h = GC.@preserve argv cargv begin
        @ccall libkitjl.kitjl_startup(length(argv)::Cint, cargv::Ptr{Cstring})::Ptr{Cvoid}
    end
    h == C_NULL && throw(KitJlError("startup", last_error()))
    return h
end

update(h::Ptr{Cvoid}) = (@ccall libkitjl.kitjl_update(h::Ptr{Cvoid})::Cvoid; nothing)
is_running(h::Ptr{Cvoid}) = (@ccall libkitjl.kitjl_is_running(h::Ptr{Cvoid})::Cint) != 0
post_quit(h::Ptr{Cvoid}, code::Integer = 0) =
    (@ccall libkitjl.kitjl_post_quit(h::Ptr{Cvoid}, Cint(code)::Cint)::Cvoid; nothing)
shutdown(h::Ptr{Cvoid}) = Int(@ccall libkitjl.kitjl_shutdown(h::Ptr{Cvoid})::Cint)

set_setting_bool(h::Ptr{Cvoid}, path::AbstractString, v::Bool) =
    (@ccall libkitjl.kitjl_set_setting_bool(h::Ptr{Cvoid}, path::Cstring, Cint(v)::Cint)::Cvoid; nothing)
set_setting_int(h::Ptr{Cvoid}, path::AbstractString, v::Integer) =
    (@ccall libkitjl.kitjl_set_setting_int(h::Ptr{Cvoid}, path::Cstring, Clonglong(v)::Clonglong)::Cvoid; nothing)
set_setting_float(h::Ptr{Cvoid}, path::AbstractString, v::Real) =
    (@ccall libkitjl.kitjl_set_setting_float(h::Ptr{Cvoid}, path::Cstring, Cdouble(v)::Cdouble)::Cvoid; nothing)
set_setting_string(h::Ptr{Cvoid}, path::AbstractString, v::AbstractString) =
    (@ccall libkitjl.kitjl_set_setting_string(h::Ptr{Cvoid}, path::Cstring, v::Cstring)::Cvoid; nothing)
get_setting_bool(h::Ptr{Cvoid}, path::AbstractString) =
    (@ccall libkitjl.kitjl_get_setting_bool(h::Ptr{Cvoid}, path::Cstring)::Cint) != 0

"""
    exec_string(h, code::AbstractString)

Run `code` in Kit's embedded Python interpreter (the scripting escape hatch).
Throws `KitJlError` if execution reports failure.
"""
function exec_string(h::Ptr{Cvoid}, code::AbstractString)
    _ensure_available("exec_string")
    rc = @ccall libkitjl.kitjl_exec_string(h::Ptr{Cvoid}, code::Cstring)::Cint
    rc == 0 || throw(KitJlError("exec_string", last_error()))
    return nothing
end

end # module LibKitJL
