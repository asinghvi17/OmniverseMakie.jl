module LibOVRTX
using CEnum
import Libdl
import Libglvnd_jll

# Resolved at runtime in __init__ so OVRTX_LIBRARY_PATH is honored (not baked
# at precompile); generated `@ccall libovrtx.sym(...)` calls use this binding.
global libovrtx::String = "libovrtx-dynamic.so"
const _OVRTX_HANDLE  = Ref{Ptr{Cvoid}}(C_NULL)
const _OPENGL_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

function __init__()
    # Load libOpenGL first and RTLD_GLOBAL (soname libOpenGL.so.0) so ovrtx's
    # usd_resolver plugin resolves GL symbols against this image via its later
    # by-soname dlopen.  The env override permits a system/driver libOpenGL.
    libgl = get(ENV, "OVRTX_LIBOPENGL_PATH", Libglvnd_jll.libOpenGL_path)
    _OPENGL_HANDLE[] = try
        Libdl.dlopen(libgl, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
    catch e
        error("LibOVRTX: failed to dlopen libOpenGL at \"$libgl\" ($e). \
               Override the path with the OVRTX_LIBOPENGL_PATH env var.")
    end
    global libovrtx = get(ENV, "OVRTX_LIBRARY_PATH", "libovrtx-dynamic.so")
    _OVRTX_HANDLE[] = try
        Libdl.dlopen(libovrtx, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
    catch e
        error("LibOVRTX: failed to dlopen libovrtx at \"$libovrtx\" ($e). \
               Set OVRTX_LIBRARY_PATH to the absolute path of libovrtx-dynamic.so.")
    end
end

include("libovrtx_api.jl")  # generated ccalls, structs, enums, const macros

# --- static const / macros Clang does not emit ------------------------------
const OVRTX_TIMEOUT_INFINITE = ovrtx_timeout_t(typemax(UInt64))
const NOSYNC = ovrtx_cuda_sync_t(0, 0)

# --- ovx_string_t (raw ptr+len; caller GC.@preserve's the backing String) ---
# `String`-only by design: only a `String` is contiguous, so
# Base.unsafe_convert(Cstring, ::String) hands back its pointer with no copy,
# and the returned ovx_string_t points into memory the CALLER can root.  A
# convenience `ovx_string(::AbstractString)` would materialize `String(s)`
# INSIDE this call, leaving the ovx_string_t pointing into a temporary the
# caller's `GC.@preserve original` never roots (use-after-free for a
# SubString etc.).  So callers must `s = String(x)`, `GC.@preserve s`, then
# `ovx_string(s)`; a non-String argument here is a MethodError by intent.
ovx_string(s::String) = ovx_string_t(Base.unsafe_convert(Cstring, s), ncodeunits(s))
function Base.String(s::ovx_string_t)
    s.ptr == C_NULL && return ""
    return unsafe_string(Ptr{UInt8}(s.ptr), s.length)
end

# --- error idiom: ovrtx returns a struct whose .status is the enum ----------
struct OVRTXError <: Exception; op::String; msg::String; end
function Base.showerror(io::IO, e::OVRTXError)
    print(io, "OVRTXError during ", e.op, ": ", e.msg)
end
function check(result, op::AbstractString)
    result.status == OVRTX_API_SUCCESS && return result
    s = ovrtx_get_last_error()  # transient thread-local ovx_string_t; copy now
    throw(OVRTXError(op, String(s)))
end

# --- version helper (ovrtx_get_version returns Cvoid via 3 out-params) ------
function version()
    major, minor, patch = Ref{UInt32}(0), Ref{UInt32}(0), Ref{UInt32}(0)
    ovrtx_get_version(major, minor, patch)
    return (major[], minor[], patch[])
end

end # module
