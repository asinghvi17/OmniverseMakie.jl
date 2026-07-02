using Test
using OmniverseMakie: OV
import OmniverseMakie.LibOVRTX as LibOVRTX

# ---------------------------------------------------------------------------
# Task A4 — GC-safe Binding finalizer + ovx_string SubString fix.
#
# 1. ovx_string (PURE).  The old `ovx_string(::Union{String,SubString{String}})` branch
#    was a guaranteed MethodError on a SubString: it called
#    Base.unsafe_convert(Cstring, ::SubString), which has NO method (a SubString carries no
#    terminating NUL).  The fix routes every non-String AbstractString through
#    `ovx_string(String(s))` (ovx_string_t is a raw ptr+len — only a materialized String is
#    contiguous + NUL-terminated), while a plain String keeps its zero-copy pointer.
# 2. destroy! finalizer-flag paths (PURE, no GPU).  The finalizer now registers
#    `from_finalizer=true`; that path must skip all ovrtx work on a closed Renderer and
#    ALWAYS mark the binding dead (`alive=false`, `map_handle=0`), never throwing.  The real
#    ovrtx teardown ccalls stay covered by the m2_binding / m2_delete GPU testsets (which use
#    the unchanged explicit `destroy!`).
# ---------------------------------------------------------------------------

@testset "A4 ovx_string: SubString no MethodError + String zero-copy (pure)" begin
    subs = SubString("abc/def", 1, 3)            # "abc", a genuine SubString{String}
    @test subs isa SubString{String}

    # RED before A4: MethodError at unsafe_convert(Cstring, ::SubString).
    # GREEN: dispatches via ovx_string(::AbstractString); length is the substring's (3), not
    # the parent's (7).
    @test LibOVRTX.ovx_string(subs) isa LibOVRTX.ovx_string_t
    @test Int(LibOVRTX.ovx_string(subs).length) == ncodeunits(subs) == 3

    # Content round-trip: the AbstractString method builds `String(subs)`, and the
    # ovx_string_t points into THAT fresh String — so preserve it (not the parent literal)
    # across String(::ovx_string_t), under GC pressure.
    backing = String(subs)
    os = LibOVRTX.ovx_string(backing)
    GC.@preserve backing begin
        GC.gc()
        @test String(os) == "abc"
    end

    # Plain String: zero-copy — the ovx_string_t points straight at the String's bytes (no
    # extra allocation / copy), pointer identity preserved.
    s = "hello/world"
    os_str = LibOVRTX.ovx_string(s)
    GC.@preserve s begin
        GC.gc()
        @test Ptr{UInt8}(os_str.ptr) == pointer(s)
        @test Int(os_str.length) == ncodeunits(s)
        @test String(os_str) == s
    end
end

@testset "A4 destroy! finalizer-flag paths (pure, no GPU)" begin
    # A closed Renderer built WITHOUT ovrtx: `Renderer` exposes only its GPU-calling inner
    # constructor, so bypass it (two isbits fields — ptr + alive — set by hand).  With the
    # Renderer closed, BOTH destroy! paths must skip every ccall yet still mark the binding
    # dead, exercising the flag logic with no GPU.
    _dead_renderer() = let r = ccall(:jl_new_struct_uninit, Any, (Any,), OV.Renderer)::OV.Renderer
        r.ptr = Ptr{LibOVRTX.ovrtx_renderer_t}(C_NULL)
        r.alive = false
        r
    end
    _binding(r; alive=true, map_handle=0) = OV.Binding(
        r, LibOVRTX.ovrtx_attribute_binding_handle_t(7), "/P", "attr",
        LibOVRTX.DLDataType(UInt8(2), UInt8(32), UInt16(3)), false,
        LibOVRTX.OVRTX_SEMANTIC_NONE, LibOVRTX.ovrtx_map_handle_t(map_handle), alive)

    leaks0 = OV.binding_finalizer_leak_count()
    @test leaks0 isa Int

    # from_finalizer path, closed Renderer: must not throw, must force the binding dead, and
    # must NOT count a leak (nothing errored — the ovrtx side was skipped).
    b1 = _binding(_dead_renderer(); map_handle=99)
    @test (OV.destroy!(b1; from_finalizer=true); true)   # no throw out of the finalizer path
    @test b1.alive == false
    @test b1.map_handle == 0                              # finally-cleared even from non-zero
    @test OV.binding_finalizer_leak_count() == leaks0

    # Idempotent: a second finalizer-path call short-circuits on the alive guard (no throw).
    @test (OV.destroy!(b1; from_finalizer=true); b1.alive == false)
    @test OV.binding_finalizer_leak_count() == leaks0

    # Explicit path (default from_finalizer=false), closed Renderer: same flag clearing, no
    # throw (renderer closed ⇒ no ccall ⇒ nothing to propagate).
    b2 = _binding(_dead_renderer(); map_handle=42)
    @test (OV.destroy!(b2); b2.alive == false)
    @test b2.map_handle == 0

    # Already-dead binding short-circuits on BOTH paths (no work, no leak).
    b3 = _binding(_dead_renderer(); alive=false)
    @test (OV.destroy!(b3; from_finalizer=true); true)
    @test (OV.destroy!(b3); true)
    @test OV.binding_finalizer_leak_count() == leaks0
end
