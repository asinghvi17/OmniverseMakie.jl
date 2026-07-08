using Test
using OmniverseMakie: OV
import OmniverseMakie.LibOVRTX as LibOVRTX
import OmniverseMakie.OV.SignalGuard as SG

# ---------------------------------------------------------------------------
# GC-safe Binding finalizer + ovx_string lifetime contract + map_binding /
# with_restored_signals guards (pure, no GPU).
#
# `ovx_string` is `String`-ONLY by design: ovx_string_t is a raw ptr+len into
# the String's bytes, so the CALLER must own + GC.@preserve that String. A
# convenience `ovx_string(::AbstractString)` would materialize `String(s)`
# inside the call, leaving the struct pointing into a temporary the caller's
# `GC.@preserve original` never roots (use-after-free for a SubString). So a
# non-String argument is a MethodError by intent, forcing call-site
# materialization. `destroy!(; from_finalizer=true)` must skip all ovrtx work
# on a closed Renderer, always mark the binding dead (`alive=false`,
# `map_handle=0`), and never throw.
# ---------------------------------------------------------------------------

@testset "A4 ovx_string: String-only, forces materialization (pure)" begin
    subs = SubString("abc/def", 1, 3)   # "abc", a genuine SubString{String}
    @test subs isa SubString{String}
    @test !(subs isa String)

    # Hazard shape: a non-String argument is a MethodError, NOT a silent
    # materialize-into-a-temp. This is what forces callers to `s = String(x)`
    # + `GC.@preserve s` (the lifetime-sound pattern).
    @test_throws MethodError LibOVRTX.ovx_string(subs)

    # The sound pattern: materialize an owned String, preserve THAT, and the
    # ovx_string_t round-trips under GC pressure.
    backing = String(subs)
    @test backing isa String
    os = LibOVRTX.ovx_string(backing)
    GC.@preserve backing begin
        GC.gc()
        @test Int(os.length) == ncodeunits(backing) == 3
        @test String(os) == "abc"
    end

    # Plain String: zero-copy — the ovx_string_t points straight at the
    # String's bytes (no extra allocation / copy), pointer identity
    # preserved.
    s = "hello/world"
    os_str = LibOVRTX.ovx_string(s)
    GC.@preserve s begin
        GC.gc()
        @test Ptr{UInt8}(os_str.ptr) == pointer(s)
        @test Int(os_str.length) == ncodeunits(s)
        @test String(os_str) == s
    end
end

@testset "A4 map_binding: closed-Renderer guard errors cleanly (pure)" begin
    # A live Binding whose Renderer has been closed: map_binding must error on
    # the `b.r.alive` check BEFORE handing a C_NULL renderer instance to
    # ovrtx_map_attribute. A red-first demo of the DEFECT is impractical (the
    # unguarded path is a C_NULL-instance ccall = probable segfault, which
    # would crash the test process), so we assert the clean-error fix directly.
    _dead_renderer() = let r = ccall(:jl_new_struct_uninit, Any, (Any,), OV.Renderer)::OV.Renderer
        r.ptr = Ptr{LibOVRTX.ovrtx_renderer_t}(C_NULL)
        r.alive = false
        r
    end
    # alive Binding (alive=true, map_handle=0) referencing the CLOSED renderer.
    b = OV.Binding(_dead_renderer(), LibOVRTX.ovrtx_attribute_binding_handle_t(7),
        "/P", "omni:xform", LibOVRTX.DLDataType(UInt8(2), UInt8(64), UInt16(16)),
        false, LibOVRTX.OVRTX_SEMANTIC_NONE, LibOVRTX.ovrtx_map_handle_t(0), true)
    @test b.alive && !b.r.alive          # the exact hazard state
    err = try; OV.map_binding(b); nothing catch e; e end
    @test err isa ErrorException
    @test occursin("closed Renderer", err.msg)   # named cleanly, no ccall
end

@testset "A4 scoped-map + write-list guards error cleanly, no ccall (pure)" begin
    # map_cpu / with_mapped_hdr / map_cuda / read_pick_hit share the
    # `with_mapped_var` map/check/unmap scope but each keeps its OWN
    # closed-Renderer guard BEFORE any ccall.  A red-first demo is impractical
    # (the unguarded path is a C_NULL-instance ccall = probable segfault), so
    # assert each guard's clean error + verbatim message directly.
    _dead_renderer() = let r = ccall(:jl_new_struct_uninit, Any, (Any,), OV.Renderer)::OV.Renderer
        r.ptr = Ptr{LibOVRTX.ovrtx_renderer_t}(C_NULL); r.alive = false; r
    end
    dead = _dead_renderer()
    sr = OV.StepResult(dead, LibOVRTX.ovrtx_step_result_handle_t(0), false)
    @test !sr.r.alive
    guards = ((() -> OV.map_cpu(sr),                        "map_cpu: the StepResult's Renderer is already closed"),
              (() -> OV.with_mapped_hdr((a, b, c) -> nothing, sr), "with_mapped_hdr on a closed Renderer"),
              (() -> OV.map_cuda(sr),                       "map_cuda on a closed Renderer"),
              (() -> OV.read_pick_hit(sr),                  "read_pick_hit on a closed Renderer"))
    for (fn, msg) in guards
        err = try; fn(); nothing catch e; e end
        @test err isa ErrorException
        @test occursin(msg, err.msg)
    end
    # The shared helpers introduced by the consolidation exist.
    @test OV.with_mapped_var isa Function
    @test OV._fetch_find_var isa Function
    @test OV._write_attribute_prims! isa Function

    # set_selection_outline_group! now delegates to the shared FFI-write core
    # (_write_attribute_prims!, prim LIST).  The length-mismatch guard and the
    # empty (n==0) early return both short-circuit BEFORE any FFI write, so
    # they are provable against a Renderer whose instance is C_NULL.
    alive_null = let r = ccall(:jl_new_struct_uninit, Any, (Any,), OV.Renderer)::OV.Renderer
        r.ptr = Ptr{LibOVRTX.ovrtx_renderer_t}(C_NULL); r.alive = true; r
    end
    mm = try; OV.set_selection_outline_group!(alive_null, ["/A", "/B"], UInt8[1]); nothing catch e; e end
    @test mm isa ErrorException
    @test occursin("prim_paths (2) and group_ids (1) length mismatch", mm.msg)
    @test OV.set_selection_outline_group!(alive_null, String[], UInt8[]) === nothing
    cl = try; OV.set_selection_outline_group!(dead, ["/A"], UInt8[1]); nothing catch e; e end
    @test cl isa ErrorException
    @test occursin("set_selection_outline_group! on a closed Renderer", cl.msg)
end

@testset "A4 with_restored_signals: lock + GC restore (pure)" begin
    # The window is serialized by a module-level ReentrantLock so concurrent
    # Renderer() creations can't interleave snapshot/restore.
    @test SG._RESTORE_LOCK isa ReentrantLock

    # 1. f's value passes through.
    @test SG.with_restored_signals(() -> 42) == 42

    # 2. Reentrant: a nested call from the same task must not deadlock (proves
    #    ReentrantLock, not a plain lock) and still passes f's value through.
    @test SG.with_restored_signals() do
        SG.with_restored_signals(() -> 7) + 1
    end == 8

    # 3. GC is re-enabled after a NORMAL return. `GC.enable(true)` returns the
    #    previous state; restore it so the probe is non-destructive.
    SG.with_restored_signals(() -> nothing)
    was = GC.enable(true); GC.enable(was)
    @test was == true

    # 4. GC is re-enabled even when f THROWS (the finally chain runs). This is
    #    the regression: before the nested `try restore finally GC.enable end`,
    #    a throw between disable and restore could leave the GC disabled.
    threw = false
    try
        SG.with_restored_signals() do
            error("boom_signals_gc")
        end
    catch e
        threw = true
        @test occursin("boom_signals_gc", sprint(showerror, e))
    end
    @test threw
    was2 = GC.enable(true); GC.enable(was2)
    @test was2 == true

    # 5. Behavioral: with the Julia SIGSEGV handler restored, a full GC (which
    #    arms the safepoint page) runs without the process dying — SIG_DFL
    #    would fatally mis-handle a safepoint fault.
    SG.with_restored_signals(() -> nothing)
    GC.gc(true)
    @test true   # reached here ⇒ handlers restored, safepoint survived
end

@testset "A4 destroy! finalizer-flag paths (pure, no GPU)" begin
    # A closed Renderer built WITHOUT ovrtx: `Renderer` exposes only its
    # GPU-calling inner constructor, so bypass it (two isbits fields — ptr +
    # alive — set by hand).  With the Renderer closed, BOTH destroy! paths
    # must skip every ccall yet still mark the binding dead, exercising the
    # flag logic with no GPU.
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

    # from_finalizer path, closed Renderer: must not throw, must force the
    # binding dead, and must NOT count a leak (nothing errored — the ovrtx
    # side was skipped).
    b1 = _binding(_dead_renderer(); map_handle=99)
    # no throw out of the finalizer path
    @test (OV.destroy!(b1; from_finalizer=true); true)
    @test b1.alive == false
    @test b1.map_handle == 0   # finally-cleared even from non-zero
    @test OV.binding_finalizer_leak_count() == leaks0

    # Idempotent: a second finalizer-path call short-circuits on the alive
    # guard (no throw).
    @test (OV.destroy!(b1; from_finalizer=true); b1.alive == false)
    @test OV.binding_finalizer_leak_count() == leaks0

    # Explicit path (default from_finalizer=false), closed Renderer: same
    # flag clearing, no throw (renderer closed ⇒ no ccall ⇒ nothing to
    # propagate).
    b2 = _binding(_dead_renderer(); map_handle=42)
    @test (OV.destroy!(b2); b2.alive == false)
    @test b2.map_handle == 0

    # Already-dead binding short-circuits on BOTH paths (no work, no leak).
    b3 = _binding(_dead_renderer(); alive=false)
    @test (OV.destroy!(b3; from_finalizer=true); true)
    @test (OV.destroy!(b3); true)
    @test OV.binding_finalizer_leak_count() == leaks0
end
