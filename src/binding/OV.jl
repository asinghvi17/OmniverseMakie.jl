module OV
import ..LibOVRTX
include("signals.jl")
using .SignalGuard: with_restored_signals

include("dlpack.jl")

# Volumes M1 — NVIDIA IndeX enablement (`_ensure_index` OncePerProcess + carb-config injection).
# Included before the Renderer struct so `_ensure_index` is defined when Renderer() calls it.
include("index_config.jl")

# Named constant so the three timeout kwarg defaults share a single source.
const _TIMEOUT_INFINITE_NS = LibOVRTX.OVRTX_TIMEOUT_INFINITE.time_out_ns

# ------------------------------------------------------------------
# Renderer — GC-aware wrapper around ovrtx_renderer_t
# ------------------------------------------------------------------

mutable struct Renderer
    ptr::Ptr{LibOVRTX.ovrtx_renderer_t}
    alive::Bool
    # selection_outline=true builds a creation-time `ovrtx_config_t` enabling the M6.B
    # outline; default false passes an EMPTY config (entry_count 0), byte-identical to the
    # pre-M6.B path.  outline_width = global thickness in px (only meaningful when on).
    #
    # Config-entry FFI (REPL-VERIFIED vs ovrtx_config.h ovrtx_config_entry_bool/_int +
    # a real create+render): each `ovrtx_config_entry_t` is an opaque 24-byte struct —
    # key_type @0 (enum UInt32), key union @4, value union @8.  Written via setproperty!
    # on a Ptr into a ZERO-INITIALIZED entries Vector (deterministic union padding), and
    # `entries` is GC.@preserve'd across ovrtx_create_renderer (which copies the config).
    # Mirrors the C inlines: bool → BOOL/key.bool_key/value.bool_value; int64 → INT64/
    # key.int64_key/value.int_value.
    function Renderer(; selection_outline::Bool=false, outline_width::Int=8)
        # Volumes M1: enable NVIDIA IndeX (once/process) BEFORE ovrtx_create_renderer so
        # carb consumes the IndeX-libs token at framework init.  No-op false unless a volume
        # env var is set — zero overhead on the non-volume path (memoized after 1st call).
        _ensure_index()
        renderer_ref = Ref{Ptr{LibOVRTX.ovrtx_renderer_t}}(C_NULL)
        _create(cfg) = with_restored_signals() do
            LibOVRTX.check(LibOVRTX.ovrtx_create_renderer(cfg, renderer_ref), "create_renderer")
        end
        if selection_outline
            entries = fill(LibOVRTX.ovrtx_config_entry_t(ntuple(_ -> UInt8(0), 24)), 2)
            GC.@preserve entries begin
                entry1 = pointer(entries, 1)
                entry1.key_type         = LibOVRTX.OVRTX_CONFIG_KEY_TYPE_BOOL
                entry1.key.bool_key     = LibOVRTX.OVRTX_CONFIG_SELECTION_OUTLINE_ENABLED
                entry1.value.bool_value = true
                entry2 = pointer(entries, 2)
                entry2.key_type         = LibOVRTX.OVRTX_CONFIG_KEY_TYPE_INT64
                entry2.key.int64_key    = LibOVRTX.OVRTX_CONFIG_SELECTION_OUTLINE_WIDTH
                entry2.value.int_value  = Int64(outline_width)
                _create(Ref(LibOVRTX.ovrtx_config_t(pointer(entries), Csize_t(2))))
            end
        else
            _create(Ref(LibOVRTX.ovrtx_config_t(Ptr{LibOVRTX.ovrtx_config_entry_t}(C_NULL), Csize_t(0))))
        end
        r = new(renderer_ref[], true)
        finalizer(close, r)
        return r
    end
end

Base.unsafe_convert(::Type{Ptr{LibOVRTX.ovrtx_renderer_t}}, r::Renderer) = r.ptr

function Base.close(r::Renderer)
    r.alive || return
    LibOVRTX.ovrtx_destroy_renderer(r.ptr)
    r.ptr = Ptr{LibOVRTX.ovrtx_renderer_t}(C_NULL)   # avoid a dangling pointer via unsafe_convert after close
    r.alive = false
    return
end

# ------------------------------------------------------------------
# Async lifecycle: enqueue (ovrtx_enqueue_result_t) -> wait_op
# ------------------------------------------------------------------

"""
    enqueue_wait(f, r::Renderer, op::AbstractString; timeout_ns) -> ovrtx_op_wait_result_t

Run one enqueue-then-wait cycle.  `f` is a THUNK returning the
`ovrtx_enqueue_result_t` (pass it as a `do` block) — the alive check runs FIRST, so a
closed Renderer errors cleanly BEFORE the enqueue ccall (whose args would otherwise
pass `C_NULL` into ovrtx).  After the wait, per-op failures reported in
`ovrtx_op_wait_result_t.error_op_ids` (e.g. a missing USD file — the enqueue AND the
wait both still report SUCCESS) are resolved via `ovrtx_get_last_op_error` and thrown
as `OVRTXError`.
"""
function enqueue_wait(f, r::Renderer, op::AbstractString;
                     timeout_ns::UInt64 = _TIMEOUT_INFINITE_NS)
    r.alive || error("enqueue_wait called on a closed Renderer")
    enq = f()
    LibOVRTX.check(enq, op)
    wait_ref = Ref{LibOVRTX.ovrtx_op_wait_result_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_wait_op(r.ptr, enq.op_index, LibOVRTX.ovrtx_timeout_t(timeout_ns), wait_ref), op * ":wait")
    wr = wait_ref[]
    if wr.num_error_ops != 0
        # Per-op failures surface ONLY here.  error_op_ids + the ovrtx_get_last_op_error
        # strings are transient thread-local data invalidated by the next ovrtx_wait_op —
        # copy each String(::ovx_string_t) immediately, same discipline as LibOVRTX.check.
        msgs = String[]
        for i in 1:wr.num_error_ops
            push!(msgs, String(LibOVRTX.ovrtx_get_last_op_error(unsafe_load(wr.error_op_ids, i))))
        end
        throw(LibOVRTX.OVRTXError(op, join(msgs, "; ")))
    end
    return wr
end

# ------------------------------------------------------------------
# open_usd! / open_usd_string! — load a stage (sync)
# ------------------------------------------------------------------

"""
    open_usd!(r::Renderer, path::AbstractString)

Open a USD stage from the file at `path`.  Synchronous (enqueue + wait);
`path` is GC.@preserve'd across the ccall + wait.
"""
function open_usd!(r::Renderer, path::AbstractString)
    GC.@preserve path begin
        enqueue_wait(r, "open_usd") do
            LibOVRTX.ovrtx_open_usd_from_file(r.ptr, LibOVRTX.ovx_string(path))
        end
    end
    return nothing
end

"""
    open_usd_string!(r::Renderer, usda::AbstractString)

Open a USD stage from an in-memory USDA string.  Synchronous.
"""
function open_usd_string!(r::Renderer, usda::AbstractString)
    GC.@preserve usda begin
        enqueue_wait(r, "open_usd_string") do
            LibOVRTX.ovrtx_open_usd_from_string(r.ptr, LibOVRTX.ovx_string(usda))
        end
    end
    return nothing
end

# ------------------------------------------------------------------
# StepResult — wraps an ovrtx_step_result_handle_t
# ------------------------------------------------------------------

mutable struct StepResult
    r::Renderer
    handle::LibOVRTX.ovrtx_step_result_handle_t
    open::Bool
end

function Base.close(sr::StepResult)
    sr.open || return
    sr.r.alive && LibOVRTX.ovrtx_destroy_results(sr.r.ptr, sr.handle)  # pool already freed if renderer closed
    sr.open = false
    return nothing
end

# ------------------------------------------------------------------
# step! — enqueue one render step, return StepResult
# ------------------------------------------------------------------

"""
    step!(r::Renderer, product::AbstractString; dt=1/60, timeout_ns) -> StepResult

Enqueue + wait one RT2 render step for `product`.  `timeout_ns` bounds the wait
(defaults to infinite; the M5 camera loop sets a finite value to avoid hanging on
a stalled step).  The backing `ovx_string_t` array and `product` are GC.@preserve'd
across the ccall + wait.  Caller must `close` the returned `StepResult` (or let the
finalizer run).
"""
function step!(r::Renderer, product::AbstractString;
              dt::Float64=1/60, timeout_ns::UInt64 = _TIMEOUT_INFINITE_NS)
    rp = LibOVRTX.ovx_string_t[ LibOVRTX.ovx_string(product) ]
    GC.@preserve product rp begin
        set = LibOVRTX.ovrtx_render_product_set_t(pointer(rp), Csize_t(1))
        h = Ref{LibOVRTX.ovrtx_step_result_handle_t}(0)
        try
            enqueue_wait(r, "step"; timeout_ns) do
                LibOVRTX.ovrtx_step(r.ptr, set, dt, h)
            end
        catch
            # A1 made enqueue_wait a thunk: the enqueue ccall fills h[] BEFORE the wait.
            # If the wait times out/fails (a finite-timeout M5 path), the results handle
            # is live but no StepResult owns it — free it best-effort, then rethrow.
            h[] != 0 && r.alive && LibOVRTX.ovrtx_destroy_results(r.ptr, h[])
            rethrow()
        end
        sr = StepResult(r, h[], true)
        finalizer(close, sr)
        return sr
    end
end

# ------------------------------------------------------------------
# _find_var — walk the nested output tree to locate a render var
# ------------------------------------------------------------------

"""
    _find_var_opt(outs::ovrtx_render_product_set_outputs_t, name) -> handle | nothing

Walk outputs → frames → render_vars; return the handle whose render_var_name ==
`name` (e.g. "LdrColor"), or `nothing`.  A missing `ovrtx_pick_hit` var (no pick
enqueued) is an expected, non-error case for `read_pick_hit`.
"""
function _find_var_opt(outs::LibOVRTX.ovrtx_render_product_set_outputs_t, name::AbstractString)
    for i in 1:outs.output_count
        product_out = unsafe_load(outs.outputs, i)             # ovrtx_render_product_output_t
        for f in 1:product_out.output_frame_count
            frame_out = unsafe_load(product_out.output_frames, f)  # ..._frame_output_t
            for v in 1:frame_out.render_var_count
                var_out = unsafe_load(frame_out.output_render_vars, v)  # ..._render_var_output_t
                String(var_out.render_var_name) == name && return var_out.output_handle
            end
        end
    end
    return nothing
end

"""
    _find_var(outs::ovrtx_render_product_set_outputs_t, name) -> handle

Like `_find_var_opt` but throws if `name` is absent (the contract map_cpu /
map_cpu_f32 / map_cuda rely on — LdrColor/HdrColor are always present).
"""
function _find_var(outs::LibOVRTX.ovrtx_render_product_set_outputs_t, name::AbstractString)
    h = _find_var_opt(outs, name)
    h === nothing && error("render var '$name' not found in step outputs")
    return h
end

# A successful ovrtx_map_render_var_output can still hand back a FAILED output whose
# .status/.error_message the map result code does NOT reflect (review: never inspected).
# Throw so a bad map surfaces instead of feeding garbage into the readback.
function _check_var_output(out::LibOVRTX.ovrtx_render_var_output_t, op::AbstractString)
    out.status == LibOVRTX.OVRTX_EVENT_FAILURE &&
        throw(LibOVRTX.OVRTXError(op, String(out.error_message)))
    return nothing
end

# ------------------------------------------------------------------
# map_cpu — fetch results, find LdrColor, map to CPU, single-pass build, unmap
# ------------------------------------------------------------------

"""
    map_cpu(sr::StepResult, name="LdrColor") -> (img::Matrix{RGBA{N0f8}}, W, H)

Fetch results, find render var `name` (kDLUInt/8 RGBA), map to CPU, and build the
`(H, W)` display matrix in ONE pass (`cwh_to_matrix`, reinterpret+permute) straight
from the still-mapped `[C,W,H]` memory — no separate byte copy.  The mapping is
released in a `finally` (a throw mid-build never leaks it); the returned `img` is
owned and valid after unmap.  Top-left origin, no y-flip.
"""
function map_cpu(sr::StepResult, name::AbstractString="LdrColor")
    sr.r.alive || error("map_cpu: the StepResult's Renderer is already closed")
    outs = Ref{LibOVRTX.ovrtx_render_product_set_outputs_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_fetch_results(sr.r.ptr, sr.handle, LibOVRTX.OVRTX_TIMEOUT_INFINITE, outs), "fetch_results")
    h = _find_var(outs[], name)
    mdesc = Ref(LibOVRTX.ovrtx_map_output_description_t(LibOVRTX.OVRTX_MAP_DEVICE_TYPE_CPU, Csize_t(0)))
    ro    = Ref{LibOVRTX.ovrtx_render_var_output_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_map_render_var_output(sr.r.ptr, h, mdesc, LibOVRTX.OVRTX_TIMEOUT_INFINITE, ro), "map_render_var_output")
    return try
        _check_var_output(ro[], "map_render_var_output($name)")
        # DLTensor shape [H,W,C]; wrap the mapped bytes as a non-owning [C,W,H] view.
        dlt = unsafe_load(unsafe_load(ro[].tensors, 1).dl)
        H, W, C = Int(unsafe_load(dlt.shape, 1)), Int(unsafe_load(dlt.shape, 2)), Int(unsafe_load(dlt.shape, 3))
        raw = unsafe_wrap(Array, Ptr{UInt8}(dlt.data), (C, W, H); own=false)
        (cwh_to_matrix(raw), W, H)          # single pass → owned Matrix, safe post-unmap
    finally
        # NOSYNC: CPU map.  Checked — an unmap failure means the build above may be invalid.
        LibOVRTX.check(LibOVRTX.ovrtx_unmap_render_var_output(sr.r.ptr, ro[].map_handle, LibOVRTX.NOSYNC),
                       "unmap_render_var_output")
    end
end

# ------------------------------------------------------------------
# with_mapped_hdr — map an HDR render var, run `f` on the mapped view, unmap
#
# HdrColor is kDLFloat/16 (verified: dtype.code=kDLFloat, dtype.bits=16).  The
# scoped/callback form owns the map lifetime (unmap in `finally`) so a throwing `f`
# never leaks it; `map_cpu_f32` is the eager-copy convenience over it.  INT-2 tonemaps
# the mapped view in place through this same signature.
# ------------------------------------------------------------------

"""
    with_mapped_hdr(f, sr::StepResult, name="HdrColor") -> f's result

Fetch results, find render var `name`, map it to CPU, and call `f(raw16, W, H)` where
`raw16` is the still-mapped, non-owning `[C=4, W, H]` `Float16` view (channel-fastest).
The mapping is released in a `finally` — even if `f` throws — so it never leaks; `f`'s
return value is passed through.  `raw16` is INVALID once `f` returns (mapped memory
dies on unmap): copy anything you keep.
"""
function with_mapped_hdr(f, sr::StepResult, name::AbstractString="HdrColor")
    sr.r.alive || error("with_mapped_hdr on a closed Renderer")
    outs = Ref{LibOVRTX.ovrtx_render_product_set_outputs_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_fetch_results(sr.r.ptr, sr.handle, LibOVRTX.OVRTX_TIMEOUT_INFINITE, outs), "fetch_results")
    h = _find_var(outs[], name)
    mdesc = Ref(LibOVRTX.ovrtx_map_output_description_t(LibOVRTX.OVRTX_MAP_DEVICE_TYPE_CPU, Csize_t(0)))
    ro    = Ref{LibOVRTX.ovrtx_render_var_output_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_map_render_var_output(sr.r.ptr, h, mdesc, LibOVRTX.OVRTX_TIMEOUT_INFINITE, ro), "map_render_var_output")
    return try
        _check_var_output(ro[], "map_render_var_output($name)")
        # DLTensor shape [H,W,C], dtype kDLFloat/16/1; wrap mapped bytes as [C,W,H] Float16.
        dlt = unsafe_load(unsafe_load(ro[].tensors, 1).dl)
        H, W, C = Int(unsafe_load(dlt.shape, 1)), Int(unsafe_load(dlt.shape, 2)), Int(unsafe_load(dlt.shape, 3))
        raw16 = unsafe_wrap(Array, Ptr{Float16}(dlt.data), (C, W, H); own=false)
        f(raw16, W, H)
    finally
        # NOSYNC: CPU map.  Checked — an unmap failure means the mapped read may be invalid.
        LibOVRTX.check(LibOVRTX.ovrtx_unmap_render_var_output(sr.r.ptr, ro[].map_handle, LibOVRTX.NOSYNC),
                       "unmap_render_var_output")
    end
end

"""
    map_cpu_f32(sr::StepResult, name) -> (pixels::Array{Float32,3}, W, H)

Eager HDR readback: `with_mapped_hdr` + a `Float16`→`Float32` copy of the mapped
view.  `pixels` is an owned `[C=4, W, H]` (channel-fastest) array.
"""
map_cpu_f32(sr::StepResult, name::AbstractString) =
    with_mapped_hdr(sr, name) do raw16, W, H
        (Float32.(raw16), W, H)
    end

# ------------------------------------------------------------------
# map_cuda / unmap_cuda — map a render var as LINEAR CUDA device memory (M6.A)
#
# Unlike map_cpu (copy+unmap inside the call), map_cuda MAPS and RETURNS the live
# device ptr + handles WITHOUT copy/unmap.  The caller (CUDA pkg ext) wraps `data`
# as CuArray{Float16}, tonemaps on-device into the GLMakie texture, then calls
# unmap_cuda gated on a copy-done event (else ovrtx reclaims the buffer mid-copy).
# No `using CUDA` (weakdep) — pure LibOVRTX ccalls returning RAW handles.
#
# Map mode OVRTX_MAP_DEVICE_TYPE_CUDA (=2, LINEAR → CUdeviceptr), NOT CUDA_ARRAY
# (=3, opaque CUarray): the tonemap needs LINEAR mem `unsafe_wrap` can wrap as a
# CuArray (an opaque CUarray cannot).  HdrColor is kDLFloat/16.  Verified RTX
# A5000: mode 2 → non-null kDLCUDA float16 [H,W,C=4] ptr + non-zero wait-event.
# (Co-loading CUDA.jl + ovrtx needs JULIA_CUDA_USE_COMPAT=false to share the
# system libcuda — see test/helpers.jl.)
# ------------------------------------------------------------------

"""
    map_cuda(sr::StepResult, name="HdrColor")
        -> (data::Ptr{Cvoid}, W, H, C, map_handle, wait_event::Csize_t)

Fetch results, find `name`, map as LINEAR CUDA device memory
(OVRTX_MAP_DEVICE_TYPE_CUDA).  Returns RAW handles — does NOT copy or unmap:

- `data`       — live `CUdeviceptr` (as `Ptr{Cvoid}`); for `HdrColor`, `C*W*H`
                 `Float16`s laid out `[C, W, H]` (channel-fastest).
- `(W, H, C)`  — DLTensor dims (shape `[H, W, C]`; `C == 4` for RGBA).
- `map_handle` — pass to `unmap_cuda`.
- `wait_event` — `CUevent` (as `Csize_t`, may be 0) to wait on BEFORE reading.

Buffer stays mapped until `unmap_cuda`.  No CUDA.jl dep — the CUDA pkg ext wraps
`data` as a `CuArray` and `wait_event` as a `CuEvent`.
"""
function map_cuda(sr::StepResult, name::AbstractString="HdrColor")
    sr.r.alive || error("map_cuda on a closed Renderer")
    outs = Ref{LibOVRTX.ovrtx_render_product_set_outputs_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_fetch_results(sr.r.ptr, sr.handle, LibOVRTX.OVRTX_TIMEOUT_INFINITE, outs), "fetch_results")

    h = _find_var(outs[], name)

    # map as LINEAR CUDA device memory (mode 2) — no copy, no unmap
    mdesc = Ref(LibOVRTX.ovrtx_map_output_description_t(LibOVRTX.OVRTX_MAP_DEVICE_TYPE_CUDA, Csize_t(0)))
    ro    = Ref{LibOVRTX.ovrtx_render_var_output_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_map_render_var_output(sr.r.ptr, h, mdesc, LibOVRTX.OVRTX_TIMEOUT_INFINITE, ro), "map_render_var_output(cuda)")

    # This path returns the LIVE mapping (no finally), so on a FAILED output release the
    # just-acquired mapping before surfacing the error (review: .status was never checked).
    if ro[].status == LibOVRTX.OVRTX_EVENT_FAILURE
        msg = String(ro[].error_message)
        LibOVRTX.ovrtx_unmap_render_var_output(sr.r.ptr, ro[].map_handle, LibOVRTX.NOSYNC)
        throw(LibOVRTX.OVRTXError("map_render_var_output(cuda:$name)", msg))
    end

    # DLTensor: shape [H,W,C], dtype kDLFloat/16/1 (HdrColor); CUdeviceptr in .data
    t0  = unsafe_load(ro[].tensors, 1)          # ovrtx_render_var_tensor_t
    dlt = unsafe_load(t0.dl)                     # DLTensor (CUdeviceptr in .data)
    H   = Int(unsafe_load(dlt.shape, 1))
    W   = Int(unsafe_load(dlt.shape, 2))
    C   = Int(unsafe_load(dlt.shape, 3))

    # RAW handles; caller reads on-device then calls unmap_cuda
    return (Ptr{Cvoid}(dlt.data), W, H, C, ro[].map_handle, ro[].cuda_sync.wait_event)
end

"""
    unmap_cuda(sr::StepResult, map_handle; stream=Csize_t(0), done_event=Csize_t(0))

Release a `map_cuda` mapping.  Builds `ovrtx_cuda_sync_t(stream, done_event)` (field
order stream, THEN event) so ovrtx waits for `done_event` (recorded after the
caller's on-device copy) on `stream` before reclaiming the buffer — else it reclaims
mid-copy.  Default `(0,0)` = NOSYNC (synchronous).  No-op if the Renderer is closed.
"""
function unmap_cuda(sr::StepResult, map_handle; stream::Csize_t = Csize_t(0), done_event::Csize_t = Csize_t(0))
    sync = LibOVRTX.ovrtx_cuda_sync_t(stream, done_event)
    sr.r.alive && LibOVRTX.ovrtx_unmap_render_var_output(sr.r.ptr, map_handle, sync)
    return nothing
end

# ==================================================================
# Picking — M6.B native ray-query pick (CPU-only)
#
# `enqueue_pick_query` registers a pixel-rect pick for the NEXT `step!` on a product;
# that step yields a synthetic `ovrtx_pick_hit` render var (CPU-mapped ONLY — ovrtx
# restriction; payload is tiny, no device path).  `read_pick_hit` decodes its uint32
# params (magic/version/hitCount) + named tensors into `PickHit`s; `resolve_prim_path`
# turns a hit's `primpath_id` into a prim-path string via the path-dictionary vtable.
#
# ABI VERIFIED (REPL vs references/.../test_picking_selection.cpp + helpers.h):
#   - params:  `ovrtx_render_var_param_t` carries VALUE name::ovx_string_t + VALUE
#              dl::DLTensor; each pick param is a scalar kDLUInt/32 (read at p.dl.data).
#   - tensors: `ovrtx_render_var_tensor_t` carries POINTER name::Ptr{ovx_string_t} +
#              POINTER dl::Ptr{DLTensor} — deref BOTH.  dtypes/shapes (n = hitCount):
#       primPath           kDLUInt/64  ndim=1 [n]
#       objectType         kDLUInt/32  ndim=1 [n]
#       geometryInstanceId kDLUInt/64  ndim=1 [n]   ← 64-bit on the wire, NOT 32
#       worldPositionM     kDLFloat/64 ndim=2 [n,3] row-major
#       worldNormal        kDLFloat/32 ndim=2 [n,3] row-major
#   - resolution: vtable get_tokens_from_paths / get_strings_from_tokens are
#     `Ptr{Cvoid}` fn-ptrs called via `@ccall $fp(...)`; out_tokens points INTO the
#     caller's token buffer → that buffer is GC.@preserve'd across BOTH calls.
# ==================================================================

"""
    enqueue_pick_query(r, product, (left,top,right,bottom); flags=UInt32(0)) -> Nothing

Register a pick over the pixel rect (left/top inclusive, right/bottom exclusive, in
RenderProduct pixel coords) for the NEXT `step!` on `product`.  Synchronous (enqueue
+ wait), so `product`'s bytes need only outlive this call.  Read via `read_pick_hit`
after the next `step!`.  Last query before a step wins.
"""
function enqueue_pick_query(r::Renderer, product::AbstractString, rect::NTuple{4,Int};
                            flags::UInt32 = UInt32(0))
    r.alive || error("enqueue_pick_query on a closed Renderer")
    prod_s = String(product)
    GC.@preserve prod_s begin
        desc = Ref(LibOVRTX.ovrtx_pick_query_desc_t(LibOVRTX.ovx_string(prod_s),
                   Int32(rect[1]), Int32(rect[2]), Int32(rect[3]), Int32(rect[4]), flags))
        enqueue_wait(r, "enqueue_pick_query") do
            LibOVRTX.ovrtx_enqueue_pick_query(r.ptr, desc)
        end
    end
    return nothing
end

"""
    PickHit

One viewport pick hit.  `primpath_id` (an `ovx_primpath_t` id) → resolve with
`resolve_prim_path`.  `instance_id` is the wire `geometryInstanceId` (64-bit; 0 for a
single non-instanced geometry).  `world_position`/`normal` are world-space (finite;
`(0,0,0)` for a prim at the origin).
"""
const PickHit = NamedTuple{(:primpath_id, :object_type, :instance_id, :world_position, :normal),
                           Tuple{UInt64, UInt32, UInt64, NTuple{3,Float64}, NTuple{3,Float32}}}

# Find a pick-hit PARAM by name, read its scalar uint32.  Params carry VALUE name +
# VALUE dl (distinct from tensors' POINTER name/dl).
function _pick_param_u32(out::LibOVRTX.ovrtx_render_var_output_t, name::AbstractString)
    for i in 1:out.num_params
        p = unsafe_load(out.params, i)               # ovrtx_render_var_param_t (by value)
        String(p.name) == name && return unsafe_load(Ptr{UInt32}(p.dl.data))
    end
    error("pick-hit param '$name' not found")
end

# Find a pick-hit TENSOR by name, return its deref'd DLTensor.  Tensors carry POINTER
# name + POINTER dl (distinct from params' VALUE name/dl).
function _pick_tensor(out::LibOVRTX.ovrtx_render_var_output_t, name::AbstractString)
    for i in 1:out.num_tensors
        t = unsafe_load(out.tensors, i)              # ovrtx_render_var_tensor_t (by value)
        t.name == C_NULL && continue
        String(unsafe_load(t.name)) == name && return unsafe_load(t.dl)
    end
    error("pick-hit tensor '$name' not found")
end

"""
    read_pick_hit(sr::StepResult) -> Vector{PickHit}

Decode the `ovrtx_pick_hit` var produced by the `step!` that consumed a prior
`enqueue_pick_query`.  One `PickHit` per hit; EMPTY when no pick was enqueued (var
absent) or the magic/version header mismatches.  CPU-only map (ovrtx restriction);
always released (try/finally).
"""
function read_pick_hit(sr::StepResult)::Vector{PickHit}
    sr.r.alive || error("read_pick_hit on a closed Renderer")
    outs = Ref{LibOVRTX.ovrtx_render_product_set_outputs_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_fetch_results(sr.r.ptr, sr.handle, LibOVRTX.OVRTX_TIMEOUT_INFINITE, outs), "fetch_results")
    h = _find_var_opt(outs[], LibOVRTX.OVRTX_RENDER_VAR_PICK_HIT)
    h === nothing && return PickHit[]                    # no pick was enqueued for this step
    mdesc = Ref(LibOVRTX.ovrtx_map_output_description_t(LibOVRTX.OVRTX_MAP_DEVICE_TYPE_CPU, Csize_t(0)))
    ro    = Ref{LibOVRTX.ovrtx_render_var_output_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_map_render_var_output(sr.r.ptr, h, mdesc, LibOVRTX.OVRTX_TIMEOUT_INFINITE, ro),
                   "map_render_var_output(pick_hit)")
    out  = ro[]
    hits = PickHit[]
    try
        magic   = _pick_param_u32(out, "magic")
        version = _pick_param_u32(out, "version")
        (magic == LibOVRTX.OVRTX_PICK_HIT_MAGIC && version == LibOVRTX.OVRTX_PICK_HIT_VERSION) || return PickHit[]
        n = Int(_pick_param_u32(out, "hitCount"))
        n == 0 && return hits
        prim = _pick_tensor(out, "primPath")             # kDLUInt/64  [n]
        otyp = _pick_tensor(out, "objectType")           # kDLUInt/32  [n]
        inst = _pick_tensor(out, "geometryInstanceId")   # kDLUInt/64  [n]
        wpos = _pick_tensor(out, "worldPositionM")       # kDLFloat/64 [n,3] row-major
        wnrm = _pick_tensor(out, "worldNormal")          # kDLFloat/32 [n,3] row-major
        pprim = Ptr{UInt64}(prim.data);  potyp = Ptr{UInt32}(otyp.data);  pinst = Ptr{UInt64}(inst.data)
        pwp   = Ptr{Float64}(wpos.data); pwn   = Ptr{Float32}(wnrm.data)
        for i in 1:n
            base = (i - 1) * 3
            push!(hits, (primpath_id    = unsafe_load(pprim, i),
                         object_type    = unsafe_load(potyp, i),
                         instance_id    = unsafe_load(pinst, i),
                         world_position = (unsafe_load(pwp, base + 1), unsafe_load(pwp, base + 2), unsafe_load(pwp, base + 3)),
                         normal         = (unsafe_load(pwn, base + 1), unsafe_load(pwn, base + 2), unsafe_load(pwn, base + 3))))
        end
    finally
        LibOVRTX.ovrtx_unmap_render_var_output(sr.r.ptr, out.map_handle, LibOVRTX.NOSYNC)
    end
    return hits
end

"""
    PathResolver

Wraps the owning `Renderer` + its `path_dictionary_instance_t` + loaded vtable so
`resolve_prim_path` can call the two path-dictionary fns via raw fn-pointers.  Build
with `path_resolver(r)`.

COMPOSITION-SCOPED: the captured dictionary context is valid only while the stage
composition is unchanged — DISCARD and rebuild after any `add_usd_reference!` /
`remove_usd!` (the Screen cache does exactly this in `path_resolver_for`).  `r` is
retained so it can be alive-checked and `GC.@preserve`d across the raw-pointer ccalls
(the vtable fns dereference the renderer-owned context).
"""
struct PathResolver
    r::Renderer
    pd::Base.RefValue{LibOVRTX.path_dictionary_instance_t}
    vt::LibOVRTX.path_dictionary_vtable_t
end

"""
    path_resolver(r::Renderer) -> PathResolver

Fetch the renderer's path dictionary + load its vtable (one fetch, reused across many
`resolve_prim_path` calls).
"""
function path_resolver(r::Renderer)
    r.alive || error("path_resolver on a closed Renderer")
    pd = Ref{LibOVRTX.path_dictionary_instance_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_get_path_dictionary(r.ptr, pd), "get_path_dictionary")
    return PathResolver(r, pd, unsafe_load(pd[].vtable))
end

"""
    resolve_prim_path(pr::PathResolver, id::UInt64) -> String

Resolve one `ovx_primpath_t` id (a `PickHit.primpath_id`) to its `/A/B/C` path via the
vtable: `get_tokens_from_paths`, then per-token `get_strings_from_tokens`, joined by
`/`.  Returns `""` if `id` is unresolvable.  Reimplements helpers.h
`docs_resolve_primpath`.
"""
function resolve_prim_path(pr::PathResolver, id::UInt64)::String
    r = pr.r
    r.alive || error("resolve_prim_path on a closed Renderer")
    ctx        = Ptr{Cvoid}(pr.pd[].context)
    idref      = Ref(id)
    token_buf  = Vector{UInt64}(undef, 64)
    out_tokens = Ref{Ptr{UInt64}}(C_NULL)
    out_ntok   = Ref{Csize_t}(0)
    out_nproc  = Ref{Csize_t}(0)
    # out_tokens points INTO token_buf; keep it (and idref) preserved across BOTH ccalls.
    # `r` is preserved too so the Renderer owning the dictionary context these raw vtable
    # pointers dereference cannot be finalized mid-ccall.
    GC.@preserve token_buf idref r begin
        res = @ccall $(pr.vt.get_tokens_from_paths)(
            ctx::Ptr{Cvoid}, idref::Ptr{UInt64}, Csize_t(1)::Csize_t,
            pointer(token_buf)::Ptr{UInt64}, Csize_t(64)::Csize_t,
            out_tokens::Ptr{Ptr{UInt64}}, out_ntok::Ptr{Csize_t}, out_nproc::Ptr{Csize_t}
            )::LibOVRTX.ovx_api_result_t
        res.status == LibOVRTX.OVX_API_SUCCESS ||
            error("path get_tokens_from_paths failed: $(String(res.error))")
        out_nproc[] == 0 && return ""                    # id absent from the dictionary
        ntok = Int(out_ntok[]); toks = out_tokens[]
        io = IOBuffer()
        for i in 1:ntok
            s  = Ref{LibOVRTX.ovx_string_t}()
            r2 = @ccall $(pr.vt.get_strings_from_tokens)(
                ctx::Ptr{Cvoid}, (toks + (i - 1) * sizeof(UInt64))::Ptr{UInt64},
                Csize_t(1)::Csize_t, s::Ptr{LibOVRTX.ovx_string_t}
                )::LibOVRTX.ovx_api_result_t
            r2.status == LibOVRTX.OVX_API_SUCCESS ||
                error("path get_strings_from_tokens failed: $(String(r2.error))")
            print(io, "/", String(s[]))
        end
        return String(take!(io))
    end
end

# ------------------------------------------------------------------
# Selection-outline writers (M6.B) — group assignment + per-group styling
# ------------------------------------------------------------------

"""
    set_selection_outline_group!(r, prim_paths, group_ids::Vector{UInt8}) -> Nothing

Write per-prim `omni:selectionOutlineGroup` (uint8) on each of `prim_paths` (parallel
`group_ids`): 0 clears, 1..255 assign a group.  One multi-prim write (kDLUInt/8,
shape [N], semantic NONE).  Group tracking is independent of the renderer's
selection-outline config, but an outline is only DRAWN when that config is enabled.
"""
function set_selection_outline_group!(r::Renderer, prim_paths::Vector{String}, group_ids::Vector{UInt8})
    r.alive || error("set_selection_outline_group! on a closed Renderer")
    n = length(prim_paths)
    n == length(group_ids) ||
        error("set_selection_outline_group!: prim_paths ($n) and group_ids ($(length(group_ids))) length mismatch")
    n == 0 && return nothing
    prim_arr = LibOVRTX.ovx_string_t[LibOVRTX.ovx_string(s) for s in prim_paths]
    name_s   = String(LibOVRTX.OVRTX_ATTR_NAME_SELECTION_OUTLINE_GROUP)
    shape    = Int64[n]
    strides  = Int64[1]
    dtype    = LibOVRTX.DLDataType(UInt8(LibOVRTX.kDLUInt), UInt8(8), UInt16(1))
    GC.@preserve prim_paths prim_arr group_ids name_s shape strides begin
        prim_list   = LibOVRTX.ovrtx_prim_list_t(pointer(prim_arr), Csize_t(n))
        attr_lookup = LibOVRTX.ovx_string_or_token_t(UInt64(0), LibOVRTX.ovx_string(name_s))
        attr_type   = LibOVRTX.ovrtx_attribute_type_t(dtype, false, LibOVRTX.OVRTX_SEMANTIC_NONE)
        bdesc = LibOVRTX.ovrtx_binding_desc_t(prim_list, LibOVRTX.ovx_primpath_list_t(0), attr_lookup,
            attr_type, LibOVRTX.OVRTX_BINDING_PRIM_MODE_EXISTING_ONLY, LibOVRTX.OVRTX_BINDING_FLAG_NONE)
        bdoh = Ref(LibOVRTX.ovrtx_binding_desc_or_handle_t(bdesc, LibOVRTX.ovrtx_attribute_binding_handle_t(0)))
        dl = LibOVRTX.DLTensor(Ptr{Cvoid}(pointer(group_ids)), LibOVRTX.DLDevice(LibOVRTX.kDLCPU, Int32(0)),
            Int32(1), dtype, pointer(shape), pointer(strides), UInt64(0))
        dl_arr = LibOVRTX.DLTensor[dl]
        GC.@preserve dl_arr begin
            ibuf = Ref(LibOVRTX.ovrtx_input_buffer_t(pointer(dl_arr), UInt64(1),
                       Ptr{UInt8}(C_NULL), Csize_t(0), LibOVRTX.NOSYNC, LibOVRTX.NOSYNC))
            enqueue_wait(r, "set_selection_outline_group") do
                LibOVRTX.ovrtx_write_attribute(r.ptr, bdoh, ibuf, LibOVRTX.OVRTX_DATA_ACCESS_SYNC)
            end
        end
    end
    return nothing
end

"""
    set_selection_group_styles!(r, group_ids::Vector{UInt8}, styles) -> Nothing

Set the visual style (outline + fill RGBA in [0,1]) per selection group id (parallel
arrays).  Per-group colors are runtime state; global width + fill mode are
renderer-creation config.  Synchronous.
"""
function set_selection_group_styles!(r::Renderer, group_ids::Vector{UInt8},
                                     styles::Vector{LibOVRTX.ovrtx_selection_group_style_t})
    r.alive || error("set_selection_group_styles! on a closed Renderer")
    length(group_ids) == length(styles) ||
        error("set_selection_group_styles!: group_ids ($(length(group_ids))) and styles ($(length(styles))) length mismatch")
    GC.@preserve group_ids styles begin
        enqueue_wait(r, "set_selection_group_styles") do
            LibOVRTX.ovrtx_set_selection_group_styles(r.ptr, pointer(group_ids),
                pointer(styles), Csize_t(length(group_ids)))
        end
    end
    return nothing
end

# ------------------------------------------------------------------
# render_hdr_to_array — convenience: warmup + map_cpu_f32 + return raw HDR
# ------------------------------------------------------------------

"""
    render_hdr_to_array(r, product; warmup=64, timeout_ns) -> Array{Float32,3}

Run `warmup` RT2 steps on `product`, then map the final `HdrColor` and return a
Float32 `(C=4, W, H)` array (channel-fastest).  HdrColor is linear kDLFloat/16,
converted to Float32.  Caller tonemaps (see `tonemap_frame`).
"""
function render_hdr_to_array(r::Renderer, product::AbstractString;
                             warmup::Int=64, timeout_ns::UInt64 = _TIMEOUT_INFINITE_NS)
    for s in 1:(warmup - 1)
        sr = step!(r, product; timeout_ns); close(sr)
    end
    sr = step!(r, product; timeout_ns)
    pixels, W, H = try
        map_cpu_f32(sr, "HdrColor")
    finally
        close(sr)
    end
    return pixels  # [C, W, H] Float32
end

# ------------------------------------------------------------------
# render_to_matrix — convenience: warmup + map + reshape
# ------------------------------------------------------------------

"""
    render_to_matrix(r, product; warmup=64, timeout_ns) -> Matrix{RGBA{N0f8}}

Run `warmup` RT2 steps (RT2 needs many samples to converge), then map the final
`LdrColor` to a `Matrix{RGBA{N0f8}}` (H, W).  Warmup frames are freed immediately.
Top-left origin (row 1 = top), NO y-flip — verified by test/m1_orientation_test.jl.
"""
function render_to_matrix(r::Renderer, product::AbstractString;
                         warmup::Int=64, timeout_ns::UInt64 = _TIMEOUT_INFINITE_NS)
    for s in 1:(warmup - 1)
        sr = step!(r, product; timeout_ns); close(sr)
    end
    sr = step!(r, product; timeout_ns)
    img, _W, _H = try
        map_cpu(sr, "LdrColor")     # single-pass: already the (H, W) RGBA matrix
    finally
        close(sr)
    end
    return img
end

# ------------------------------------------------------------------
# reset! — restart RT2 accumulation (call after any geometry/camera change)
# ------------------------------------------------------------------

"""
    reset!(r::Renderer; time=0.0)

Enqueue + wait an RT2 accumulation reset.  Call after any geometry/camera change so
the path-tracer restarts fresh.
"""
# Diagnostic hook: fired (with no args) on every `reset!`.  `nothing` (default) → zero overhead;
# a test sets it to a counter to assert accumulate-across-frames suppresses per-frame resets.
# Mirrors compute.jl's `_PUSH_OBSERVER`.
const _RESET_OBSERVER = Ref{Any}(nothing)

function reset!(r::Renderer; time::Float64=0.0)
    enqueue_wait(r, "reset") do
        LibOVRTX.ovrtx_reset(r.ptr, time)
    end
    ob = _RESET_OBSERVER[]
    ob === nothing || ob()
    return nothing
end

# ------------------------------------------------------------------
# _write_attribute! — shared private helper for FFI attribute writes
# ------------------------------------------------------------------

# Write one attribute (fixed-size or array) to `prim` via a DLTensor over `data`.
# `data` must be a contiguous OWNED Vector backing the DLTensor; the caller
# preprocesses (transpose/flatten for xform, dtype inference for arrays).
function _write_attribute!(r::Renderer, prim::AbstractString, attr_name::AbstractString,
                           dtype::LibOVRTX.DLDataType, is_array::Bool, semantic,
                           data::AbstractVector, shape::Vector{Int64})
    strides = Int64[1]
    prim_s   = String(prim)
    prim_ovx = LibOVRTX.ovx_string(prim_s)
    prim_arr = LibOVRTX.ovx_string_t[prim_ovx]
    name_s   = String(attr_name)
    name_ovx = LibOVRTX.ovx_string(name_s)
    GC.@preserve prim_s prim_arr name_s data shape strides begin
        prim_list   = LibOVRTX.ovrtx_prim_list_t(pointer(prim_arr), Csize_t(1))
        attr_lookup = LibOVRTX.ovx_string_or_token_t(UInt64(0), name_ovx)
        attr_type   = LibOVRTX.ovrtx_attribute_type_t(dtype, is_array, semantic)
        bdesc = LibOVRTX.ovrtx_binding_desc_t(
            prim_list,
            LibOVRTX.ovx_primpath_list_t(0),
            attr_lookup,
            attr_type,
            LibOVRTX.OVRTX_BINDING_PRIM_MODE_EXISTING_ONLY,
            LibOVRTX.OVRTX_BINDING_FLAG_NONE,
        )
        bdoh  = Ref(LibOVRTX.ovrtx_binding_desc_or_handle_t(bdesc, LibOVRTX.ovrtx_attribute_binding_handle_t(0)))
        dl = LibOVRTX.DLTensor(
            Ptr{Cvoid}(pointer(data)),
            LibOVRTX.DLDevice(LibOVRTX.kDLCPU, Int32(0)),
            Int32(1),
            dtype,
            pointer(shape),
            pointer(strides),
            UInt64(0),
        )
        dl_arr = LibOVRTX.DLTensor[dl]
        GC.@preserve dl_arr begin
            ibuf = Ref(LibOVRTX.ovrtx_input_buffer_t(
                pointer(dl_arr),
                UInt64(1),
                Ptr{UInt8}(C_NULL),
                Csize_t(0),
                LibOVRTX.NOSYNC,
                LibOVRTX.NOSYNC,
            ))
            enqueue_wait(r, "write_attribute($attr_name)") do
                LibOVRTX.ovrtx_write_attribute(r.ptr, bdoh, ibuf, LibOVRTX.OVRTX_DATA_ACCESS_SYNC)
            end
        end
    end
    return nothing
end

# ------------------------------------------------------------------
# write_xform! — write a 4×4 transform to a USD prim (hot-path)
# ------------------------------------------------------------------

"""
    write_xform!(r::Renderer, prim, mat::AbstractMatrix{Float64})

Write a 4×4 row-major transform to `prim`'s `omni:xform` (translation in the last
row).  Reimplements the `ovrtx_set_xform_mat` inline; `mat` + `prim` GC.@preserve'd
across the write + wait.
"""
function write_xform!(r::Renderer, prim::AbstractString, mat::AbstractMatrix{Float64})
    @assert size(mat) == (4, 4) "write_xform! requires a 4×4 matrix, got $(size(mat))"

    # Julia is column-major; transpose to row-major (16 flat Float64) for ovrtx.
    mat_rowmajor = vec(collect(mat'))

    # dtype kDLFloat/64/lanes=16 → the 4×4 as a single element
    dtype = LibOVRTX.DLDataType(UInt8(LibOVRTX.kDLFloat), UInt8(64), UInt16(16))

    _write_attribute!(r, prim, "omni:xform", dtype, false, LibOVRTX.OVRTX_SEMANTIC_XFORM_MAT4x4, mat_rowmajor, Int64[1])
    return nothing
end

# ------------------------------------------------------------------
# bind_material! — write the `material:binding` relationship (M3.1)
# ------------------------------------------------------------------

"""
    bind_material!(r::Renderer, geom_prim, material_prim)

Write the USD `material:binding` relationship on `geom_prim` → `material_prim` (an
absolute path, e.g. `/World/Looks/Mat_42`, which MUST already exist on the stage).
Mirrors `ovrtx_set_path_attributes`: one `ovx_string_t` path element (128-bit
ptr+len), dtype {kDLUInt,128,1}, is_array=true, shape [1], OVRTX_SEMANTIC_PATH_STRING.
`material_prim` is GC.@preserve'd across the call.  Call `OV.reset!` after.
"""
function bind_material!(r::Renderer, geom_prim::AbstractString, material_prim::AbstractString)
    r.alive || error("bind_material! on a closed Renderer")
    mat_s = String(material_prim)
    dtype = LibOVRTX.DLDataType(UInt8(LibOVRTX.kDLUInt), UInt8(128), UInt16(1))
    GC.@preserve mat_s begin
        data = LibOVRTX.ovx_string_t[LibOVRTX.ovx_string(mat_s)]
        _write_attribute!(r, geom_prim, "material:binding", dtype, true,
                          LibOVRTX.OVRTX_SEMANTIC_PATH_STRING, data, Int64[1])
    end
    return nothing
end

# ------------------------------------------------------------------
# write_shader_input! — live re-write of an OmniPBR shader input (M3.4)
# ------------------------------------------------------------------

"""
    write_shader_input!(r::Renderer, shader_prim::AbstractString, name::AbstractString,
                        value::Union{Float32,NTuple{3,Float32}})

Live re-write of OmniPBR `inputs:<name>` on the OPEN stage `shader_prim` (e.g.
`/World/Looks/Mat_<id>/Shader`) — the M3.4 material-edit path.
- `Float32` scalar → dtype {kDLFloat,32,1}, data=[v].
- `color3f` `NTuple{3,Float32}` → dtype {kDLFloat,32,3} (3 lanes), data=[r,g,b].
Both is_array=false, shape=[1]; the backing Float32 vector is GC.@preserve'd.  Does
NOT reset — caller restarts RT2 accumulation (one `OV.reset!` per changed frame).
"""
function write_shader_input!(r::Renderer, shader_prim::AbstractString, name::AbstractString,
                             value::Union{Float32,NTuple{3,Float32}})
    r.alive || error("write_shader_input! on a closed Renderer")
    attr_name = "inputs:" * String(name)
    if value isa Float32
        dtype = LibOVRTX.DLDataType(UInt8(LibOVRTX.kDLFloat), UInt8(32), UInt16(1))
        data  = Float32[value]
    else
        dtype = LibOVRTX.DLDataType(UInt8(LibOVRTX.kDLFloat), UInt8(32), UInt16(3))
        data  = Float32[value[1], value[2], value[3]]
    end
    GC.@preserve data begin
        _write_attribute!(r, shader_prim, attr_name, dtype, false, LibOVRTX.OVRTX_SEMANTIC_NONE,
                          data, Int64[1])
    end
    return nothing
end

# ------------------------------------------------------------------
# write_array_attribute! — write an array attribute (e.g. points)
# ------------------------------------------------------------------

# Float32-aggregate lane count: Point3f/Vec3f → 3, Point2f → 2, etc.; 0 if ET is not
# a fixed-size Float32 aggregate.  Structural (`eltype(ET) === Float32`) → no
# GeometryBasics/StaticArrays import needed.
function _f32_lanes(::Type{ET}) where {ET}
    (isconcretetype(ET) && isbitstype(ET) && eltype(ET) === Float32) || return 0
    n, r = divrem(sizeof(ET), sizeof(Float32))
    return r == 0 ? n : 0
end

"""
    write_array_attribute!(r::Renderer, prim, name, arr::AbstractArray)

Write an array attribute (e.g. `points`) to `prim`; element DLDataType inferred from
`eltype(arr)`.  Fixed-size Float32 aggregates (`Point3f`/`Vec3f`/…) go zero-copy as a
multi-lane kDLFloat/32 tensor: `reinterpret(Float32,·)` flattens components, lanes =
sizeof(eltype)/4, shape = [length(arr)] (one element per point).  Array + prim + name
GC.@preserve'd across the write + wait.
"""
function write_array_attribute!(r::Renderer, prim::AbstractString,
                                name::AbstractString, arr::AbstractArray)
    ET    = eltype(arr)
    lanes = _f32_lanes(ET)
    if lanes >= 2
        # Float32 aggregate: reinterpret components to a flat owned Float32 vector;
        # one tensor element per aggregate (multi-lane).
        src   = arr isa Vector ? arr : collect(arr)
        data  = collect(reinterpret(Float32, src))   # length = lanes * length(arr)
        dtype = LibOVRTX.DLDataType(UInt8(LibOVRTX.kDLFloat), UInt8(32), UInt16(lanes))
        _write_attribute!(r, prim, name, dtype, true, LibOVRTX.OVRTX_SEMANTIC_NONE, data, Int64[length(arr)])
        return nothing
    end

    # Scalar element types: one lane each.
    dtype = if ET === Float32
        LibOVRTX.DLDataType(UInt8(LibOVRTX.kDLFloat), UInt8(32), UInt16(1))
    elseif ET === Float64
        LibOVRTX.DLDataType(UInt8(LibOVRTX.kDLFloat), UInt8(64), UInt16(1))
    elseif ET === Int32
        LibOVRTX.DLDataType(UInt8(LibOVRTX.kDLInt), UInt8(32), UInt16(1))
    elseif ET === Int64
        LibOVRTX.DLDataType(UInt8(LibOVRTX.kDLInt), UInt8(64), UInt16(1))
    elseif ET === UInt32
        LibOVRTX.DLDataType(UInt8(LibOVRTX.kDLUInt), UInt8(32), UInt16(1))
    elseif ET === UInt64
        LibOVRTX.DLDataType(UInt8(LibOVRTX.kDLUInt), UInt8(64), UInt16(1))
    else
        error("write_array_attribute!: unsupported element type $ET")
    end

    data = collect(arr)  # ensure contiguous, owned copy

    _write_attribute!(r, prim, name, dtype, true, LibOVRTX.OVRTX_SEMANTIC_NONE, data, Int64[length(data)])
    return nothing
end


# ------------------------------------------------------------------
# add_usd_reference! / remove_usd! — per-plot USD layer management
# ------------------------------------------------------------------

"""
    add_usd_reference!(r::Renderer, usda, prim_path) -> ovrtx_usd_handle_t

Add a USD layer (in-memory USDA string) to the running stage under `prim_path`;
return an opaque handle for `remove_usd!`.  Both strings are converted to owned
`String`s and GC.@preserve'd across the call + wait (the `ovx_string_t`s reference
Julia heap memory).
"""
function add_usd_reference!(r::Renderer, usda::AbstractString, prim_path::AbstractString)
    r.alive || error("add_usd_reference! on a closed Renderer")
    layer_s = String(usda)
    path_s  = String(prim_path)
    h = Ref{LibOVRTX.ovrtx_usd_handle_t}(0)
    GC.@preserve layer_s path_s begin
        enqueue_wait(r, "add_usd_reference") do
            LibOVRTX.ovrtx_add_usd_reference_from_string(
                r.ptr, LibOVRTX.ovx_string(layer_s), LibOVRTX.ovx_string(path_s), h)
        end
    end
    return h[]
end

"""
    remove_usd!(r::Renderer, handle::ovrtx_usd_handle_t) -> Nothing

Remove the USD layer previously added via `add_usd_reference!`.
"""
function remove_usd!(r::Renderer, handle::LibOVRTX.ovrtx_usd_handle_t)
    enqueue_wait(r, "remove_usd") do
        LibOVRTX.ovrtx_remove_usd(r.ptr, handle)
    end
    return nothing
end

# ==================================================================
# Binding — persistent attribute bindings (M2.4 hot path)
#
# `ovrtx_create_attribute_binding` locks prim + attr name + element type so per-frame
# writes skip rebuilding the descriptor.  Two tiers (both validated by the M2.4 spike):
#   - fixed-size (`omni:xform`): map_binding/unmap! → ZERO-COPY write into ovrtx's
#     internal buffer (created with OVRTX_BINDING_FLAG_OPTIMIZE).
#   - array (`points`): write_binding! copies a fresh tensor through the handle.
#
# GC: the prim path + attr name `String`s are retained on the struct (rooted for the
# binding's lifetime); every FFI buffer/desc is GC.@preserve'd across the ccall + wait.
# ==================================================================

"""
    Binding

A persistent ovrtx attribute binding (handle from `ovrtx_create_attribute_binding`),
reused across frames; release with `destroy!` (finalizer is a backstop).  `map_handle`
is non-zero only while a `map_binding`/`unmap!` pair is outstanding.
"""
mutable struct Binding
    r::Renderer
    handle::LibOVRTX.ovrtx_attribute_binding_handle_t
    prim::String
    attr_name::String
    dtype::LibOVRTX.DLDataType
    is_array::Bool
    semantic::LibOVRTX.ovrtx_attribute_semantic_t
    map_handle::LibOVRTX.ovrtx_map_handle_t   # non-zero while mapped
    alive::Bool
end

# A zeroed `ovrtx_binding_desc_t`, IGNORED whenever binding_handle != 0 (header: "If
# binding_handle is non-zero it will be used, otherwise binding_desc") — so every
# map/write through a handle passes this.
_empty_binding_desc() = LibOVRTX.ovrtx_binding_desc_t(
    LibOVRTX.ovrtx_prim_list_t(Ptr{LibOVRTX.ovx_string_t}(C_NULL), Csize_t(0)),
    LibOVRTX.ovx_primpath_list_t(0),
    LibOVRTX.ovx_string_or_token_t(UInt64(0), LibOVRTX.ovx_string_t(reinterpret(Cstring, C_NULL), Csize_t(0))),
    LibOVRTX.ovrtx_attribute_type_t(LibOVRTX.DLDataType(UInt8(0), UInt8(0), UInt16(0)), false, LibOVRTX.OVRTX_SEMANTIC_NONE),
    LibOVRTX.OVRTX_BINDING_PRIM_MODE_EXISTING_ONLY,
    LibOVRTX.OVRTX_BINDING_FLAG_NONE,
)

_desc_or_handle(b::Binding) = LibOVRTX.ovrtx_binding_desc_or_handle_t(_empty_binding_desc(), b.handle)

"""
    create_binding(r, prim, name, dtype; array=false, semantic=OVRTX_SEMANTIC_NONE,
                   optimize=false) -> Binding

Create a persistent binding locking `prim`'s `name` attribute to `dtype` (lane-based:
{kDLFloat,64,16} for a 4×4 double, {kDLFloat,32,3} for point3f[]).  `array=true` binds
a variable-length array; `optimize=true` sets OVRTX_BINDING_FLAG_OPTIMIZE for the main
hot binding.  Synchronous; prim strings GC.@preserve'd.
"""
function create_binding(r::Renderer, prim::AbstractString, name::AbstractString,
                        dtype::LibOVRTX.DLDataType; array::Bool=false,
                        semantic=LibOVRTX.OVRTX_SEMANTIC_NONE, optimize::Bool=false)
    r.alive || error("create_binding on a closed Renderer")
    prim_s   = String(prim)
    name_s   = String(name)
    prim_arr = LibOVRTX.ovx_string_t[LibOVRTX.ovx_string(prim_s)]
    handle   = Ref{LibOVRTX.ovrtx_attribute_binding_handle_t}(0)
    flags    = optimize ? LibOVRTX.OVRTX_BINDING_FLAG_OPTIMIZE : LibOVRTX.OVRTX_BINDING_FLAG_NONE
    GC.@preserve prim_s prim_arr name_s begin
        prim_list   = LibOVRTX.ovrtx_prim_list_t(pointer(prim_arr), Csize_t(1))
        attr_lookup = LibOVRTX.ovx_string_or_token_t(UInt64(0), LibOVRTX.ovx_string(name_s))
        attr_type   = LibOVRTX.ovrtx_attribute_type_t(dtype, array, semantic)
        bdesc = LibOVRTX.ovrtx_binding_desc_t(prim_list, LibOVRTX.ovx_primpath_list_t(0),
            attr_lookup, attr_type, LibOVRTX.OVRTX_BINDING_PRIM_MODE_EXISTING_ONLY, flags)
        bref = Ref(bdesc)
        GC.@preserve bref begin
            enqueue_wait(r, "create_attribute_binding($name)") do
                LibOVRTX.ovrtx_create_attribute_binding(r.ptr, bref, handle)
            end
        end
    end
    b = Binding(r, handle[], prim_s, name_s, dtype, array, semantic,
                LibOVRTX.ovrtx_map_handle_t(0), true)
    finalizer(b -> destroy!(b; from_finalizer=true), b)
    return b
end

"""
    map_binding(b::Binding; device=kDLCPU, device_id=0) -> Ptr{Cvoid}

Map the binding's internal buffer for a ZERO-COPY write; return the data pointer
(valid ONLY until `unmap!`).  Stashes the map handle on `b`.  Synchronous.
"""
function map_binding(b::Binding; device=LibOVRTX.kDLCPU, device_id::Integer=0)
    b.alive || error("map_binding on a destroyed Binding")
    b.map_handle == 0 || error("map_binding: already mapped (call unmap! first)")
    bdoh  = Ref(_desc_or_handle(b))
    mdesc = LibOVRTX.ovrtx_mapping_desc_t(Int32(device), Int32(device_id))
    out   = Ref{LibOVRTX.ovrtx_attribute_mapping_t}()
    GC.@preserve bdoh begin
        LibOVRTX.check(LibOVRTX.ovrtx_map_attribute(b.r.ptr, bdoh, mdesc, out), "map_attribute($(b.attr_name))")
    end
    m = out[]
    b.map_handle = m.map_handle
    return Ptr{Cvoid}(m.dl.data)
end

# Finalizer-path teardown must not wedge GC and must never let an error escape (a throw would
# abort the finalizer before we clear `alive`, and finalizers must not throw).  So the
# finalizer path (from_finalizer=true) waits under this FINITE bound instead of INFINITE and
# swallows any failure — 5 s is generous for a normal unmap/destroy yet keeps GC bounded.
const _FINALIZER_TIMEOUT_NS = UInt64(5_000_000_000)

# Bindings whose finalizer-path teardown swallowed an error (a leaked GPU handle — the
# renderer close frees the whole pool regardless).  A bare counter, NOT a @warn: logging from
# a finalizer would build a String and do I/O, both of which task-switch (illegal there).
const _BINDING_FINALIZER_LEAKS = Ref{Int}(0)

"""
    binding_finalizer_leak_count() -> Int

How many `Binding`s had their GC-finalizer teardown swallow an ovrtx error (handle leaked,
reclaimed at renderer close).  Observability for the otherwise-silent finalizer path.
"""
binding_finalizer_leak_count() = _BINDING_FINALIZER_LEAKS[]

"""
    unmap!(b::Binding; from_finalizer::Bool=false)

Commit + release a `map_binding` mapping (async unmap + wait).  No-op when not mapped or the
Renderer is closed.  `from_finalizer=true` bounds the wait (`_FINALIZER_TIMEOUT_NS`) for the
GC-finalizer teardown path, where the caller (`destroy!`) swallows any resulting error.
"""
function unmap!(b::Binding; from_finalizer::Bool=false)
    b.map_handle == 0 && return nothing
    if b.r.alive
        timeout_ns = from_finalizer ? _FINALIZER_TIMEOUT_NS : _TIMEOUT_INFINITE_NS
        enqueue_wait(b.r, "unmap_attribute"; timeout_ns) do
            LibOVRTX.ovrtx_unmap_attribute(b.r.ptr, b.map_handle, LibOVRTX.NOSYNC)
        end
    end
    b.map_handle = LibOVRTX.ovrtx_map_handle_t(0)
    return nothing
end

"""
    write_mapped_xform!(b::Binding, mat::AbstractMatrix{Float64})

ZERO-COPY 4×4 transform write through the MAPPED fixed-size binding `b`: map → store
16 row-major doubles → unmap.  `mat` is USD row-vector form (translation last row),
identical to `write_xform!`; round-trips byte-for-byte.
"""
function write_mapped_xform!(b::Binding, mat::AbstractMatrix{Float64})
    @assert size(mat) == (4, 4) "write_mapped_xform! requires a 4×4 matrix, got $(size(mat))"
    mat_rowmajor = vec(collect(mat'))             # 16 row-major Float64
    ptr = Ptr{Cdouble}(map_binding(b))
    try
        GC.@preserve mat_rowmajor begin
            @inbounds for i in 1:16
                unsafe_store!(ptr, mat_rowmajor[i], i)
            end
        end
    finally
        unmap!(b)
    end
    return nothing
end

"""
    write_binding!(b::Binding, data::AbstractVector, shape::Vector{Int64})

Write `data` through the persistent binding handle (array tier).  `data` is a
contiguous, owned vector matching `b.dtype` (e.g. flattened Float32 for point3f[]);
`shape` is the per-prim element count ([npoints]).  GC.@preserve discipline mirrors
`_write_attribute!`.
"""
function write_binding!(b::Binding, data::AbstractVector, shape::Vector{Int64})
    b.alive || error("write_binding! on a destroyed Binding")
    strides = Int64[1]
    bdoh    = Ref(_desc_or_handle(b))
    GC.@preserve data shape strides bdoh begin
        dl = LibOVRTX.DLTensor(Ptr{Cvoid}(pointer(data)), LibOVRTX.DLDevice(LibOVRTX.kDLCPU, Int32(0)),
                        Int32(1), b.dtype, pointer(shape), pointer(strides), UInt64(0))
        dl_arr = LibOVRTX.DLTensor[dl]
        GC.@preserve dl_arr begin
            ibuf = Ref(LibOVRTX.ovrtx_input_buffer_t(pointer(dl_arr), UInt64(1),
                       Ptr{UInt8}(C_NULL), Csize_t(0), LibOVRTX.NOSYNC, LibOVRTX.NOSYNC))
            enqueue_wait(b.r, "write_attribute(binding:$(b.attr_name))") do
                LibOVRTX.ovrtx_write_attribute(b.r.ptr, bdoh, ibuf, LibOVRTX.OVRTX_DATA_ACCESS_SYNC)
            end
        end
    end
    return nothing
end

# The ovrtx side of teardown: unmap-if-mapped, then destroy the handle, waiting under
# `timeout_ns`.  NO flag-clearing here — `destroy!` owns `alive`/`map_handle`, so its throwing
# and swallowing paths share this.  No-op once the Renderer is closed (GPU pool already freed).
function _destroy_binding_ovrtx!(b::Binding, timeout_ns::UInt64, from_finalizer::Bool)
    b.r.alive || return nothing
    b.map_handle == 0 || unmap!(b; from_finalizer)
    enqueue_wait(b.r, "destroy_attribute_binding"; timeout_ns) do
        LibOVRTX.ovrtx_destroy_attribute_binding(b.r.ptr, b.handle)
    end
    return nothing
end

"""
    destroy!(b::Binding; from_finalizer::Bool=false)

Release the persistent binding (`ovrtx_destroy_attribute_binding`).  Idempotent; unmaps
first if mapped; no-op once the Renderer is closed.

Explicit calls wait INFINITE and propagate ovrtx errors.  The GC finalizer registers
`from_finalizer=true`: that path waits under `_FINALIZER_TIMEOUT_NS` and swallows any error
AFTER marking the binding dead (`alive=false`, `map_handle=0`) — leaking one handle beats
wedging GC on a stuck queue or throwing out of a finalizer, and renderer close frees the
pool.  Each swallowed error bumps `binding_finalizer_leak_count()`.
"""
function destroy!(b::Binding; from_finalizer::Bool=false)
    b.alive || return nothing
    if from_finalizer
        try
            _destroy_binding_ovrtx!(b, _FINALIZER_TIMEOUT_NS, true)
        catch
            _BINDING_FINALIZER_LEAKS[] += 1
        finally
            b.map_handle = LibOVRTX.ovrtx_map_handle_t(0)
            b.alive = false
        end
    else
        _destroy_binding_ovrtx!(b, _TIMEOUT_INFINITE_NS, false)
        b.map_handle = LibOVRTX.ovrtx_map_handle_t(0)
        b.alive = false
    end
    return nothing
end

end # module OV
