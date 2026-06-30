module OV
import ..LibOVRTX
include("signals.jl")
using .SignalGuard: with_restored_signals

include("dlpack.jl")

# Named constant so the three timeout kwarg defaults share a single source.
const _TIMEOUT_INFINITE_NS = LibOVRTX.OVRTX_TIMEOUT_INFINITE.time_out_ns

# ------------------------------------------------------------------
# Renderer — GC-aware wrapper around ovrtx_renderer_t
# ------------------------------------------------------------------

mutable struct Renderer
    ptr::Ptr{LibOVRTX.ovrtx_renderer_t}
    alive::Bool
    function Renderer()
        cfg  = Ref(LibOVRTX.ovrtx_config_t(Ptr{LibOVRTX.ovrtx_config_entry_t}(C_NULL), Csize_t(0)))
        rref = Ref{Ptr{LibOVRTX.ovrtx_renderer_t}}(C_NULL)
        with_restored_signals() do
            LibOVRTX.check(LibOVRTX.ovrtx_create_renderer(cfg, rref), "create_renderer")
        end
        r = new(rref[], true)
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

function enqueue_wait(r::Renderer, enq, op::AbstractString;
                     timeout_ns::UInt64 = _TIMEOUT_INFINITE_NS)
    r.alive || error("enqueue_wait called on a closed Renderer")
    LibOVRTX.check(enq, op)
    wr = Ref{LibOVRTX.ovrtx_op_wait_result_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_wait_op(r.ptr, enq.op_index, LibOVRTX.ovrtx_timeout_t(timeout_ns), wr), op * ":wait")
    return wr[]
end

# ------------------------------------------------------------------
# open_usd! / open_usd_string! — load a stage (sync)
# ------------------------------------------------------------------

"""
    open_usd!(r::Renderer, path::AbstractString)

Open a USD stage from the file at `path`.  Synchronous: enqueues then waits.
The `path` string is preserved across the ccall and the wait via `GC.@preserve`.
"""
function open_usd!(r::Renderer, path::AbstractString)
    GC.@preserve path begin
        enqueue_wait(r, LibOVRTX.ovrtx_open_usd_from_file(r.ptr, LibOVRTX.ovx_string(path)), "open_usd")
    end
    return nothing
end

"""
    open_usd_string!(r::Renderer, usda::AbstractString)

Open a USD stage from an in-memory USDA string.  Synchronous.
"""
function open_usd_string!(r::Renderer, usda::AbstractString)
    GC.@preserve usda begin
        enqueue_wait(r, LibOVRTX.ovrtx_open_usd_from_string(r.ptr, LibOVRTX.ovx_string(usda)), "open_usd_string")
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

Enqueue and wait for one RT2 render step for the given render product path.
Both the backing `ovx_string_t` array and the product `String` are preserved
across the ccall and the wait.  `timeout_ns` sets the maximum wait for the
async step to complete; it defaults to the infinite-wait constant (preserving
back-compat) and is set to a bounded value by the M5 interactive camera loop
to prevent hanging on a slow or stalled render step.

Returns a `StepResult`; the caller is responsible for closing it (or letting
the finalizer run).
"""
function step!(r::Renderer, product::AbstractString;
              dt::Float64=1/60, timeout_ns::UInt64 = _TIMEOUT_INFINITE_NS)
    rp = LibOVRTX.ovx_string_t[ LibOVRTX.ovx_string(product) ]
    GC.@preserve product rp begin
        set = LibOVRTX.ovrtx_render_product_set_t(pointer(rp), Csize_t(1))
        h = Ref{LibOVRTX.ovrtx_step_result_handle_t}(0)
        enqueue_wait(r, LibOVRTX.ovrtx_step(r.ptr, set, dt, h), "step"; timeout_ns)
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

Walk outputs → output_frames → output_render_vars and return the output handle
whose render_var_name matches `name` (e.g. "LdrColor"), or `nothing` if absent.
Used by `read_pick_hit`, where a missing `ovrtx_pick_hit` var (no pick enqueued)
is an expected, non-error case.
"""
function _find_var_opt(outs::LibOVRTX.ovrtx_render_product_set_outputs_t, name::AbstractString)
    for i in 1:outs.output_count
        po = unsafe_load(outs.outputs, i)            # ovrtx_render_product_output_t
        for f in 1:po.output_frame_count
            fr = unsafe_load(po.output_frames, f)    # ovrtx_render_product_frame_output_t
            for v in 1:fr.render_var_count
                rv = unsafe_load(fr.output_render_vars, v)  # ovrtx_render_product_render_var_output_t
                String(rv.render_var_name) == name && return rv.output_handle
            end
        end
    end
    return nothing
end

"""
    _find_var(outs::ovrtx_render_product_set_outputs_t, name) -> ovrtx_render_var_output_handle_t

Like `_find_var_opt` but throws if `name` is not found (the contract `map_cpu`,
`map_cpu_f32`, and `map_cuda` rely on — LdrColor/HdrColor are always present).
"""
function _find_var(outs::LibOVRTX.ovrtx_render_product_set_outputs_t, name::AbstractString)
    h = _find_var_opt(outs, name)
    h === nothing && error("render var '$name' not found in step outputs")
    return h
end

# ------------------------------------------------------------------
# map_cpu — fetch results, find LdrColor, map to CPU, copy, unmap
# ------------------------------------------------------------------

"""
    map_cpu(sr::StepResult, name="LdrColor") -> (pixels::Array{UInt8,3}, W::Int, H::Int)

Fetch the step results from `sr`, find the render var `name`, map it to CPU
memory, **copy** the pixels (mandatory — mapped memory is invalid after unmap),
unmap, and return `(pixels, W, H)`.

`pixels` is a 3-D `Array{UInt8}` with layout `[C=4, W, H]` (channel-fastest).
"""
function map_cpu(sr::StepResult, name::AbstractString="LdrColor")
    sr.r.alive || error("map_cpu: the StepResult's Renderer is already closed")
    # 1. fetch
    outs = Ref{LibOVRTX.ovrtx_render_product_set_outputs_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_fetch_results(sr.r.ptr, sr.handle, LibOVRTX.OVRTX_TIMEOUT_INFINITE, outs), "fetch_results")

    # 2. walk the output tree
    h = _find_var(outs[], name)

    # 3. map to CPU
    mdesc = Ref(LibOVRTX.ovrtx_map_output_description_t(LibOVRTX.OVRTX_MAP_DEVICE_TYPE_CPU, Csize_t(0)))
    ro    = Ref{LibOVRTX.ovrtx_render_var_output_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_map_render_var_output(sr.r.ptr, h, mdesc, LibOVRTX.OVRTX_TIMEOUT_INFINITE, ro), "map_render_var_output")

    # 4. decode the DLTensor (shape = [H, W, C], dtype = kDLUInt/8/1)
    t0  = unsafe_load(ro[].tensors, 1)          # ovrtx_render_var_tensor_t
    dlt = unsafe_load(t0.dl)                    # DLTensor (Ptr{DLTensor} → value)
    H   = Int(unsafe_load(dlt.shape, 1))
    W   = Int(unsafe_load(dlt.shape, 2))
    C   = Int(unsafe_load(dlt.shape, 3))

    # 5. wrap as non-owning view [C, W, H], then COPY before unmap
    raw    = unsafe_wrap(Array, Ptr{UInt8}(dlt.data), (C, W, H); own=false)
    pixels = copy(raw)   # own the data BEFORE we unmap

    # 6. unmap (NOSYNC — CPU, no CUDA stream needed)
    LibOVRTX.ovrtx_unmap_render_var_output(sr.r.ptr, ro[].map_handle, LibOVRTX.NOSYNC)

    return (pixels, W, H)
end

# ------------------------------------------------------------------
# map_cpu_f32 — like map_cpu but returns a Float32 [C,W,H] array.
# HdrColor is kDLFloat/16 (verified: dtype.code=kDLFloat, dtype.bits=16);
# we read via Ptr{Float16} and convert to Float32 for the tonemap path.
# ------------------------------------------------------------------

"""
    map_cpu_f32(sr::StepResult, name) -> (pixels::Array{Float32,3}, W::Int, H::Int)

Like `map_cpu` but returns a `Float32` array suitable for the HDR tonemap path.
`HdrColor` is kDLFloat/16 on the wire; we read via `Ptr{Float16}` and
convert to Float32 before returning (copy happens before unmap).

`pixels` is a 3-D `Array{Float32}` with layout `[C=4, W, H]` (channel-fastest).
"""
function map_cpu_f32(sr::StepResult, name::AbstractString)
    sr.r.alive || error("map_cpu_f32 on a closed Renderer")
    # 1. fetch
    outs = Ref{LibOVRTX.ovrtx_render_product_set_outputs_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_fetch_results(sr.r.ptr, sr.handle, LibOVRTX.OVRTX_TIMEOUT_INFINITE, outs), "fetch_results")

    # 2. walk the output tree
    h = _find_var(outs[], name)

    # 3. map to CPU
    mdesc = Ref(LibOVRTX.ovrtx_map_output_description_t(LibOVRTX.OVRTX_MAP_DEVICE_TYPE_CPU, Csize_t(0)))
    ro    = Ref{LibOVRTX.ovrtx_render_var_output_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_map_render_var_output(sr.r.ptr, h, mdesc, LibOVRTX.OVRTX_TIMEOUT_INFINITE, ro), "map_render_var_output")

    # 4. decode the DLTensor (shape = [H, W, C], dtype = kDLFloat/16/1 for HdrColor)
    t0  = unsafe_load(ro[].tensors, 1)
    dlt = unsafe_load(t0.dl)
    H   = Int(unsafe_load(dlt.shape, 1))
    W   = Int(unsafe_load(dlt.shape, 2))
    C   = Int(unsafe_load(dlt.shape, 3))

    # 5. wrap as non-owning Float16 view [C, W, H]; convert to Float32 and copy before unmap.
    raw16  = unsafe_wrap(Array, Ptr{Float16}(dlt.data), (C, W, H); own=false)
    pixels = Float32.(raw16)   # own the data BEFORE we unmap; Float16 → Float32

    # 6. unmap (NOSYNC — CPU, no CUDA stream needed)
    LibOVRTX.ovrtx_unmap_render_var_output(sr.r.ptr, ro[].map_handle, LibOVRTX.NOSYNC)

    return (pixels, W, H)
end

# ------------------------------------------------------------------
# map_cuda / unmap_cuda — map a render var as LINEAR CUDA device memory (M6.A)
#
# Unlike map_cpu (which copies then unmaps inside the call), map_cuda MAPS and
# RETURNS the live device pointer + handles WITHOUT copying or unmapping.  The
# caller (the CUDA package extension, Task 4) wraps `data` as a CuArray{Float16},
# tonemaps float16→RGBA8 on-device straight into the GLMakie texture, then calls
# `unmap_cuda` (gated on a copy-done event so ovrtx does not reclaim the buffer
# mid-copy).  NO `using CUDA` here — CUDA is a weakdep; these are pure LibOVRTX
# ccalls returning RAW handles (Ptr / Csize_t).
#
# Map mode is OVRTX_MAP_DEVICE_TYPE_CUDA (=2, LINEAR device memory → a CUdeviceptr),
# NOT CUDA_ARRAY (=3, an opaque CUarray).  HdrColor is kDLFloat/16, and the tonemap
# step needs LINEAR memory that `unsafe_wrap` can wrap as a CuArray — an opaque
# CUarray cannot be wrapped.  REPL-verified on an RTX A5000: mode 2 yields a
# non-null kDLCUDA float16 [H,W,C=4] device pointer + a non-zero wait-event.
# (Co-loading CUDA.jl + ovrtx requires JULIA_CUDA_USE_COMPAT=false so both share
# the system libcuda — see test/helpers.jl.)
# ------------------------------------------------------------------

"""
    map_cuda(sr::StepResult, name="HdrColor")
        -> (data::Ptr{Cvoid}, W::Int, H::Int, C::Int, map_handle, wait_event::Csize_t)

Fetch the step results from `sr`, find the render var `name`, and map it as
**linear CUDA device memory** (mode `OVRTX_MAP_DEVICE_TYPE_CUDA`).  Returns RAW
handles — does **not** copy and does **not** unmap:

- `data`       — a live `CUdeviceptr` (cast to `Ptr{Cvoid}`); for `HdrColor` it
                 points at `C*W*H` `Float16`s laid out `[C, W, H]` (channel-fastest).
- `(W, H, C)`  — DLTensor dims (shape is `[H, W, C]`; `C == 4` for an RGBA frame).
- `map_handle` — the ovrtx map handle; pass it to `unmap_cuda`.
- `wait_event` — the `CUevent` (as `Csize_t`, may be 0) to `cuStreamWaitEvent` on
                 BEFORE reading the buffer.

The buffer stays mapped and live until the caller calls
`unmap_cuda(sr, map_handle; ...)`.  No CUDA.jl dependency — the CUDA package
extension (Task 4) wraps `data` as a `CuArray` and `wait_event` as a `CuEvent`.
"""
function map_cuda(sr::StepResult, name::AbstractString="HdrColor")
    sr.r.alive || error("map_cuda on a closed Renderer")
    # 1. fetch
    outs = Ref{LibOVRTX.ovrtx_render_product_set_outputs_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_fetch_results(sr.r.ptr, sr.handle, LibOVRTX.OVRTX_TIMEOUT_INFINITE, outs), "fetch_results")

    # 2. walk the output tree
    h = _find_var(outs[], name)

    # 3. map as LINEAR CUDA device memory (mode 2) — no copy, no unmap
    mdesc = Ref(LibOVRTX.ovrtx_map_output_description_t(LibOVRTX.OVRTX_MAP_DEVICE_TYPE_CUDA, Csize_t(0)))
    ro    = Ref{LibOVRTX.ovrtx_render_var_output_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_map_render_var_output(sr.r.ptr, h, mdesc, LibOVRTX.OVRTX_TIMEOUT_INFINITE, ro), "map_render_var_output(cuda)")

    # 4. decode the DLTensor (shape = [H, W, C], dtype = kDLFloat/16/1 for HdrColor)
    t0  = unsafe_load(ro[].tensors, 1)          # ovrtx_render_var_tensor_t
    dlt = unsafe_load(t0.dl)                     # DLTensor (CUdeviceptr in .data)
    H   = Int(unsafe_load(dlt.shape, 1))
    W   = Int(unsafe_load(dlt.shape, 2))
    C   = Int(unsafe_load(dlt.shape, 3))

    # 5. return RAW handles; caller reads on-device then calls unmap_cuda.
    return (Ptr{Cvoid}(dlt.data), W, H, C, ro[].map_handle, ro[].cuda_sync.wait_event)
end

"""
    unmap_cuda(sr::StepResult, map_handle; stream=Csize_t(0), done_event=Csize_t(0))

Release a `map_cuda` mapping.  Constructs an `ovrtx_cuda_sync_t(stream, done_event)`
(field order is **stream first, then the event**) so ovrtx waits for `done_event`
— a `CUevent` the caller records after its on-device copy — on `stream` before
reclaiming the buffer (spike §3: event-gated unmap, else ovrtx reclaims mid-copy).
The default `(0, 0)` is `NOSYNC` (synchronous).  No-op if the Renderer is already
closed.
"""
function unmap_cuda(sr::StepResult, map_handle; stream::Csize_t = Csize_t(0), done_event::Csize_t = Csize_t(0))
    sync = LibOVRTX.ovrtx_cuda_sync_t(stream, done_event)
    sr.r.alive && LibOVRTX.ovrtx_unmap_render_var_output(sr.r.ptr, map_handle, sync)
    return nothing
end

# ==================================================================
# Picking — M6.B native ray-query pick (CPU-only)
#
# `enqueue_pick_query` registers a pixel-rect pick for the NEXT `step!` on a
# product; that step yields a synthetic `ovrtx_pick_hit` render var (CPU-mapped
# ONLY — an ovrtx restriction; the payload is tiny so there is no device path).
# `read_pick_hit` decodes its uint32 params (magic/version/hitCount) and named
# tensors into `PickHit`s; `path_resolver`/`resolve_prim_path` turn a hit's
# `primpath_id` into the authored prim-path string via the path-dictionary vtable.
#
# ABI VERIFIED in a REPL against references/ovrtx/tests/docs/c/
# test_picking_selection.cpp + helpers.h (docs_resolve_primpath):
#   - params: `ovrtx_render_var_param_t` carries a VALUE `name::ovx_string_t` and
#     a VALUE `dl::DLTensor`; each pick param is a scalar kDLUInt/32 (read at
#     p.dl.data).  (C `find_param` uses `param.name` / `&param.dl`.)
#   - tensors: `ovrtx_render_var_tensor_t` carries a POINTER `name::Ptr{ovx_string_t}`
#     and a POINTER `dl::Ptr{DLTensor}` — deref both.  (C `find_tensor` uses
#     `*tensor.name` / `tensor.dl`.)  Verified dtypes/shapes (n = hitCount):
#       primPath           kDLUInt/64  ndim=1 [n]
#       objectType         kDLUInt/32  ndim=1 [n]
#       geometryInstanceId kDLUInt/64  ndim=1 [n]   ← 64-bit on the wire, NOT 32
#       worldPositionM     kDLFloat/64 ndim=2 [n,3] row-major
#       worldNormal        kDLFloat/32 ndim=2 [n,3] row-major
#   - resolution: `path_dictionary_vtable_t.get_tokens_from_paths` /
#     `.get_strings_from_tokens` are `Ptr{Cvoid}` fn-pointers called through
#     `@ccall $fp(...)`.  `out_tokens_per_path` points INTO the caller's token
#     buffer, so the buffer is `GC.@preserve`d across BOTH calls.
# ==================================================================

"""
    enqueue_pick_query(r::Renderer, product, (left,top,right,bottom); flags=UInt32(0)) -> Nothing

Register a pick query over the pixel rect (left/top inclusive, right/bottom
exclusive, in RenderProduct pixel coords) for the NEXT `step!` that renders
`product`.  Synchronous (enqueue + wait, mirroring the C picking reference which
waits on the enqueue op before stepping), so the `product` bytes only need to
outlive this call.  Read the result after the next `step!` with `read_pick_hit`.
If several queries are enqueued for one product before a step, the last wins.
"""
function enqueue_pick_query(r::Renderer, product::AbstractString, rect::NTuple{4,Int};
                            flags::UInt32 = UInt32(0))
    r.alive || error("enqueue_pick_query on a closed Renderer")
    prod_s = String(product)
    GC.@preserve prod_s begin
        desc = Ref(LibOVRTX.ovrtx_pick_query_desc_t(LibOVRTX.ovx_string(prod_s),
                   Int32(rect[1]), Int32(rect[2]), Int32(rect[3]), Int32(rect[4]), flags))
        enqueue_wait(r, LibOVRTX.ovrtx_enqueue_pick_query(r.ptr, desc), "enqueue_pick_query")
    end
    return nothing
end

"""
    PickHit

One viewport pick hit.  `primpath_id` is an `ovx_primpath_t` id — resolve it to a
prim-path string with `resolve_prim_path`.  `instance_id` is the wire
`geometryInstanceId` (a 64-bit id; `0` for a single non-instanced geometry).
`world_position`/`normal` are in world space (the C reference asserts only that
they are finite — for a prim at the world origin they read `(0,0,0)`).
"""
const PickHit = NamedTuple{(:primpath_id, :object_type, :instance_id, :world_position, :normal),
                           Tuple{UInt64, UInt32, UInt64, NTuple{3,Float64}, NTuple{3,Float32}}}

# Find a pick-hit PARAM by name and read its scalar uint32 value.  Params carry a
# VALUE name + VALUE dl (verified), distinct from tensors.
function _pick_param_u32(out::LibOVRTX.ovrtx_render_var_output_t, name::AbstractString)
    for i in 1:out.num_params
        p = unsafe_load(out.params, i)               # ovrtx_render_var_param_t (by value)
        String(p.name) == name && return unsafe_load(Ptr{UInt32}(p.dl.data))
    end
    error("pick-hit param '$name' not found")
end

# Find a pick-hit TENSOR by name and return its DLTensor (deref'd).  Tensors carry
# a POINTER name + POINTER dl (verified), distinct from params.
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

Decode the `ovrtx_pick_hit` render var produced by the `step!` that consumed a
prior `enqueue_pick_query`.  Returns one `PickHit` per hit; an EMPTY vector when
no pick was enqueued for this step (the var is absent) or the magic/version
header does not match.  CPU-only map (an ovrtx restriction); the map is always
released (try/finally), never `map_cuda`.
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

Wraps the renderer's `path_dictionary_instance_t` (from `ovrtx_get_path_dictionary`)
plus its loaded vtable, so `resolve_prim_path` can dispatch the two `static inline`
path-dictionary functions through their raw function pointers.  Build one with
`path_resolver(r)`; valid while the renderer's stage composition is unchanged.
"""
struct PathResolver
    pd::Base.RefValue{LibOVRTX.path_dictionary_instance_t}
    vt::LibOVRTX.path_dictionary_vtable_t
end

"""
    path_resolver(r::Renderer) -> PathResolver

Fetch the renderer's path dictionary and load its vtable (one fetch, reused for
many `resolve_prim_path` calls).
"""
function path_resolver(r::Renderer)
    r.alive || error("path_resolver on a closed Renderer")
    pd = Ref{LibOVRTX.path_dictionary_instance_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_get_path_dictionary(r.ptr, pd), "get_path_dictionary")
    return PathResolver(pd, unsafe_load(pd[].vtable))
end

"""
    resolve_prim_path(pr::PathResolver, id::UInt64) -> String

Resolve one `ovx_primpath_t` id (a `PickHit.primpath_id`) to its `/A/B/C` prim
path via the path-dictionary vtable: `get_tokens_from_paths`, then per token
`get_strings_from_tokens`, joining the pieces with `/`.  Returns `""` if the
dictionary cannot resolve `id`.  Reimplements helpers.h `docs_resolve_primpath`.
"""
function resolve_prim_path(pr::PathResolver, id::UInt64)::String
    ctx        = Ptr{Cvoid}(pr.pd[].context)
    idref      = Ref(id)
    token_buf  = Vector{UInt64}(undef, 64)
    out_tokens = Ref{Ptr{UInt64}}(C_NULL)
    out_ntok   = Ref{Csize_t}(0)
    out_nproc  = Ref{Csize_t}(0)
    # out_tokens points INTO token_buf; keep it (and idref) preserved across BOTH ccalls.
    GC.@preserve token_buf idref begin
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
    set_selection_outline_group!(r, prim_paths::Vector{String}, group_ids::Vector{UInt8}) -> Nothing

Write the per-prim `omni:selectionOutlineGroup` (uint8) attribute on each prim in
`prim_paths` (parallel `group_ids`): group `0` clears, `1..255` assign a selection
group.  One multi-prim `ovrtx_write_attribute` (kDLUInt/8, shape `[N]`, semantic
NONE), mirroring the C `ovrtx_set_selection_outline_group` inline helper.  Group
tracking works regardless of the renderer's selection-outline config; an outline
is only DRAWN when that config was enabled at renderer creation.
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
            enqueue_wait(r, LibOVRTX.ovrtx_write_attribute(r.ptr, bdoh, ibuf, LibOVRTX.OVRTX_DATA_ACCESS_SYNC),
                         "set_selection_outline_group")
        end
    end
    return nothing
end

"""
    set_selection_group_styles!(r, group_ids::Vector{UInt8}, styles::Vector{ovrtx_selection_group_style_t}) -> Nothing

Set the visual style (outline + fill RGBA, each in `[0,1]`) for each selection
group id (parallel arrays).  Per-group colors are runtime state; global outline
width and fill mode are renderer-creation config.  Synchronous (enqueue + wait).
"""
function set_selection_group_styles!(r::Renderer, group_ids::Vector{UInt8},
                                     styles::Vector{LibOVRTX.ovrtx_selection_group_style_t})
    r.alive || error("set_selection_group_styles! on a closed Renderer")
    length(group_ids) == length(styles) ||
        error("set_selection_group_styles!: group_ids ($(length(group_ids))) and styles ($(length(styles))) length mismatch")
    GC.@preserve group_ids styles begin
        enqueue_wait(r, LibOVRTX.ovrtx_set_selection_group_styles(r.ptr, pointer(group_ids),
                     pointer(styles), Csize_t(length(group_ids))), "set_selection_group_styles")
    end
    return nothing
end

# ------------------------------------------------------------------
# render_hdr_to_array — convenience: warmup + map_cpu_f32 + return raw HDR
# ------------------------------------------------------------------

"""
    render_hdr_to_array(r::Renderer, product::AbstractString; warmup=64, timeout_ns) -> Array{Float32,3}

Run `warmup` RT2 steps on `product`, then map the final frame's `HdrColor` output
and return a `Float32` array of size `(C=4, W, H)` (channel-fastest).
The caller is responsible for tonemapping (see `tonemap_frame`).

`HdrColor` is a linear-space kDLFloat/16 output (verified via the C API tests);
the data is converted to Float32 before return.
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
    render_to_matrix(r::Renderer, product::AbstractString; warmup=64, timeout_ns) -> Matrix{RGBA{N0f8}}

Run `warmup` RT2 steps on `product` (RT2 needs many samples to converge),
then map the final frame's `LdrColor` output to a `Matrix{RGBA{N0f8}}` of
size `(H, W)`.  `timeout_ns` is passed to each `step!` call; it defaults to
the infinite-wait constant (preserving back-compat) and is set to a bounded
value by the M5 interactive camera loop.

Warmup frames are destroyed immediately.  The final `StepResult` is closed
after the pixel copy.

Orientation: the returned matrix is top-left-origin (row 1 = top of the image,
right-side-up).  No vertical flip is applied.  Verified empirically by
`test/m1_orientation_test.jl` (red_row ≈ 103 < blue_row ≈ 306 for boxes at
world +Z vs −Z).
"""
function render_to_matrix(r::Renderer, product::AbstractString;
                         warmup::Int=64, timeout_ns::UInt64 = _TIMEOUT_INFINITE_NS)
    for s in 1:(warmup - 1)
        sr = step!(r, product; timeout_ns); close(sr)
    end
    sr = step!(r, product; timeout_ns)
    pixels, W, H = try
        map_cpu(sr, "LdrColor")
    finally
        close(sr)
    end
    return cwh_to_matrix(pixels)
end

# ------------------------------------------------------------------
# reset! — restart RT2 accumulation (call after any geometry/camera change)
# ------------------------------------------------------------------

"""
    reset!(r::Renderer; time=0.0)

Enqueue and wait for an RT2 accumulation reset.  Must be called after any
geometry or camera change so the path-tracer starts fresh.
"""
function reset!(r::Renderer; time::Float64=0.0)
    enqueue_wait(r, LibOVRTX.ovrtx_reset(r.ptr, time), "reset")
    return nothing
end

# ------------------------------------------------------------------
# _write_attribute! — shared private helper for FFI attribute writes
# ------------------------------------------------------------------

# Write one attribute (fixed-size or array) to `prim` via a DLTensor over `data`.
# `data` must be a contiguous, OWNED Vector whose bytes back the DLTensor; the
# caller preprocesses (transpose/flatten for xform, dtype inference for arrays).
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
            enqueue_wait(r, LibOVRTX.ovrtx_write_attribute(r.ptr, bdoh, ibuf, LibOVRTX.OVRTX_DATA_ACCESS_SYNC),
                         "write_attribute($attr_name)")
        end
    end
    return nothing
end

# ------------------------------------------------------------------
# write_xform! — write a 4×4 transform to a USD prim (hot-path)
# ------------------------------------------------------------------

"""
    write_xform!(r::Renderer, prim::AbstractString, mat::AbstractMatrix{Float64})

Write a 4×4 row-major transform matrix to the `omni:xform` attribute of `prim`.
Translation lives in the last row (row 4, columns 1–3).

Reimplements the static-inline `ovrtx_set_xform_mat` in pure Julia.
The matrix data AND the prim string are preserved across the write + wait via
`GC.@preserve`.
"""
function write_xform!(r::Renderer, prim::AbstractString, mat::AbstractMatrix{Float64})
    @assert size(mat) == (4, 4) "write_xform! requires a 4×4 matrix, got $(size(mat))"

    # Row-major 4×4 flat array (16 elements).  Julia matrices are column-major,
    # so we transpose to get row-major order as expected by ovrtx.
    M = vec(collect(mat'))  # length-16 Vector{Float64}, row-major

    # dtype: kDLFloat / 64 bits / lanes=16 → encodes the 4×4 as a single element
    dtype = LibOVRTX.DLDataType(UInt8(LibOVRTX.kDLFloat), UInt8(64), UInt16(16))

    _write_attribute!(r, prim, "omni:xform", dtype, false, LibOVRTX.OVRTX_SEMANTIC_XFORM_MAT4x4, M, Int64[1])
    return nothing
end

# ------------------------------------------------------------------
# bind_material! — write the `material:binding` relationship (M3.1)
# ------------------------------------------------------------------

"""
    bind_material!(r::Renderer, geom_prim::AbstractString, material_prim::AbstractString)

Write the USD `material:binding` relationship on the GEOMETRY prim `geom_prim`,
pointing it at the material prim `material_prim` (an absolute prim path, e.g.
`/World/Looks/Mat_42`).  The material prim MUST already exist on the open stage.

`material:binding` is a USD relationship — a single-element array of PATHS per prim —
so this mirrors the C `ovrtx_set_path_attributes` helper: one `ovx_string_t` element
(the material path, 128 bits = ptr+len) with dtype `{kDLUInt,128,1}`, `is_array=true`,
`shape=[1]`, and the `OVRTX_SEMANTIC_PATH_STRING` semantic.  Mechanically the same
shape as the `visibility` TOKEN_STRING write, but `is_array=true` (a relationship array)
and the PATH semantic.

The material-path `String` is preserved across the FFI call + wait (the `ovx_string_t`
references its bytes), exactly like the token-string write.  As with every open-stage
edit, call `OV.reset!` after binding to restart RT2 accumulation before stepping.
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

Live re-write of an OmniPBR shader input `inputs:<name>` on the OPEN stage `shader_prim`
(e.g. `/World/Looks/Mat_<id>/Shader`) — the M3.4 material-edit diff path.  Mirrors the
M0 `_write_attribute!`/`write_xform!` fixed-size pattern:

- a `Float32` scalar (`metallic_constant`, `reflection_roughness_constant`,
  `opacity_constant`, …) → `dtype = {kDLFloat,32,1}`, `is_array=false`, `data=[v]`,
  `shape=[1]`.
- a `color3f` `NTuple{3,Float32}` (`diffuse_color_constant`, `emissive_color`) →
  `dtype = {kDLFloat,32,3}` (3 lanes, like the xform's 16), `is_array=false`,
  `data=[r,g,b]`, `shape=[1]`.

The backing `Float32` vector is `GC.@preserve`d across the write + wait.  The
material-editor render-test proves `inputs:diffuse_color_constant` (and shader inputs
generally) are live-writable on an OPEN stage.  As with every open-stage edit, the
caller restarts RT2 accumulation (one `OV.reset!` per changed frame — the M2 contract);
this function does NOT reset.
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

# Float32-aggregate lane count: `Point3f`/`Vec3f` → 3, `Point2f` → 2, etc.
# 0 for anything that is NOT a fixed-size Float32 aggregate (so the scalar paths or
# the error branch handle it).  Detected structurally (`eltype(ET) === Float32`) so
# OV.jl needs no GeometryBasics/StaticArrays import.
function _f32_lanes(::Type{ET}) where {ET}
    (isconcretetype(ET) && isbitstype(ET) && eltype(ET) === Float32) || return 0
    n, r = divrem(sizeof(ET), sizeof(Float32))
    return r == 0 ? n : 0
end

"""
    write_array_attribute!(r::Renderer, prim::AbstractString,
                           name::AbstractString, arr::AbstractArray)

Write an array attribute (e.g. `points`) to `prim`.
The element DLDataType is inferred from the Julia element type of `arr`.

`Vector{Point3f}` / `Vector{Vec3f}` (and other fixed-size `Float32` aggregates) are
written zero-copy as a multi-lane `kDLFloat/32` tensor: `reinterpret(Float32, …)`
flattens the components and the DLTensor carries `lanes = sizeof(eltype)/4` with
`shape = [length(arr)]` (one element per point), so `point3f[]` round-trips
correctly (M2 hot-path needs point arrays).

The array data, prim string, and attribute name are preserved across the
write + wait via `GC.@preserve`.
"""
function write_array_attribute!(r::Renderer, prim::AbstractString,
                                name::AbstractString, arr::AbstractArray)
    ET    = eltype(arr)
    lanes = _f32_lanes(ET)
    if lanes >= 2
        # Float32 aggregate (Point3f/Vec3f/…): reinterpret components to a flat,
        # owned Float32 vector; one tensor element per aggregate (multi-lane).
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
    add_usd_reference!(r::Renderer, usda::AbstractString, prim_path::AbstractString)
        -> ovrtx_usd_handle_t

Add a USD layer (given as an in-memory USDA string) to the running stage under the
prefix `prim_path`.  Returns an opaque handle for later `remove_usd!` calls.

Both strings are converted to owned `String`s and preserved across the FFI call and
the wait (the `ovx_string_t` structs reference Julia heap memory).
"""
function add_usd_reference!(r::Renderer, usda::AbstractString, prim_path::AbstractString)
    r.alive || error("add_usd_reference! on a closed Renderer")
    layer_s = String(usda)
    path_s  = String(prim_path)
    h = Ref{LibOVRTX.ovrtx_usd_handle_t}(0)
    GC.@preserve layer_s path_s begin
        enqueue_wait(r,
            LibOVRTX.ovrtx_add_usd_reference_from_string(
                r.ptr, LibOVRTX.ovx_string(layer_s), LibOVRTX.ovx_string(path_s), h),
            "add_usd_reference")
    end
    return h[]
end

"""
    remove_usd!(r::Renderer, handle::ovrtx_usd_handle_t) -> Nothing

Remove the USD layer previously added via `add_usd_reference!`.
"""
function remove_usd!(r::Renderer, handle::LibOVRTX.ovrtx_usd_handle_t)
    enqueue_wait(r, LibOVRTX.ovrtx_remove_usd(r.ptr, handle), "remove_usd")
    return nothing
end

# ==================================================================
# Binding — persistent attribute bindings (M2.4 hot path)
#
# `ovrtx_create_attribute_binding` locks a prim + attribute name + element type so
# repeated per-frame writes skip rebuilding the binding descriptor.  Two hot-path
# tiers (ARCHITECTURE §6), both VALIDATED on a referenced plot prim by the M2.4
# binding spike (omni:xform map round-trip byte-exact; points write-through-handle
# scaled + reverted exactly):
#   - fixed-size (`omni:xform`): `map_binding`/`unmap!` → ZERO-COPY write straight
#     into ovrtx's internal buffer (created with OVRTX_BINDING_FLAG_OPTIMIZE).
#   - array (`points`): `write_binding!` copies a fresh tensor through the handle.
#
# GC discipline mirrors `_write_attribute!`: the prim path + attribute name `String`s
# are retained on the struct so they stay rooted for the binding's lifetime, and
# every FFI buffer/desc is `GC.@preserve`d across the ccall + wait.
# ==================================================================

"""
    Binding

A persistent ovrtx attribute binding (handle from `ovrtx_create_attribute_binding`).
Reused across frames; released by `destroy!` (finalizer is a backstop).  `map_handle`
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

# A zeroed `ovrtx_binding_desc_t`; it is IGNORED whenever `binding_handle != 0`
# (the header: "If binding_handle is non-zero then it will be used, otherwise
# binding_desc will be used"), so every map/write through a handle passes this.
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
`{kDLFloat,64,16}` for a 4×4 double, `{kDLFloat,32,3}` for `point3f[]`).  `array=true`
binds a variable-length array attribute; `optimize=true` sets
`OVRTX_BINDING_FLAG_OPTIMIZE` for the primary high-volume hot binding.  Synchronous
(enqueue + wait); the prim list strings are preserved across the call.
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
            enqueue_wait(r, LibOVRTX.ovrtx_create_attribute_binding(r.ptr, bref, handle),
                         "create_attribute_binding($name)")
        end
    end
    b = Binding(r, handle[], prim_s, name_s, dtype, array, semantic,
                LibOVRTX.ovrtx_map_handle_t(0), true)
    finalizer(destroy!, b)
    return b
end

"""
    map_binding(b::Binding; device=kDLCPU, device_id=0) -> Ptr{Cvoid}

Map the binding's internal buffer for a ZERO-COPY write; returns the data pointer
(valid ONLY until `unmap!`).  Stashes the map handle on `b`.  `ovrtx_map_attribute`
is synchronous (no enqueue/wait).
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

"""
    unmap!(b::Binding)

Commit + release a mapping from `map_binding` (async unmap + wait).  No-op when not
mapped or when the Renderer is already closed.
"""
function unmap!(b::Binding)
    b.map_handle == 0 && return nothing
    b.r.alive && enqueue_wait(b.r, LibOVRTX.ovrtx_unmap_attribute(b.r.ptr, b.map_handle, LibOVRTX.NOSYNC),
                              "unmap_attribute")
    b.map_handle = LibOVRTX.ovrtx_map_handle_t(0)
    return nothing
end

"""
    write_mapped_xform!(b::Binding, mat::AbstractMatrix{Float64})

ZERO-COPY write of a 4×4 transform through the MAPPED fixed-size binding `b`:
map → store the 16 row-major doubles into the internal buffer → unmap.  `mat` is in
USD row-vector form (translation in the last row), identical to `write_xform!`; the
16 doubles written are `vec(collect(mat'))`, so it round-trips byte-for-byte.
"""
function write_mapped_xform!(b::Binding, mat::AbstractMatrix{Float64})
    @assert size(mat) == (4, 4) "write_mapped_xform! requires a 4×4 matrix, got $(size(mat))"
    M   = vec(collect(mat'))                      # 16 row-major Float64
    ptr = Ptr{Cdouble}(map_binding(b))
    try
        GC.@preserve M begin
            @inbounds for i in 1:16
                unsafe_store!(ptr, M[i], i)
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
contiguous, owned vector whose bytes match `b.dtype` (e.g. a flattened `Float32`
buffer for a `point3f[]` binding); `shape` is the per-prim element count
(`[npoints]`).  Mirrors `_write_attribute!`'s `GC.@preserve` discipline.
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
            enqueue_wait(b.r, LibOVRTX.ovrtx_write_attribute(b.r.ptr, bdoh, ibuf, LibOVRTX.OVRTX_DATA_ACCESS_SYNC),
                         "write_attribute(binding:$(b.attr_name))")
        end
    end
    return nothing
end

"""
    destroy!(b::Binding)

Release the persistent binding (`ovrtx_destroy_attribute_binding`).  Idempotent;
unmaps first if still mapped; a no-op once the Renderer is closed (the GPU pool is
already freed).
"""
function destroy!(b::Binding)
    b.alive || return nothing
    if b.r.alive
        b.map_handle == 0 || unmap!(b)
        enqueue_wait(b.r, LibOVRTX.ovrtx_destroy_attribute_binding(b.r.ptr, b.handle),
                     "destroy_attribute_binding")
    end
    b.map_handle = LibOVRTX.ovrtx_map_handle_t(0)
    b.alive = false
    return nothing
end

end # module OV
