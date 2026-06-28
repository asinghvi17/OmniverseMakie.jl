module OV
using ..LibOVRTX
const L = LibOVRTX
include("signals.jl")
using .SignalGuard: with_restored_signals

include("dlpack.jl")

# ------------------------------------------------------------------
# Renderer — GC-aware wrapper around ovrtx_renderer_t
# ------------------------------------------------------------------

mutable struct Renderer
    ptr::Ptr{L.ovrtx_renderer_t}
    alive::Bool
    function Renderer()
        cfg  = Ref(L.ovrtx_config_t(Ptr{L.ovrtx_config_entry_t}(C_NULL), Csize_t(0)))
        rref = Ref{Ptr{L.ovrtx_renderer_t}}(C_NULL)
        with_restored_signals() do
            L.check(L.ovrtx_create_renderer(cfg, rref), "create_renderer")
        end
        r = new(rref[], true)
        finalizer(close, r)
        return r
    end
end

Base.unsafe_convert(::Type{Ptr{L.ovrtx_renderer_t}}, r::Renderer) = r.ptr

function Base.close(r::Renderer)
    r.alive || return
    L.ovrtx_destroy_renderer(r.ptr); r.alive = false; return
end

# ------------------------------------------------------------------
# Async lifecycle: enqueue (ovrtx_enqueue_result_t) -> wait_op
# ------------------------------------------------------------------

function enqueue_wait(r::Renderer, enq, op::AbstractString)
    L.check(enq, op)
    wr = Ref{L.ovrtx_op_wait_result_t}()
    L.check(L.ovrtx_wait_op(r.ptr, enq.op_index, L.OVRTX_TIMEOUT_INFINITE, wr), op * ":wait")
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
        enqueue_wait(r, L.ovrtx_open_usd_from_file(r.ptr, L.ovx_string(path)), "open_usd")
    end
    return nothing
end

"""
    open_usd_string!(r::Renderer, usda::AbstractString)

Open a USD stage from an in-memory USDA string.  Synchronous.
"""
function open_usd_string!(r::Renderer, usda::AbstractString)
    GC.@preserve usda begin
        enqueue_wait(r, L.ovrtx_open_usd_from_string(r.ptr, L.ovx_string(usda)), "open_usd_string")
    end
    return nothing
end

# ------------------------------------------------------------------
# StepResult — wraps an ovrtx_step_result_handle_t
# ------------------------------------------------------------------

mutable struct StepResult
    r::Renderer
    handle::L.ovrtx_step_result_handle_t
    open::Bool
end

function Base.close(sr::StepResult)
    sr.open || return
    L.ovrtx_destroy_results(sr.r.ptr, sr.handle)
    sr.open = false
    return nothing
end

# ------------------------------------------------------------------
# step! — enqueue one render step, return StepResult
# ------------------------------------------------------------------

"""
    step!(r::Renderer, product::AbstractString; dt=1/60) -> StepResult

Enqueue and wait for one RT2 render step for the given render product path.
Both the backing `ovx_string_t` array and the product `String` are preserved
across the ccall and the wait.

Returns a `StepResult`; the caller is responsible for closing it (or letting
the finalizer run).
"""
function step!(r::Renderer, product::AbstractString; dt::Float64=1/60)
    rp = L.ovx_string_t[ L.ovx_string(product) ]
    GC.@preserve product rp begin
        set = L.ovrtx_render_product_set_t(pointer(rp), Csize_t(1))
        h = Ref{L.ovrtx_step_result_handle_t}(0)
        enqueue_wait(r, L.ovrtx_step(r.ptr, set, dt, h), "step")
        sr = StepResult(r, h[], true)
        finalizer(close, sr)
        return sr
    end
end

# ------------------------------------------------------------------
# _find_var — walk the nested output tree to locate a render var
# ------------------------------------------------------------------

"""
    _find_var(outs::ovrtx_render_product_set_outputs_t, name) -> ovrtx_render_var_output_handle_t

Walk outputs → output_frames → output_render_vars and return the output handle
whose render_var_name matches `name` (e.g. "LdrColor").  Throws if not found.
"""
function _find_var(outs::L.ovrtx_render_product_set_outputs_t, name::AbstractString)
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
    error("render var '$name' not found in step outputs")
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
    # 1. fetch
    outs = Ref{L.ovrtx_render_product_set_outputs_t}()
    L.check(L.ovrtx_fetch_results(sr.r.ptr, sr.handle, L.OVRTX_TIMEOUT_INFINITE, outs), "fetch_results")

    # 2. walk the output tree
    h = _find_var(outs[], name)

    # 3. map to CPU
    mdesc = Ref(L.ovrtx_map_output_description_t(L.OVRTX_MAP_DEVICE_TYPE_CPU, Csize_t(0)))
    ro    = Ref{L.ovrtx_render_var_output_t}()
    L.check(L.ovrtx_map_render_var_output(sr.r.ptr, h, mdesc, L.OVRTX_TIMEOUT_INFINITE, ro), "map_render_var_output")

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
    L.ovrtx_unmap_render_var_output(sr.r.ptr, ro[].map_handle, L.NOSYNC)

    return (pixels, W, H)
end

# ------------------------------------------------------------------
# render_to_matrix — convenience: warmup + map + reshape
# ------------------------------------------------------------------

"""
    render_to_matrix(r::Renderer, product::AbstractString; warmup=64) -> Matrix{RGBA{N0f8}}

Run `warmup` RT2 steps on `product` (RT2 needs many samples to converge),
then map the final frame's `LdrColor` output to a `Matrix{RGBA{N0f8}}` of
size `(H, W)`.

Warmup frames are destroyed immediately.  The final `StepResult` is closed
after the pixel copy.
"""
function render_to_matrix(r::Renderer, product::AbstractString; warmup::Int=64)
    for s in 1:(warmup - 1)
        sr = step!(r, product)
        close(sr)
    end
    sr = step!(r, product)
    pixels, W, H = try
        map_cpu(sr, "LdrColor")
    finally
        close(sr)
    end
    return cwh_to_matrix(pixels)
end

end # module OV
