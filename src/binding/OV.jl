module OV
using ..LibOVRTX
const L = LibOVRTX
include("signals.jl")
using .SignalGuard: with_restored_signals

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

end # module OV
