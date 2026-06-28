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
    L.ovrtx_destroy_renderer(r.ptr)
    r.ptr = Ptr{L.ovrtx_renderer_t}(C_NULL)   # avoid a dangling pointer via unsafe_convert after close
    r.alive = false
    return
end

# ------------------------------------------------------------------
# Async lifecycle: enqueue (ovrtx_enqueue_result_t) -> wait_op
# ------------------------------------------------------------------

function enqueue_wait(r::Renderer, enq, op::AbstractString)
    r.alive || error("enqueue_wait called on a closed Renderer")
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
    sr.r.alive && L.ovrtx_destroy_results(sr.r.ptr, sr.handle)  # pool already freed if renderer closed
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
    sr.r.alive || error("map_cpu: the StepResult's Renderer is already closed")
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

Orientation: the returned matrix is top-left-origin (row 1 = top of the image,
right-side-up).  No vertical flip is applied.  Verified empirically by
`test/m1_orientation_test.jl` (red_row ≈ 103 < blue_row ≈ 306 for boxes at
world +Z vs −Z).
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

# ------------------------------------------------------------------
# reset! — restart RT2 accumulation (call after any geometry/camera change)
# ------------------------------------------------------------------

"""
    reset!(r::Renderer; time=0.0)

Enqueue and wait for an RT2 accumulation reset.  Must be called after any
geometry or camera change so the path-tracer starts fresh.
"""
function reset!(r::Renderer; time::Float64=0.0)
    enqueue_wait(r, L.ovrtx_reset(r.ptr, time), "reset")
    return nothing
end

# ------------------------------------------------------------------
# _write_attribute! — shared private helper for FFI attribute writes
# ------------------------------------------------------------------

# Write one attribute (fixed-size or array) to `prim` via a DLTensor over `data`.
# `data` must be a contiguous, OWNED Vector whose bytes back the DLTensor; the
# caller preprocesses (transpose/flatten for xform, dtype inference for arrays).
function _write_attribute!(r::Renderer, prim::AbstractString, attr_name::AbstractString,
                           dtype::L.DLDataType, is_array::Bool, semantic,
                           data::AbstractVector, shape::Vector{Int64})
    strides = Int64[1]
    prim_s   = String(prim)
    prim_ovx = L.ovx_string(prim_s)
    prim_arr = L.ovx_string_t[prim_ovx]
    name_s   = String(attr_name)
    name_ovx = L.ovx_string(name_s)
    GC.@preserve prim_s prim_arr name_s data shape strides begin
        prim_list   = L.ovrtx_prim_list_t(pointer(prim_arr), Csize_t(1))
        attr_lookup = L.ovx_string_or_token_t(UInt64(0), name_ovx)
        attr_type   = L.ovrtx_attribute_type_t(dtype, is_array, semantic)
        bdesc = L.ovrtx_binding_desc_t(
            prim_list,
            L.ovx_primpath_list_t(0),
            attr_lookup,
            attr_type,
            L.OVRTX_BINDING_PRIM_MODE_EXISTING_ONLY,
            L.OVRTX_BINDING_FLAG_NONE,
        )
        bdoh  = Ref(L.ovrtx_binding_desc_or_handle_t(bdesc, L.ovrtx_attribute_binding_handle_t(0)))
        dl = L.DLTensor(
            Ptr{Cvoid}(pointer(data)),
            L.DLDevice(L.kDLCPU, Int32(0)),
            Int32(1),
            dtype,
            pointer(shape),
            pointer(strides),
            UInt64(0),
        )
        dl_arr = L.DLTensor[dl]
        GC.@preserve dl_arr begin
            ibuf = Ref(L.ovrtx_input_buffer_t(
                pointer(dl_arr),
                UInt64(1),
                Ptr{UInt8}(C_NULL),
                Csize_t(0),
                L.NOSYNC,
                L.NOSYNC,
            ))
            enqueue_wait(r, L.ovrtx_write_attribute(r.ptr, bdoh, ibuf, L.OVRTX_DATA_ACCESS_SYNC),
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
    dtype = L.DLDataType(UInt8(L.kDLFloat), UInt8(64), UInt16(16))

    _write_attribute!(r, prim, "omni:xform", dtype, false, L.OVRTX_SEMANTIC_XFORM_MAT4x4, M, Int64[1])
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
    dtype = L.DLDataType(UInt8(L.kDLUInt), UInt8(128), UInt16(1))
    GC.@preserve mat_s begin
        data = L.ovx_string_t[L.ovx_string(mat_s)]
        _write_attribute!(r, geom_prim, "material:binding", dtype, true,
                          L.OVRTX_SEMANTIC_PATH_STRING, data, Int64[1])
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
        dtype = L.DLDataType(UInt8(L.kDLFloat), UInt8(32), UInt16(1))
        data  = Float32[value]
    else
        dtype = L.DLDataType(UInt8(L.kDLFloat), UInt8(32), UInt16(3))
        data  = Float32[value[1], value[2], value[3]]
    end
    GC.@preserve data begin
        _write_attribute!(r, shader_prim, attr_name, dtype, false, L.OVRTX_SEMANTIC_NONE,
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
        dtype = L.DLDataType(UInt8(L.kDLFloat), UInt8(32), UInt16(lanes))
        _write_attribute!(r, prim, name, dtype, true, L.OVRTX_SEMANTIC_NONE, data, Int64[length(arr)])
        return nothing
    end

    # Scalar element types: one lane each.
    dtype = if ET === Float32
        L.DLDataType(UInt8(L.kDLFloat), UInt8(32), UInt16(1))
    elseif ET === Float64
        L.DLDataType(UInt8(L.kDLFloat), UInt8(64), UInt16(1))
    elseif ET === Int32
        L.DLDataType(UInt8(L.kDLInt), UInt8(32), UInt16(1))
    elseif ET === Int64
        L.DLDataType(UInt8(L.kDLInt), UInt8(64), UInt16(1))
    elseif ET === UInt32
        L.DLDataType(UInt8(L.kDLUInt), UInt8(32), UInt16(1))
    elseif ET === UInt64
        L.DLDataType(UInt8(L.kDLUInt), UInt8(64), UInt16(1))
    else
        error("write_array_attribute!: unsupported element type $ET")
    end

    data = collect(arr)  # ensure contiguous, owned copy

    _write_attribute!(r, prim, name, dtype, true, L.OVRTX_SEMANTIC_NONE, data, Int64[length(data)])
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
    h = Ref{L.ovrtx_usd_handle_t}(0)
    GC.@preserve layer_s path_s begin
        enqueue_wait(r,
            L.ovrtx_add_usd_reference_from_string(
                r.ptr, L.ovx_string(layer_s), L.ovx_string(path_s), h),
            "add_usd_reference")
    end
    return h[]
end

"""
    remove_usd!(r::Renderer, handle::ovrtx_usd_handle_t) -> Nothing

Remove the USD layer previously added via `add_usd_reference!`.
"""
function remove_usd!(r::Renderer, handle::L.ovrtx_usd_handle_t)
    enqueue_wait(r, L.ovrtx_remove_usd(r.ptr, handle), "remove_usd")
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
    handle::L.ovrtx_attribute_binding_handle_t
    prim::String
    attr_name::String
    dtype::L.DLDataType
    is_array::Bool
    semantic::L.ovrtx_attribute_semantic_t
    map_handle::L.ovrtx_map_handle_t   # non-zero while mapped
    alive::Bool
end

# A zeroed `ovrtx_binding_desc_t`; it is IGNORED whenever `binding_handle != 0`
# (the header: "If binding_handle is non-zero then it will be used, otherwise
# binding_desc will be used"), so every map/write through a handle passes this.
_empty_binding_desc() = L.ovrtx_binding_desc_t(
    L.ovrtx_prim_list_t(Ptr{L.ovx_string_t}(C_NULL), Csize_t(0)),
    L.ovx_primpath_list_t(0),
    L.ovx_string_or_token_t(UInt64(0), L.ovx_string_t(reinterpret(Cstring, C_NULL), Csize_t(0))),
    L.ovrtx_attribute_type_t(L.DLDataType(UInt8(0), UInt8(0), UInt16(0)), false, L.OVRTX_SEMANTIC_NONE),
    L.OVRTX_BINDING_PRIM_MODE_EXISTING_ONLY,
    L.OVRTX_BINDING_FLAG_NONE,
)

_desc_or_handle(b::Binding) = L.ovrtx_binding_desc_or_handle_t(_empty_binding_desc(), b.handle)

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
                        dtype::L.DLDataType; array::Bool=false,
                        semantic=L.OVRTX_SEMANTIC_NONE, optimize::Bool=false)
    r.alive || error("create_binding on a closed Renderer")
    prim_s   = String(prim)
    name_s   = String(name)
    prim_arr = L.ovx_string_t[L.ovx_string(prim_s)]
    handle   = Ref{L.ovrtx_attribute_binding_handle_t}(0)
    flags    = optimize ? L.OVRTX_BINDING_FLAG_OPTIMIZE : L.OVRTX_BINDING_FLAG_NONE
    GC.@preserve prim_s prim_arr name_s begin
        prim_list   = L.ovrtx_prim_list_t(pointer(prim_arr), Csize_t(1))
        attr_lookup = L.ovx_string_or_token_t(UInt64(0), L.ovx_string(name_s))
        attr_type   = L.ovrtx_attribute_type_t(dtype, array, semantic)
        bdesc = L.ovrtx_binding_desc_t(prim_list, L.ovx_primpath_list_t(0),
            attr_lookup, attr_type, L.OVRTX_BINDING_PRIM_MODE_EXISTING_ONLY, flags)
        bref = Ref(bdesc)
        GC.@preserve bref begin
            enqueue_wait(r, L.ovrtx_create_attribute_binding(r.ptr, bref, handle),
                         "create_attribute_binding($name)")
        end
    end
    b = Binding(r, handle[], prim_s, name_s, dtype, array, semantic,
                L.ovrtx_map_handle_t(0), true)
    finalizer(destroy!, b)
    return b
end

"""
    map_binding(b::Binding; device=kDLCPU, device_id=0) -> Ptr{Cvoid}

Map the binding's internal buffer for a ZERO-COPY write; returns the data pointer
(valid ONLY until `unmap!`).  Stashes the map handle on `b`.  `ovrtx_map_attribute`
is synchronous (no enqueue/wait).
"""
function map_binding(b::Binding; device=L.kDLCPU, device_id::Integer=0)
    b.alive || error("map_binding on a destroyed Binding")
    b.map_handle == 0 || error("map_binding: already mapped (call unmap! first)")
    bdoh  = Ref(_desc_or_handle(b))
    mdesc = L.ovrtx_mapping_desc_t(Int32(device), Int32(device_id))
    out   = Ref{L.ovrtx_attribute_mapping_t}()
    GC.@preserve bdoh begin
        L.check(L.ovrtx_map_attribute(b.r.ptr, bdoh, mdesc, out), "map_attribute($(b.attr_name))")
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
    b.r.alive && enqueue_wait(b.r, L.ovrtx_unmap_attribute(b.r.ptr, b.map_handle, L.NOSYNC),
                              "unmap_attribute")
    b.map_handle = L.ovrtx_map_handle_t(0)
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
        dl = L.DLTensor(Ptr{Cvoid}(pointer(data)), L.DLDevice(L.kDLCPU, Int32(0)),
                        Int32(1), b.dtype, pointer(shape), pointer(strides), UInt64(0))
        dl_arr = L.DLTensor[dl]
        GC.@preserve dl_arr begin
            ibuf = Ref(L.ovrtx_input_buffer_t(pointer(dl_arr), UInt64(1),
                       Ptr{UInt8}(C_NULL), Csize_t(0), L.NOSYNC, L.NOSYNC))
            enqueue_wait(b.r, L.ovrtx_write_attribute(b.r.ptr, bdoh, ibuf, L.OVRTX_DATA_ACCESS_SYNC),
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
        enqueue_wait(b.r, L.ovrtx_destroy_attribute_binding(b.r.ptr, b.handle),
                     "destroy_attribute_binding")
    end
    b.map_handle = L.ovrtx_map_handle_t(0)
    b.alive = false
    return nothing
end

end # module OV
