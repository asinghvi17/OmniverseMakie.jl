module OmniverseMakieCUDADirectExt

# GPU-direct plot updates: push mesh data for a displayed plot straight from
# CUDA device arrays through the persistent ovrtx attribute bindings as
# kDLCUDA DLTensors — no host roundtrip.  Loads with CUDA alone (the sibling
# OmniverseMakieCUDAExt viewport blit additionally needs GLMakie).

using OmniverseMakie, CUDA
using OmniverseMakie: OV
import Makie

# Flatten a device array to Float32 (xyzxyz…): Float32 arrays pass through,
# 3-component aggregates (Point3f/Vec3f/NTuple) reinterpret in place.
_flat32(a::CuArray{Float32}) = vec(a)
_flat32(a::CuArray{<:Union{Makie.Point3f,Makie.Vec3f,NTuple{3,Float32}}}) =
    reinterpret(Float32, vec(a))
_flat32(a) = throw(ArgumentError(
    "gpu_update_mesh! takes CuArrays of Float32 (flat xyz) or Point3f/Vec3f, got $(typeof(a))"))

# One device write through binding `b`: `flat` is 3·npts Float32s on device.
function _write_device!(b, flat, npts::Int, what::AbstractString)
    length(flat) == 3 * npts || throw(ArgumentError(
        "$(what): expected 3·$(npts) = $(3npts) Float32s, got $(length(flat))"))
    ptr = reinterpret(Ptr{Cvoid}, pointer(flat))
    dev = CUDA.deviceid(CUDA.device())
    OV.write_binding_device!(b, ptr, npts, flat; device_id = dev)
    return nothing
end

"""
    gpu_update_mesh!(screen, plot; points, sync = true) -> Nothing

Update a displayed mesh plot's `points` directly from a CUDA device array
(no host copy): a `CuArray` of `3·npoints` `Float32`s (flat xyz) or
`npoints` `Point3f`/`Vec3f`, where `npoints` is frozen at author time.
Data must be in the plot's post-transform Float32 space (what
`plot.positions_transformed_f32c` would hold — world coordinates for an
untransformed root-scene plot; the model matrix stays separate).

GPU inputs use ovrtx's ASYNC access: `sync = true` (default) runs
`CUDA.synchronize()` first so producing kernels are complete; the call
blocks until the engine has consumed the buffer, so it is immediately
reusable.  Makie's CPU-side plot data is NOT updated — it goes stale, and a
later Makie-side edit of the same attribute overwrites these writes (the
same trade-off as GLMakie's raw-`GLBuffer` inputs).

No `normals` path: vertex-normal writes are pixel-inert in standalone ovrtx
(authored or live, RT2 and PathTracing — the engine shades meshes from its
own geometry-derived normals; tripwired in test/live/gpu_direct_test.jl).
"""
function OmniverseMakie.gpu_update_mesh!(screen::OmniverseMakie.Screen, plot;
                                         points = nothing, sync::Bool = true)
    robj = get(screen.plot2robj, objectid(plot), nothing)
    robj === nothing && throw(ArgumentError(
        "gpu_update_mesh!: plot is not displayed on this Screen — the stage authors \
         lazily, so render once (e.g. `Makie.colorbuffer(screen)`) before pushing \
         device data"))
    npts = get(robj.meta, :mesh_npoints, -1)
    npts > 0 || throw(ArgumentError(
        "gpu_update_mesh! needs a mesh-like plot with a frozen point count \
         (a materialized plot such as Surface has none; got $(typeof(plot)))"))
    sync && CUDA.synchronize()
    if points !== nothing
        b = get(robj.bindings, :positions_transformed_f32c, nothing)
        b === nothing && throw(ArgumentError(
            "gpu_update_mesh!: plot has no live points binding"))
        _write_device!(b, _flat32(points), npts, "points")
    end
    OmniverseMakie._set_requires_update!(screen)
    return nothing
end

end # module OmniverseMakieCUDADirectExt
