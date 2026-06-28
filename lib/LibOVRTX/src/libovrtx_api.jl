using CEnum: CEnum, @cenum


"""
    ovx_string_t

Null-terminated string with explicit length.

Prefer the length field over relying on the null terminator:

```c++
{.c}
 printf("%.*s\\n", (int)str.length, str.ptr);
```

```c++
{.cpp}
 std::string_view sv(str.ptr, str.length);
```
"""
struct ovx_string_t
    ptr::Cstring
    length::Csize_t
end

"""
    DLPackVersion

The DLPack version.

A change in major version indicates that we have changed the data layout of the ABI - [`DLManagedTensorVersioned`](@ref).

A change in minor version indicates that we have added new code, such as a new device type, but the ABI is kept the same.

If an obtained DLPack tensor has a major version that disagrees with the version number specified in this header file (i.e. major != [`DLPACK_MAJOR_VERSION`](@ref)), the consumer must call the deleter (and it is safe to do so). It is not safe to access any other fields as the memory layout will have changed.

In the case of a minor version mismatch, the tensor can be safely used as long as the consumer knows how to interpret all fields. Minor version updates indicate the addition of enumeration values.
"""
struct DLPackVersion
    major::UInt32
    minor::UInt32
end

@cenum DLDeviceType::UInt32 begin
    kDLCPU = 1
    kDLCUDA = 2
    kDLCUDAHost = 3
    kDLOpenCL = 4
    kDLVulkan = 7
    kDLMetal = 8
    kDLVPI = 9
    kDLROCM = 10
    kDLROCMHost = 11
    kDLExtDev = 12
    kDLCUDAManaged = 13
    kDLOneAPI = 14
    kDLWebGPU = 15
    kDLHexagon = 16
    kDLMAIA = 17
    kDLTrn = 18
end

"""
    DLDevice

A Device for Tensor and operator.
"""
struct DLDevice
    device_type::DLDeviceType
    device_id::Int32
end

"""
    DLDataTypeCode

The type code options [`DLDataType`](@ref).
"""
@cenum DLDataTypeCode::UInt32 begin
    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLOpaqueHandle = 3
    kDLBfloat = 4
    kDLComplex = 5
    kDLBool = 6
    kDLFloat8_e3m4 = 7
    kDLFloat8_e4m3 = 8
    kDLFloat8_e4m3b11fnuz = 9
    kDLFloat8_e4m3fn = 10
    kDLFloat8_e4m3fnuz = 11
    kDLFloat8_e5m2 = 12
    kDLFloat8_e5m2fnuz = 13
    kDLFloat8_e8m0fnu = 14
    kDLFloat6_e2m3fn = 15
    kDLFloat6_e3m2fn = 16
    kDLFloat4_e2m1fn = 17
end

"""
    DLDataType

The data type the tensor can hold. The data type is assumed to follow the native endian-ness. An explicit error message should be raised when attempting to export an array with non-native endianness

Examples - float: type\\_code = 2, bits = 32, lanes = 1 - float4(vectorized 4 float): type\\_code = 2, bits = 32, lanes = 4 - int8: type\\_code = 0, bits = 8, lanes = 1 - std::complex<float>: type\\_code = 5, bits = 64, lanes = 1 - bool: type\\_code = 6, bits = 8, lanes = 1 (as per common array library convention, the underlying storage size of bool is 8 bits) - float8\\_e4m3: type\\_code = 8, bits = 8, lanes = 1 (packed in memory) - float6\\_e3m2fn: type\\_code = 16, bits = 6, lanes = 1 (packed in memory) - float4\\_e2m1fn: type\\_code = 17, bits = 4, lanes = 1 (packed in memory)

When a sub-byte type is packed, DLPack requires the data to be in little bit-endian, i.e., for a packed data set D ((D >> (i * bits)) && bit\\_mask) stores the i-th element.
"""
struct DLDataType
    code::UInt8
    bits::UInt8
    lanes::UInt16
end

"""
    DLTensor

Plain C Tensor object, does not manage memory.
"""
struct DLTensor
    data::Ptr{Cvoid}
    device::DLDevice
    ndim::Int32
    dtype::DLDataType
    shape::Ptr{Int64}
    strides::Ptr{Int64}
    byte_offset::UInt64
end

"""
    DLManagedTensor

C Tensor object, manage memory of [`DLTensor`](@ref). This data structure is intended to facilitate the borrowing of [`DLTensor`](@ref) by another framework. It is not meant to transfer the tensor. When the borrowing framework doesn't need the tensor, it should call the deleter to notify the host that the resource is no longer needed.

!!! note

    This data structure is used as Legacy [`DLManagedTensor`](@ref) in DLPack exchange and is deprecated after DLPack v0.8 Use [`DLManagedTensorVersioned`](@ref) instead. This data structure may get renamed or deleted in future versions.

# See also
[`DLManagedTensorVersioned`](@ref)
"""
struct DLManagedTensor
    dl_tensor::DLTensor
    manager_ctx::Ptr{Cvoid}
    deleter::Ptr{Cvoid}
end

"""
    DLManagedTensorVersioned

A versioned and managed C Tensor object, manage memory of [`DLTensor`](@ref).

This data structure is intended to facilitate the borrowing of [`DLTensor`](@ref) by another framework. It is not meant to transfer the tensor. When the borrowing framework doesn't need the tensor, it should call the deleter to notify the host that the resource is no longer needed.

!!! note

    This is the current standard DLPack exchange data structure.
"""
struct DLManagedTensorVersioned
    version::DLPackVersion
    manager_ctx::Ptr{Cvoid}
    deleter::Ptr{Cvoid}
    flags::UInt64
    dl_tensor::DLTensor
end

# typedef int ( * DLPackManagedTensorAllocator ) ( // DLTensor * prototype , DLManagedTensorVersioned * * out , void * error_ctx , // void ( * SetError ) ( void * error_ctx , const char * kind , const char * message ) // )
"""
Request a producer library to create a new tensor.

Create a new [`DLManagedTensorVersioned`](@ref) within the context of the producer library. The allocation is defined via the prototype [`DLTensor`](@ref).

This function is exposed by the framework through the [`DLPackExchangeAPI`](@ref).

!!! note

    - As a C function, must not thrown C++ exceptions. - Error propagation via SetError to avoid any direct need of Python API. Due to this `SetError` may have to ensure the GIL is held since it will presumably set a Python error.

# Arguments
* `prototype`: The prototype [`DLTensor`](@ref). Only the dtype, ndim, shape, and device fields are used.
* `out`: The output [`DLManagedTensorVersioned`](@ref).
* `error_ctx`: Context for `SetError`.
* `SetError`: The function to set the error.
# Returns
The owning [`DLManagedTensorVersioned`](@ref)* or NULL on failure. SetError is called exactly when NULL is returned (the implementer must ensure this).
# See also
[`DLPackExchangeAPI`](@ref)
"""
const DLPackManagedTensorAllocator = Ptr{Cvoid}

# typedef int ( * DLPackManagedTensorFromPyObjectNoSync ) ( // void * py_object , // DLManagedTensorVersioned * * out // )
"""
Exports a PyObject* Tensor/NDArray to a [`DLManagedTensorVersioned`](@ref).

This function does not perform any stream synchronization. The consumer should query [`DLPackCurrentWorkStream`](@ref) to get the current work stream and launch kernels on it.

This function is exposed by the framework through the [`DLPackExchangeAPI`](@ref).

!!! note

    - As a C function, must not thrown C++ exceptions.

# Arguments
* `py_object`: The Python object to convert. Must have the same type as the one the [`DLPackExchangeAPI`](@ref) was discovered from.
* `out`: The output [`DLManagedTensorVersioned`](@ref).
# Returns
The owning [`DLManagedTensorVersioned`](@ref)* or NULL on failure with a Python exception set. If the data cannot be described using DLPack this should be a BufferError if possible.
# See also
[`DLPackExchangeAPI`](@ref), [`DLPackCurrentWorkStream`](@ref)
"""
const DLPackManagedTensorFromPyObjectNoSync = Ptr{Cvoid}

# typedef int ( * DLPackDLTensorFromPyObjectNoSync ) ( // void * py_object , // DLTensor * out // )
"""
Exports a PyObject* Tensor/NDArray to a provided [`DLTensor`](@ref).

This function provides a faster interface for temporary, non-owning, exchange. The producer (implementer) still owns the memory of data, strides, shape. The liveness of the [`DLTensor`](@ref) and the data it views is only guaranteed until control is returned.

This function currently assumes that the producer (implementer) can fill in the [`DLTensor`](@ref) shape and strides without the need for temporary allocations.

This function does not perform any stream synchronization. The consumer should query [`DLPackCurrentWorkStream`](@ref) to get the current work stream and launch kernels on it.

This function is exposed by the framework through the [`DLPackExchangeAPI`](@ref).

!!! note

    - As a C function, must not thrown C++ exceptions.

# Arguments
* `py_object`: The Python object to convert. Must have the same type as the one the [`DLPackExchangeAPI`](@ref) was discovered from.
* `out`: The output [`DLTensor`](@ref), whose space is pre-allocated on stack.
# Returns
0 on success, -1 on failure with a Python exception set.
# See also
[`DLPackExchangeAPI`](@ref), [`DLPackCurrentWorkStream`](@ref)
"""
const DLPackDLTensorFromPyObjectNoSync = Ptr{Cvoid}

# typedef int ( * DLPackCurrentWorkStream ) ( // DLDeviceType device_type , // int32_t device_id , // void * * out_current_stream // )
"""
Obtain the current work stream of a device.

Obtain the current work stream of a device from the producer framework. For example, it should map to torch.cuda.current\\_stream in PyTorch.

When device\\_type is kDLCPU, the consumer do not have to query the stream and the producer can simply return NULL when queried. The consumer do not have to do anything on stream sync or setting. So CPU only framework can just provide a dummy implementation that always set out\\_current\\_stream[0] to NULL.

!!! note

    - As a C function, must not thrown C++ exceptions.

# Arguments
* `device_type`: The device type.
* `device_id`: The device id.
* `out_current_stream`: The output current work stream.
# Returns
0 on success, -1 on failure with a Python exception set.
# See also
[`DLPackExchangeAPI`](@ref)
"""
const DLPackCurrentWorkStream = Ptr{Cvoid}

# typedef int ( * DLPackManagedTensorToPyObjectNoSync ) ( // DLManagedTensorVersioned * tensor , // void * * out_py_object // )
"""
Imports a [`DLManagedTensorVersioned`](@ref) to a PyObject* Tensor/NDArray.

Convert an owning [`DLManagedTensorVersioned`](@ref)* to the Python tensor of the producer (implementer) library with the correct type.

This function does not perform any stream synchronization.

This function is exposed by the framework through the [`DLPackExchangeAPI`](@ref).

# Arguments
* `tensor`: The [`DLManagedTensorVersioned`](@ref) to convert the ownership of the tensor is stolen.
* `out_py_object`: The output Python object.
# Returns
0 on success, -1 on failure with a Python exception set.
# See also
[`DLPackExchangeAPI`](@ref)
"""
const DLPackManagedTensorToPyObjectNoSync = Ptr{Cvoid}

"""
    DLPackExchangeAPIHeader

[`DLPackExchangeAPI`](@ref) stable header.

# See also
[`DLPackExchangeAPI`](@ref)
"""
struct DLPackExchangeAPIHeader
    version::DLPackVersion
    prev_api::Ptr{DLPackExchangeAPIHeader}
end

"""
    DLPackExchangeAPI

Framework-specific function pointers table for DLPack exchange.

Additionally to `\\_\\_dlpack\\_\\_()` we define a C function table sharable by

Python implementations via `__dlpack_c_exchange_api__`. This attribute must be set on the type as a Python PyCapsule with name "dlpack\\_exchange\\_api".

A consumer library may use a pattern such as:

```c++

  PyObject *api_capsule = PyObject_GetAttrString(
    (PyObject *)Py_TYPE(tensor_obj), "__dlpack_c_exchange_api__")
  );
  if (api_capsule == NULL) { goto handle_error; }
  MyDLPackExchangeAPI *api = (MyDLPackExchangeAPI *)PyCapsule_GetPointer(
    api_capsule, "dlpack_exchange_api"
  );
  Py_DECREF(api_capsule);
  if (api == NULL) { goto handle_error; }

```

Note that this must be defined on the type. The consumer should look up the attribute on the type and may cache the result for each unique type.

The precise API table is given by:

```c++
 struct MyDLPackExchangeAPI : public DLPackExchangeAPI {
   MyDLPackExchangeAPI() {
     header.version.major = DLPACK_MAJOR_VERSION;
     header.version.minor = DLPACK_MINOR_VERSION;
     header.prev_version_api = nullptr;

     managed_tensor_allocator = MyDLPackManagedTensorAllocator;
     managed_tensor_from_py_object_no_sync = MyDLPackManagedTensorFromPyObjectNoSync;
     managed_tensor_to_py_object_no_sync = MyDLPackManagedTensorToPyObjectNoSync;
     dltensor_from_py_object_no_sync = MyDLPackDLTensorFromPyObjectNoSync;
     current_work_stream = MyDLPackCurrentWorkStream;
  }

  static const DLPackExchangeAPI* Global() {
     static MyDLPackExchangeAPI inst;
     return &inst;
  }
 };
```

Guidelines for leveraging [`DLPackExchangeAPI`](@ref):

There are generally two kinds of consumer needs for DLPack exchange: - N0: library support, where consumer.kernel(x, y, z) would like to run a kernel with the data from x, y, z. The consumer is also expected to run the kernel with the same stream context as the producer. For example, when x, y, z is torch.Tensor, consumer should query exchange\\_api->current\\_work\\_stream to get the current stream and launch the kernel with the same stream. This setup is necessary for no synchronization in kernel launch and maximum compatibility with CUDA graph capture in the producer. This is the desirable behavior for library extension support for frameworks like PyTorch. - N1: data ingestion and retention

Note that obj.\\_\\_dlpack\\_\\_() API should provide useful ways for N1. The primary focus of the current [`DLPackExchangeAPI`](@ref) is to enable faster exchange N0 with the support of the function pointer current\\_work\\_stream.

Array/Tensor libraries should statically create and initialize this structure then return a pointer to [`DLPackExchangeAPI`](@ref) as an int value in Tensor/Array. The [`DLPackExchangeAPI`](@ref)* must stay alive throughout the lifetime of the process.

One simple way to do so is to create a static instance of [`DLPackExchangeAPI`](@ref) within the framework and return a pointer to it. The following code shows an example to do so in C++. It should also be reasonably easy to do so in other languages.
"""
struct DLPackExchangeAPI
    header::DLPackExchangeAPIHeader
    managed_tensor_allocator::DLPackManagedTensorAllocator
    managed_tensor_from_py_object_no_sync::DLPackManagedTensorFromPyObjectNoSync
    managed_tensor_to_py_object_no_sync::DLPackManagedTensorToPyObjectNoSync
    dltensor_from_py_object_no_sync::DLPackDLTensorFromPyObjectNoSync
    current_work_stream::DLPackCurrentWorkStream
end

const ovx_token_t = UInt64

const ovx_primpath_t = UInt64

const ovx_primpath_list_t = UInt64

@cenum ovx_api_status_t::UInt32 begin
    OVX_API_SUCCESS = 0
    OVX_API_ERROR = 1
end

struct ovx_api_result_t
    status::ovx_api_status_t
    error::ovx_string_t
end

struct ovx_string_or_token_t
    token::ovx_token_t
    string::ovx_string_t
end

struct ovx_string_or_prim_path_t
    path::ovx_primpath_t
    string::ovx_string_t
end

mutable struct path_dictionary_context_t end

struct path_dictionary_vtable_t
    create_tokens_from_strings::Ptr{Cvoid}
    create_paths_from_tokens::Ptr{Cvoid}
    create_paths_from_strings::Ptr{Cvoid}
    create_path_list_from_paths::Ptr{Cvoid}
    create_path_list_from_strings::Ptr{Cvoid}
    destroy_path_list::Ptr{Cvoid}
    get_strings_from_tokens::Ptr{Cvoid}
    get_tokens_from_paths::Ptr{Cvoid}
    get_num_paths_from_path_list::Ptr{Cvoid}
    get_paths_from_path_list::Ptr{Cvoid}
    release_error::Ptr{Cvoid}
end

struct path_dictionary_instance_t
    vtable::Ptr{path_dictionary_vtable_t}
    context::Ptr{path_dictionary_context_t}
end

mutable struct ovrtx_renderer_t end

"""
Handle representing a USD stage.
"""
const ovrtx_usd_handle_t = UInt64

"""
Handle representing an event.
"""
const ovrtx_event_handle_t = UInt64

"""
Handle representing a persistent attribute binding.
"""
const ovrtx_attribute_binding_handle_t = UInt64

"""
Handle representing a resource mapping that can be used to unmap it.
"""
const ovrtx_map_handle_t = UInt64

"""
Handle to the result of a ovrtx_step() operation.
"""
const ovrtx_step_result_handle_t = UInt64

"""
Handle to a rendered output; pass to ovrtx_map_render_var_output() to access its data.
"""
const ovrtx_render_var_output_handle_t = UInt64

"""
Handle to the mapping of a rendered output that can be used to unmap it with ovrtx_unmap_render_var_output().
"""
const ovrtx_render_var_output_map_handle_t = UInt64

"""
Identifier of a particular asynchronous operation such as ovrtx_open_usd_from_file() that can be used to poll or wait.
"""
const ovrtx_op_id_t = UInt64

"""
    ovrtx_api_status_t

Return status from a synchronous function call.
"""
@cenum ovrtx_api_status_t::UInt32 begin
    OVRTX_API_SUCCESS = 0
    OVRTX_API_ERROR = 1
    OVRTX_API_TIMEOUT = 2
end

"""
    ovrtx_result_t

Result from a synchronous function call. The status of the call can be checked with the ovrtx_result_t::status member.
"""
struct ovrtx_result_t
    status::ovrtx_api_status_t
end

"""
    ovrtx_enqueue_result_t

Result from an asynchronous function call.

Contains the API call status and the operation index. A non-zero operation index can be used to poll or wait on completion, while [`OVRTX_INVALID_HANDLE`](@ref) means the operation could not be enqueued.

Note that if OVRTX\\_CONFIG\\_SYNC\\_MODE is active any error from the async operation itself will cause status to be OVRTX\\_API\\_ERROR and the error is obtainable using [`ovrtx_get_last_error`](@ref)(). In addition to this a valid operation index is still delivered and can be waited on (even though the operation has already failed).
"""
struct ovrtx_enqueue_result_t
    status::ovrtx_api_status_t
    op_index::ovrtx_op_id_t
end

@cenum ovrtx_event_status_t::UInt32 begin
    OVRTX_EVENT_PENDING = 0
    OVRTX_EVENT_COMPLETED = 1
    OVRTX_EVENT_FAILURE = 2
end

"""
    ovrtx_op_wait_result_t

Result of waiting on an ovrtx_op_id_t with ovrtx_wait_op().
"""
struct ovrtx_op_wait_result_t
    error_op_ids::Ptr{ovrtx_op_id_t}
    num_error_ops::Csize_t
    lowest_pending_op_id::ovrtx_op_id_t
end

"""
    ovrtx_timeout_t

Represents a timeout duration in nanoseconds.
"""
struct ovrtx_timeout_t
    time_out_ns::UInt64
end

"""
    ovrtx_cuda_sync_t

Represents a CUDA event to wait for on a particular stream.
"""
struct ovrtx_cuda_sync_t
    stream::Csize_t
    wait_event::Csize_t
end

struct ovrtx_device_t
    device_type::Int32
    device_id::Int32
end

struct ovrtx_event_description_t
    device::ovrtx_device_t
end

"""
    ovrtx_write_bits_t

` ovrtx_attribute_types Stage attribute writer types`

@{
"""
@cenum ovrtx_write_bits_t::UInt32 begin
    OVRTX_DIRTY_MASK_REPLACE = 0
    OVRTX_DIRTY_MASK_OR = 1
    OVRTX_DIRTY_MASK_AND = 2
end

@cenum ovrtx_data_access_t::UInt32 begin
    OVRTX_DATA_ACCESS_ASYNC = 0
    OVRTX_DATA_ACCESS_SYNC = 1
end

"""
    ovrtx_prim_list_t

A list of paths to prims in the runtime stage.
"""
struct ovrtx_prim_list_t
    prim_paths::Ptr{ovx_string_t}
    num_paths::Csize_t
end

struct ovrtx_mapping_desc_t
    device_type::Int32
    device_id::Int32
end

"""
    ovrtx_binding_prim_mode_t

Describes how to handle attempts to write to paths in the runtime stage that do not exist.
"""
@cenum ovrtx_binding_prim_mode_t::UInt32 begin
    OVRTX_BINDING_PRIM_MODE_EXISTING_ONLY = 0
    OVRTX_BINDING_PRIM_MODE_MUST_EXIST = 1
    OVRTX_BINDING_PRIM_MODE_CREATE_NEW = 2
end

"""
    ovrtx_attribute_semantic_t

Used to differentiate the intended data layout and usage of a given attribute type.

For example a transform attribute can be written as different data layouts that are automatically converted into renderer's data layout. This enum is used to differentiate between data layouts written to the same attribute.
"""
@cenum ovrtx_attribute_semantic_t::UInt32 begin
    OVRTX_SEMANTIC_NONE = 0
    OVRTX_SEMANTIC_XFORM_MAT4x4 = 1
    OVRTX_SEMANTIC_XFORM_POS3d_ROT4f_SCALE3f = 2
    OVRTX_SEMANTIC_XFORM_POS3d_ROT3x3f = 3
    OVRTX_SEMANTIC_PATH_STRING = 4
    OVRTX_SEMANTIC_TOKEN_STRING = 5
    OVRTX_SEMANTIC_TOKEN_ID = 6
    OVRTX_SEMANTIC_PATH_ID = 7
    OVRTX_SEMANTIC_TAG = 8
end

struct ovrtx_xform_matrix44d_t
    v::NTuple{16, Cdouble}
end

struct ovrtx_xform_pos3d_rot4f_scale3f_t
    position::NTuple{3, Cdouble}
    rot_quat_xyzw::NTuple{4, Cfloat}
    scale::NTuple{3, Cfloat}
    padding::UInt32
end

struct ovrtx_xform_pos3d_rot3x3f_t
    position::NTuple{3, Cdouble}
    rot_matrix::NTuple{9, Cfloat}
    padding::UInt32
end

"""
    ovrtx_attribute_type_t

Describes the type of an attribute to be written to the runtime stage.
"""
struct ovrtx_attribute_type_t
    dtype::DLDataType
    is_array::Bool
    semantic::ovrtx_attribute_semantic_t
end

"""
    ovrtx_binding_flag_t

Flags giving hints to the renderer about the expected use of a binding.
"""
@cenum ovrtx_binding_flag_t::UInt32 begin
    OVRTX_BINDING_FLAG_NONE = 0
    OVRTX_BINDING_FLAG_OPTIMIZE = 1
end

"""
    ovrtx_binding_desc_t

Describes a binding to an attribute on a list of prims so that they can be written to.
"""
struct ovrtx_binding_desc_t
    prim_list::ovrtx_prim_list_t
    prims_list_handle::ovx_primpath_list_t
    attribute_name::ovx_string_or_token_t
    attribute_type::ovrtx_attribute_type_t
    prim_mode::ovrtx_binding_prim_mode_t
    flags::ovrtx_binding_flag_t
end

"""
    ovrtx_binding_desc_or_handle_t

Represents either an [`ovrtx_binding_desc_t`](@ref) or an [`ovrtx_attribute_binding_handle_t`](@ref) allowing either to be passed to  ovrtx_write_attribute() and ovrtx_map_attribute().  The use of persistent bindings allows for more optimal writes in the renderer when an attribute will be written to  repeatedly.

If ovrtx_binding_desc_or_handle_t::binding_handle is non-zero then it will be used, otherwise  ovrtx_binding_desc_or_handle_t::binding_desc will be used.
"""
struct ovrtx_binding_desc_or_handle_t
    binding_desc::ovrtx_binding_desc_t
    binding_handle::ovrtx_attribute_binding_handle_t
end

struct ovrtx_input_buffer_t
    tensors::Ptr{DLTensor}
    tensor_count::UInt64
    dirty_bits::Ptr{UInt8}
    dirty_bits_size::Csize_t
    access_cuda_sync::ovrtx_cuda_sync_t
    done_cuda_sync::ovrtx_cuda_sync_t
end

"""
    ovrtx_read_dest_t

Optional destination for ovrtx_read_attribute() scalar read data.

When the `tensor` pointer is non-null the read writes directly into the caller-provided buffer instead of allocating internal storage. The tensor must be pre-allocated with shape [prim\\_count] and a matching element size, including [`DLTensor`](@ref)::dtype.lanes for multi-component attributes. For GPU tensors (`kDLCUDA`) the data is copied via cudaMemcpy. Must be NULL for array attributes (variable-length per prim).
"""
struct ovrtx_read_dest_t
    tensor::Ptr{DLTensor}
    access_cuda_sync::ovrtx_cuda_sync_t
    done_cuda_sync::ovrtx_cuda_sync_t
end

"""
    ovrtx_output_buffer_t

!!! compat "Deprecated"

    Use [`ovrtx_render_var_output_t`](@ref) tensors[]->dl instead. Kept for source compatibility with existing consumers.
"""
struct ovrtx_output_buffer_t
    dl::DLTensor
    cuda_sync::ovrtx_cuda_sync_t
end

"""
    ovrtx_render_var_param_t

Named render variable param represented as a [`DLTensor`](@ref) with labels.

The [`DLTensor`](@ref)'s dtype encodes the value type (e.g. kDLFloat/32 for float, kDLUInt/64 for uint64\\_t) and shape encodes scalar vs. array (e.g. {1} for scalar, {N} for array, {4,4} for a matrix). Param values are always CPU-resident: param.device is {kDLCPU, 0}.
"""
struct ovrtx_render_var_param_t
    dl::DLTensor
    name::ovx_string_t
    doc::ovx_string_t
end

"""
    ovrtx_attribute_mapping_t

Mapped attribute that can be written to until unmapped.
"""
struct ovrtx_attribute_mapping_t
    map_handle::ovrtx_map_handle_t
    dl::DLTensor
end

"""
Handle to the result of a ovrtx_query_prims() operation.
"""
const ovrtx_query_handle_t = UInt64

@cenum ovrtx_filter_kind_t::UInt32 begin
    OVRTX_FILTER_PRIM_TYPE = 0
    OVRTX_FILTER_HAS_ATTRIBUTE = 1
end

struct ovrtx_filter_t
    kind::ovrtx_filter_kind_t
    name::ovx_string_or_token_t
end

@cenum ovrtx_attribute_filter_mode_t::UInt32 begin
    OVRTX_ATTRIBUTE_FILTER_NONE = 0
    OVRTX_ATTRIBUTE_FILTER_ALL = 1
    OVRTX_ATTRIBUTE_FILTER_SPECIFIC = 2
end

struct ovrtx_attribute_filter_t
    mode::ovrtx_attribute_filter_mode_t
    attribute_names::Ptr{ovx_string_or_token_t}
    attribute_name_count::Csize_t
end

struct ovrtx_query_desc_t
    require_all::Ptr{ovrtx_filter_t}
    require_all_count::Csize_t
    require_any::Ptr{ovrtx_filter_t}
    require_any_count::Csize_t
    exclude::Ptr{ovrtx_filter_t}
    exclude_count::Csize_t
    attribute_filter::ovrtx_attribute_filter_t
end

"""
    ovrtx_attribute_desc_t

Describes a single attribute on a group of prims in the query result. The name is a token that can be resolved to a string via the path dictionary's get\\_strings\\_from\\_tokens.
"""
struct ovrtx_attribute_desc_t
    name::ovx_token_t
    type::ovrtx_attribute_type_t
end

"""
    ovrtx_query_prim_group_t

A group of prims sharing the same attribute schema, returned by ovrtx_fetch_query_results().

All prims in a group are guaranteed to have the same set of attributes. The ovrtx_query_prim_group_t::prim_list_handle can be plugged directly into ovrtx_binding_desc_t::prims_list_handle for subsequent read or write operations.
"""
struct ovrtx_query_prim_group_t
    prim_count::Csize_t
    prim_list_handle::ovx_primpath_list_t
    attributes::Ptr{ovrtx_attribute_desc_t}
    attribute_count::Csize_t
end

"""
    ovrtx_query_result_t

Result of a ovrtx_query_prims() operation retrieved via ovrtx_fetch_query_results().

Contains one group per matching bucket. All pointers are valid until ovrtx_release_query_results() is called.
"""
struct ovrtx_query_result_t
    groups::Ptr{ovrtx_query_prim_group_t}
    group_count::Csize_t
    total_prim_count::Csize_t
end

"""
Handle to the result of a ovrtx_read_attribute() operation.
"""
const ovrtx_read_handle_t = UInt64

"""
Handle to a fetched read result that can be released.
"""
const ovrtx_read_map_handle_t = UInt64

"""
    ovrtx_read_output_t

Attribute read output retrieved via ovrtx_fetch_read_result().

For scalar attributes ovrtx_read_output_t::buffer_count is 1 and the single tensor has shape [prim\\_count]. For array attributes ovrtx_read_output_t::buffer_count equals ovrtx_read_output_t::prim_count with one tensor per prim of variable length. Multi-component C attribute tensors encode component count in [`DLTensor`](@ref)::dtype.lanes.

When a user-provided destination tensor was passed to ovrtx_read_attribute(), buffer\\_count is 0 (the data was written directly into the caller's tensor).
"""
struct ovrtx_read_output_t
    map_handle::ovrtx_read_map_handle_t
    buffers::Ptr{ovrtx_output_buffer_t}
    buffer_count::Csize_t
    prim_count::Csize_t
    is_array::Bool
end

"""
    ovrtx_render_product_set_t

Set of RenderProducts that will be stepped.
"""
struct ovrtx_render_product_set_t
    render_products::Ptr{ovx_string_t}
    num_render_products::Csize_t
end

"""
    ovrtx_renderer_event_status_t

Enum representing the status of an asynchronous operation.
"""
@cenum ovrtx_renderer_event_status_t::UInt32 begin
    OVRTX_RENDERER_EVENT_PENDING = 0
    OVRTX_RENDERER_EVENT_COMPLETED = 1
    OVRTX_RENDERER_EVENT_FAILED = 2
end

"""
    ovrtx_render_product_render_var_output_t

Name an associated handle of a particular RenderVar's output in a RenderProduct output.
"""
struct ovrtx_render_product_render_var_output_t
    render_var_name::ovx_string_t
    output_handle::ovrtx_render_var_output_handle_t
end

"""
    ovrtx_render_product_frame_output_t

Output of a particular RenderProduct for a particular frame.

May contain one or more RenderVar outputs in ovrtx_render_product_frame_output_t::output_render_vars which may each be mapped to get access to the output data using ovrtx_map_render_var_output().
"""
struct ovrtx_render_product_frame_output_t
    frame_start_time::Cdouble
    frame_end_time::Cdouble
    output_render_vars::Ptr{ovrtx_render_product_render_var_output_t}
    render_var_count::Csize_t
end

"""
    ovrtx_render_product_output_t

The output of a particular RenderProduct for a particular ovrtx_step() operation.
"""
struct ovrtx_render_product_output_t
    render_product_path::ovx_string_t
    output_frames_produced::Cfloat
    output_frames::Ptr{ovrtx_render_product_frame_output_t}
    output_frame_count::Csize_t
end

"""
    ovrtx_render_product_set_outputs_t

The set of RenderProduct outputs for a ovrtx_step() operation.

Depending on the sensor configuration, each ovrtx_step() may produce zero or more frames for each RenderProduct in the ovrtx_render_product_set_t passed to ovrtx_step().
"""
struct ovrtx_render_product_set_outputs_t
    status::ovrtx_event_status_t
    error_message::ovx_string_t
    simulation_start_time::Cdouble
    simulation_end_time::Cdouble
    outputs::Ptr{ovrtx_render_product_output_t}
    output_count::Csize_t
    start_time::Cdouble
    end_time::Cdouble
end

"""
    ovrtx_pick_query_desc_t

Pick rectangle in RenderProduct pixel coordinates (inclusive left/top, exclusive right/bottom).
"""
struct ovrtx_pick_query_desc_t
    render_product_path::ovx_string_t
    left::Int32
    top::Int32
    right::Int32
    bottom::Int32
    flags::UInt32
end

"""
    ovrtx_selection_fill_mode_t

Selection-outline interior (fill) mode. Mirrors `OutlineMode` in the underlying RTX shader. Set globally via OVRTX_CONFIG_SELECTION_FILL_MODE at renderer creation.
"""
@cenum ovrtx_selection_fill_mode_t::UInt32 begin
    OVRTX_SELECTION_FILL_MODE_EDGE_ONLY = 0
    OVRTX_SELECTION_FILL_MODE_GLOBAL = 1
    OVRTX_SELECTION_FILL_MODE_GROUP_OUTLINE_COLOR = 2
    OVRTX_SELECTION_FILL_MODE_GROUP_FILL_COLOR = 3
end

"""
    ovrtx_selection_group_style_t

Visual styling for one selection-outline group. RGBA components in [0..1].
"""
struct ovrtx_selection_group_style_t
    outline_color::NTuple{4, Cfloat}
    fill_color::NTuple{4, Cfloat}
end

"""
    ovrtx_map_device_type_t

Specifies which device (CPU or GPU) should be used to map a given output.
"""
@cenum ovrtx_map_device_type_t::UInt32 begin
    OVRTX_MAP_DEVICE_TYPE_DEFAULT = 0
    OVRTX_MAP_DEVICE_TYPE_CPU = 1
    OVRTX_MAP_DEVICE_TYPE_CUDA = 2
    OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY = 3
end

"""
    ovrtx_map_output_description_t

Description of the device and synchronization for mapping an output to be passed to ovrtx_map_render_var_output().

The device\\_type applies to tensors only. Param entries are always mapped to CPU regardless of this setting.
"""
struct ovrtx_map_output_description_t
    device_type::ovrtx_map_device_type_t
    sync_stream::Csize_t
end

"""
    ovrtx_render_var_tensor_t

One tensor slot in a mapped render variable output (DLPack view plus labels).

Lifetime: pointers are valid from [`ovrtx_map_render_var_output`](@ref)() until [`ovrtx_unmap_render_var_output`](@ref)().
"""
struct ovrtx_render_var_tensor_t
    dl::Ptr{DLTensor}
    name::Ptr{ovx_string_t}
    doc::Ptr{ovx_string_t}
end

"""
    ovrtx_render_var_output_t

The output of a particular RenderVar for a particular RenderProduct on a particular frame.

Contains zero or more named tensor slots (ovrtx_render_var_tensor_t) and zero or more named param entries. Tensor data may reside on CPU or CUDA depending on the map request. Params are always CPU-resident.

Lifetime: valid from [`ovrtx_map_render_var_output`](@ref)() until [`ovrtx_unmap_render_var_output`](@ref)().

!!! note

    ABI break from pre-0.3: the [`ovrtx_output_buffer_t`](@ref) buffer field has been removed. Consumers must migrate to the tensors[]/params[] layout.
"""
struct ovrtx_render_var_output_t
    status::ovrtx_event_status_t
    error_message::ovx_string_t
    map_handle::ovrtx_render_var_output_map_handle_t
    cuda_sync::ovrtx_cuda_sync_t
    name::ovx_string_t
    type::ovx_string_t
    doc::ovx_string_t
    version::Cint
    num_tensors::Csize_t
    tensors::Ptr{ovrtx_render_var_tensor_t}
    num_params::Csize_t
    params::Ptr{ovrtx_render_var_param_t}
end

"""
    ovrtx_op_counter_t

Named resource counter for tracking progress of specific resource types.

Counter semantics: - name: Identifies the resource type (e.g., "shaders", "textures", "materials") - current: Number of items processed so far - total: Total items to process, or 0 if the total is not yet known
"""
struct ovrtx_op_counter_t
    name::ovx_string_t
    current::UInt64
    total::UInt64
end

"""
    ovrtx_op_status_t

Operation status information for long-running operations.

Progress semantics: - Range [0.0, 1.0] where 1.0 = complete - Negative value indicates indeterminate progress
"""
struct ovrtx_op_status_t
    op_id::ovrtx_op_id_t
    state::ovrtx_event_status_t
    progress::Cdouble
    counters::Ptr{ovrtx_op_counter_t}
    counter_count::Csize_t
end

"""
    ovrtx_log_severity_t

Log severity levels for operation messages.
"""
@cenum ovrtx_log_severity_t::Int32 begin
    OVRTX_LOG_INFO = -1
    OVRTX_LOG_WARNING = 0
    OVRTX_LOG_ERROR = 1
    OVRTX_LOG_FATAL = 2
end

# typedef void ( * ovrtx_log_callback_t ) ( ovrtx_log_severity_t severity , double timestamp , ovx_string_t message , void * user_data )
"""
Callback function type for receiving log messages.

The callback is process-global (see [`ovrtx_set_log_callback`](@ref)) and may be invoked from any thread. The implementation serializes invocations so the callback body itself does not need its own mutex, but it must still be thread-safe with respect to whatever data it touches outside.

The message string is only valid for the duration of the callback. If the message needs to be retained, copy it before returning.

# Arguments
* `severity`: Severity level of the message
* `timestamp`: Wall-clock time in seconds since the epoch
* `message`: Log message text (valid only during callback)
* `user_data`: User-provided context from [`ovrtx_set_log_callback`](@ref)
"""
const ovrtx_log_callback_t = Ptr{Cvoid}

"""
    ovrtx_config_key_type_t

Key type tag for ovrtx_config_entry_t; selects which key and value union members are valid.
"""
@cenum ovrtx_config_key_type_t::UInt32 begin
    OVRTX_CONFIG_KEY_TYPE_BOOL = 0
    OVRTX_CONFIG_KEY_TYPE_INT64 = 1
    OVRTX_CONFIG_KEY_TYPE_UINT64 = 2
    OVRTX_CONFIG_KEY_TYPE_DOUBLE = 3
    OVRTX_CONFIG_KEY_TYPE_STRING = 4
    OVRTX_CONFIG_KEY_TYPE_BLOB = 5
    OVRTX_CONFIG_KEY_TYPE_COUNT = 6
end

"""
    ovrtx_config_bool_t

Boolean config keys. Value type: bool. Used at init and create\\_renderer (must match).
"""
@cenum ovrtx_config_bool_t::UInt32 begin
    OVRTX_CONFIG_SYNC_MODE = 0
    OVRTX_CONFIG_ENABLE_PROFILING = 1
    OVRTX_CONFIG_READ_GPU_TRANSFORMS = 2
    OVRTX_CONFIG_KEEP_SYSTEM_ALIVE = 3
    OVRTX_CONFIG_USE_VULKAN = 4
    OVRTX_CONFIG_SELECTION_OUTLINE_ENABLED = 5
    OVRTX_CONFIG_ENABLE_GEOMETRY_STREAMING = 6
    OVRTX_CONFIG_ENABLE_GEOMETRY_STREAMING_LOD = 7
    OVRTX_CONFIG_ENABLE_SPG = 8
    OVRTX_CONFIG_ENABLE_MOTION_BVH = 9
    OVRTX_CONFIG_BOOL_COUNT = 10
end

"""
    ovrtx_config_string_t

String config keys. Value type: [`ovx_string_t`](@ref). Used at init and create\\_renderer (must match).
"""
@cenum ovrtx_config_string_t::UInt32 begin
    OVRTX_CONFIG_BINARY_PACKAGE_ROOT_PATH = 0
    OVRTX_CONFIG_LOG_FILE_PATH = 1
    OVRTX_CONFIG_LOG_LEVEL = 2
    OVRTX_CONFIG_ACTIVE_CUDA_GPUS = 3
    OVRTX_CONFIG_STRING_COUNT = 4
end

"""
    ovrtx_config_int64_t

Int64 config keys. Value type: int64\\_t. Used at create\\_renderer.
"""
@cenum ovrtx_config_int64_t::UInt32 begin
    OVRTX_CONFIG_SELECTION_OUTLINE_WIDTH = 0
    OVRTX_CONFIG_SELECTION_FILL_MODE = 1
    OVRTX_CONFIG_INT64_COUNT = 2
end

"""
    ovrtx_config_uint64_t

Uint64 config keys (reserved for future use). Value type: uint64\\_t.
"""
@cenum ovrtx_config_uint64_t::UInt32 begin
    OVRTX_CONFIG_UINT64_COUNT = 0
end

"""
    ovrtx_config_double_t

Double config keys (reserved for future use). Value type: double.
"""
@cenum ovrtx_config_double_t::UInt32 begin
    OVRTX_CONFIG_DOUBLE_COUNT = 0
end

"""
    ovrtx_config_blob_t

Blob config keys (reserved for future use). Value type: ptr + size.
"""
@cenum ovrtx_config_blob_t::UInt32 begin
    OVRTX_CONFIG_BLOB_COUNT = 0
end

struct var"##Ctag#277"
    data::NTuple{4, UInt8}
end

function Base.getproperty(x::Ptr{var"##Ctag#277"}, f::Symbol)
    f === :bool_key && return Ptr{ovrtx_config_bool_t}(x + 0)
    f === :int64_key && return Ptr{ovrtx_config_int64_t}(x + 0)
    f === :uint64_key && return Ptr{ovrtx_config_uint64_t}(x + 0)
    f === :double_key && return Ptr{ovrtx_config_double_t}(x + 0)
    f === :string_key && return Ptr{ovrtx_config_string_t}(x + 0)
    f === :blob_key && return Ptr{ovrtx_config_blob_t}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#277", f::Symbol)
    r = Ref{var"##Ctag#277"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#277"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#277"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::var"##Ctag#277", private::Bool = false)
    (:bool_key, :int64_key, :uint64_key, :double_key, :string_key, :blob_key, if private
            fieldnames(typeof(x))
        else
            ()
        end...)
end

struct var"##Ctag#278"
    data::NTuple{16, UInt8}
end

function Base.getproperty(x::Ptr{var"##Ctag#278"}, f::Symbol)
    f === :bool_value && return Ptr{Bool}(x + 0)
    f === :int_value && return Ptr{Int64}(x + 0)
    f === :uint_value && return Ptr{UInt64}(x + 0)
    f === :double_value && return Ptr{Cdouble}(x + 0)
    f === :string_value && return Ptr{ovx_string_t}(x + 0)
    f === :blob_value && return Ptr{var"##Ctag#279"}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#278", f::Symbol)
    r = Ref{var"##Ctag#278"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#278"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#278"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::var"##Ctag#278", private::Bool = false)
    (:bool_value, :int_value, :uint_value, :double_value, :string_value, :blob_value, if private
            fieldnames(typeof(x))
        else
            ()
        end...)
end

"""
    ovrtx_config_entry_t

A config entry. key\\_type selects which member of key and value is valid.
"""
struct ovrtx_config_entry_t
    data::NTuple{24, UInt8}
end

function Base.getproperty(x::Ptr{ovrtx_config_entry_t}, f::Symbol)
    f === :key_type && return Ptr{ovrtx_config_key_type_t}(x + 0)
    f === :key && return Ptr{var"##Ctag#277"}(x + 4)
    f === :value && return Ptr{var"##Ctag#278"}(x + 8)
    return getfield(x, f)
end

function Base.getproperty(x::ovrtx_config_entry_t, f::Symbol)
    r = Ref{ovrtx_config_entry_t}(x)
    ptr = Base.unsafe_convert(Ptr{ovrtx_config_entry_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{ovrtx_config_entry_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::ovrtx_config_entry_t, private::Bool = false)
    (:key_type, :key, :value, if private
            fieldnames(typeof(x))
        else
            ()
        end...)
end

"""
    ovrtx_config_t

Config passed to ovrtx_initialize() and ovrtx_create_renderer(). Non-null required; empty (entry\\_count 0) means defaults.
"""
struct ovrtx_config_t
    entries::Ptr{ovrtx_config_entry_t}
    entry_count::Csize_t
end

"""
    ovrtx_get_version(out_major, out_minor, out_patch)

Get the ovrtx API version that the library was compiled with.

This function can be called at any time, including before [`ovrtx_initialize`](@ref)(). It can be used to verify that the loaded library matches the header version at runtime.

The compile-time version is also available via [`OVRTX_VERSION_MAJOR`](@ref), [`OVRTX_VERSION_MINOR`](@ref), and [`OVRTX_VERSION_PATCH`](@ref) macros.

# Arguments
* `out_major`:\\[out\\] Major version number
* `out_minor`:\\[out\\] Minor version number
* `out_patch`:\\[out\\] Patch version number
"""
function ovrtx_get_version(out_major, out_minor, out_patch)
    @ccall libovrtx.ovrtx_get_version(out_major::Ptr{UInt32}, out_minor::Ptr{UInt32}, out_patch::Ptr{UInt32})::Cvoid
end

"""
    ovrtx_register_schema_paths(config)

Register ovrtx's USD schema and plugin discovery paths with the process's USD plugin search environment, without loading USD or initializing the renderer.

This is intended for applications that share an OpenUSD runtime between multiple subsystems (for example ovrtx and ovphysx). USD's schema registry is populated only once for the process, so every subsystem that contributes schema/plugin paths must have published them before the registry is first consulted (typically when the first stage is opened). Each subsystem calls its own equivalent (e.g. `ovphysx\\_prepare\\_usd\\_plugins()`, `[`ovrtx_register_schema_paths`](@ref)(...)`) before any of them initialize, after which the order of initialize calls no longer matters.

Binary package root resolution (highest precedence first): 1. The `OMNI_USD_PLUGINS_BASE_PATH` environment variable, if set. 2. `OVRTX_CONFIG_BINARY_PACKAGE_ROOT_PATH` from `config`, if `config` is non-null and contains that entry. 3. The directory of the loaded ovrtx loader library (default).

Notes: - Safe to call before ovrtx_initialize() and before ovrtx_create_renderer(). - Idempotent for matching roots: the first call performs registration; subsequent calls with the same effective root are no-ops. - **First-call wins.** Once schema/plugin paths have been registered against an effective binary package root, subsequent calls (here, or via ovrtx_initialize() / ovrtx_create_renderer()) that resolve to a different effective root log a warning to stderr and are no-ops; `PXR_PLUGINPATH_NAME` stays anchored at the first-registered root (the contract is one-shot per process, since USD's plug system reads it once during static initialization). - Calling this after USD has already been loaded and the schema registry populated has no retroactive effect on previously-discovered schemas. - This function does not allocate the ovrtx system; it only adjusts process-global environment used by USD's plugin discovery.

# Arguments
* `config`: Optional configuration (may be NULL). When non-null, the `OVRTX_CONFIG_BINARY_PACKAGE_ROOT_PATH` entry, if present, anchors the bundled `usd\\_plugins/` tree. Pass the same config (or one with an equivalent `binary_package_root_path`) that you will subsequently supply to ovrtx_initialize() / ovrtx_create_renderer().
# Returns
Always **OVRTX\\_API\\_SUCCESS**. This API does not surface failure through the return code or ovrtx_get_last_error(); a mismatched root logs a warning to stderr instead. (This is the one exception to the file-level return-value rule above.)
"""
function ovrtx_register_schema_paths(config)
    @ccall libovrtx.ovrtx_register_schema_paths(config::Ptr{ovrtx_config_t})::ovrtx_result_t
end

"""
    ovrtx_initialize(config)

Initialize the ovrtx loader or increase its ref count.

It is allowed to call this function multiple times and for each successful call a corresponding call to [`ovrtx_shutdown`](@ref)() is required.

Note that explicit initialization is not required: creating a render instance with [`ovrtx_create_renderer`](@ref)() will also initialize the system if needed. Calling [`ovrtx_initialize`](@ref)() and [`ovrtx_shutdown`](@ref)() can be used to prevent system shutdown and initialization if the renderer is recreated multiple times.

# Arguments
* `config`: Configuration for the ovrtx system (see ovrtx_config.h). Must be non-null; may be empty (entry\\_count 0) for defaults.
# Returns
- **OVRTX\\_API\\_SUCCESS** if the system was initialized or ref-count was increased successfully, - **OVRTX\\_API\\_ERROR** if initialization failed.
"""
function ovrtx_initialize(config)
    @ccall libovrtx.ovrtx_initialize(config::Ptr{ovrtx_config_t})::ovrtx_result_t
end

# no prototype is found for this function at ovrtx.h:143:20, please use with caution
"""
    ovrtx_shutdown()

Shuts down the ovrtx system or decreases it's ref count. One call per call to [`ovrtx_initialize`](@ref)() is required. Note that any render instances that have not been destroyed will keep the system alive until they are destroyed.

# Returns
- **OVRTX\\_API\\_SUCCESS** if the system was released successfully (though might still be loaded), - **OVRTX\\_API\\_ERROR** if the system shutdown failed or was not initialized.
"""
function ovrtx_shutdown()
    @ccall libovrtx.ovrtx_shutdown()::ovrtx_result_t
end

"""
    ovrtx_create_renderer(config, out_renderer)

Create a new renderer instance. System initialization is done automatically if [`ovrtx_initialize`](@ref)() has not been called yet, but in that case the config must contain both initialization and renderer settings. The system will be kept running until a corresponding call to [`ovrtx_destroy_renderer`](@ref)().

# Arguments
* `config`: Configuration for the renderer (see ovrtx_config.h). Must be non-null; may be empty (entry\\_count 0) for defaults. Keys are enum-based (e.g. OVRTX\\_CONFIG\\_SYNC\\_MODE, OVRTX\\_CONFIG\\_LOG\\_FILE\\_PATH, OVRTX\\_CONFIG\\_ACTIVE\\_CUDA\\_GPUS); build entries with ovrtx\\_config\\_entry\\_bool() and ovrtx\\_config\\_entry\\_string().
* `out_renderer`: [out] Renderer instance
# Returns
- **OVRTX\\_API\\_SUCCESS** if the renderer was created successfully, - **OVRTX\\_API\\_ERROR** if the renderer creation failed.
"""
function ovrtx_create_renderer(config, out_renderer)
    @ccall libovrtx.ovrtx_create_renderer(config::Ptr{ovrtx_config_t}, out_renderer::Ptr{Ptr{ovrtx_renderer_t}})::ovrtx_result_t
end

"""
    ovrtx_destroy_renderer(renderer)

Destroy a renderer instance.

# Arguments
* `renderer`: Renderer instance to destroy
# Returns
- **OVRTX\\_API\\_SUCCESS** if the renderer was destroyed successfully, - **OVRTX\\_API\\_ERROR** if the renderer destruction failed.
"""
function ovrtx_destroy_renderer(renderer)
    @ccall libovrtx.ovrtx_destroy_renderer(renderer::Ptr{ovrtx_renderer_t})::ovrtx_result_t
end

"""
    ovrtx_open_usd_from_file(instance, file_name)

Enqueue an asynchronous operation to open a USD file as the root layer of the runtime stage.

This resets the current stage to empty and then loads the given file as the root sublayer. Only one root layer can be active at a time; call this again to replace it.

Note that errors occuring during loading (including a given USD file not being found) will be reported through the ovrtx_op_wait_result_t::error_op_ids list.

# Arguments
* `instance`: Renderer instance
* `file_name`: Path to the USD file to open
# Returns
- **OVRTX\\_API\\_SUCCESS** if the operation was enqueued successfully. - **OVRTX\\_API\\_ERROR** if the operation was not enqueued successfully.
"""
function ovrtx_open_usd_from_file(instance, file_name)
    @ccall libovrtx.ovrtx_open_usd_from_file(instance::Ptr{ovrtx_renderer_t}, file_name::ovx_string_t)::ovrtx_enqueue_result_t
end

"""
    ovrtx_open_usd_from_string(instance, root_layer_content)

Enqueue an asynchronous operation to open a USD stage from inline USDA content.

This resets the current stage to empty and then loads the given layer content as the root sublayer. Only one root layer can be active at a time; call this again to replace it.

# Arguments
* `instance`: Renderer instance
* `root_layer_content`: USDA content string for the root layer
# Returns
- **OVRTX\\_API\\_SUCCESS** if the operation was enqueued successfully. - **OVRTX\\_API\\_ERROR** if the operation was not enqueued successfully.
"""
function ovrtx_open_usd_from_string(instance, root_layer_content)
    @ccall libovrtx.ovrtx_open_usd_from_string(instance::Ptr{ovrtx_renderer_t}, root_layer_content::ovx_string_t)::ovrtx_enqueue_result_t
end

"""
    ovrtx_add_usd_reference_from_file(instance, layer_file, prefix_path, out_handle)

Enqueue an asynchronous operation to add a USD file as a reference at the given prim path.

A new prim is created at `prefix_path` and the layer is added as a reference on that prim. The prefix path must be an absolute prim path (starting with '/') and must not already exist.

`out_handle` is reserved when the add operation is enqueued. A non-zero handle does not mean the USD was loaded. In normal async mode, execution errors are reported through ovrtx_wait_op(). If OVRTX_CONFIG_SYNC_MODE is active, this function returns OVRTX_API_ERROR for those execution errors. Details can be queried with ovrtx_get_last_error().

# Arguments
* `instance`: Renderer instance
* `layer_file`: Path to the USD file to add as a reference
* `prefix_path`: Absolute prim path where the reference will be created
* `out_handle`: [out] Reserved handle for the added reference. It may be used to queue dependent stream-ordered operations, but does not by itself indicate the load succeeded.
# Returns
- **OVRTX\\_API\\_SUCCESS** if the operation was enqueued successfully. - **OVRTX\\_API\\_ERROR** if the operation was not enqueued successfully, or if an execution error occurs in sync mode.
"""
function ovrtx_add_usd_reference_from_file(instance, layer_file, prefix_path, out_handle)
    @ccall libovrtx.ovrtx_add_usd_reference_from_file(instance::Ptr{ovrtx_renderer_t}, layer_file::ovx_string_t, prefix_path::ovx_string_t, out_handle::Ptr{ovrtx_usd_handle_t})::ovrtx_enqueue_result_t
end

"""
    ovrtx_add_usd_reference_from_string(instance, layer_content, prefix_path, out_handle)

Enqueue an asynchronous operation to add inline USDA content as a reference at the given prim path.

A new prim is created at `prefix_path` and the layer content is added as a reference on that prim. The prefix path must be an absolute prim path (starting with '/') and must not already exist.

`out_handle` is reserved when the add operation is enqueued. A non-zero handle does not mean the USD was loaded. In normal async mode, execution errors are reported through ovrtx_wait_op(). If OVRTX_CONFIG_SYNC_MODE is active, this function returns OVRTX_API_ERROR for those execution errors. Details can be queried with ovrtx_get_last_error().

# Arguments
* `instance`: Renderer instance
* `layer_content`: USDA content string for the reference layer
* `prefix_path`: Absolute prim path where the reference will be created
* `out_handle`: [out] Reserved handle for the added reference. It may be used to queue dependent stream-ordered operations, but does not by itself indicate the load succeeded.
# Returns
- **OVRTX\\_API\\_SUCCESS** if the operation was enqueued successfully. - **OVRTX\\_API\\_ERROR** if the operation was not enqueued successfully, or if an execution error occurs in sync mode.
"""
function ovrtx_add_usd_reference_from_string(instance, layer_content, prefix_path, out_handle)
    @ccall libovrtx.ovrtx_add_usd_reference_from_string(instance::Ptr{ovrtx_renderer_t}, layer_content::ovx_string_t, prefix_path::ovx_string_t, out_handle::Ptr{ovrtx_usd_handle_t})::ovrtx_enqueue_result_t
end

"""
    ovrtx_remove_usd(instance, add_usd_handle)

Enqueue an asynchronous operation to remove a previously added USD reference from the runtime stage. All prims added to the stage during the add operation will be removed.

# Arguments
* `instance`: Renderer instance
* `add_usd_handle`: Handle obtained from [`ovrtx_add_usd_reference_from_file`](@ref)() or [`ovrtx_add_usd_reference_from_string`](@ref)()
# Returns
- **OVRTX\\_API\\_SUCCESS** if the usd file was removed successfully, - **OVRTX\\_API\\_ERROR** if the usd file removal failed.
"""
function ovrtx_remove_usd(instance, add_usd_handle)
    @ccall libovrtx.ovrtx_remove_usd(instance::Ptr{ovrtx_renderer_t}, add_usd_handle::ovrtx_usd_handle_t)::ovrtx_enqueue_result_t
end

"""
    ovrtx_clone_usd(instance, source_path_in_usd, target_paths, num_target_paths)

Enqueue an asynchronous operation to clone the subtree under the source path to one or more target paths in the runtime stage representation. The source path must exist in the stage. The target paths must not already exist in the stage.

# Arguments
* `instance`: Renderer instance
* `source_path_in_usd`: Path to the source path to clone
* `target_paths`: Array of target paths to clone to
* `num_target_paths`: Number of target paths to clone to
# Returns
- **OVRTX\\_API\\_SUCCESS** if the path was cloned successfully, - **OVRTX\\_API\\_ERROR** if the path cloning failed.
"""
function ovrtx_clone_usd(instance, source_path_in_usd, target_paths, num_target_paths)
    @ccall libovrtx.ovrtx_clone_usd(instance::Ptr{ovrtx_renderer_t}, source_path_in_usd::ovx_string_t, target_paths::Ptr{ovx_string_t}, num_target_paths::Csize_t)::ovrtx_enqueue_result_t
end

"""
    ovrtx_reset_stage(instance)

Enqueue an asynchronous operation to reset the runtime stage representation to an empty stage.

# Arguments
* `instance`: Renderer instance
# Returns
- **OVRTX\\_API\\_SUCCESS** if the stage was reset successfully, - **OVRTX\\_API\\_ERROR** if the stage reset failed.
"""
function ovrtx_reset_stage(instance)
    @ccall libovrtx.ovrtx_reset_stage(instance::Ptr{ovrtx_renderer_t})::ovrtx_enqueue_result_t
end

"""
    ovrtx_update_stage_from_usd_time(instance, usd_time)

Enqueue an asynchronous operation to update the runtime stage representation from a specific USD time. This operation will update all time-sampled attributes in the runtime stage representation to the provided USD time.

# Arguments
* `instance`: Renderer instance
* `usd_time`: USD time to update the stage to
# Returns
- **OVRTX\\_API\\_SUCCESS** if the stage was updated from the USD time successfully, - **OVRTX\\_API\\_ERROR** if the stage update from USD time failed.
"""
function ovrtx_update_stage_from_usd_time(instance, usd_time)
    @ccall libovrtx.ovrtx_update_stage_from_usd_time(instance::Ptr{ovrtx_renderer_t}, usd_time::Cdouble)::ovrtx_enqueue_result_t
end

"""
    ovrtx_write_attribute(instance, binding_handle_or_desc, data_array, data_access)

Enqueue an asynchronous write operation from source data into the system's stage representation. This write operation is not fully executed when the call returns and inputs with asynchronous access must remain valid until the stream execution of this operation has completed.

# Arguments
* `instance`: Renderer instance
* `binding_handle_or_desc`: Handle or description of the binding to write to. This binding defines the layout of the input data, both in terms of attribute data encoding as well as the layout of prims. The binding can be generated in place or a persistent handle can be provided to manually manage the lifetime.
* `data_array`: Source data to write. Based on the input\\_access this source is used during the execution of the write operation, not during the enqueue. So it is important that the data source remains valid until the write operation has completed. This can be determined through stream synchronization events using ovrtx\\_signal\\_event() and ovrtx\\_wait\\_all\\_events(). When using asynchronous access of GPU input data, the cuda synchronization event must be signaled when the input data is ready to be accessed.
* `data_access`: Determines the time of access to the input data. With asynchronous access, the lifetime must be managed by the user, while synchronous access incurs a synchronous copy during this call but prevents any access after this call returns.
# Returns
- **OVRTX\\_API\\_SUCCESS** if the attribute was written successfully, - **OVRTX\\_API\\_ERROR** if the attribute write failed.
"""
function ovrtx_write_attribute(instance, binding_handle_or_desc, data_array, data_access)
    @ccall libovrtx.ovrtx_write_attribute(instance::Ptr{ovrtx_renderer_t}, binding_handle_or_desc::Ptr{ovrtx_binding_desc_or_handle_t}, data_array::Ptr{ovrtx_input_buffer_t}, data_access::ovrtx_data_access_t)::ovrtx_enqueue_result_t
end

"""
    ovrtx_map_attribute(instance, binding_handle_or_desc, mapping_desc, out_attribute_mapping)

Immediately provedes internal memory according to the binding description to be written to by the user and later applied to the stage representation via [`ovrtx_unmap_attribute`](@ref)()

# Arguments
* `instance`: Renderer instance
* `binding_handle_or_desc`: Handle or description of the binding to use to determine the layout of the data to write. The binding can be generated in place or a persistent handle can be provided to manually manage the lifetime.
* `mapping_desc`: Description of the mapping to use
* `out_attribute_mapping`: [out] Handle to the attribute mapping
# Returns
- **OVRTX\\_API\\_SUCCESS** if the attribute was mapped successfully, - **OVRTX\\_API\\_ERROR** if the attribute mapping failed.
"""
function ovrtx_map_attribute(instance, binding_handle_or_desc, mapping_desc, out_attribute_mapping)
    @ccall libovrtx.ovrtx_map_attribute(instance::Ptr{ovrtx_renderer_t}, binding_handle_or_desc::Ptr{ovrtx_binding_desc_or_handle_t}, mapping_desc::ovrtx_mapping_desc_t, out_attribute_mapping::Ptr{ovrtx_attribute_mapping_t})::ovrtx_result_t
end

"""
    ovrtx_unmap_attribute(instance, map_handle, cuda_sync)

Enqueue an asynchronous operation to take the data written by the user and do whatever necessary to apply it to the system's stage representation. Note that while the map operation is not asynchronous, the unmap operation is and it determines the logical order of applying the written data to the stage. Multiple mappings can be outstanding on the same stage data with the effects on the stage representation depending on the order of the unmap operations. The data written by the user must be ready when the unmap operation is called for CPU data and when the cuda synchronization event is signaled for GPU data.

# Arguments
* `instance`: Renderer instance
* `map_handle`: Handle to the attribute mapping to unmap
* `cuda_sync`: optional cuda synchronization to wait for before the mapped memory is accessed during the application of the written data to the stage representation.
# Returns
- **OVRTX\\_API\\_SUCCESS** if the attribute was unmapped successfully, - **OVRTX\\_API\\_ERROR** if the attribute unmap failed.
"""
function ovrtx_unmap_attribute(instance, map_handle, cuda_sync)
    @ccall libovrtx.ovrtx_unmap_attribute(instance::Ptr{ovrtx_renderer_t}, map_handle::ovrtx_map_handle_t, cuda_sync::ovrtx_cuda_sync_t)::ovrtx_enqueue_result_t
end

"""
    ovrtx_create_attribute_binding(instance, description, out_attribute_binding_handle)

Enqueue an asynchronous operation to create a persistent attribute binding that binds a list of prims to a buffer layout. This operation is an optimization to manage the lifetime of internal resources used to perform write or map operations using this binding.

# Arguments
* `instance`: Renderer instance
* `description`: Description of the binding to create
* `out_attribute_binding_handle`: [out] Handle to the attribute binding
# Returns
- **OVRTX\\_API\\_SUCCESS** if the attribute binding was created successfully, - **OVRTX\\_API\\_ERROR** if the attribute binding creation failed.
"""
function ovrtx_create_attribute_binding(instance, description, out_attribute_binding_handle)
    @ccall libovrtx.ovrtx_create_attribute_binding(instance::Ptr{ovrtx_renderer_t}, description::Ptr{ovrtx_binding_desc_t}, out_attribute_binding_handle::Ptr{ovrtx_attribute_binding_handle_t})::ovrtx_enqueue_result_t
end

"""
    ovrtx_destroy_attribute_binding(instance, binding_handle)

Enqueue an asynchronous operation to destroy a persistent attribute binding.

# Arguments
* `instance`: Renderer instance
* `binding_handle`: Handle to the attribute binding to destroy
# Returns
- **OVRTX\\_API\\_SUCCESS** if the attribute binding was destroyed successfully, - **OVRTX\\_API\\_ERROR** if the attribute binding destruction failed.
"""
function ovrtx_destroy_attribute_binding(instance, binding_handle)
    @ccall libovrtx.ovrtx_destroy_attribute_binding(instance::Ptr{ovrtx_renderer_t}, binding_handle::ovrtx_attribute_binding_handle_t)::ovrtx_enqueue_result_t
end

"""
    ovrtx_read_attribute(instance, binding_handle_or_desc, read_dest, out_read_handle)

Enqueue an asynchronous stream-ordered read of attribute values from the runtime stage.

The binding identifies which prims and which attribute to read. It reuses the same ovrtx_binding_desc_or_handle_t used for write operations: - ovrtx_binding_desc_t::prims_list_handle or ovrtx_binding_desc_t::prim_list identifies the prims (typically obtained from a prior ovrtx_query_prims() result). - ovrtx_binding_desc_t::attribute_name selects the attribute. - ovrtx_binding_desc_t::attribute_type serves as an optional type hint for reads; if zero-initialized the system returns the native type. - ovrtx_binding_desc_t::prim_mode determines behavior for missing prims (EXISTING\\_ONLY skips them, MUST\\_EXIST errors, CREATE\\_NEW is not supported).

Persistent binding handles (ovrtx_attribute_binding_handle_t) optimize repeated reads.

The read sees the stage as-if all prior stream-ordered operations have completed.

# Arguments
* `instance`: Renderer instance
* `binding_handle_or_desc`: Binding identifying prims and attribute to read
* `read_dest`: Optional destination tensor for scalar reads (NULL = allocate internally). Must be NULL for array attributes. GPU tensors (kDLCUDA) are supported.
* `out_read_handle`:\\[out\\] Handle to the read result for use with ovrtx_fetch_read_result()
# Returns
- **OVRTX\\_API\\_SUCCESS** if the read was enqueued successfully, - **OVRTX\\_API\\_ERROR** if the read failed to enqueue.
"""
function ovrtx_read_attribute(instance, binding_handle_or_desc, read_dest, out_read_handle)
    @ccall libovrtx.ovrtx_read_attribute(instance::Ptr{ovrtx_renderer_t}, binding_handle_or_desc::Ptr{ovrtx_binding_desc_or_handle_t}, read_dest::Ptr{ovrtx_read_dest_t}, out_read_handle::Ptr{ovrtx_read_handle_t})::ovrtx_enqueue_result_t
end

"""
    ovrtx_fetch_read_result(instance, read_handle, timeout, out_read_output)

Fetch the results of a prior ovrtx_read_attribute().

This operation is synchronous and will block until the read completes or the timeout is reached. Passing 0 as the timeout makes it a non-blocking poll.

For scalar attributes the result contains a single tensor of shape [prim\\_count]. For array attributes the result contains one tensor per prim with variable length. Multi-component C attribute tensors encode component count in [`DLTensor`](@ref)::dtype.lanes: for example, a point3f[] array with 10 points is shape [10] with lanes=3, and a 4x4 double matrix attribute for N prims is shape [N] with lanes=16. When a destination tensor was provided to ovrtx_read_attribute(), buffer\\_count is 0.

All pointers in the output are valid until ovrtx_release_read_result() is called.

# Arguments
* `instance`: Renderer instance
* `read_handle`: Handle obtained from ovrtx_read_attribute()
* `timeout`: Timeout for the operation
* `out_read_output`:\\[out\\] Read data
# Returns
- **OVRTX\\_API\\_SUCCESS** if the read result was fetched successfully, - **OVRTX\\_API\\_ERROR** if the fetch failed, - **OVRTX\\_API\\_TIMEOUT** if the result could not be obtained within the timeout.
"""
function ovrtx_fetch_read_result(instance, read_handle, timeout, out_read_output)
    @ccall libovrtx.ovrtx_fetch_read_result(instance::Ptr{ovrtx_renderer_t}, read_handle::ovrtx_read_handle_t, timeout::ovrtx_timeout_t, out_read_output::Ptr{ovrtx_read_output_t})::ovrtx_result_t
end

"""
    ovrtx_release_read_result(instance, map_handle, before_destroy_cuda_sync)

Release resources from a prior ovrtx_fetch_read_result().

After this call all pointers in the previously returned ovrtx_read_output_t become invalid.

# Arguments
* `instance`: Renderer instance
* `map_handle`: Handle from ovrtx_read_output_t::map_handle
* `before_destroy_cuda_sync`: Optional CUDA synchronization to wait for before freeing GPU resources (zero-initialized = no sync)
# Returns
- **OVRTX\\_API\\_SUCCESS** if the result was released successfully, - **OVRTX\\_API\\_ERROR** if the release failed.
"""
function ovrtx_release_read_result(instance, map_handle, before_destroy_cuda_sync)
    @ccall libovrtx.ovrtx_release_read_result(instance::Ptr{ovrtx_renderer_t}, map_handle::ovrtx_read_map_handle_t, before_destroy_cuda_sync::ovrtx_cuda_sync_t)::ovrtx_result_t
end

"""
    ovrtx_query_prims(instance, query_desc, out_query_handle)

Enqueue an asynchronous stream-ordered query of the runtime stage.

The query finds all prims matching the filter criteria in the ovrtx_query_desc_t and groups them by attribute schema (all prims in a group share the same attributes).

# Arguments
* `instance`: Renderer instance
* `query_desc`: Description of the query (filters, attribute reporting)
* `out_query_handle`:\\[out\\] Handle for use with ovrtx_fetch_query_results()
# Returns
- **OVRTX\\_API\\_SUCCESS** if the query was enqueued successfully, - **OVRTX\\_API\\_ERROR** if the query failed to enqueue.
"""
function ovrtx_query_prims(instance, query_desc, out_query_handle)
    @ccall libovrtx.ovrtx_query_prims(instance::Ptr{ovrtx_renderer_t}, query_desc::Ptr{ovrtx_query_desc_t}, out_query_handle::Ptr{ovrtx_query_handle_t})::ovrtx_enqueue_result_t
end

"""
    ovrtx_fetch_query_results(instance, query_handle, timeout, out_result)

Retrieve the results of a prior ovrtx_query_prims() operation.

This operation is synchronous and will block until the query completes or the timeout is reached. Passing 0 as the timeout makes it a non-blocking poll.

The result contains one ovrtx_query_prim_group_t per matching bucket. Each group has a ovrtx_query_prim_group_t::prim_list_handle that can be plugged directly into ovrtx_binding_desc_t::prims_list_handle for subsequent read or write operations.

All pointers in the result are valid until ovrtx_release_query_results() is called.

# Arguments
* `instance`: Renderer instance
* `query_handle`: Handle obtained from ovrtx_query_prims()
* `timeout`: Timeout for the operation
* `out_result`:\\[out\\] Query result containing prim groups
# Returns
- **OVRTX\\_API\\_SUCCESS** if the results were retrieved successfully, - **OVRTX\\_API\\_ERROR** if the retrieval failed, - **OVRTX\\_API\\_TIMEOUT** if the result could not be obtained within the timeout.
"""
function ovrtx_fetch_query_results(instance, query_handle, timeout, out_result)
    @ccall libovrtx.ovrtx_fetch_query_results(instance::Ptr{ovrtx_renderer_t}, query_handle::ovrtx_query_handle_t, timeout::ovrtx_timeout_t, out_result::Ptr{ovrtx_query_result_t})::ovrtx_result_t
end

"""
    ovrtx_release_query_results(instance, query_handle)

Release all resources associated with a prior ovrtx_query_prims() result.

This destroys all ovrtx_query_prim_group_t::prim_list_handle values in the result, frees attribute descriptor arrays, and releases internal resources.

After this call all pointers in the previously returned ovrtx_query_result_t become invalid.

# Arguments
* `instance`: Renderer instance
* `query_handle`: Handle obtained from ovrtx_query_prims()
# Returns
- **OVRTX\\_API\\_SUCCESS** if the results were released successfully, - **OVRTX\\_API\\_ERROR** if the release failed.
"""
function ovrtx_release_query_results(instance, query_handle)
    @ccall libovrtx.ovrtx_release_query_results(instance::Ptr{ovrtx_renderer_t}, query_handle::ovrtx_query_handle_t)::ovrtx_result_t
end

"""
    ovrtx_get_path_dictionary(instance, out_path_dictionary)

Obtain the renderer's path dictionary for converting between handles and strings.

The path dictionary can be used to: - Convert ovrtx_query_prim_group_t::prim_list_handle to string paths via get\\_paths\\_from\\_path\\_list / get\\_strings\\_from\\_tokens. - Pre-resolve filter names to tokens via create\\_tokens\\_from\\_strings for repeated queries. - Build prim lists from strings via create\\_path\\_list\\_from\\_strings.

The returned instance is valid for the lifetime of the renderer.

# Arguments
* `instance`: Renderer instance
* `out_path_dictionary`:\\[out\\] The renderer's path dictionary
# Returns
- **OVRTX\\_API\\_SUCCESS** if the path dictionary was retrieved successfully, - **OVRTX\\_API\\_ERROR** if the retrieval failed.
"""
function ovrtx_get_path_dictionary(instance, out_path_dictionary)
    @ccall libovrtx.ovrtx_get_path_dictionary(instance::Ptr{ovrtx_renderer_t}, out_path_dictionary::Ptr{path_dictionary_instance_t})::ovrtx_result_t
end

"""
    ovrtx_step(instance, render_products, delta_time, out_step_result_handle)

Enqueue an asynchronous operation that will perform a sensor simulation step for all render products in the provided render product set. The simulation step will be performed for the time span [last\\_step\\_time, last\\_step\\_time + delta\\_time], where last\\_step\\_time is determined by the history of previous calls to [`ovrtx_step`](@ref)() or [`ovrtx_reset`](@ref)(). When performing the sensor simulation, the result of prior stream ordered operations affecting the stage since the last call of [`ovrtx_step`](@ref)() will be considered the state of the stage at time (last\\_step\\_time + delta\\_time). After the simulation step was executed, last\\_step\\_time will be updated to (last\\_step\\_time + delta\\_time) for the next call to [`ovrtx_step`](@ref)().

# Arguments
* `instance`: Renderer instance
* `render_products`: Render products to simulate during this simulation step. Accumulated sensor rendering history for all render products not in the provided set will be discarded.
* `delta_time`: Time step to simulate
* `out_step_result_handle`: [out] Handle to the step result
# Returns
- **OVRTX\\_API\\_SUCCESS** if the step was enqueued successfully, - **OVRTX\\_API\\_ERROR** if the step enqueue failed.
"""
function ovrtx_step(instance, render_products, delta_time, out_step_result_handle)
    @ccall libovrtx.ovrtx_step(instance::Ptr{ovrtx_renderer_t}, render_products::ovrtx_render_product_set_t, delta_time::Cdouble, out_step_result_handle::Ptr{ovrtx_step_result_handle_t})::ovrtx_enqueue_result_t
end

"""
    ovrtx_enqueue_pick_query(instance, desc)

Enqueue a pick query for the next ovrtx_step() that renders the given RenderProduct. Results appear as the multi-tensor render variable OVRTX_RENDER_VAR_PICK_HIT, with named CPU tensors (``primPath``, ``objectType``, ``geometryInstanceId``, ``worldPositionM``, ``worldNormal``) and ``uint32`` params (``magic`` = OVRTX_PICK_HIT_MAGIC, ``version`` = OVRTX_PICK_HIT_VERSION, ``hitCount``). The ``primPath`` column stores ovx_primpath_t ids; resolve strings via ovrtx_get_path_dictionary() if needed. If multiple queries are enqueued for the same RenderProduct before a step, the last one wins.
"""
function ovrtx_enqueue_pick_query(instance, desc)
    @ccall libovrtx.ovrtx_enqueue_pick_query(instance::Ptr{ovrtx_renderer_t}, desc::Ptr{ovrtx_pick_query_desc_t})::ovrtx_enqueue_result_t
end

"""
    ovrtx_set_selection_group_styles(instance, group_ids, styles, count)

Set per-group visual styling (outline color and fill color) for one or more selection groups.

Group ids are uint8 (0..255) and match the value written to a prim's `omni`:selectionOutlineGroup attribute (see OVRTX_ATTR_NAME_SELECTION_OUTLINE_GROUP). `group_ids` and `styles` are parallel arrays of length `count`.

The operation is stream-ordered: it takes effect on the next ovrtx_step that occurs after this op completes. If multiple writes target the same group id (in this call or across calls), the last writer wins.

Global state (outline thickness, fill mode) is configured via OVRTX_CONFIG_SELECTION_OUTLINE_WIDTH and OVRTX_CONFIG_SELECTION_FILL_MODE at renderer creation time.

Outline dashing is **not supported** by the underlying renderer.

# Arguments
* `instance`: Renderer instance
* `group_ids`: Array of `count` group ids
* `styles`: Array of `count` styles, parallel to `group_ids`
* `count`: Number of (group\\_id, style) pairs
# Returns
- **OVRTX\\_API\\_SUCCESS** if the operation was enqueued successfully, - **OVRTX\\_API\\_ERROR** if the operation failed to enqueue (null arguments, etc.).
"""
function ovrtx_set_selection_group_styles(instance, group_ids, styles, count)
    @ccall libovrtx.ovrtx_set_selection_group_styles(instance::Ptr{ovrtx_renderer_t}, group_ids::Ptr{UInt8}, styles::Ptr{ovrtx_selection_group_style_t}, count::Csize_t)::ovrtx_enqueue_result_t
end

"""
    ovrtx_reset(instance, time)

Enqueue an asynchronous operation to reset the accumulated sensor rendering history for all render products and start future sensor simulation steps at the provided time. After the reset was executed, last\\_step\\_time will be updated to the provided time for the next call to [`ovrtx_step`](@ref)().

# Arguments
* `instance`: Renderer instance
* `time`: Time to reset the simulation to
# Returns
- **OVRTX\\_API\\_SUCCESS** if the reset was enqueued successfully, - **OVRTX\\_API\\_ERROR** if the reset enqueue failed.
"""
function ovrtx_reset(instance, time)
    @ccall libovrtx.ovrtx_reset(instance::Ptr{ovrtx_renderer_t}, time::Cdouble)::ovrtx_enqueue_result_t
end

"""
    ovrtx_fetch_results(instance, result_handle, timeout, out_render_product_set_outputs)

Query the results of a prior enqueued step operation. This operation is synchronous and will block until the results are available or the timeout has passed. By passing 0 as the timeout this operation becomes a non-blocking poll operation that returns immediately if the results are not yet available. The complete production of render outputs is not determined by the completion of the asynchronous [`ovrtx_step`](@ref)() operation within the stream, so it is not possible to ensure this operations returns the results immediately by waiting for a stream synchronization event signaled after the [`ovrtx_step`](@ref)() operation. The result of this operation contains information about what results each render product has produced. Each render product can have produced 0-n frames of output for m render vars. This operation doesn't return the actual output data, but rather a handle to the output data which can then be mapped to retrieve the actual output data. All strings and pointers inside the result are valid until the result is destroyed by [`ovrtx_destroy_results`](@ref)().

# Arguments
* `instance`: Renderer instance
* `result_handle`: Handle to the step result to query
* `timeout`: Timeout for the operation. Passing 0 will make the operation non-blocking and return immediately with the current status of the operation.
* `out_render_product_set_outputs`: [out] Render product set outputs
# Returns
- **OVRTX\\_API\\_SUCCESS** if the render product set outputs were retrieved successfully, - **OVRTX\\_API\\_ERROR** if the operation failed, - **OVRTX\\_API\\_TIMEOUT** if the result could not be obtained within the timeout.
"""
function ovrtx_fetch_results(instance, result_handle, timeout, out_render_product_set_outputs)
    @ccall libovrtx.ovrtx_fetch_results(instance::Ptr{ovrtx_renderer_t}, result_handle::ovrtx_step_result_handle_t, timeout::ovrtx_timeout_t, out_render_product_set_outputs::Ptr{ovrtx_render_product_set_outputs_t})::ovrtx_result_t
end

"""
    ovrtx_map_render_var_output(instance, output_handle, map_output_desc, timeout, out_render_var_output)

Maps the rendered output into user accessible memory when the result are available. This operation is synchronous and will block until the output is mapped or the timeout has passed. By passing 0 as the timeout this operation becomes a non-blocking poll operation that returns immediately if the output is not yet mapped. The complete production of render outputs is not determined by the completion of the asynchronous [`ovrtx_step`](@ref)() operation within the stream, so it is not possible to ensure this operations returns the results immediately by waiting for a stream synchronization event signaled after the [`ovrtx_step`](@ref)() operation. The result of this operation contains the actual rendered output memory, but it can contain cuda events that must be synchronized to before actually accessing the output memory. Calling map on a output\\_handle that is part of a step result after calling destroy\\_results on the step result will return an error.

Tensor layout contract for ovrtx_render_var_output_t::tensors "[`ovrtx_render_var_output_t`](@ref)::tensors": - A render variable carries `num_tensors` named tensor slots; each `tensors[i].dl` describes one slot. - Image-shaped tensors are channel-last with: - `ndim = 3` - `shape = [height, width, channels]` - `dtype.lanes = 1` - Non-image tensors may use different ranks/layouts (for example, 1D buffers for point clouds). - Strides, when provided, are in elements (DLPack convention). - Params (ovrtx_render_var_output_t::params) are always CPU-resident DLPack tensors; their shape encodes scalar vs. array (e.g. `{1}` for scalar, `{N}` for array).

# Arguments
* `instance`: Renderer instance
* `output_handle`: Handle to the output to map
* `map_output_desc`: Description of the output to map
* `timeout`: Timeout for the operation. Passing 0 will make the operation non-blocking and return immediately with the current status of the operation.
* `out_render_var_output`: [out] Mapped render variable output
# Returns
- **OVRTX\\_API\\_SUCCESS** if the output was mapped successfully. - **OVRTX\\_API\\_ERROR** if the output mapping failed. - **OVRTX\\_API\\_TIMEOUT** if the result could not be obtained within the timeout.
"""
function ovrtx_map_render_var_output(instance, output_handle, map_output_desc, timeout, out_render_var_output)
    @ccall libovrtx.ovrtx_map_render_var_output(instance::Ptr{ovrtx_renderer_t}, output_handle::ovrtx_render_var_output_handle_t, map_output_desc::Ptr{ovrtx_map_output_description_t}, timeout::ovrtx_timeout_t, out_render_var_output::Ptr{ovrtx_render_var_output_t})::ovrtx_result_t
end

"""
    ovrtx_unmap_render_var_output(instance, map_handle, before_destroy_cuda_sync)

Unmaps the render variable output and frees resources associated with the prior [`ovrtx_map_render_var_output`](@ref)(). When this is called all access to the buffer provided by map\\_render\\_var\\_output must be done. This call determines the lifetime of the resources made accessible through the prior map\\_render\\_var\\_output call and this lifetime is independent of [`ovrtx_destroy_results`](@ref)(). It is safe to call unmap on a map\\_handle that was produced from a step result that has been destroyed by destroy\\_results.

# Arguments
* `instance`: Renderer instance.
* `map_handle`: Handle to the map to unmap.
* `before_destroy_cuda_sync`: CUDA synchronization to wait for before the mapped memory is destroyed.
# Returns
- **OVRTX\\_API\\_SUCCESS** if the output was unmapped successfully, - **OVRTX\\_API\\_ERROR** if the output unmapping failed.
"""
function ovrtx_unmap_render_var_output(instance, map_handle, before_destroy_cuda_sync)
    @ccall libovrtx.ovrtx_unmap_render_var_output(instance::Ptr{ovrtx_renderer_t}, map_handle::ovrtx_render_var_output_map_handle_t, before_destroy_cuda_sync::ovrtx_cuda_sync_t)::ovrtx_result_t
end

"""
    ovrtx_destroy_results(instance, result_handle)

Releases all resources associated with the result of a sensor simulation step, with the exception of resources provided by calls to [`ovrtx_map_render_var_output`](@ref)(). Those are only released through [`ovrtx_unmap_render_var_output`](@ref)().

# Arguments
* `instance`: Renderer instance
* `result_handle`: Handle to the step result to destroy
# Returns
- **OVRTX\\_API\\_SUCCESS** if the step result was destroyed successfully, - **OVRTX\\_API\\_ERROR** if the step result destruction failed.
"""
function ovrtx_destroy_results(instance, result_handle)
    @ccall libovrtx.ovrtx_destroy_results(instance::Ptr{ovrtx_renderer_t}, result_handle::ovrtx_step_result_handle_t)::ovrtx_result_t
end

"""
    ovrtx_wait_op(instance, op_id, time_out, out_wait_result)

Wait for completion of all operations up to and including the specified operation id. This operation is synchronous and will block until the operations are completed or the timeout has passed. Passing 0 as the timeout makes the operation a non-blocking poll. The out structure returns any errors observed since the last wait call and, on timeout, the lowest still-pending op id. For each op id in out\\_wait\\_result->error\\_op\\_ids, call [`ovrtx_get_last_op_error`](@ref)(op\\_id) to get the corresponding error string. Both out\\_wait\\_result->error\\_op\\_ids and strings returned by [`ovrtx_get_last_op_error`](@ref)() are transient thread-local data and are invalidated by the next call to [`ovrtx_wait_op`](@ref)() on the same thread. Consume/copy them before the next wait call.

# Arguments
* `instance`: Renderer instance
* `op_id`: Non-zero operation id to wait for. OVRTX_INVALID_HANDLE is not waitable
* `time_out`: Timeout for the operation
* `out_wait_result`: [out] Wait result information (error op ids and lowest pending op id)
# Returns
- **OVRTX\\_API\\_SUCCESS** if the operations were waited for successfully, - **OVRTX\\_API\\_ERROR** if the wait failed (e.g., invalid op id), - **OVRTX\\_API\\_TIMEOUT** if not all operations completed within the timeout.
"""
function ovrtx_wait_op(instance, op_id, time_out, out_wait_result)
    @ccall libovrtx.ovrtx_wait_op(instance::Ptr{ovrtx_renderer_t}, op_id::ovrtx_op_id_t, time_out::ovrtx_timeout_t, out_wait_result::Ptr{ovrtx_op_wait_result_t})::ovrtx_result_t
end

"""
    ovrtx_query_op_status(instance, op_id, out_status)

Query the status of a long-running operation.

This operation is synchronous and returns immediately with the current status of the specified operation. The returned status structure contains progress information and named resource counters. All strings and pointers in out\\_status are valid until [`ovrtx_release_op_status`](@ref) is called. This function must be paired with [`ovrtx_release_op_status`](@ref).

Counter semantics: - name: Identifies the resource type (e.g., "shaders", "textures", "materials") - current: Number of items processed so far - total: Total items to process, or 0 if the total is not yet known - The set of counters varies by operation type

# Arguments
* `instance`: Renderer instance
* `op_id`: Operation ID to query (from [`ovrtx_enqueue_result_t`](@ref).op\\_index)
* `out_status`: [out] Status information for the operation
# Returns
OVRTX\\_API\\_SUCCESS if status was retrieved successfully, OVRTX\\_API\\_ERROR if op\\_id is invalid or query failed
"""
function ovrtx_query_op_status(instance, op_id, out_status)
    @ccall libovrtx.ovrtx_query_op_status(instance::Ptr{ovrtx_renderer_t}, op_id::ovrtx_op_id_t, out_status::Ptr{ovrtx_op_status_t})::ovrtx_result_t
end

"""
    ovrtx_release_op_status(instance, status)

Release resources associated with a previously queried operation status.

After this call, all pointers in the status structure become invalid. This must be called for every successful [`ovrtx_query_op_status`](@ref) call.

# Arguments
* `instance`: Renderer instance
* `status`: Status structure to release (previously obtained from [`ovrtx_query_op_status`](@ref))
# Returns
OVRTX\\_API\\_SUCCESS if released successfully, OVRTX\\_API\\_ERROR if release failed
"""
function ovrtx_release_op_status(instance, status)
    @ccall libovrtx.ovrtx_release_op_status(instance::Ptr{ovrtx_renderer_t}, status::Ptr{ovrtx_op_status_t})::ovrtx_result_t
end

"""
    ovrtx_query_extension(name, vtable)

Query for an internal extension interface by name.

# Arguments
* `name`: The name of the extension
* `vtable`: [out] Vtable with function pointers for the extension
# Returns
- **OVRTX\\_API\\_SUCCESS** if the extension was queried successfully, - **OVRTX\\_API\\_ERROR** if the extension is unavailable or if the system is not initialized yet.
"""
function ovrtx_query_extension(name, vtable)
    @ccall libovrtx.ovrtx_query_extension(name::Cstring, vtable::Ptr{Ptr{Cvoid}})::ovrtx_result_t
end

# no prototype is found for this function at ovrtx.h:875:18, please use with caution
"""
    ovrtx_get_last_error()

Returns the error string for the latest API call on the calling thread. The string is valid until the next API call on the same thread.
"""
function ovrtx_get_last_error()
    @ccall libovrtx.ovrtx_get_last_error()::ovx_string_t
end

"""
    ovrtx_get_last_op_error(op_id)

Returns the error string for the provided operation id from the last call to [`ovrtx_wait_op`](@ref) on the calling thread. The string is valid until the next call to [`ovrtx_wait_op`](@ref) on the same thread.

# Arguments
* `op_id`: Operation id to get the error string for
"""
function ovrtx_get_last_op_error(op_id)
    @ccall libovrtx.ovrtx_get_last_op_error(op_id::ovrtx_op_id_t)::ovx_string_t
end

"""
    ovrtx_set_log_callback(severity, channel_filter, callback, user_data)

Set a process-global callback that receives every carb log message produced by ovrtx (and by any plugin or framework code loaded under it).

The callback is shared across the whole ovrtx process, not per-renderer: its lifetime is tied to [`ovrtx_initialize`](@ref) / [`ovrtx_shutdown`](@ref), not to any particular renderer instance. As a result the callback receives messages emitted before the first [`ovrtx_create_renderer`](@ref) (framework startup, plugin loading, the OVRTX logging banner) as well as messages emitted during and after [`ovrtx_destroy_renderer`](@ref) (e.g. asset eviction during teardown).

Only one callback can be active at a time; calling this function replaces the previous callback. Pass NULL as `callback` to disable delivery.

Calling this function before [`ovrtx_initialize`](@ref) or after [`ovrtx_shutdown`](@ref) returns OVRTX\\_API\\_ERROR with a descriptive error string accessible via [`ovrtx_get_last_error`](@ref)().

The callback does NOT carry per-renderer attribution. ovrtx may add a v2 callback type with (renderer, op\\_id) once per-op TLS plumbing exists.

Thread safety: The callback may be invoked from any thread. The implementation guarantees that callbacks are serialized for the process (no concurrent invocations).

Channel filter syntax --------------------- `channel_filter` is a comma-separated list of `<channel\\_prefix>=<level>` entries (RUST\\_LOG-style). The channel prefix is matched against carb's dotted source name; longest matching prefix wins. Per-channel rules override `severity` for matched channels. Channels not matched by any rule use `severity` as their threshold. The empty / NULL filter is equivalent to "every channel uses @p severity".

Accepted level names (case-insensitive): "verbose" (alias "debug"), "info", "warn" (alias "warning"), "error", "fatal".

Whitespace around tokens and trailing commas are tolerated; empty entries are skipped. Malformed entries (missing `=`, unknown level name, empty channel name) cause this function to return OVRTX\\_API\\_ERROR with a descriptive [`ovrtx_get_last_error`](@ref)() string and leave the previously installed callback state unchanged.

Examples: - `""` no rules; every channel uses `severity` - `"omni.usd=error"` omni.usd* uses error+, everything else uses `severity` - `"carb=warn,carb.tasking=verbose"` carb.tasking uses verbose+, other carb.* uses warn+, everything else uses `severity`

# Arguments
* `severity`: Default severity threshold applied to channels not matched by an explicit rule in `channel_filter`. Use OVRTX\\_LOG\\_INFO to receive INFO and above by default; OVRTX\\_LOG\\_ERROR to filter out everything but errors by default. Note that [`ovrtx_log_severity_t`](@ref) does not expose carb's lower verbose / debug levels: those are dropped by the default and can only be received by an explicit per-channel rule like "carb.tasking=verbose" in `channel_filter`.
* `channel_filter`: Optional comma-separated `<channel>=<level>` list. Pass NULL to apply `severity` uniformly.
* `callback`: Callback function to receive log messages, or NULL to disable
* `user_data`: User-provided context passed to each callback invocation
# Returns
OVRTX\\_API\\_SUCCESS if callback was set successfully, OVRTX\\_API\\_ERROR if the system is not initialized or the filter string failed to parse
"""
function ovrtx_set_log_callback(severity, channel_filter, callback, user_data)
    @ccall libovrtx.ovrtx_set_log_callback(severity::ovrtx_log_severity_t, channel_filter::Ptr{ovx_string_t}, callback::ovrtx_log_callback_t, user_data::Ptr{Cvoid})::ovrtx_result_t
end

"""
    ovrtx_flush_log(timeout)

Flush all pending log messages through the global callback.

This operation blocks until all log messages generated up to this point have been delivered through the log callback. This is useful when you need to ensure all logs have been processed before proceeding (e.g., after an operation completes or fails, or before tearing the system down).

If no log callback is set, this function returns immediately with success.

Calling this function before [`ovrtx_initialize`](@ref) or after [`ovrtx_shutdown`](@ref) returns OVRTX\\_API\\_ERROR.

Note: This only flushes messages generated before this call. Messages generated concurrently or after this call may not be included.

# Arguments
* `timeout`: Maximum time to wait for flush to complete
# Returns
OVRTX\\_API\\_SUCCESS if all pending logs were flushed, OVRTX\\_API\\_TIMEOUT if flush did not complete within timeout, OVRTX\\_API\\_ERROR if flush failed (system not initialized)
"""
function ovrtx_flush_log(timeout)
    @ccall libovrtx.ovrtx_flush_log(timeout::ovrtx_timeout_t)::ovrtx_result_t
end

struct var"##Ctag#279"
    data::Ptr{Cvoid}
    size::Csize_t
end
function Base.getproperty(x::Ptr{var"##Ctag#279"}, f::Symbol)
    f === :data && return Ptr{Ptr{Cvoid}}(x + 0)
    f === :size && return Ptr{Csize_t}(x + 8)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#279", f::Symbol)
    r = Ref{var"##Ctag#279"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#279"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#279"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


const OVRTX_VERSION_MAJOR = 0

const OVRTX_VERSION_MINOR = 3

const OVRTX_VERSION_PATCH = 0

const DLPACK_MAJOR_VERSION = 1

const DLPACK_MINOR_VERSION = 3

const DLPACK_FLAG_BITMASK_READ_ONLY = Culong(1) << Culong(0)

const DLPACK_FLAG_BITMASK_IS_COPIED = Culong(1) << Culong(1)

const DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED = Culong(1) << Culong(2)

const OVRTX_INVALID_HANDLE = 0

const OVRTX_ATTR_NAME_SELECTION_OUTLINE_GROUP = "omni:selectionOutlineGroup"

const OVRTX_ATTR_NAME_PICKABLE = "omni:pickable"

const OVRTX_RENDER_VAR_PICK_HIT = "ovrtx_pick_hit"

const OVRTX_PICK_FLAG_GIZMO = Cuint(1) << 0

const OVRTX_PICK_FLAG_INCLUDE_TRACKED_INFO = Cuint(1) << 1

const OVRTX_PICK_HIT_MAGIC = Cuint(0x56505448)

const OVRTX_PICK_HIT_VERSION = Cuint(1)

# exports
const PREFIXES = ["ovrtx_", "ovx_", "OVRTX_", "OVX_", "path_dictionary_", "DL", "kDL"]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end

