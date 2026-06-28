/* Umbrella header for Clang.jl generation of LibOVRTX.
 *
 * We intentionally DO NOT include ovrtx/ovrtx_attributes.h: that header is C++
 * (uses nullptr / new / delete / brace-init) and contains only `static inline`
 * helpers, which Clang.jl cannot wrap anyway. We reimplement those in Julia.
 *
 * ovrtx.h transitively pulls in ovrtx_types.h, which pulls in dlpack.h,
 * ovx/types.h and the path_dictionary headers (incl. the vtable struct).
 */
#include "ovrtx/ovrtx_types.h"
#include "ovrtx/ovrtx_config.h"
#include "ovrtx/ovrtx.h"
