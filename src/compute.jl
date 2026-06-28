# compute.jl — M2 render-object handles + (M2.2+) ComputePipeline diff nodes.
#
# In M2.1 this file holds only the per-plot render-object record, `OvrtxRObj`,
# which the open-stage `Screen` stores in `plot2robj`.  M2.2 adds the
# `:ovrtx_renderobject` diff node + `push_to_ovrtx!`; M2.3 fills `bindings`.
#
# NOTE: included inside the OmniverseMakie module, BEFORE screen.jl, because the
# `Screen` struct references `OvrtxRObj` in its `plot2robj` field type.

"""
    OvrtxRObj

Per-plot render object created when a plot's USD reference is authored on the
open stage.  Records:

- `prim_path`  — the USD prim the reference was added at (`/World/plot_<id>`).
- `usd_handle` — the `ovrtx_usd_handle_t` returned by `OV.add_usd_reference!`
                 (used by `OV.remove_usd!` on delete — M2.4).
- `bindings`   — persistent attribute bindings keyed by attribute name; empty in
                 M2.1, filled by the hot-path work in M2.3.
"""
mutable struct OvrtxRObj
    prim_path::String
    usd_handle::UInt64
    bindings::Dict{Symbol,Any}
end

OvrtxRObj(prim_path::AbstractString, usd_handle::Integer) =
    OvrtxRObj(String(prim_path), UInt64(usd_handle), Dict{Symbol,Any}())
