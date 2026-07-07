# Mesh translation. `to_ovrtx_object(screen, scene, plot)` is the generic
# fallback (returns nothing). Mesh has non-empty consumed_inputs, so it takes
# the live author_usd_prim! path (compute.jl); only empty-consumed_inputs
# types (::Surface / unknown) reach to_ovrtx_object.

"""
    to_ovrtx_object(screen, scene, plot) -> Union{UInt64, Nothing}

Generic fallback: returns `nothing` so an unknown atomic plot is silently
skipped (the rest of the scene still renders). Supported types build live
via `author_usd_prim!` (compute.jl); only empty-`consumed_inputs` types
(`::Surface`, and unknown here) reach this.
"""
to_ovrtx_object(screen, scene, plot) = nothing
