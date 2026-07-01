# Mesh translation.  `to_ovrtx_object(screen, scene, plot)` is the generic
# fallback (returns nothing).  Included in the OmniverseMakie module after
# materials.jl.  Scope: Makie, GeometryBasics, OV, usda_mesh, displaycolor_for.
#
# The M1.5 to_ovrtx_object(::Mesh) method is gone: Mesh has non-empty
# consumed_inputs, so it takes the live author_usd_prim! path (compute.jl).
# Only empty-consumed_inputs types (::Surface / unknown) reach to_ovrtx_object.

"""
    to_ovrtx_object(screen, scene, plot) -> Union{UInt64, Nothing}

Generic fallback: returns `nothing` so an unknown atomic plot is silently skipped (the
rest of the scene still renders).  Supported types build live via `author_usd_prim!`
(compute.jl); only NO-`consumed_inputs` types (`::Surface`, and unknown here) reach here.
"""
to_ovrtx_object(screen, scene, plot) = nothing
