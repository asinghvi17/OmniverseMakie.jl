# Mesh translation for OmniverseMakie.
#
# Provides:
#   to_ovrtx_object(screen, scene, plot) — generic fallback (= nothing)
#
# NOTE: included inside the OmniverseMakie module, after materials.jl.
#       Makie, GeometryBasics, OV, usda_mesh, displaycolor_for are all in scope.
#
# M2.3 dead-path consolidation: the M1.5 `to_ovrtx_object(::Makie.Mesh)` build method
# was UNREACHABLE after M2.2 (Mesh has non-empty `consumed_inputs`, so it always takes
# the live `author_usd_prim!` diff-node path in compute.jl; only empty-`consumed_inputs`
# types — `::Surface` / unknown — reach `to_ovrtx_object`).  It was removed; the generic
# `= nothing` fallback below keeps unknown atomic plots silently skipped.

"""
    to_ovrtx_object(screen, scene, plot) -> Union{UInt64, Nothing}

Generic fallback: returns `nothing` so an unknown atomic plot is silently skipped
(the rest of the scene still renders).  The live build path for supported types is
`author_usd_prim!` (compute.jl); only plot types with NO tracked `consumed_inputs`
(`::Surface` in primitives.jl, and unknown types here) route through `to_ovrtx_object`.
"""
to_ovrtx_object(screen, scene, plot) = nothing
