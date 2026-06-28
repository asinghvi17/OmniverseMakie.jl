# Mesh translation for OmniverseMakie.
#
# Provides:
#   to_ovrtx_object(screen, scene, plot) — atomic Makie plot → ovrtx USD geometry
#   to_ovrtx_object(screen, scene, plot::Makie.Mesh) — UsdGeomMesh reference
#
# NOTE: included inside the OmniverseMakie module, after materials.jl.
#       Makie, GeometryBasics, OV, usda_mesh, displaycolor_for are all in scope.

"""
    to_ovrtx_object(screen, scene, plot) -> Union{UInt64, Nothing}

Translate one ATOMIC Makie plot into ovrtx USD geometry: author a USD reference
under `/World/plot_<objectid(plot)>` and return its `ovrtx_usd_handle_t`.

The generic method returns `nothing` for plot types not yet supported, so an
unknown atomic plot is silently skipped and the rest of the scene still renders.
M1.5 implements `::Makie.Mesh`; M1.7 adds scatter / meshscatter / lines / surface.
"""
to_ovrtx_object(screen, scene, plot) = nothing

"""
    to_ovrtx_object(screen, scene, plot::Makie.Mesh) -> UInt64

Author the plot's `GeometryBasics.Mesh` as a `UsdGeomMesh` USD reference and
return the `ovrtx_usd_handle_t`.

Mesh data (Makie converts mesh args to `GLTriangleFace` + per-vertex normals):
- `points`  — `GeometryBasics.coordinates` (one `Point3f` per vertex)
- `faces`   — `GeometryBasics.faces` (triangles); USD `faceVertexIndices` are
  **0-based**, so each index is emitted as `Int(GeometryBasics.raw(i))` (the
  underlying 0-based stored value; `value`/`Int` would be 1-based → garbled mesh).
- `normals` — `GeometryBasics.normals` (one per vertex → `interpolation="vertex"`)
- `model`   — `plot.model[]` (Makie column-vector `Mat4f`; `usda_matrix4d` transposes)
- colour    — via `displaycolor_for` → constant or per-vertex `primvars:displayColor`
"""
function to_ovrtx_object(screen, scene, plot::Makie.Mesh)
    mesh   = plot[1][]                          # GeometryBasics.Mesh (converted args)
    points = GeometryBasics.coordinates(mesh)
    normals = GeometryBasics.normals(mesh)

    # 0-based USD face indices (raw == 0-based; value/Int == 1-based).
    faces0 = [Int[Int(GeometryBasics.raw(i)) for i in f] for f in GeometryBasics.faces(mesh)]

    values, interp = displaycolor_for(plot, length(points))

    usda = usda_mesh(points, faces0, normals, values;
                     model                = plot.model[],
                     normal_interpolation = "vertex",   # Makie meshes carry per-vertex normals
                     color_interpolation  = interp)

    return OV.add_usd_reference!(screen.renderer, usda, plot_prim_path(plot))
end
