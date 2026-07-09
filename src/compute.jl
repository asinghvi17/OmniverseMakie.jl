# Per-plot render-object record `OvrtxRObj` + the `:ovrtx_renderobject` diff
# node: pushes one minimal C write per changed plot attribute on the open
# stage instead of re-authoring.  Included BEFORE screen.jl —
# `Screen.plot2robj`'s field type references `OvrtxRObj`.

"""
    OvrtxRObj

Per-plot render object recorded when a plot's USD reference is authored on
the open stage.  Fields:

- `prim_path`  — USD prim the reference was added at (`/World/plot_<id>`).
- `usd_handle` — `ovrtx_usd_handle_t` from `OV.add_usd_reference!` (for
                 `OV.remove_usd!`).
- `bindings`   — persistent attribute bindings keyed by driving
                 compute-output name (hot path); empty until
                 `bind_hot_attributes!`, freed by `destroy_bindings!`.
- `material_shader` — materialized plot: pre-authored OmniPBR shader prim
                 path (`/World/Looks/Mat_<id>/Shader`), else `nothing`.
                 `push_to_ovrtx!` uses it to route a live `color`/`material`
                 edit to `write_shader_input!` instead of `displayColor`.
- `plot`       — source Makie plot, for pick resolution.  Set at the
                 `plot2robj` insert sites so it rides that map's lifecycle.
- `meta`       — per-plot state that is NOT a destroyable GPU binding.  A
                 `Volume` build records `:vdb_tmp` (screen-owned temp
                 `.nvdb`, removed by `destroy_bindings!` on close/delete)
                 + `:volume_prim`.  Empty for non-volume plots.
"""
mutable struct OvrtxRObj
    prim_path::String
    usd_handle::UInt64
    bindings::Dict{Symbol,OV.Binding}   # always OV.create_binding results
    material_shader::Union{String,Nothing}
    plot::Union{Nothing,Makie.AbstractPlot}
    meta::Dict{Symbol,Any}
end

OvrtxRObj(prim_path::AbstractString, usd_handle::Integer) =
    OvrtxRObj(String(prim_path), UInt64(usd_handle), Dict{Symbol,OV.Binding}(), nothing, nothing,
              Dict{Symbol,Any}())

# ==================================================================
# :ovrtx_renderobject diff node + push_to_ovrtx! (diff driver)
#
# Each atomic plot gets a ComputePipeline node over the owning screen
# (`:ovrtx_screen`) + its resolved compute outputs.  First resolve (or a new
# owning screen) builds the USD reference; later resolves push one minimal C
# write per changed output — no re-author (writes on a referenced prim on
# the open stage are honored, so the diff path writes in place).
# colorbuffer pulls every node each frame; a clean graph is a no-op, any
# change flips screen.requires_update → one OV.reset! for the frame.
# ==================================================================

# Single source of truth for a plot's USD prim path: nested under the owning
# scene's `def Scope` via screen.scene2scope (`/World` for the root; a scene
# not in the map falls back to `/World`).  objectid keys are screen-stable.
plot_prim_path(scene2scope::AbstractDict, scene, plot) =
    string(get(scene2scope, objectid(scene), "/World"), "/plot_", objectid(plot))

# ------------------------------------------------------------------
# consumed_inputs — per-type Makie compute outputs the diff node tracks
# ------------------------------------------------------------------

# Only outputs that BOTH resolve by default AND have a `push_to_ovrtx!`
# route are tracked (one minimal write per change).  Surface/unknown types
# get no node (`Symbol[]`) → built once via `to_ovrtx_object`.  Dispatch is
# on TYPE and the node registers for EVERY plot of that type, so a tracked
# input must resolve for non-materialized plots too, else
# `register_computation!` fails for all plots of the type.  `:material`
# resolves only for Mesh/MeshScatter, so it is tracked there only; a live
# material-PARAM edit on a materialized Scatter/Lines/LineSegments is a
# no-op (a live `color` edit still works via `:scaled_color`).
consumed_inputs(::Makie.Mesh)         = [:positions_transformed_f32c, :model_f32c, :faces, :normals, :scaled_color, :material, :visible]
consumed_inputs(::Makie.Scatter)      = [:positions_transformed_f32c, :model_f32c, :scaled_color, :markersize, :visible]
consumed_inputs(::Makie.MeshScatter)  = [:positions_transformed_f32c, :model_f32c, :scaled_color, :markersize, :rotation, :material, :visible]
consumed_inputs(::Makie.Lines)        = [:positions_transformed_f32c, :model_f32c, :scaled_color, :linewidth, :visible]
consumed_inputs(::Makie.LineSegments) = [:positions_transformed_f32c, :model_f32c, :scaled_color, :linewidth, :visible]
# `volume!` renders via the UsdVol → IndeX Direct path.  Tracked: `:visible`
# (hide/reshow) and `:volume` (the converted scalar data; its push writes a
# fresh temp `.nvdb` + reloads via `reload_volume_data!`).  IndeX Direct is
# grayscale-only, so colormap/colorrange are untracked — read off the plot
# at build/reload time.
consumed_inputs(::Makie.Volume)       = [:visible, :volume]
consumed_inputs(::Any)                = Symbol[]

# ------------------------------------------------------------------
# displayColor from the resolved :scaled_color output
# ------------------------------------------------------------------

# `(values, interpolation)` for `usda_*` from a resolved `:scaled_color`:
#   single Colorant → constant; vector/array of Colorants → per-vertex.
_displaycolor_from_scaled(c::Colorant, _n) = (_rgb(c), "constant")
_displaycolor_from_scaled(cs::AbstractVector{<:Colorant}, _n) = ([_rgb(c) for c in cs], "vertex")
_displaycolor_from_scaled(cs::AbstractArray{<:Colorant}, _n)  = ([_rgb(c) for c in vec(cs)], "vertex")
# Fallback (e.g. a single packed value): treat as one constant colour.
_displaycolor_from_scaled(c, _n) = (_rgb(Makie.to_color(c)), "constant")

# A NUMERIC `scaled_color` vector (numbers + colormap) must be mapped
# through the plot's colormap (`_map_through_colormap`, the shared NaN-safe
# mapper); a Colorant or scalar defers to `_displaycolor_from_scaled`.
function _scaled_to_display(plot, sc, n)
    if sc isa AbstractVector{<:Real}
        return (_map_through_colormap(plot, sc), "vertex")
    end
    return _displaycolor_from_scaled(sc, n)
end

# ------------------------------------------------------------------
# texcoords (`st` UV primvar) for a textured materialized mesh
# ------------------------------------------------------------------

# Per-vertex UVs for a texture-materialized mesh, read off the plot's
# `:texturecoordinates` output (not a tracked diff output).  `nothing` (→
# `usda_mesh` omits `st`) when no texture, no texcoords, or count mismatch.
function _texcoords_for(plot, npoints::Int)
    _needs_texcoords(plot)              || return nothing
    haskey(plot, :texturecoordinates)  || return nothing
    tc = Makie.to_value(plot[:texturecoordinates])
    (tc isa AbstractVector && length(tc) == npoints) || return nothing
    return tc
end

# ------------------------------------------------------------------
# _add_materialized_reference! — shared epilogue for a MATERIALIZED plot
# ------------------------------------------------------------------

# Shared epilogue for a MATERIALIZED plot build: add the USD reference
# (unless the caller already holds its handle), bind the pre-authored
# OmniPBR material, wrap in an `OvrtxRObj`, and record the shader prim so a
# live `color`/`material` edit re-writes it in place instead of
# `displayColor`.  Dispatch: an `AbstractString` USDA layer is referenced at
# `path` first; an `Integer` handle is an already-referenced prim (Surface),
# wrapped as-is.
#
# This is the single place the `"/Shader"` suffix lives; it MUST match the
# shader prim name in `def Shader "Shader"` emitted by `_usda_mdl_material`
# (materials.jl) — a rename there silently breaks every materialized plot's
# live edit.
function _add_materialized_reference!(screen, path, usda::AbstractString, plot)
    h = OV.add_usd_reference!(screen.renderer, usda, path)
    return _add_materialized_reference!(screen, path, h, plot)
end

function _add_materialized_reference!(screen, path, handle::Integer, plot)
    OV.bind_material!(screen.renderer, path, material_prim_path(plot))
    robj = OvrtxRObj(path, handle)
    robj.material_shader = material_prim_path(plot) * "/Shader"
    return robj
end

# ------------------------------------------------------------------
# author_usd_prim! — BUILD branch: USDA emitters fed from resolved `args`
# ------------------------------------------------------------------

"""
    author_usd_prim!(screen, scene, plot, args) -> Union{OvrtxRObj,Nothing}

Build a plot's USD reference on the OPEN stage from its resolved compute
outputs (`args`), returning the `OvrtxRObj` (or `nothing` for an empty plot).

Points come from `:positions_transformed_f32c` (model-local), the transform
from `:model_f32c` (composed world), colour from `:scaled_color`; world =
`model_f32c · positions`.  Feeds the `usda_mesh` / `_usda_pointinstancer` /
`_usda_basiscurves` emitters.  `scene` is threaded so the reference is added
at the nested scope path `plot_prim_path(screen.scene2scope, scene, plot)`.
"""
function author_usd_prim!(screen, scene, plot::Makie.Mesh, args)
    points  = args[:positions_transformed_f32c]
    isempty(points) && return nothing
    normals = args[:normals]
    face_counts, face_indices = _flat_faces(args[:faces])   # flatten once
    path    = plot_prim_path(screen.scene2scope, scene, plot)

    if is_materialized(plot)
        # Materialized: emit geometry WITHOUT `displayColor`; bind the OmniPBR
        # material pre-authored into /World/Looks.  A textured material samples
        # the `st` UV primvar, authored from the plot's `:texturecoordinates`.
        texcoords = _texcoords_for(plot, length(points))
        usda = usda_mesh(points, face_counts, face_indices, normals, nothing;
                         model                = args[:model_f32c],
                         normal_interpolation = "vertex",
                         texcoords            = texcoords)
        robj = _add_materialized_reference!(screen, path, usda, plot)
    else
        # Non-materialized: the USD-native `displayColor` path.
        values, interp = _scaled_to_display(plot, args[:scaled_color], length(points))
        usda = usda_mesh(points, face_counts, face_indices, normals, values;
                         model                = args[:model_f32c],
                         normal_interpolation = "vertex",
                         color_interpolation  = interp)
        h = OV.add_usd_reference!(screen.renderer, usda, path)
        robj = OvrtxRObj(path, h)
    end
    robj.meta[:mesh_npoints] = length(points)  # frozen live-push gate size
    return robj
end

function author_usd_prim!(screen, scene, plot::Makie.Scatter, args)
    pos = args[:positions_transformed_f32c]
    n   = length(pos)
    n == 0 && return nothing
    scales = _scales_for(plot.markersize[], n)
    path   = plot_prim_path(screen.scene2scope, scene, plot)
    if is_materialized(plot)
        # ovrtx does NOT honor a material binding on a PointInstancer, so a
        # materialized scatter renders as ONE merged UsdGeomMesh of tessellated
        # unit-sphere markers, without displayColor, bound to the material.
        sphere_mesh    = GeometryBasics.normal_mesh(GeometryBasics.Tesselation(GeometryBasics.Sphere(GeometryBasics.Point3f(0), 1f0), 16))
        sphere_pts     = GeometryBasics.coordinates(sphere_mesh)
        sphere_normals = GeometryBasics.normals(sphere_mesh)
        sphere_counts, sphere_indices = _flat_faces(GeometryBasics.faces(sphere_mesh))
        merged_pts, merged_counts, merged_indices, merged_normals = _merged_instances_mesh(sphere_pts, sphere_counts, sphere_indices, sphere_normals, pos, scales, nothing)
        usda = usda_mesh(merged_pts, merged_counts, merged_indices, merged_normals, nothing; model = args[:model_f32c], normal_interpolation = "vertex")
        return _add_materialized_reference!(screen, path, usda, plot)
    end
    values, interp = _scaled_to_display(plot, args[:scaled_color], n)
    proto_color     = interp == "constant" ? values  : nothing
    instancer_color = interp == "constant" ? nothing : values
    usda = _usda_pointinstancer(pos, scales, nothing, instancer_color,
                                _sphere_proto_body(proto_color); model = args[:model_f32c])
    h = OV.add_usd_reference!(screen.renderer, usda, path)
    robj = OvrtxRObj(path, h)
    robj.meta[:instancer_npoints] = n  # frozen live-push gate size
    robj.meta[:instancer_current_npoints] = n
    return robj
end

function author_usd_prim!(screen, scene, plot::Makie.MeshScatter, args)
    pos = args[:positions_transformed_f32c]
    n   = length(pos)
    n == 0 && return nothing
    marker = plot.marker[]
    marker_mesh    = marker isa GeometryBasics.Mesh ? marker : GeometryBasics.mesh(marker)
    marker_pts     = GeometryBasics.coordinates(marker_mesh)
    marker_normals = GeometryBasics.normals(marker_mesh)
    marker_counts, marker_indices = _flat_faces(GeometryBasics.faces(marker_mesh))
    scales       = _scales_for(plot.markersize[], n)
    orientations = _orientations_for(plot.rotation[], n)
    path         = plot_prim_path(screen.scene2scope, scene, plot)
    if is_materialized(plot)
        # ovrtx does NOT honor a material binding on a PointInstancer, so a
        # materialized meshscatter renders as ONE merged UsdGeomMesh of the
        # marker copies, without displayColor, bound to the material.
        merged_pts, merged_counts, merged_indices, merged_normals = _merged_instances_mesh(marker_pts, marker_counts, marker_indices, marker_normals, pos, scales, orientations)
        usda = usda_mesh(merged_pts, merged_counts, merged_indices, merged_normals, nothing; model = args[:model_f32c], normal_interpolation = "vertex")
        return _add_materialized_reference!(screen, path, usda, plot)
    end
    values, interp = _scaled_to_display(plot, args[:scaled_color], n)
    proto_color     = interp == "constant" ? values  : nothing
    instancer_color = interp == "constant" ? nothing : values
    proto = _mesh_proto_body(marker_pts, marker_counts, marker_indices, marker_normals, proto_color)
    usda  = _usda_pointinstancer(pos, scales, orientations, instancer_color, proto;
                                 model = args[:model_f32c])
    h = OV.add_usd_reference!(screen.renderer, usda, path)
    robj = OvrtxRObj(path, h)
    robj.meta[:instancer_npoints] = n  # frozen live-push gate size
    robj.meta[:instancer_current_npoints] = n
    return robj
end

function author_usd_prim!(screen, scene, plot::Makie.Lines, args)
    pts = args[:positions_transformed_f32c]
    # NaN-separated polyline (Makie's broken-line idiom): split into
    # contiguous finite runs ≥2, each a BasisCurves curve; no finite run →
    # `nothing`.  `keep` filters the per-vertex colour by the same mask.
    fpts, counts, keep = _split_nan_runs(pts)
    isempty(counts) && return nothing
    width = _curve_width(fpts, plot.linewidth[])
    path  = plot_prim_path(screen.scene2scope, scene, plot)
    if is_materialized(plot)
        # Materialized: bind the OmniPBR material; emit the curve WITHOUT
        # `displayColor` (the `nothing` sentinel).
        usda = _usda_basiscurves(fpts, counts, width, nothing, "constant"; model = args[:model_f32c])
        robj = _add_materialized_reference!(screen, path, usda, plot)
        robj.meta[:curve_npoints] = length(fpts)  # frozen live-push gate size
        robj.meta[:curve_points] = fpts
        return robj
    end
    values, interp = _scaled_to_display(plot, args[:scaled_color], length(fpts))
    values = interp == "vertex" ? values[keep] : values
    usda  = _usda_basiscurves(fpts, counts, width, values, interp; model = args[:model_f32c])
    h = OV.add_usd_reference!(screen.renderer, usda, path)
    robj = OvrtxRObj(path, h)
    robj.meta[:curve_npoints] = length(fpts)
    robj.meta[:curve_points] = fpts
    return robj
end

function author_usd_prim!(screen, scene, plot::Makie.LineSegments, args)
    pts = args[:positions_transformed_f32c]
    # Drop segments with a non-finite endpoint; `keep` filters colours.
    seg_pts, counts, keep = _finite_segments(pts)
    isempty(counts) && return nothing
    width = _curve_width(seg_pts, plot.linewidth[])
    path  = plot_prim_path(screen.scene2scope, scene, plot)
    if is_materialized(plot)
        usda = _usda_basiscurves(seg_pts, counts, width, nothing, "constant"; model = args[:model_f32c])
        robj = _add_materialized_reference!(screen, path, usda, plot)
        robj.meta[:curve_npoints] = length(seg_pts)
        robj.meta[:curve_points] = seg_pts
        return robj
    end
    values, interp = _scaled_to_display(plot, args[:scaled_color], length(seg_pts))
    values = interp == "vertex" ? values[keep] : values
    usda  = _usda_basiscurves(seg_pts, counts, width, values, interp; model = args[:model_f32c])
    h = OV.add_usd_reference!(screen.renderer, usda, path)
    robj = OvrtxRObj(path, h)
    robj.meta[:curve_npoints] = length(seg_pts)
    robj.meta[:curve_points] = seg_pts
    return robj
end

# Resolve a `Volume`'s transfer-function domain as Float64: an explicit
# `(lo,hi)` passes through; `Automatic()` uses `_resolve_colorrange` (NaN-safe
# finite extrema).  Cosmetic under IndeX Direct (default grayscale TF).
function _volume_colorrange(plot, scalars)
    cr = Makie.to_value(plot.colorrange)
    cr isa Makie.Automatic || return (Float64(first(cr)), Float64(last(cr)))
    lo, hi = _resolve_colorrange(plot, scalars)
    return (Float64(lo), Float64(hi))
end

# `volume!(x,y,z,::Array{Float32,3})` build: read the scalar field + axis
# ranges + colormap off the plot, write the dense array to a screen-owned
# temp `.nvdb` (`NanoVDBWriter.save_nanovdb`), and author it via
# `_vdb_volume_usda` + `add_usd_reference!` on the OPEN stage (the returned
# handle is needed for `remove_usd!`).  The writer maps data[i,j,k] with
# i→x, j→y, k→z, matching Makie's `volume!` — no index remap.  IndeX Direct
# renders the default grayscale density TF (the authored colormap is ignored).
# The temp `.nvdb` lands on `robj.meta[:vdb_tmp]`, removed by
# `destroy_bindings!` on close/delete.
function author_usd_prim!(screen, scene, plot::Makie.Volume, args)
    OV._index_enabled() || error(
        "author_usd_prim!(::Volume): volume rendering requires NVIDIA IndeX, which is not enabled.  " *
        "Set OMNIVERSEMAKIE_INDEX_LIBS (or OMNIVERSEMAKIE_OVRTX_CONFIG) BEFORE creating the Screen, " *
        "then re-create it.")
    scalars = Float32.(Makie.to_value(plot[4]))
    # An all-zero field authors nothing: IndeX renders uniform/zero density
    # fully transparent, so there is no geometry.  A later live fill
    # (`plot[4][] = nonzero`) rebuilds via the diff node's late-build path.
    all(iszero, scalars) && return nothing
    xr = Makie.to_value(plot[1]); yr = Makie.to_value(plot[2]); zr = Makie.to_value(plot[3])
    origin = GeometryBasics.Point3f(first(xr), first(yr), first(zr))
    extent = GeometryBasics.Vec3f(last(xr) - first(xr), last(yr) - first(yr), last(zr) - first(zr))
    tmp   = tempname() * ".nvdb"
    built = false
    # If save/add throws before the robj owns `tmp`, remove the orphan here
    # (nothing else records or removes it).
    try
        NanoVDBWriter.save_nanovdb(tmp, scalars, origin, extent)
        path = plot_prim_path(screen.scene2scope, scene, plot)
        usda = _vdb_volume_usda(tmp; prim_path = path, field = "density",
                                colormap = Makie.to_value(plot.colormap),
                                colorrange = _volume_colorrange(plot, scalars),
                                bounds = (origin, origin .+ extent))
        h = OV.add_usd_reference!(screen.renderer, usda, path)
        robj = OvrtxRObj(path, h)
        robj.meta[:vdb_tmp]     = tmp   # cleaned by destroy_bindings!
        robj.meta[:volume_prim] = path  # the authored UsdVol prim
        built = true
        return robj
    finally
        built || rm(tmp; force = true)
    end
end

# ------------------------------------------------------------------
# push_to_ovrtx! — route ONE changed output to the right minimal C write
# ------------------------------------------------------------------

# Makie column-vector model matrix → USD row-vector 4×4 (Float64),
# translation in the last row — the form `OV.write_xform!` expects.
_model_to_usd_xform(m) = Float64.(collect(m'))

# Plot-type hook to tweak `:model_f32c` before it becomes `omni:xform`:
# identity for every plot; usdplot.jl specializes `::USDPlot` to fold the
# `up = :y` +90° X rotation in (author-time and live writes agree).
_usdplot_model(plot, model) = model

# Constant or per-vertex displayColor → a 3-lane color3f[] write on the
# referenced prim; one element per colour, `shape = [ncolors]`.
function _push_displaycolor!(r, prim, plot, scaled_color)
    # a numeric scaled_color maps through the plot colormap
    values, _ = _scaled_to_display(plot, scaled_color, 0)
    # An EMPTY per-vertex color has nothing to write; the `[values]` wrap +
    # `c[1]` under @inbounds below would be UB.  Skip (no write) and report
    # unrouted so the caller burns no RT2 reset.
    values isa AbstractVector && isempty(values) && return false
    rgbs = values isa AbstractVector && first(values) isa Union{Tuple,AbstractVector} ?
        values : [values]
    flat = Vector{Float32}(undef, 3 * length(rgbs))
    @inbounds for (i, c) in enumerate(rgbs)
        flat[3i-2] = Float32(c[1]); flat[3i-1] = Float32(c[2]); flat[3i] = Float32(c[3])
    end
    dtype = LibOVRTX.DLDataType(UInt8(LibOVRTX.kDLFloat), UInt8(32), UInt16(3))
    OV._write_attribute!(r, prim, "primvars:displayColor", dtype, true,
                         LibOVRTX.OVRTX_SEMANTIC_NONE, flat, Int64[length(rgbs)])
    return true
end

# USD `visibility` token (`inherited`/`invisible`): a TOKEN_STRING write
# needs a 128-bit element = one `ovx_string_t` (ptr+len); GC.@preserve BOTH
# the struct vector AND the backing String across the FFI call.
function _push_visibility!(r, prim, visible::Bool)
    tok = visible ? "inherited" : "invisible"
    dtype = LibOVRTX.DLDataType(UInt8(LibOVRTX.kDLUInt), UInt8(128), UInt16(1))
    GC.@preserve tok begin
        data = LibOVRTX.ovx_string_t[LibOVRTX.ovx_string(tok)]
        OV._write_attribute!(r, prim, "visibility", dtype, false,
                             LibOVRTX.OVRTX_SEMANTIC_TOKEN_STRING, data, Int64[1])
    end
    return nothing
end

# Topology change (rare): faceVertexIndices (0-based) + faceVertexCounts.
function _push_faces!(r, prim, faces)
    idx    = Int32[Int32(GeometryBasics.raw(i)) for f in faces for i in f]
    counts = Int32[Int32(length(f)) for f in faces]
    OV.write_array_attribute!(r, prim, "faceVertexIndices", idx)
    OV.write_array_attribute!(r, prim, "faceVertexCounts", counts)
    return nothing
end

# Write a resolved positions output through the persistent point3f[] array
# binding (one element per point, shape = [npoints]) as a reinterpreted
# Float32 view, no copy: `pointer` on a ReinterpretArray over a Vector is the
# parent buffer's pointer, and preserving the view roots the parent across
# the ccall.
function _push_points_binding!(binding, pts::AbstractVector)
    src  = pts isa Vector ? pts : collect(pts)
    data = reinterpret(Float32, src)
    OV.write_binding!(binding, data, Int64[length(src)])
    return nothing
end

# NaN-aware re-split of a live positions edit, dispatched on the curve plot
# type (mirrors the author): Lines → contiguous finite runs; LineSegments →
# per-segment endpoint filter.
_curve_split(::Makie.Lines, pts)        = _split_nan_runs(pts)
_curve_split(::Makie.LineSegments, pts) = _finite_segments(pts)

# Live positions edit for a Lines/LineSegments BasisCurves prim.  A NaN edit
# (Makie's broken-line idiom) can change the topology on any frame, so
# re-split and re-write `curveVertexCounts` unconditionally alongside
# `points` — it is tiny, and this stays correct across both finite→NaN
# splits and NaN→finite merges.  The persistent `points` binding is sized
# once at author time; the frozen `robj.meta[:curve_npoints]` gates it.
function _push_curve_positions!(screen, robj, plot, binding, value)
    r = screen.renderer; prim = robj.prim_path
    fpts, counts, _ = _curve_split(plot, value)
    OV.write_array_attribute!(r, prim, "curveVertexCounts", Int32.(counts))
    # Gate on the frozen author-time bound size (never updated on a push):
    # the zero-copy binding is written only at exactly that length; any other
    # count takes the one-shot `write_array_attribute!` resize path.
    if binding !== nothing && length(fpts) == get(robj.meta, :curve_npoints, -1)
        _push_points_binding!(binding, fpts)
    else
        OV.write_array_attribute!(r, prim, "points", fpts)
    end
    robj.meta[:curve_points] = fpts
    return nothing
end

# Live positions edit for a Mesh `points` array.  The persistent binding is
# sized once at author time; the frozen `robj.meta[:mesh_npoints]` gates it —
# a vertex-count-changing edit takes the one-shot `write_array_attribute!`
# resize path instead of writing through the author-time-sized binding (the
# documented-unreliable resize-through-binding path).  Mirrors the curve/
# instancer gates.  `:faces`/`:normals` push separately when they change.
function _push_mesh_positions!(screen, robj, binding, value)
    r = screen.renderer; prim = robj.prim_path
    if binding !== nothing && length(value) == get(robj.meta, :mesh_npoints, -1)
        _push_points_binding!(binding, value)
    else
        OV.write_array_attribute!(r, prim, "points", value)
    end
    return nothing
end

# Live positions edit for a non-materialized Scatter/MeshScatter
# UsdGeomPointInstancer.  The per-instance attr is `positions`, NOT `points`
# (ovrtx silently drops writes to nonexistent attrs).  The frozen
# `robj.meta[:instancer_npoints]` gates the binding to its author-time
# length; any other count takes the one-shot path and rewrites the coupled
# per-instance arrays that must match positions.
function _push_instancer_positions!(screen, robj, plot, binding, value)
    r = screen.renderer; prim = robj.prim_path
    n = length(value)
    if binding !== nothing && length(value) == get(robj.meta, :instancer_npoints, -1)
        _push_points_binding!(binding, value)
    else
        OV.write_array_attribute!(r, prim, "positions", value)
        OV.write_array_attribute!(r, prim, "protoIndices", fill(Int32(0), n))
        _push_instancer_scales!(r, prim, plot, n)
        plot isa Makie.MeshScatter && _push_instancer_orientations!(r, prim, plot, n)
    end
    robj.meta[:instancer_current_npoints] = n
    return nothing
end

function _push_instancer_scales!(r, prim, plot, n::Integer)
    OV.write_array_attribute!(r, prim, "scales", _scales_for(plot.markersize[], Int(n)))
    return nothing
end

function _push_instancer_orientations!(r, prim, plot, n::Integer)
    _write_orientations!(r, prim, _orientations_or_identity(plot.rotation[], Int(n)))
    return nothing
end

function _write_orientations!(r, prim, orientations)
    data = Vector{Float16}(undef, 4 * length(orientations))
    @inbounds for (i, q) in enumerate(orientations)
        j = 4i - 3
        data[j]     = Float16(q[1])
        data[j + 1] = Float16(q[2])
        data[j + 2] = Float16(q[3])
        data[j + 3] = Float16(q[4])
    end
    dtype = LibOVRTX.DLDataType(UInt8(LibOVRTX.kDLFloat), UInt8(16), UInt16(4))
    OV._write_attribute!(r, prim, "orientations", dtype, true,
                         LibOVRTX.OVRTX_SEMANTIC_NONE, data, Int64[length(orientations)])
    return nothing
end

function _orientations_or_identity(rotation, n::Integer)
    orientations = try
        _orientations_for(rotation, Int(n))
    catch err
        err isa InterruptException && rethrow()
        if err isa ArgumentError
            @warn "OmniverseMakie: live `rotation` vector length does not match the \
                   current meshscatter instance count — using identity orientations." maxlog=1
            nothing
        else
            rethrow()
        end
    end
    return orientations === nothing ? fill((1.0f0, 0.0f0, 0.0f0, 0.0f0), Int(n)) : orientations
end

function _push_curve_width!(r, prim, robj, plot, linewidth)
    pts = get(robj.meta, :curve_points, nothing)
    pts === nothing && return false
    isempty(pts) && return false
    OV.write_array_attribute!(r, prim, "widths", Float32[_curve_width(pts, linewidth)])
    return true
end

# Diagnostic hook: called with the attribute name on every `push_to_ovrtx!`
# write.  `nothing` (default) → no overhead.
const _PUSH_OBSERVER = Ref{Any}(nothing)

# Diagnostic hook: called with the OmniPBR input name on every live
# shader-input write.  `nothing` (default) → no overhead.
const _SHADER_WRITE_OBSERVER = Ref{Any}(nothing)

# Write one OmniPBR shader input live + fire the shader-write observer
# (per-input granularity; one `:material` push may re-write several inputs).
function _write_shader_input!(r, shader_prim::AbstractString, input_name::AbstractString, value)
    OV.write_shader_input!(r, shader_prim, input_name, value)
    ob = _SHADER_WRITE_OBSERVER[]
    ob === nothing || ob(input_name)
    return nothing
end

# Constant `(r,g,b)` base colour from a resolved `:scaled_color` for a
# materialized plot: a single colour is used directly; a per-vertex colour
# collapses to its average (OmniPBR has no per-vertex diffuse base).
function _materialized_base_rgb(plot, scaled_color)
    # a numeric scaled_color maps through the plot colormap
    values, interp = _scaled_to_display(plot, scaled_color, 0)
    if interp == "constant"
        return (Float32(values[1]), Float32(values[2]), Float32(values[3]))
    end
    # An EMPTY per-vertex color has no average to take (`sum` over ∅ throws);
    # fall back to neutral white so the material keeps a valid base colour.
    isempty(values) && return (1f0, 1f0, 1f0)
    @warn "OmniverseMakie: per-vertex `color` on a materialized plot — a live edit uses a \
           constant AVERAGE base colour (OmniPBR has no per-vertex diffuse base)."
    n = length(values)
    return (Float32(sum(v[1] for v in values) / n),
            Float32(sum(v[2] for v in values) / n),
            Float32(sum(v[3] for v in values) / n))
end

# Route a changed `:material` to live shader-input writes on the plot's
# material kind (OmniPBR `_merge_material_input!` / OmniGlass
# `_merge_glass_input!`).  Texture-asset live-swaps are warn+skipped.
function _push_material!(r, shader_prim::AbstractString, kind::Symbol, material_attrs)
    inputs = Dict{String,Any}()
    for k in keys(material_attrs)
        if kind === :glass
            _merge_glass_input!(inputs, Symbol(k), Makie.to_value(material_attrs[k]))
        else
            # plot = nothing: this live path writes only scalar/color3f
            # inputs and warn+skips texture swaps, so the plot threaded into
            # `_texture_asset_for` is irrelevant here.
            _merge_material_input!(inputs, Symbol(k), Makie.to_value(material_attrs[k]), nothing, false, false)
        end
    end
    for (input_name, v) in inputs
        if v isa AbstractString
            @warn "OmniverseMakie: live texture-asset swap (`$(input_name)`) is not supported \
                   — skipped."
        elseif v isa NTuple{3}
            _write_shader_input!(r, shader_prim, input_name,
                                 (Float32(v[1]), Float32(v[2]), Float32(v[3])))
        elseif v isa NTuple{2}
            # float2 (UV tiling) — live-writable like scalars.
            _write_shader_input!(r, shader_prim, input_name, (Float32(v[1]), Float32(v[2])))
        elseif v isa Real
            _write_shader_input!(r, shader_prim, input_name, Float32(v))
        else
            @warn "OmniverseMakie: live material input `$(input_name)` has unsupported type \
                   $(typeof(v)) — skipped."
        end
    end
    return nothing
end

"""
    push_to_ovrtx!(screen, robj, plot, name::Symbol, value) -> Bool

Route ONE changed compute output to its minimal in-place USD write on
`robj.prim_path` (no re-author).  Returns `true` if a write was issued (an
unrouted `name` → `false`):

- `:model_f32c`                 → `OV.write_xform!` (`omni:xform`, composed
                                  world transform)
- `:positions_transformed_f32c` → `points` array write (Mesh); `points` +
                                  `curveVertexCounts` (Lines/LineSegments);
                                  `positions` (Scatter/MeshScatter
                                  UsdGeomPointInstancer); a MATERIALIZED
                                  scatter/meshscatter (merged mesh) is
                                  warn+skipped (needs a re-author)
- `:normals`                    → `normals` array write
- `:faces`                      → `faceVertexIndices` + `faceVertexCounts`
- `:markersize`                 → `scales` on a Scatter/MeshScatter
                                  UsdGeomPointInstancer
- `:rotation`                   → `orientations` on a MeshScatter
                                  UsdGeomPointInstancer
- `:linewidth`                  → `widths` on a Lines/LineSegments
                                  BasisCurves prim
- `:scaled_color`               → `primvars:displayColor`, or (materialized)
                                  the shader's base-colour input
- `:material`                   → each changed scalar/color3f shader input
                                  on a MATERIALIZED plot's `Mat_<id>/Shader`
                                  (no-op otherwise)
- `:volume`                     → live density edit: fresh temp `.nvdb` +
                                  reload (`reload_volume_data!`)
- `:visible`                    → `visibility` token
"""
function push_to_ovrtx!(screen, robj::OvrtxRObj, plot, name::Symbol, value)
    r       = screen.renderer
    prim    = robj.prim_path
    binding = get(robj.bindings, name, nothing)   # hot-path binding or nothing
    routed  = true
    if name === :model_f32c
        # omni:xform — zero-copy through the mapped binding when one exists,
        # else the one-shot `write_attribute` path.  `_usdplot_model` folds a
        # USDPlot `up = :y` correction in (identity for other plot types).
        m = _usdplot_model(plot, value)
        if binding === nothing
            OV.write_xform!(r, prim, _model_to_usd_xform(m))
        else
            OV.write_mapped_xform!(binding, _model_to_usd_xform(m))
        end
    elseif name === :positions_transformed_f32c
        if plot isa Union{Makie.Lines,Makie.LineSegments}
            # BasisCurves: NaN-aware re-split (see _push_curve_positions!).
            _push_curve_positions!(screen, robj, plot, binding, value)
        elseif plot isa Union{Makie.Scatter,Makie.MeshScatter}
            # PointInstancer: the per-instance attr is `positions`, NOT
            # `points`.  A materialized scatter is a merged mesh instead —
            # warn+skip (routed=false → no reset burn); needs a re-author.
            if is_materialized(plot)
                @warn "OmniverseMakie: live position edits on a materialized scatter need a \
                       re-author — skipped." maxlog=1
                routed = false
            else
                _push_instancer_positions!(screen, robj, plot, binding, value)
            end
        else
            # mesh `points`: frozen-count gate (binding when the vertex count
            # is unchanged, one-shot resize otherwise — see
            # `_push_mesh_positions!`).
            _push_mesh_positions!(screen, robj, binding, value)
        end
    elseif name === :normals
        OV.write_array_attribute!(r, prim, "normals", value)
    elseif name === :faces
        _push_faces!(r, prim, value)
    elseif name === :scaled_color
        if robj.material_shader === nothing
            routed = _push_displaycolor!(r, prim, plot, value)   # displayColor (false on empty)
        elseif _plot_color(plot) isa AbstractMatrix
            # Image `color` = a texture asset input.  ovrtx cannot live-swap
            # a texture asset via the FFI, the OmniPBR material is baked into
            # the root layer (not a removable reference), and writing
            # `diffuse_color_constant` would be inert (the texture overrides
            # it) → warn + skip; recreate the Screen to change the texture.
            @warn "OmniverseMakie: live update of an image `color` (texture) is not supported — \
                   ovrtx can't live-swap a texture asset; recreate the Screen to change it. Skipped." maxlog=1
            routed = false
        else
            # Materialized: re-write the base colour in place (constant only)
            # on the material kind (glass_color vs diffuse_color_constant).
            _write_shader_input!(r, robj.material_shader, _base_color_input(_material_kind(plot)),
                                 _materialized_base_rgb(plot, value))
        end
    elseif name === :markersize
        if plot isa Union{Makie.Scatter,Makie.MeshScatter} && !is_materialized(plot)
            n = get(robj.meta, :instancer_current_npoints, get(robj.meta, :instancer_npoints, 0))
            n > 0 || return false
            OV.write_array_attribute!(r, prim, "scales", _scales_for(value, n))
        else
            @warn "OmniverseMakie: live `markersize` edits on a materialized scatter need a \
                   re-author — skipped." maxlog=1
            routed = false
        end
    elseif name === :rotation
        if plot isa Makie.MeshScatter && !is_materialized(plot)
            n = get(robj.meta, :instancer_current_npoints, get(robj.meta, :instancer_npoints, 0))
            n > 0 || return false
            _write_orientations!(r, prim, _orientations_or_identity(value, n))
        else
            @warn "OmniverseMakie: live `rotation` edits on a materialized scatter need a \
                   re-author — skipped." maxlog=1
            routed = false
        end
    elseif name === :linewidth
        if plot isa Union{Makie.Lines,Makie.LineSegments}
            routed = _push_curve_width!(r, prim, robj, plot, value)
        else
            routed = false
        end
    elseif name === :material
        # A live material edit re-writes the pre-authored shader's inputs.
        if robj.material_shader === nothing
            # A plain plot gaining a material at runtime is a true material
            # swap (needs a root re-author) — unsupported.
            @warn "OmniverseMakie: live `material` edit on a non-materialized plot is not \
                   supported (a runtime material swap needs a root re-author) — skipped."
            routed = false
        else
            _push_material!(r, robj.material_shader, _material_kind(plot), value)
        end
    elseif name === :volume
        # Live density edit: fresh temp `.nvdb` + reload (remove_usd! +
        # add_usd_reference!; a filePath write does NOT update).  On reload
        # failure keep the last good frame (warn once, don't thrash).
        try
            reload_volume_data!(screen, robj, plot, value)
        catch e
            @warn "OmniverseMakie: live volume data reload failed — keeping the last frame." exception=e maxlog=1
            routed = false
        end
    elseif name === :visible
        _push_visibility!(r, prim, value)
    else
        routed = false
    end
    if routed
        ob = _PUSH_OBSERVER[]
        ob === nothing || ob(name)
    end
    return routed
end

# ------------------------------------------------------------------
# bind_hot_attributes! — create the persistent hot-path bindings
# ------------------------------------------------------------------

# The array attribute a persistent `:positions_transformed_f32c` binding
# backs (`nothing` = none): `points` (Mesh/BasisCurves), `positions` (a
# non-materialized Scatter/MeshScatter instancer).  A materialized scatter's
# live-positions route warn+skips, so it takes no positions binding.
_points_binding_attr(::Makie.Mesh)         = "points"
_points_binding_attr(::Makie.Lines)        = "points"
_points_binding_attr(::Makie.LineSegments) = "points"
_points_binding_attr(p::Union{Makie.Scatter,Makie.MeshScatter}) = is_materialized(p) ? nothing : "positions"
_points_binding_attr(::Any)                = nothing

"""
    bind_hot_attributes!(screen, robj, plot, args) -> OvrtxRObj

Create the per-plot persistent attribute bindings once (right after the USD
reference is authored), keyed in `robj.bindings` by the driving
compute-output name so `push_to_ovrtx!` routes a changed attribute through
the binding instead of re-authoring:

- `:model_f32c` → an `omni:xform` binding (`OVRTX_BINDING_FLAG_OPTIMIZE`)
  written zero-copy via `map_attribute` (every plot type).
- `:positions_transformed_f32c` → a `point3f[]` array binding on the attr
  `_points_binding_attr` picks (`points` for mesh/curve, `positions` for a
  non-materialized instancer; none for a materialized scatter).

Both target `robj.prim_path`; a map/write through a persistent binding on a
referenced prim honors the edit.  Released by `destroy_bindings!` on
`close(Screen)` / per-plot `delete!`.
"""
function bind_hot_attributes!(screen, robj::OvrtxRObj, plot, args)
    r    = screen.renderer
    prim = robj.prim_path
    # Tier 1 (universal): omni:xform, zero-copy map, OPTIMIZE.
    robj.bindings[:model_f32c] = OV.create_binding(
        r, prim, "omni:xform",
        LibOVRTX.DLDataType(UInt8(LibOVRTX.kDLFloat), UInt8(64), UInt16(16));
        array = false, semantic = LibOVRTX.OVRTX_SEMANTIC_XFORM_MAT4x4, optimize = true)
    # Tier 2 (mesh/curve `points` or non-materialized instancer `positions`):
    # point3f[] array, bind + write.  `_points_binding_attr` picks the attr
    # (or `nothing` → skip).
    pname = _points_binding_attr(plot)
    if pname !== nothing
        robj.bindings[:positions_transformed_f32c] = OV.create_binding(
            r, prim, pname,
            LibOVRTX.DLDataType(UInt8(LibOVRTX.kDLFloat), UInt8(32), UInt16(3));
            array = true, semantic = LibOVRTX.OVRTX_SEMANTIC_NONE, optimize = false)
    end
    return robj
end

# A `Volume`'s UsdVol prim carries NO xform (placement lives in the `.nvdb`'s
# baked voxel→world Map, and extra metadata on a volume layer can render it
# black); its tracked inputs are one-shot writes — no binding to create.
bind_hot_attributes!(screen, robj::OvrtxRObj, ::Makie.Volume, args) = robj

"""
    destroy_bindings!(robj::OvrtxRObj) -> OvrtxRObj

Destroy + clear every persistent binding on `robj`; called by `close(Screen)`
and per-plot `delete!`.  Safe after the Renderer is closed (`OV.destroy!` is
a no-op then).  Also removes the screen-owned temp `.nvdb` on
`robj.meta[:vdb_tmp]` (`force=true` → idempotent across teardown sites).
"""
function destroy_bindings!(robj::OvrtxRObj)
    for b in values(robj.bindings)
        b isa OV.Binding && OV.destroy!(b)
    end
    empty!(robj.bindings)
    tmp = get(robj.meta, :vdb_tmp, nothing)
    if tmp isa AbstractString
        rm(tmp; force = true)
        delete!(robj.meta, :vdb_tmp)
    end
    # usdplot: detach any bind_usd! observable listeners this robj owns so a
    # deleted/closed plot leaves no dangling Observable references.
    ofs = get(robj.meta, :usd_binding_obsfuncs, nothing)
    if ofs isa AbstractDict
        for of in values(ofs)
            Makie.Observables.off(of)
        end
        delete!(robj.meta, :usd_binding_obsfuncs)
    end
    return robj
end

# ------------------------------------------------------------------
# register_ovrtx_robj! — register the diff node, force first resolve (build)
# ------------------------------------------------------------------

# Record the plot↔render-object links (`robj.plot` back-reference + forward
# `plot2robj` + reverse `path2plot`) so a built reference is pickable.  One
# place — register time AND late (empty→fill) rebuild — so they never drift.
function _register_robj_maps!(screen, plot, robj::OvrtxRObj)
    robj.plot = plot
    screen.plot2robj[objectid(plot)] = robj
    screen.path2plot[robj.prim_path] = objectid(plot)
    # A (late-)built reference is a stage-composition change → drop the
    # cached PathResolver + flag structural_dirty (accumulate mode).  The
    # remove side lives in _teardown_usd_reference! / reload_volume_data!.
    _note_composition_change!(screen)
    return robj
end

"""
    register_ovrtx_robj!(screen, scene, plot) -> Union{OvrtxRObj,Nothing}

Register the plot's `:ovrtx_renderobject` diff node (once) and force a
resolve, which builds the USD reference via `author_usd_prim!` on `screen`'s
open stage and records the `OvrtxRObj` in `screen.plot2robj`.  Later
resolves (pulled by `colorbuffer`) push minimal writes.

The node consumes a `:ovrtx_screen` input carrying the owning screen.  A
plot's compute graph OUTLIVES a `Screen` (a `Figure` can be rendered by
several transient screens), so pointing `:ovrtx_screen` at a new screen
marks the node dirty and rebuilds the reference on that screen's fresh stage
instead of pushing diffs to a closed renderer; re-rendering the same screen
leaves it unchanged so only real edits drive writes.

Single live-screen model: `:ovrtx_screen` holds ONE screen, so this supports
SEQUENTIAL screens (each new screen supersedes a closed one).  Two screens
displaying the same figure CONCURRENTLY is not: the last to register wins all
live edits and the earlier screen renders an ever-staler stage, so
re-pointing away from a still-open screen `@warn`s once (full multi-screen
diff routing is out of scope).

A plot type with no tracked inputs (empty `consumed_inputs` —
Surface/unknown) gets no node and is built once per screen via
`to_ovrtx_object`.
"""
function register_ovrtx_robj!(screen, scene, plot)
    inputs = consumed_inputs(plot)
    if isempty(inputs)
        h = to_ovrtx_object(screen, scene, plot)
        h === nothing && return nothing
        path = plot_prim_path(screen.scene2scope, scene, plot)
        # A materialized no-diff-node plot (Surface) gets its pre-authored
        # material bound via the shared epilogue (`to_ovrtx_object` returned
        # the handle `h`, not a USDA string).  No diff node ⇒ static material.
        robj = is_materialized(plot) ?
            _add_materialized_reference!(screen, path, h, plot) :
            OvrtxRObj(path, h)
        return _register_robj_maps!(screen, plot, robj)
    end

    attr = plot.attributes
    # Per-screen build context: the diff node consumes this; pointing it at a
    # new screen marks it dirty → rebuild on that screen's stage.  Single
    # live-screen model: the node has ONE `:ovrtx_screen`, so if the plot is
    # already displayed on a DIFFERENT, still-open screen, live edits now route
    # here and the earlier screen shows a frozen stage — warn once.
    if haskey(attr, :ovrtx_screen)
        prev = attr[:ovrtx_screen][]
        if prev isa Screen && prev !== screen && isopen(prev)
            @warn "OmniverseMakie: this plot is now displayed on a second live Screen; live \
                   edits route to the most recent Screen and the earlier one shows a frozen \
                   stage (concurrent multi-screen routing is unsupported)." maxlog=1
        end
        setproperty!(attr, :ovrtx_screen, screen)
    else
        ComputePipeline.add_input!(attr, :ovrtx_screen, screen)
    end

    if !haskey(attr, :ovrtx_renderobject)
        node_inputs = Symbol[:ovrtx_screen; inputs...]
        ComputePipeline.register_computation!(attr, node_inputs, [:ovrtx_renderobject]) do args, changed, last
            scr = args[:ovrtx_screen]
            local robj
            dirty = false   # did this resolve change the stage?
            if isnothing(last) || changed[:ovrtx_screen]
                # `scene` captured from the arg; on a rebuild for a new
                # screen, scr.scene2scope (same objectids) yields the same
                # nested path — the reference re-nests identically.
                robj = author_usd_prim!(scr, scene, plot, args)   # (re)build
                # create the persistent hot-path bindings once
                robj === nothing || bind_hot_attributes!(scr, robj, plot, args)
                dirty = true   # a (re)build authored geometry
            else
                robj = last.ovrtx_renderobject
                if robj === nothing
                    # Late (empty→fill) build: the plot was authored empty, so
                    # no reference exists on this screen.  A data edit that
                    # fills it builds now — the same path as first resolve
                    # (author + bind + pick maps) — else the fill is silently
                    # dropped; a still-empty build retries on the next fill.
                    if any(n -> n !== :ovrtx_screen && changed[n], keys(args))
                        robj = author_usd_prim!(scr, scene, plot, args)
                        if robj !== nothing
                            bind_hot_attributes!(scr, robj, plot, args)
                            _register_robj_maps!(scr, plot, robj)
                            dirty = true   # late build authored geometry
                        end
                    end
                else
                    for name in keys(args)
                        name === :ovrtx_screen && continue
                        changed[name] || continue   # minimal-delta gate
                        # A skipped push (push_to_ovrtx! → false) writes
                        # nothing to the stage, so it must NOT flip
                        # requires_update — a reset for a no-op burns a frame.
                        push_to_ovrtx!(scr, robj, plot, name, args[name]) && (dirty = true)
                    end
                end
            end
            # Only a REAL stage change resets RT2.  Guarded on `dirty`:
            # sibling plots share screen.requires_update within one pull, so
            # this only sets the flag (under edit_lock);
            # _sync_and_needs_reset! owns the atomic read-and-clear.
            scr === nothing || !dirty || _set_requires_update!(scr)
            return (robj,)
        end
    end

    built = attr[:ovrtx_renderobject][]   # force resolve → (re)build
    built === nothing || _register_robj_maps!(screen, plot, built)  # pick maps
    return built
end

# ------------------------------------------------------------------
# pull_ovrtx_nodes! — colorbuffer per-frame node resolution
# ------------------------------------------------------------------

# Resolve one plot's `:ovrtx_renderobject` node (recursing composites): a
# clean node is a no-op; a dirty one pushes per changed input.  A failed
# resolve is marked resolved so it doesn't re-throw every frame.
function _pull_plot_node!(screen, plot)
    if isempty(plot.plots)
        attr = plot.attributes
        if haskey(attr, :ovrtx_renderobject)
            try
                attr[:ovrtx_renderobject][]
            catch e
                # Never swallow an interrupt; mark the node resolved otherwise
                # so a persistent failure does not re-throw every frame.
                e isa InterruptException && rethrow()
                @warn "OmniverseMakie: :ovrtx_renderobject resolve failed" plot=typeof(plot) err=e maxlog=1
                ComputePipeline.mark_resolved!(attr.outputs[:ovrtx_renderobject])
            end
        end
    else
        foreach(p -> _pull_plot_node!(screen, p), plot.plots)
    end
    return
end

"""
    pull_ovrtx_nodes!(screen, scene) -> Nothing

Resolve every plot's `:ovrtx_renderobject` diff node (scene + children) before a render.
No-op for a static graph; any change flips `screen.requires_update` (via the node callback)
so `colorbuffer` issues one `OV.reset!`.
"""
function pull_ovrtx_nodes!(screen, scene)
    for plot in scene.plots
        _pull_plot_node!(screen, plot)
    end
    foreach(child -> pull_ovrtx_nodes!(screen, child), scene.children)
    return
end
