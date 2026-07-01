# Per-plot render-object record `OvrtxRObj` + the M2.2 diff driver: the
# `:ovrtx_renderobject` node pushes one minimal C write per changed plot attribute on the
# already-open stage instead of re-authoring.  M2.4 fills `bindings` (hot path).
# Included BEFORE screen.jl: `Screen.plot2robj`'s field type references `OvrtxRObj`.

"""
    OvrtxRObj

Per-plot render object recorded when a plot's USD reference is authored on the open
stage.  Fields:

- `prim_path`  — USD prim the reference was added at (`/World/plot_<id>`).
- `usd_handle` — `ovrtx_usd_handle_t` from `OV.add_usd_reference!` (for `OV.remove_usd!`).
- `bindings`   — persistent attribute bindings keyed by driving compute-output name (the
                 M2.4 hot path); empty until `bind_hot_attributes!`, freed by `destroy_bindings!`.
- `material_shader` — MATERIALIZED plot (M3): pre-authored OmniPBR shader prim path
                 (`/World/Looks/Mat_<id>/Shader`), else `nothing`.  `push_to_ovrtx!` uses it to
                 route a live `color`/`material` edit to `write_shader_input!` (M3.4) instead of
                 `displayColor`.
- `plot`       — M6.B: source Makie plot, for pick resolution (`_plot_for_objectid` → `Plot`).
                 Set at the `plot2robj` insert sites so it rides that map's lifecycle.
- `meta`       — per-plot state that is NOT a destroyable GPU binding (unlike `bindings`).  A
                 `Volume` build records `:vdb_tmp` (screen-owned temp `.nvdb`, removed by
                 `destroy_bindings!` on close/delete) + `:volume_prim` (Tasks 4/5).  Empty
                 for non-volume plots.
"""
mutable struct OvrtxRObj
    prim_path::String
    usd_handle::UInt64
    bindings::Dict{Symbol,Any}
    material_shader::Union{String,Nothing}
    plot::Union{Nothing,Makie.AbstractPlot}
    meta::Dict{Symbol,Any}
end

OvrtxRObj(prim_path::AbstractString, usd_handle::Integer) =
    OvrtxRObj(String(prim_path), UInt64(usd_handle), Dict{Symbol,Any}(), nothing, nothing,
              Dict{Symbol,Any}())

# ==================================================================
# M2.2 — :ovrtx_renderobject diff node + push_to_ovrtx! (diff driver)
#
# Each atomic plot gets a ComputePipeline node whose inputs are the owning screen
# (`:ovrtx_screen`) + the plot's RESOLVED compute outputs.  Callback:
#   - first resolve OR new owning screen: BUILDS the USD reference (author_usd_prim!).
#   - later resolves, same screen: pushes ONE minimal C write per CHANGED output
#     (push_to_ovrtx!) — no re-author.
# colorbuffer pulls every node each frame (pull_ovrtx_nodes!); a clean graph is a no-op,
# any change flips screen.requires_update → one OV.reset! for the frame.
#
# Referenced-prim writes on the open stage are spike-proven honored, so the diff path
# writes IN PLACE (no remove_usd!/re-reference fallback needed).
# ==================================================================

# Single source of truth for a plot's USD prim path.  Scope-aware (M2.3): the plot nests
# under its owning scene's `def Scope`, looked up in screen.scene2scope (`/World` for the
# root scene, `/World/Scene_<id>/…` for a subscene).  Keys are stable `objectid`s, so the
# path is identical across screens (a rebuild recomputes it).  A scene not yet in the map
# (subscene added live after authoring) falls back to `/World` (renders flat).
plot_prim_path(scene2scope::AbstractDict, scene, plot) =
    string(get(scene2scope, objectid(scene), "/World"), "/plot_", objectid(plot))

# ------------------------------------------------------------------
# consumed_inputs — per-type Makie compute outputs the diff node tracks
# ------------------------------------------------------------------

# Only outputs that BOTH resolve by default AND have a `push_to_ovrtx!` route are listed
# (each tracked change → exactly one minimal write).  Scatter/Lines size/rotation/linewidth
# diffing is deferred to the M2.3 hot path (no clean in-place USD route yet); the build
# still reads them.  Surface + unknown types get NO node (`Symbol[]`) → built once via M1
# `to_ovrtx_object`.
# M3.4: `:material` is tracked so a live `plot.material[]` edit fires the node and re-writes
# the pre-authored OmniPBR shader inputs.  `consumed_inputs` dispatches on TYPE only and the
# node registers for EVERY plot of that type, so a tracked input MUST resolve for a
# non-materialized plot too — else `register_computation!` raises "Inputs [:material] not
# found" and breaks ALL plots of that type.  `:material` resolves for a non-materialized
# Mesh/MeshScatter (→ `nothing`) but NOT for Scatter/Lines/LineSegments (Makie registers it
# there only with a `material=` kwarg) → tracked for Mesh + MeshScatter ONLY.  Hence a live
# material-PARAM edit on a MATERIALIZED Scatter/Lines/LineSegments is a no-op (documented —
# test/m3_material_live_test.jl); a live `color` edit still works (`:scaled_color` tracked).
consumed_inputs(::Makie.Mesh)         = [:positions_transformed_f32c, :model_f32c, :faces, :normals, :scaled_color, :material, :visible]
consumed_inputs(::Makie.Scatter)      = [:positions_transformed_f32c, :model_f32c, :scaled_color, :visible]
consumed_inputs(::Makie.MeshScatter)  = [:positions_transformed_f32c, :model_f32c, :scaled_color, :material, :visible]
consumed_inputs(::Makie.Lines)        = [:positions_transformed_f32c, :model_f32c, :scaled_color, :visible]
consumed_inputs(::Makie.LineSegments) = [:positions_transformed_f32c, :model_f32c, :scaled_color, :visible]
# Volumes M2: `volume!` renders via the M1 UsdVol→IndeX-Direct path (author_usd_prim!(::Volume)).
# Tracked: `:visible` (resolves for every plot; push = hide/reshow) and `:volume` (the CONVERTED
# scalar-data output — `plot[4][] = arr` re-resolves it; SPIKE-VERIFIED to resolve as
# `Array{Float32,3}` and fire `changed[:volume]`; its push re-writes a fresh temp `.nvdb` + RELOADS
# it via `reload_volume_data!` — the M2 LIVE-DATA path).  Colors are OFF for M2 (IndeX Direct =
# grayscale), so `colormap`/`colorrange` are NOT tracked (a live colormap edit is a moot no-op under
# Direct); both are read off the plot at build + reload time.
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

# M4: a NUMERIC `scaled_color` vector (numbers + colormap) must be mapped THROUGH the plot's
# colormap — the bare `_displaycolor_from_scaled` fallback `_rgb(to_color(::Vector{Float32}))`
# can't (it `red()`s the whole vector → MethodError).  Resolve via `colormap` + `colorrange`
# (`interpolated_getindex`); a Colorant or scalar `scaled_color` defers to
# `_displaycolor_from_scaled`.
function _scaled_to_display(plot, sc, n)
    if sc isa AbstractVector{<:Real}
        cmap   = Makie.to_colormap(plot.colormap[])
        crange = _colorrange(plot, sc)
        return ([_rgb(Makie.interpolated_getindex(cmap, Float32(v), crange)) for v in sc], "vertex")
    end
    return _displaycolor_from_scaled(sc, n)
end

# ------------------------------------------------------------------
# texcoords (`st` UV primvar) for a textured materialized mesh (M3.3)
# ------------------------------------------------------------------

# Per-vertex UVs for an image-/texture-materialized mesh, read off the plot's
# `:texturecoordinates` output (`Vector{Vec2f}`); NOT a tracked `consumed_inputs` diff output.
# Returns `nothing` (→ `usda_mesh` OMITS `st`) when the material samples no texture, the plot
# exposes no texcoords, or their count != the vertex count (skipped, not mis-authored).
function _texcoords_for(plot, npoints::Int)
    _needs_texcoords(plot)              || return nothing
    haskey(plot, :texturecoordinates)  || return nothing
    tc = Makie.to_value(plot[:texturecoordinates])
    (tc isa AbstractVector && length(tc) == npoints) || return nothing
    return tc
end

# ------------------------------------------------------------------
# author_usd_prim! — BUILD branch: M1 emitters fed from resolved `args`
# ------------------------------------------------------------------

"""
    author_usd_prim!(screen, scene, plot, args) -> Union{OvrtxRObj,Nothing}

Build a plot's USD reference on the OPEN stage from its RESOLVED compute outputs
(`args`), returning the `OvrtxRObj` (or `nothing` for an empty plot).

Points come from `:positions_transformed_f32c` (model-LOCAL), the transform from
`:model_f32c` (COMPOSED world), colour from `:scaled_color`; world = `model_f32c · positions`.
Reuses the M1 emitters (`usda_mesh`/`_usda_pointinstancer`/`_usda_basiscurves`) fed from
`args`.  `scene` is threaded so the reference is added at the nested scope path
`plot_prim_path(screen.scene2scope, scene, plot)` (subscene grouping, M2.3).
"""
function author_usd_prim!(screen, scene, plot::Makie.Mesh, args)
    points  = args[:positions_transformed_f32c]
    isempty(points) && return nothing
    normals = args[:normals]
    face_indices = [Int[Int(GeometryBasics.raw(i)) for i in f] for f in args[:faces]]
    path    = plot_prim_path(screen.scene2scope, scene, plot)

    if is_materialized(plot)
        # M3.2: emit geometry WITHOUT `displayColor`, BIND the OmniPBR material pre-authored
        # at open-time into /World/Looks (before the stage opened, so the runtime bind_material!
        # takes).  M3.3: a textured material samples the mesh's `st` UV primvar → author it from
        # the plot's per-vertex `:texturecoordinates` (read directly; NOT a tracked diff output).
        texcoords = _texcoords_for(plot, length(points))
        usda = usda_mesh(points, face_indices, normals, nothing;
                         model                = args[:model_f32c],
                         normal_interpolation = "vertex",
                         texcoords            = texcoords)
        h = OV.add_usd_reference!(screen.renderer, usda, path)
        OV.bind_material!(screen.renderer, path, material_prim_path(plot))
        robj = OvrtxRObj(path, h)
        # M3.4: record the shader prim so a live `color`/`material` edit re-writes it in place.
        robj.material_shader = material_prim_path(plot) * "/Shader"
        return robj
    end

    # Non-materialized: the M1 USD-native `displayColor` path, byte-unchanged.
    values, interp = _scaled_to_display(plot, args[:scaled_color], length(points))
    usda = usda_mesh(points, face_indices, normals, values;
                     model                = args[:model_f32c],
                     normal_interpolation = "vertex",
                     color_interpolation  = interp)
    h = OV.add_usd_reference!(screen.renderer, usda, path)
    return OvrtxRObj(path, h)
end

function author_usd_prim!(screen, scene, plot::Makie.Scatter, args)
    pos = args[:positions_transformed_f32c]
    n   = length(pos)
    n == 0 && return nothing
    scales = _scales_for(plot.markersize[], n)
    path   = plot_prim_path(screen.scene2scope, scene, plot)
    if is_materialized(plot)
        # M3.5: ovrtx does NOT honor a material binding on a PointInstancer, so a MATERIALIZED
        # scatter renders as ONE merged `UsdGeomMesh` of tessellated unit-sphere markers
        # (documented fallback) — WITHOUT `displayColor`, BOUND to the pre-authored material.
        # `material_shader` wires the M3.4 live path.
        sphere_mesh    = GeometryBasics.normal_mesh(GeometryBasics.Tesselation(GeometryBasics.Sphere(GeometryBasics.Point3f(0), 1f0), 16))
        sphere_pts     = GeometryBasics.coordinates(sphere_mesh)
        sphere_normals = GeometryBasics.normals(sphere_mesh)
        sphere_faces   = [Int[Int(GeometryBasics.raw(i)) for i in f] for f in GeometryBasics.faces(sphere_mesh)]
        merged_pts, merged_faces, merged_normals = _merged_instances_mesh(sphere_pts, sphere_faces, sphere_normals, pos, scales, nothing)
        usda = usda_mesh(merged_pts, merged_faces, merged_normals, nothing; model = args[:model_f32c], normal_interpolation = "vertex")
        h = OV.add_usd_reference!(screen.renderer, usda, path)
        OV.bind_material!(screen.renderer, path, material_prim_path(plot))
        robj = OvrtxRObj(path, h)
        robj.material_shader = material_prim_path(plot) * "/Shader"
        return robj
    end
    values, interp = _scaled_to_display(plot, args[:scaled_color], n)
    proto_color     = interp == "constant" ? values  : nothing
    instancer_color = interp == "constant" ? nothing : values
    usda = _usda_pointinstancer(pos, scales, nothing, instancer_color,
                                _sphere_proto_body(proto_color); model = args[:model_f32c])
    h = OV.add_usd_reference!(screen.renderer, usda, path)
    return OvrtxRObj(path, h)
end

function author_usd_prim!(screen, scene, plot::Makie.MeshScatter, args)
    pos = args[:positions_transformed_f32c]
    n   = length(pos)
    n == 0 && return nothing
    marker = plot.marker[]
    marker_mesh    = marker isa GeometryBasics.Mesh ? marker : GeometryBasics.mesh(marker)
    marker_pts     = GeometryBasics.coordinates(marker_mesh)
    marker_normals = GeometryBasics.normals(marker_mesh)
    marker_faces   = [Int[Int(GeometryBasics.raw(i)) for i in f] for f in GeometryBasics.faces(marker_mesh)]
    scales       = _scales_for(plot.markersize[], n)
    orientations = _orientations_for(plot.rotation[], n)
    path         = plot_prim_path(screen.scene2scope, scene, plot)
    if is_materialized(plot)
        # M3.5: ovrtx does NOT honor a material binding on a PointInstancer, so a MATERIALIZED
        # meshscatter renders as ONE merged `UsdGeomMesh` of the marker copies (documented
        # fallback) — WITHOUT `displayColor`, BOUND to the pre-authored material.
        # `material_shader` wires the M3.4 live path.
        merged_pts, merged_faces, merged_normals = _merged_instances_mesh(marker_pts, marker_faces, marker_normals, pos, scales, orientations)
        usda = usda_mesh(merged_pts, merged_faces, merged_normals, nothing; model = args[:model_f32c], normal_interpolation = "vertex")
        h = OV.add_usd_reference!(screen.renderer, usda, path)
        OV.bind_material!(screen.renderer, path, material_prim_path(plot))
        robj = OvrtxRObj(path, h)
        robj.material_shader = material_prim_path(plot) * "/Shader"
        return robj
    end
    values, interp = _scaled_to_display(plot, args[:scaled_color], n)
    proto_color     = interp == "constant" ? values  : nothing
    instancer_color = interp == "constant" ? nothing : values
    proto = _mesh_proto_body(marker_pts, marker_faces, marker_normals, proto_color)
    usda  = _usda_pointinstancer(pos, scales, orientations, instancer_color, proto;
                                 model = args[:model_f32c])
    h = OV.add_usd_reference!(screen.renderer, usda, path)
    return OvrtxRObj(path, h)
end

function author_usd_prim!(screen, scene, plot::Makie.Lines, args)
    pts = args[:positions_transformed_f32c]
    n   = length(pts)
    n < 2 && return nothing
    width = _curve_width(pts, plot.linewidth[])
    path  = plot_prim_path(screen.scene2scope, scene, plot)
    if is_materialized(plot)
        # M3.5: bind the OmniPBR material and emit the curve WITHOUT `displayColor` (the
        # `nothing` sentinel).  `material_shader` wires the M3.4 live-edit path.
        usda = _usda_basiscurves(pts, [n], width, nothing, "constant"; model = args[:model_f32c])
        h = OV.add_usd_reference!(screen.renderer, usda, path)
        OV.bind_material!(screen.renderer, path, material_prim_path(plot))
        robj = OvrtxRObj(path, h)
        robj.material_shader = material_prim_path(plot) * "/Shader"
        return robj
    end
    values, interp = _scaled_to_display(plot, args[:scaled_color], n)
    usda  = _usda_basiscurves(pts, [n], width, values, interp; model = args[:model_f32c])
    h = OV.add_usd_reference!(screen.renderer, usda, path)
    return OvrtxRObj(path, h)
end

function author_usd_prim!(screen, scene, plot::Makie.LineSegments, args)
    pts  = args[:positions_transformed_f32c]
    nseg = length(pts) ÷ 2
    nseg < 1 && return nothing
    seg_pts = pts[1:2*nseg]
    width = _curve_width(pts, plot.linewidth[])
    path  = plot_prim_path(screen.scene2scope, scene, plot)
    if is_materialized(plot)
        usda = _usda_basiscurves(seg_pts, fill(2, nseg), width, nothing, "constant"; model = args[:model_f32c])
        h = OV.add_usd_reference!(screen.renderer, usda, path)
        OV.bind_material!(screen.renderer, path, material_prim_path(plot))
        robj = OvrtxRObj(path, h)
        robj.material_shader = material_prim_path(plot) * "/Shader"
        return robj
    end
    values, interp = _scaled_to_display(plot, args[:scaled_color], length(seg_pts))
    usda  = _usda_basiscurves(seg_pts, fill(2, nseg), width, values, interp; model = args[:model_f32c])
    h = OV.add_usd_reference!(screen.renderer, usda, path)
    return OvrtxRObj(path, h)
end

# Volumes M2: resolve a `Volume`'s transfer-function domain.  `colorrange` defaults to
# `Automatic()` → derive from the scalar-field extrema; an explicit `(lo,hi)` passes through.
# Only cosmetic under IndeX Direct (default grayscale TF — M1 constraint #3); authored for M2's
# composite-colour path (Task 3).
function _volume_colorrange(plot, scalars)
    cr = Makie.to_value(plot.colorrange)
    cr isa Makie.Automatic && return (Float64(minimum(scalars)), Float64(maximum(scalars)))
    return (Float64(first(cr)), Float64(last(cr)))
end

# Volumes M2 — `volume!(x,y,z,::Array{Float32,3})` build.  Reads the scalar field + axis ranges +
# colormap off the plot (scalars `plot[4]`, ranges `plot[1..3]`, `plot.colormap`/`.colorrange`),
# writes the dense array to a SCREEN-OWNED temp `.nvdb` (`NanoVDBWriter.save_nanovdb`), and authors
# it via M1's `_vdb_volume_usda` + `add_usd_reference!` on the OPEN stage (returns the `usd_handle`
# Task 5's `remove_usd!` needs — hence not `author_vdb_volume!`, which returns a String prim path).
#
# The writer maps `data[i,j,k]` → NanoVDB `Coord(i-1,j-1,k-1)` → world center `origin + (i-½)·dx`
# (voxel size `extent ./ size(data)`), i.e. i→x, j→y, k→z — matches Makie's `volume!(x,y,z,vol)`,
# so no i/j/k→Coord remap is needed (VERIFIED by the asymmetric-octant orientation test: low-octant
# mass renders below-centre, where the axes + camera place it).
#
# Grayscale only (Task 2): IndeX Direct renders the default density TF; the authored colormap is
# M1's reusable primitive for Task 3's composite path.  The temp `.nvdb` is recorded on
# `robj.meta[:vdb_tmp]`, removed by `destroy_bindings!` on close/delete (no leak).
function author_usd_prim!(screen, scene, plot::Makie.Volume, args)
    OV._index_enabled() || error(
        "author_usd_prim!(::Volume): volume rendering requires NVIDIA IndeX, which is not enabled.  " *
        "Set OMNIVERSEMAKIE_INDEX_LIBS (or OMNIVERSEMAKIE_OVRTX_CONFIG) BEFORE creating the Screen, " *
        "then re-create it.")
    scalars = Float32.(Makie.to_value(plot[4]))
    # KNOWN LIMITATION (grayscale): an all-zero field authors NOTHING (no render object), so a
    # later live FILL (`plot[4][] = nonzero`) fires `:volume` but the push finds no robj — an
    # empty→filled transition on the SAME live screen renders nothing.  Self-heals on a NEW screen
    # (re-authored from now-non-empty data); build-then-edit works.  Rebuild-on-fill deferred.
    all(iszero, scalars) && return nothing
    xr = Makie.to_value(plot[1]); yr = Makie.to_value(plot[2]); zr = Makie.to_value(plot[3])
    origin = GeometryBasics.Point3f(first(xr), first(yr), first(zr))
    extent = GeometryBasics.Vec3f(last(xr) - first(xr), last(yr) - first(yr), last(zr) - first(zr))
    tmp   = tempname() * ".nvdb"
    built = false
    # Symmetric with reload_volume_data!'s hardening: if save/add throws before the robj owns
    # `tmp`, GC the orphan here (nothing else records or removes it).
    try
        NanoVDBWriter.save_nanovdb(tmp, scalars, origin, extent)
        path = plot_prim_path(screen.scene2scope, scene, plot)
        usda = _vdb_volume_usda(tmp; prim_path = path, field = "density",
                                colormap = Makie.to_value(plot.colormap),
                                colorrange = _volume_colorrange(plot, scalars))
        h = OV.add_usd_reference!(screen.renderer, usda, path)
        robj = OvrtxRObj(path, h)
        robj.meta[:vdb_tmp]     = tmp                  # cleaned by destroy_bindings! on close/delete
        robj.meta[:volume_prim] = path                 # Tasks 4/5: the authored UsdVol prim
        built = true
        return robj
    finally
        built || rm(tmp; force = true)
    end
end

# ------------------------------------------------------------------
# push_to_ovrtx! — route ONE changed output to the right minimal C write
# ------------------------------------------------------------------

# Makie column-vector model matrix → USD row-vector 4×4 (Float64), translation in the last
# ROW — the form `OV.write_xform!` expects (matches `usda_matrix4d`'s transpose).
_model_to_usd_xform(m) = Float64.(collect(m'))

# Constant or per-vertex displayColor → a 3-lane color3f[] write on the referenced prim
# (spike-proven honored).  One element per colour, `shape = [ncolors]`.
function _push_displaycolor!(r, prim, plot, scaled_color)
    values, _ = _scaled_to_display(plot, scaled_color, 0)   # numeric `scaled_color` → plot colormap
    rgbs = values isa AbstractVector && !isempty(values) && first(values) isa Union{Tuple,AbstractVector} ?
        values : [values]
    flat = Vector{Float32}(undef, 3 * length(rgbs))
    @inbounds for (i, c) in enumerate(rgbs)
        flat[3i-2] = Float32(c[1]); flat[3i-1] = Float32(c[2]); flat[3i] = Float32(c[3])
    end
    dtype = LibOVRTX.DLDataType(UInt8(LibOVRTX.kDLFloat), UInt8(32), UInt16(3))
    OV._write_attribute!(r, prim, "primvars:displayColor", dtype, true,
                         LibOVRTX.OVRTX_SEMANTIC_NONE, flat, Int64[length(rgbs)])
    return nothing
end

# USD `visibility` token (`inherited`/`invisible`) — a TOKEN_STRING write needs a 128-bit
# element = one `ovx_string_t` (ptr+len); GC.@preserve BOTH the struct vector AND the backing
# String across the FFI call (spike-proven hides/reshows the prim).
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

# Topology change (rare): faceVertexIndices (0-based) + matching faceVertexCounts.
function _push_faces!(r, prim, faces)
    idx    = Int32[Int32(GeometryBasics.raw(i)) for f in faces for i in f]
    counts = Int32[Int32(length(f)) for f in faces]
    OV.write_array_attribute!(r, prim, "faceVertexIndices", idx)
    OV.write_array_attribute!(r, prim, "faceVertexCounts", counts)
    return nothing
end

# M2.4 hot path: flatten a resolved positions output into an owned Float32 buffer and write it
# through the persistent `point3f[]` array binding (one element per point; `shape = [npoints]`).
# Mirrors `write_array_attribute!`'s `reinterpret(Float32, …)` so lanes/shape match the bound type.
function _push_points_binding!(binding, pts::AbstractVector)
    src  = pts isa Vector ? pts : collect(pts)
    data = collect(reinterpret(Float32, src))
    OV.write_binding!(binding, data, Int64[length(src)])
    return nothing
end

# Diagnostic hook: called with the attribute name on every `push_to_ovrtx!` write.
# `nothing` (default) → no overhead.  test/m2_diffnode_test.jl asserts EXACTLY ONE write per edit.
const _PUSH_OBSERVER = Ref{Any}(nothing)

# Diagnostic hook (M3.4): called with the OmniPBR INPUT name on every live shader-input write.
# `nothing` (default) → no overhead.  test/m3_material_live_test.jl asserts exactly the changed
# inputs were written (one write per changed param).
const _SHADER_WRITE_OBSERVER = Ref{Any}(nothing)

# Write one OmniPBR shader input live + fire the shader-write observer (per-input granularity;
# a single `:material` push may re-write several inputs).
function _write_shader_input!(r, shader_prim::AbstractString, input_name::AbstractString, value)
    OV.write_shader_input!(r, shader_prim, input_name, value)
    ob = _SHADER_WRITE_OBSERVER[]
    ob === nothing || ob(input_name)
    return nothing
end

# Constant `(r,g,b)` base colour from a resolved `:scaled_color` for a MATERIALIZED plot: a
# single colour is used directly; a per-vertex colour collapses to its average (OmniPBR has no
# per-vertex diffuse base — the M3.2 fallback).
function _materialized_base_rgb(plot, scaled_color)
    values, interp = _scaled_to_display(plot, scaled_color, 0)   # numeric `scaled_color` → plot colormap
    if interp == "constant"
        return (Float32(values[1]), Float32(values[2]), Float32(values[3]))
    end
    @warn "OmniverseMakie: per-vertex `color` on a materialized plot — a live edit uses a \
           constant AVERAGE base colour (OmniPBR has no per-vertex diffuse base)."
    n = max(length(values), 1)
    return (Float32(sum(v[1] for v in values) / n),
            Float32(sum(v[2] for v in values) / n),
            Float32(sum(v[3] for v in values) / n))
end

# Route a changed `:material` to live shader-input writes on the plot's material KIND (OmniPBR
# via `_merge_material_input!`, OmniGlass via `_merge_glass_input!`) — `write_shader_input!` each
# SCALAR/color3f input.  Texture-asset live-swaps are OUT of M3.4 scope → `@warn`ed + skipped.
function _push_material!(r, shader_prim::AbstractString, kind::Symbol, material_attrs)
    inputs = Dict{String,Any}()
    for k in keys(material_attrs)
        if kind === :glass
            _merge_glass_input!(inputs, Symbol(k), Makie.to_value(material_attrs[k]))
        else
            _merge_material_input!(inputs, Symbol(k), Makie.to_value(material_attrs[k]), false, false)
        end
    end
    for (input_name, v) in inputs
        if v isa AbstractString
            @warn "OmniverseMakie: live texture-asset swap (`$(input_name)`) is not supported \
                   (M3.4 scope) — skipped."
        elseif v isa NTuple{3}
            _write_shader_input!(r, shader_prim, input_name,
                                 (Float32(v[1]), Float32(v[2]), Float32(v[3])))
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

Route ONE changed compute output to its minimal in-place USD write on `robj.prim_path`
(no re-author).  Returns `true` if a write was issued (an unrouted `name` → `false`):

- `:model_f32c`                 → `OV.write_xform!` (`omni:xform`, composed world transform)
- `:positions_transformed_f32c` → `points` array write
- `:normals`                    → `normals` array write
- `:faces`                      → `faceVertexIndices` + `faceVertexCounts`
- `:scaled_color`               → `primvars:displayColor`, OR (MATERIALIZED, M3.4) the OmniPBR
                                  `inputs:diffuse_color_constant` shader input
- `:material` (M3.4)            → each changed OmniPBR scalar/color3f shader input on a
                                  MATERIALIZED plot's `Mat_<id>/Shader` (no-op otherwise)
- `:volume` (Volumes M2, T5)    → live density edit: fresh temp `.nvdb` + RELOAD
                                  (`reload_volume_data!`: remove_usd! + add_usd_reference!)
- `:visible`                    → `visibility` token
"""
function push_to_ovrtx!(screen, robj::OvrtxRObj, plot, name::Symbol, value)
    r       = screen.renderer
    prim    = robj.prim_path
    binding = get(robj.bindings, name, nothing)   # M2.4: persistent hot-path binding (or nothing)
    routed  = true
    if name === :model_f32c
        # omni:xform — write zero-copy through the mapped binding when one exists (created once
        # by bind_hot_attributes!), else the M0 one-shot `write_attribute` path.
        if binding === nothing
            OV.write_xform!(r, prim, _model_to_usd_xform(value))
        else
            OV.write_mapped_xform!(binding, _model_to_usd_xform(value))
        end
    elseif name === :positions_transformed_f32c
        # points — persistent `bind_array_attribute` + write when bound, else one-shot.
        if binding === nothing
            OV.write_array_attribute!(r, prim, "points", value)
        else
            _push_points_binding!(binding, value)
        end
    elseif name === :normals
        OV.write_array_attribute!(r, prim, "normals", value)
    elseif name === :faces
        _push_faces!(r, prim, value)
    elseif name === :scaled_color
        if robj.material_shader === nothing
            _push_displaycolor!(r, prim, plot, value)            # USD-native displayColor
        else
            # M3.4: materialized → re-write the base colour in place (constant only) on the correct
            # material KIND (OmniGlass `glass_color` vs OmniPBR `diffuse_color_constant`).
            _write_shader_input!(r, robj.material_shader, _base_color_input(_material_kind(plot)),
                                 _materialized_base_rgb(plot, value))
        end
    elseif name === :material
        # M3.4: a live `plot.material[]` edit re-writes the pre-authored shader's inputs.
        if robj.material_shader === nothing
            # A plain plot gaining a material at runtime is a true material SWAP (needs a
            # pre-authored material → root re-author) — out of M3.4 scope.
            @warn "OmniverseMakie: live `material` edit on a non-materialized plot is not \
                   supported (a runtime material swap needs a root re-author) — skipped."
            routed = false
        else
            _push_material!(r, robj.material_shader, _material_kind(plot), value)
        end
    elseif name === :volume
        # Volumes M2 (Task 5): live density edit — fresh temp `.nvdb` + RELOAD (remove_usd! +
        # add_usd_reference!; a filePath write does NOT update).  On reload failure keep the last
        # good frame (warn once, don't thrash).
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
# bind_hot_attributes! — create the persistent hot-path bindings (M2.4)
# ------------------------------------------------------------------

# Plot types whose `:positions_transformed_f32c` route writes a real `points` array on a
# UsdGeomMesh/UsdGeomBasisCurves prim → a persistent array binding is valid.  Scatter/MeshScatter
# author a UsdGeomPointInstancer (per-instance `positions`, not `points`), which the push route
# doesn't yet write in place → NO array binding, only the xform binding (behaviour = M2.3).
_points_binding_attr(::Makie.Mesh)         = "points"
_points_binding_attr(::Makie.Lines)        = "points"
_points_binding_attr(::Makie.LineSegments) = "points"
_points_binding_attr(::Any)                = nothing

"""
    bind_hot_attributes!(screen, robj, plot, args) -> OvrtxRObj

Create the per-plot persistent attribute bindings ONCE (right after the USD reference is
authored), keyed in `robj.bindings` by the driving compute-output name so `push_to_ovrtx!`
routes a changed attribute through the binding instead of re-authoring:

- `:model_f32c` → an `omni:xform` binding (`OVRTX_BINDING_FLAG_OPTIMIZE`) written ZERO-COPY
  via `map_attribute` (every plot type — the primary hot binding).
- `:positions_transformed_f32c` → a `point3f[] points` array binding via `bind_array_attribute`
  + `write` (mesh/curve only — see `_points_binding_attr`).

Both target `robj.prim_path`; the M2.4 spike VALIDATED that map/write through a persistent
binding on a referenced prim honors the edit.  Released by `destroy_bindings!` on
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
    # Tier 2 (mesh / curve points): point3f[] array, bind + write.
    pname = _points_binding_attr(plot)
    if pname !== nothing
        robj.bindings[:positions_transformed_f32c] = OV.create_binding(
            r, prim, pname,
            LibOVRTX.DLDataType(UInt8(LibOVRTX.kDLFloat), UInt8(32), UInt16(3));
            array = true, semantic = LibOVRTX.OVRTX_SEMANTIC_NONE, optimize = false)
    end
    return robj
end

# Volumes M2: a `Volume`'s UsdVol prim carries NO `xformOp:transform`/`omni:xform` (placement lives
# in the `.nvdb`'s baked voxel→world Map, NOT a USD xform — and extra metadata on a volume layer can
# render it BLACK), and `consumed_inputs(::Volume)` tracks only `:visible` (a one-shot write, not a
# binding), so NO hot-path attribute to bind — skip the xform binding.  Tasks 4/5 can grow this.
bind_hot_attributes!(screen, robj::OvrtxRObj, ::Makie.Volume, args) = robj

"""
    destroy_bindings!(robj::OvrtxRObj) -> OvrtxRObj

Destroy + clear every persistent binding on `robj` (M2.4 lifetime).  Called by `close(Screen)`
and per-plot `delete!`.  Safe after the Renderer is closed (`OV.destroy!` is a no-op then).

Volumes M2: also removes the screen-owned temp `.nvdb` on `robj.meta[:vdb_tmp]` so it does not
leak; `force=true` makes this idempotent across every teardown site.
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
    return robj
end

# ------------------------------------------------------------------
# register_ovrtx_robj! — register the diff node, force first resolve (build)
# ------------------------------------------------------------------

"""
    register_ovrtx_robj!(screen, scene, plot) -> Union{OvrtxRObj,Nothing}

Register the plot's `:ovrtx_renderobject` diff node (once) and force a resolve, which BUILDS
the USD reference via `author_usd_prim!` on `screen`'s open stage and records the `OvrtxRObj`
in `screen.plot2robj`.  Later resolves (pulled by `colorbuffer`) push minimal writes.

The node consumes a `:ovrtx_screen` input carrying the OWNING screen.  A plot's compute graph
OUTLIVES a `Screen` (a `Figure` can be rendered by several transient screens), so pointing
`:ovrtx_screen` at a NEW screen marks the node dirty and REBUILDS the reference on that
screen's fresh stage instead of pushing diffs to a closed renderer; re-rendering the SAME
screen leaves it unchanged so only real edits drive writes.

A plot type with no tracked inputs (empty `consumed_inputs` — Surface/unknown) gets no node
and is built once per screen via the M1 `to_ovrtx_object` path.
"""
function register_ovrtx_robj!(screen, scene, plot)
    inputs = consumed_inputs(plot)
    if isempty(inputs)
        h = to_ovrtx_object(screen, scene, plot)
        h === nothing && return nothing
        path = plot_prim_path(screen.scene2scope, scene, plot)
        robj = OvrtxRObj(path, h)
        if is_materialized(plot)
            # M3.5: a materialized no-diff-node plot (Surface) gets its pre-authored OmniPBR
            # material BOUND here (`to_ovrtx_object` already emitted geometry WITHOUT
            # `displayColor`).  No diff node ⇒ STATIC material (no live edit) — fine for M3.5.
            OV.bind_material!(screen.renderer, path, material_prim_path(plot))
            robj.material_shader = material_prim_path(plot) * "/Shader"
        end
        robj.plot = plot                                    # M6.B: objectid→Plot for pick resolution
        screen.plot2robj[objectid(plot)] = robj
        screen.path2plot[robj.prim_path] = objectid(plot)   # M6.B: keep the reverse map in lockstep
        return robj
    end

    attr = plot.attributes
    # Per-screen build context: the diff node consumes this; pointing it at a new screen
    # marks it dirty → rebuild on that screen's stage (see docstring).
    if haskey(attr, :ovrtx_screen)
        setproperty!(attr, :ovrtx_screen, screen)
    else
        ComputePipeline.add_input!(attr, :ovrtx_screen, screen)
    end

    if !haskey(attr, :ovrtx_renderobject)
        node_inputs = Symbol[:ovrtx_screen; inputs...]
        ComputePipeline.register_computation!(attr, node_inputs, [:ovrtx_renderobject]) do args, changed, last
            scr = args[:ovrtx_screen]
            local robj
            if isnothing(last) || changed[:ovrtx_screen]
                # `scene` captured from the arg; on a rebuild for a NEW screen, scr.scene2scope
                # (same objectids) yields the same nested path — reference re-nests identically.
                robj = author_usd_prim!(scr, scene, plot, args)      # (RE)BUILD on the active screen
                # M2.4: create the persistent hot-path bindings ONCE on the fresh reference.
                robj === nothing || bind_hot_attributes!(scr, robj, plot, args)
            else
                robj = last.ovrtx_renderobject
                if robj !== nothing
                    for name in keys(args)
                        name === :ovrtx_screen && continue
                        changed[name] || continue                    # minimal-delta gate
                        push_to_ovrtx!(scr, robj, plot, name, args[name])
                    end
                end
            end
            scr === nothing || (scr.requires_update = true)
            return (robj,)
        end
    end

    built = attr[:ovrtx_renderobject][]                              # force resolve → (re)build on `screen`
    if built !== nothing
        built.plot = plot                                           # M6.B: objectid→Plot for pick resolution
        screen.plot2robj[objectid(plot)] = built
        screen.path2plot[built.prim_path] = objectid(plot)          # M6.B: reverse map in lockstep
    end
    return built
end

# ------------------------------------------------------------------
# pull_ovrtx_nodes! — colorbuffer per-frame node resolution
# ------------------------------------------------------------------

# Resolve one plot's `:ovrtx_renderobject` node (recursing composites).  A clean node is a
# no-op; a dirty one fires `push_to_ovrtx!` per changed input and flips `screen.requires_update`.
# A failed resolve is marked resolved so it doesn't re-throw every frame (the frame still renders).
function _pull_plot_node!(screen, plot)
    if isempty(plot.plots)
        attr = plot.attributes
        if haskey(attr, :ovrtx_renderobject)
            try
                attr[:ovrtx_renderobject][]
            catch e
                @debug "OmniverseMakie: :ovrtx_renderobject resolve failed" plot=typeof(plot) err=e
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
