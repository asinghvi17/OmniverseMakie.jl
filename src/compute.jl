# compute.jl — M2 render-object handles + ComputePipeline diff nodes.
#
# Holds the per-plot render-object record `OvrtxRObj` (stored in `Screen.plot2robj`)
# and the M2.2 diff driver: `consumed_inputs`, `author_usd_prim!`, `push_to_ovrtx!`,
# `register_ovrtx_robj!`, `pull_ovrtx_nodes!` — the `:ovrtx_renderobject` node that,
# on an already-open stage, pushes one minimal C write per changed plot attribute
# instead of re-authoring.  M2.4 fills `bindings` for the hot path.
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
- `bindings`   — persistent attribute bindings keyed by the driving compute-output
                 name; empty until M2.4's `bind_hot_attributes!` fills it (the hot
                 path), destroyed by `destroy_bindings!`.
- `material_shader` — for a MATERIALIZED plot (M3), the PRE-AUTHORED OmniPBR shader prim
                 path (`/World/Looks/Mat_<id>/Shader`); `nothing` for a non-materialized
                 plot.  Set in `author_usd_prim!`'s materialized branch and read by
                 `push_to_ovrtx!` to route a live `color`/`material` edit to a
                 `write_shader_input!` on the open stage (M3.4) instead of `displayColor`.
"""
mutable struct OvrtxRObj
    prim_path::String
    usd_handle::UInt64
    bindings::Dict{Symbol,Any}
    material_shader::Union{String,Nothing}
end

OvrtxRObj(prim_path::AbstractString, usd_handle::Integer) =
    OvrtxRObj(String(prim_path), UInt64(usd_handle), Dict{Symbol,Any}(), nothing)

# ==================================================================
# M2.2 — the :ovrtx_renderobject diff node + push_to_ovrtx! (diff driver)
#
# Each atomic plot gets a ComputePipeline node (`:ovrtx_renderobject`) whose inputs
# are the owning screen (`:ovrtx_screen`) plus the plot's RESOLVED compute outputs
# (positions/model/color/visibility/…).  Its callback:
#   - first resolve OR a new owning screen (`:ovrtx_screen` changed): BUILDS the USD
#     reference (`author_usd_prim!`, the M1 emitters fed from `args`) on that screen's
#     stage and returns the `OvrtxRObj`.
#   - later resolves on the SAME screen: for each CHANGED compute output, pushes ONE
#     minimal C write to the open stage (`push_to_ovrtx!`) — no re-author.
# `colorbuffer` pulls every node each frame (`pull_ovrtx_nodes!`); a clean graph is a
# no-op, any change flips `screen.requires_update` so the frame does one `OV.reset!`.
#
# Referenced-prim writes (write_xform!/array/displayColor/visibility on /World/plot_<id>)
# are spike-proven honored on the open stage, so the diff path writes in place (no
# remove_usd!/re-reference fallback needed).
# ==================================================================

# Single source of truth for a plot's USD prim path (was reconstructed in 6 emitters
# + insert! before M2.2).  `author_usd_prim!`/`register_ovrtx_robj!` own it.
#
# M2.3: scope-aware.  The plot nests under its owning scene's `def Scope`, looked up
# in the screen's `scene2scope` map (authored into the root by `author_root_from_scene!`,
# before any reference is added).  `scene2scope[objectid(scene)]` is `/World` for the
# root scene and `/World/Scene_<id>/…` for subscenes — derived from stable `objectid`s,
# so it is identical across screens (a rebuild on a new screen recomputes the same
# path).  A scene not yet in the map (e.g. a subscene added live after authoring) falls
# back to `/World`, so the plot still renders flat.
plot_prim_path(scene2scope::AbstractDict, scene, plot) =
    string(get(scene2scope, objectid(scene), "/World"), "/plot_", objectid(plot))

# ------------------------------------------------------------------
# consumed_inputs — per-type Makie compute outputs the diff node tracks
# ------------------------------------------------------------------

# Only outputs that BOTH resolve by default (verified) AND have a `push_to_ovrtx!`
# route are listed, so every tracked change maps to exactly one minimal write.
# Scatter/Lines size/rotation/linewidth diffing is deferred to the M2.3 hot path
# (those have no clean in-place USD route yet) — their build still reads them from
# the plot.  Surface (no `*_f32c`/`faces` outputs) + unknown types get NO node
# (`Symbol[]`) and build once via the M1 `to_ovrtx_object` path.
# M3.4: `:material` is tracked so a live `plot.material[]` edit fires the diff node and
# re-writes the pre-authored OmniPBR shader inputs (push route below).  It resolves as a
# graph output for every Mesh AND MeshScatter (a plain plot's value is `nothing` / an
# empty material and never changes, so it never pushes — no regression to the displayColor
# path).  `consumed_inputs` dispatches on TYPE only and `register_ovrtx_robj!` registers
# the node for EVERY plot of that type (materialized or not), so a tracked input MUST
# resolve for a NON-materialized plot too — else `register_computation!` raises "Inputs
# [:material] not found" and breaks ALL plots of that type.
# M3 final-review (MCP-verified): `:material` resolves for a non-materialized Mesh /
# MeshScatter (→ `nothing`) but NOT for a non-materialized Scatter / Lines / LineSegments
# (Makie only registers `:material` on those when a `material=` kwarg is given).  So
# `:material` is tracked for Mesh + MeshScatter only; a live material-PARAM edit on a
# MATERIALIZED Scatter/Lines/LineSegments is therefore still a no-op (documented
# limitation — see test/m3_material_live_test.jl).  A live `color` edit on those types
# still works (their materialized build sets `robj.material_shader` and `:scaled_color`
# is tracked).
consumed_inputs(::Makie.Mesh)         = [:positions_transformed_f32c, :model_f32c, :faces, :normals, :scaled_color, :material, :visible]
consumed_inputs(::Makie.Scatter)      = [:positions_transformed_f32c, :model_f32c, :scaled_color, :visible]
consumed_inputs(::Makie.MeshScatter)  = [:positions_transformed_f32c, :model_f32c, :scaled_color, :material, :visible]
consumed_inputs(::Makie.Lines)        = [:positions_transformed_f32c, :model_f32c, :scaled_color, :visible]
consumed_inputs(::Makie.LineSegments) = [:positions_transformed_f32c, :model_f32c, :scaled_color, :visible]
consumed_inputs(::Any)                = Symbol[]

# ------------------------------------------------------------------
# displayColor from the resolved :scaled_color output
# ------------------------------------------------------------------

# `(values, interpolation)` for `usda_*`, from a resolved `:scaled_color`:
#   single Colorant → constant; vector/array of Colorants → per-vertex.
_displaycolor_from_scaled(c::Colorant, _n) = (_rgb(c), "constant")
_displaycolor_from_scaled(cs::AbstractVector{<:Colorant}, _n) = ([_rgb(c) for c in cs], "vertex")
_displaycolor_from_scaled(cs::AbstractArray{<:Colorant}, _n)  = ([_rgb(c) for c in vec(cs)], "vertex")
# Fallback (e.g. a single packed value): treat as one constant colour.
_displaycolor_from_scaled(c, _n) = (_rgb(Makie.to_color(c)), "constant")

# M4 follow-up: a NUMERIC `scaled_color` vector (colour = numbers + a colormap, as on a
# colour-mapped `meshscatter!`/`lines!`/`linesegments!`/per-vertex `mesh!`) must be mapped
# THROUGH the plot's colormap — the bare `_displaycolor_from_scaled` fallback
# `_rgb(to_color(::Vector{Float32}))` cannot (it `red()`s the whole vector → MethodError).
# Resolve it here from the plot's `colormap` + `colorrange` (mirroring the M1
# `displaycolor_for`/`_displaycolor` numeric path via `interpolated_getindex`); a Colorant or
# scalar `scaled_color` defers to the byte-unchanged `_displaycolor_from_scaled`.
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

# Per-vertex UVs for an image-/texture-materialized mesh, read DIRECTLY off the plot's
# `:texturecoordinates` compute output (`Vector{Vec2f}` — Makie's `decompose_uv`); NOT a
# tracked `consumed_inputs` diff output (it has no push route).  Returns `nothing` (→
# `usda_mesh` OMITS `st`) when the material samples no texture, the plot exposes no
# texcoords, or their count != the vertex count (so only a clean per-vertex `st` is
# authored; a mismatch is skipped rather than mis-authored).
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
(`args`), returning the recording `OvrtxRObj` (or `nothing` for an empty plot).

Geometry points come from `:positions_transformed_f32c` (model-LOCAL) and the
transform from `:model_f32c` (the COMPOSED world transform — closing the M1
scene-transform gap), so world = `model_f32c · positions`.  Colour comes from
`:scaled_color`.  The emitters are M1's (`usda_mesh` / `_usda_pointinstancer` /
`_usda_basiscurves`), reused verbatim but fed from `args` instead of the plot.

`scene` (the plot's owning Makie scene) is threaded through so the reference is added
at the NESTED scope path `plot_prim_path(screen.scene2scope, scene, plot)`
(`/World/Scene_<id>/plot_<id>` for a subscene) — M2.3 subscene grouping.
"""
function author_usd_prim!(screen, scene, plot::Makie.Mesh, args)
    points  = args[:positions_transformed_f32c]
    isempty(points) && return nothing
    normals = args[:normals]
    faces0  = [Int[Int(GeometryBasics.raw(i)) for i in f] for f in args[:faces]]
    path    = plot_prim_path(screen.scene2scope, scene, plot)

    if is_materialized(plot)
        # M3.2: emit the geometry WITHOUT `displayColor` and BIND the OmniPBR material
        # PRE-AUTHORED at open-time (`materialized_looks_usda` →
        # `material_prim_path(plot)`).  The material was composed into /World/Looks before
        # the stage opened, so this runtime `bind_material!` takes (M3.1-validated).
        # M3.3: a textured material (image `color` / `*_texture`) samples the mesh's `st`
        # UV primvar, so author it from Makie's per-vertex `:texturecoordinates` (read
        # DIRECTLY off the plot — it is NOT a tracked `consumed_inputs` diff output).
        texcoords = _texcoords_for(plot, length(points))
        usda = usda_mesh(points, faces0, normals, nothing;
                         model                = args[:model_f32c],
                         normal_interpolation = "vertex",
                         texcoords            = texcoords)
        h = OV.add_usd_reference!(screen.renderer, usda, path)
        OV.bind_material!(screen.renderer, path, material_prim_path(plot))
        robj = OvrtxRObj(path, h)
        # M3.4: record the pre-authored OmniPBR shader prim so a live `color`/`material`
        # edit re-writes its inputs in place (push_to_ovrtx! material routing).
        robj.material_shader = material_prim_path(plot) * "/Shader"
        return robj
    end

    # Non-materialized: the M1 USD-native `displayColor` path, byte-unchanged.
    values, interp = _scaled_to_display(plot, args[:scaled_color], length(points))
    usda = usda_mesh(points, faces0, normals, values;
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
        # M3.5: ovrtx does NOT honor a material binding on a PointInstancer, so a
        # MATERIALIZED scatter renders as ONE merged `UsdGeomMesh` of tessellated unit-sphere
        # markers (the documented fallback) — WITHOUT `displayColor`, BOUND to the
        # pre-authored material like any Mesh.  `material_shader` wires the M3.4 live path.
        sm  = GeometryBasics.normal_mesh(GeometryBasics.Tesselation(GeometryBasics.Sphere(GeometryBasics.Point3f(0), 1f0), 16))
        smp = GeometryBasics.coordinates(sm)
        smn = GeometryBasics.normals(sm)
        smf = [Int[Int(GeometryBasics.raw(i)) for i in f] for f in GeometryBasics.faces(sm)]
        mP, mF, mN = _merged_instances_mesh(smp, smf, smn, pos, scales, nothing)
        usda = usda_mesh(mP, mF, mN, nothing; model = args[:model_f32c], normal_interpolation = "vertex")
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
    gm     = marker isa GeometryBasics.Mesh ? marker : GeometryBasics.mesh(marker)
    mpts   = GeometryBasics.coordinates(gm)
    mnrm   = GeometryBasics.normals(gm)
    mfaces = [Int[Int(GeometryBasics.raw(i)) for i in f] for f in GeometryBasics.faces(gm)]
    scales       = _scales_for(plot.markersize[], n)
    orientations = _orientations_for(plot.rotation[], n)
    path         = plot_prim_path(screen.scene2scope, scene, plot)
    if is_materialized(plot)
        # M3.5: ovrtx does NOT honor an OmniPBR material binding on a PointInstancer, so a
        # MATERIALIZED meshscatter is rendered as ONE merged `UsdGeomMesh` of the marker
        # copies (the documented fallback) — emitted WITHOUT `displayColor` and BOUND to the
        # PRE-AUTHORED material like any Mesh.  `material_shader` wires the M3.4 live path.
        mP, mF, mN = _merged_instances_mesh(mpts, mfaces, mnrm, pos, scales, orientations)
        usda = usda_mesh(mP, mF, mN, nothing; model = args[:model_f32c], normal_interpolation = "vertex")
        h = OV.add_usd_reference!(screen.renderer, usda, path)
        OV.bind_material!(screen.renderer, path, material_prim_path(plot))
        robj = OvrtxRObj(path, h)
        robj.material_shader = material_prim_path(plot) * "/Shader"
        return robj
    end
    values, interp = _scaled_to_display(plot, args[:scaled_color], n)
    proto_color     = interp == "constant" ? values  : nothing
    instancer_color = interp == "constant" ? nothing : values
    proto = _mesh_proto_body(mpts, mfaces, mnrm, proto_color)
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
        # M3.5: bind the OmniPBR material (base colour + emissive/opacity) to the curve and
        # emit it WITHOUT `displayColor` (the `nothing` sentinel).  `material_shader` wires
        # the M3.4 live-edit path.
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
    pts2  = pts[1:2*nseg]
    width = _curve_width(pts, plot.linewidth[])
    path  = plot_prim_path(screen.scene2scope, scene, plot)
    if is_materialized(plot)
        usda = _usda_basiscurves(pts2, fill(2, nseg), width, nothing, "constant"; model = args[:model_f32c])
        h = OV.add_usd_reference!(screen.renderer, usda, path)
        OV.bind_material!(screen.renderer, path, material_prim_path(plot))
        robj = OvrtxRObj(path, h)
        robj.material_shader = material_prim_path(plot) * "/Shader"
        return robj
    end
    values, interp = _scaled_to_display(plot, args[:scaled_color], length(pts2))
    usda  = _usda_basiscurves(pts2, fill(2, nseg), width, values, interp; model = args[:model_f32c])
    h = OV.add_usd_reference!(screen.renderer, usda, path)
    return OvrtxRObj(path, h)
end

# ------------------------------------------------------------------
# push_to_ovrtx! — route ONE changed output to the right minimal C write
# ------------------------------------------------------------------

# Makie column-vector model matrix → USD row-vector 4×4 (Float64), translation in the
# last ROW — the form `OV.write_xform!` expects (matches `usda_matrix4d`'s transpose).
_model_to_usd_xform(m) = Float64.(collect(m'))

# Constant or per-vertex displayColor → a 3-lane color3f[] write on the referenced prim
# (spike-proven honored).  One element per colour, `shape = [ncolors]`.
function _push_displaycolor!(r, prim, scaled_color)
    values, _ = _displaycolor_from_scaled(scaled_color, 0)
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

# USD `visibility` token (`inherited`/`invisible`) — a TOKEN_STRING write needs a
# 128-bit element = one `ovx_string_t` (ptr+len); preserve BOTH the struct vector and
# the backing String across the FFI call (spike-proven hides/reshows the prim).
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

# M2.4 hot path: flatten a resolved positions output (`Vector{Point3f}`) into an owned
# Float32 buffer and write it through the persistent `point3f[]` array binding (one
# tensor element per point; `shape = [npoints]`).  Mirrors `write_array_attribute!`'s
# structural `reinterpret(Float32, …)` so the lanes/shape match the bound element type.
function _push_points_binding!(binding, pts::AbstractVector)
    src  = pts isa Vector ? pts : collect(pts)
    data = collect(reinterpret(Float32, src))
    OV.write_binding!(binding, data, Int64[length(src)])
    return nothing
end

# Diagnostic hook: called with the attribute name on every `push_to_ovrtx!` write.
# `nothing` (default) → no overhead.  `test/m2_diffnode_test.jl` installs a counter
# to assert EXACTLY ONE minimal write fires per attribute edit.
const _PUSH_OBSERVER = Ref{Any}(nothing)

# Diagnostic hook (M3.4): called with the OmniPBR INPUT name on every live shader-input
# write (`write_shader_input!`).  `nothing` (default) → no overhead.
# `test/m3_material_live_test.jl` installs a recorder to assert EXACTLY the changed
# inputs were written (one write per changed param) on a `color`/`material` edit.
const _SHADER_WRITE_OBSERVER = Ref{Any}(nothing)

# Write one OmniPBR shader input live + fire the shader-write observer.  Mirrors how
# `push_to_ovrtx!` fires `_PUSH_OBSERVER`, but at the granularity of individual shader
# inputs (a single `:material` push may re-write several inputs).
function _write_shader_input!(r, shader_prim::AbstractString, input_name::AbstractString, value)
    OV.write_shader_input!(r, shader_prim, input_name, value)
    ob = _SHADER_WRITE_OBSERVER[]
    ob === nothing || ob(input_name)
    return nothing
end

# Constant `(r,g,b)` Float32 base colour from a resolved `:scaled_color` for a
# MATERIALIZED plot: a single colour is used directly; a per-vertex colour collapses to
# its average (OmniPBR has no per-vertex diffuse base — the M3.2 stretch fallback).
function _materialized_base_rgb(scaled_color)
    values, interp = _displaycolor_from_scaled(scaled_color, 0)
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

# Route a changed `:material` (the resolved `Attributes`) to live OmniPBR shader-input
# writes: map each material key to its OmniPBR input name (reusing the M3.2
# `_merge_material_input!` mapping), then `write_shader_input!` each SCALAR / color3f
# input.  Texture-asset live-swaps (`diffuse_texture`, …) are OUT of M3.4 scope (only
# scalar/color3f writes are proven here) — a texture-bearing change is `@warn`ed + skipped.
function _push_material!(r, shader_prim::AbstractString, material_attrs)
    inputs = Dict{String,Any}()
    for k in keys(material_attrs)
        _merge_material_input!(inputs, Symbol(k), Makie.to_value(material_attrs[k]), false, false)
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
    push_to_ovrtx!(screen, robj, name::Symbol, value) -> Bool

Route ONE changed compute output to its minimal in-place USD write on the referenced
prim `robj.prim_path` (no re-author):

- `:model_f32c`                 → `OV.write_xform!` (`omni:xform`, composed world transform)
- `:positions_transformed_f32c` → `points` array write
- `:normals`                    → `normals` array write
- `:faces`                      → `faceVertexIndices` + `faceVertexCounts`
- `:scaled_color`               → `primvars:displayColor`, OR (MATERIALIZED plot, M3.4)
                                  the OmniPBR `inputs:diffuse_color_constant` shader input
- `:material` (M3.4)            → each changed OmniPBR scalar/color3f shader input on a
                                  MATERIALIZED plot's `Mat_<id>/Shader` (no-op otherwise)
- `:visible`                    → `visibility` token

Returns `true` if a write was issued (an unrouted `name` is a no-op → `false`).
"""
function push_to_ovrtx!(screen, robj::OvrtxRObj, name::Symbol, value)
    r       = screen.renderer
    prim    = robj.prim_path
    binding = get(robj.bindings, name, nothing)   # M2.4: persistent hot-path binding (or nothing)
    routed  = true
    if name === :model_f32c
        # omni:xform — when a persistent binding exists (created once by
        # bind_hot_attributes!), write zero-copy through the MAPPED binding; otherwise
        # the M0 one-shot `write_attribute` path (correct, non-zero-copy fallback).
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
            _push_displaycolor!(r, prim, value)                  # USD-native displayColor (unchanged)
        else
            # M3.4: materialized → re-write the OmniPBR base colour in place (constant only).
            _write_shader_input!(r, robj.material_shader, "diffuse_color_constant",
                                 _materialized_base_rgb(value))
        end
    elseif name === :material
        # M3.4: a live `plot.material[]` edit re-writes the pre-authored shader's inputs.
        if robj.material_shader === nothing
            # A plain plot gaining a material at runtime is a true material SWAP (needs a
            # pre-authored material → a root re-author) — out of M3.4 scope.
            @warn "OmniverseMakie: live `material` edit on a non-materialized plot is not \
                   supported (a runtime material swap needs a root re-author) — skipped."
            routed = false
        else
            _push_material!(r, robj.material_shader, value)
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

# Plot types whose `:positions_transformed_f32c` route writes a real `points` array on
# a UsdGeomMesh / UsdGeomBasisCurves prim, so a persistent array binding is valid.
# Scatter/MeshScatter author a UsdGeomPointInstancer (per-instance `positions`, not
# `points`), which M2.2's push route does not yet write in place, so they get NO array
# binding — only the universal xform binding — leaving their positions on the existing
# path (behaviour identical to M2.3).
_points_binding_attr(::Makie.Mesh)         = "points"
_points_binding_attr(::Makie.Lines)        = "points"
_points_binding_attr(::Makie.LineSegments) = "points"
_points_binding_attr(::Any)                = nothing

"""
    bind_hot_attributes!(screen, robj, plot, args) -> OvrtxRObj

Create the per-plot persistent attribute bindings ONCE (right after the plot's USD
reference is authored) and store them in `robj.bindings`, keyed by the compute-output
name that drives them so `push_to_ovrtx!` routes a changed attribute through the
binding instead of re-authoring:

- `:model_f32c` → an `omni:xform` binding (`OVRTX_BINDING_FLAG_OPTIMIZE`) written
  ZERO-COPY via `map_attribute` (every plot type — the primary hot binding).
- `:positions_transformed_f32c` → a `point3f[] points` array binding written via
  `bind_array_attribute` + `write` (mesh / curve only — see `_points_binding_attr`).

Both target the referenced prim `robj.prim_path`; the M2.4 binding spike VALIDATED
that map/write through a persistent binding on a referenced prim honors the edit
(xform map round-trip byte-exact; points write-through-handle scaled then reverted
exactly).  Released by `destroy_bindings!` on `close(Screen)` / per-plot `delete!`.
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

"""
    destroy_bindings!(robj::OvrtxRObj) -> OvrtxRObj

Destroy + clear every persistent attribute binding on `robj` (M2.4 lifetime).  Called
by `close(Screen)` and (M2.5) per-plot `delete!`.  Safe to call after the Renderer is
closed (`OV.destroy!` is a no-op then).
"""
function destroy_bindings!(robj::OvrtxRObj)
    for b in values(robj.bindings)
        b isa OV.Binding && OV.destroy!(b)
    end
    empty!(robj.bindings)
    return robj
end

# ------------------------------------------------------------------
# register_ovrtx_robj! — register the diff node, force first resolve (build)
# ------------------------------------------------------------------

"""
    register_ovrtx_robj!(screen, scene, plot) -> Union{OvrtxRObj,Nothing}

Register the plot's `:ovrtx_renderobject` diff node (once on the plot's compute
graph) and force a resolve, which BUILDS the USD reference via `author_usd_prim!`
on `screen`'s open stage and records the `OvrtxRObj` in `screen.plot2robj`.
Subsequent resolves (pulled by `colorbuffer`) push minimal writes for changed inputs.

The node consumes a `:ovrtx_screen` input carrying the OWNING screen.  The plot's
compute graph OUTLIVES a `Screen` (a `Figure` can be rendered by several transient
screens — `save`, `record`, then interactive), so when a NEW screen takes over,
setting `:ovrtx_screen` to that screen marks it dirty (different object) and the node
REBUILDS the reference on the new screen's fresh stage instead of pushing diffs to a
closed renderer.  Re-rendering the SAME screen leaves `:ovrtx_screen` unchanged, so
only real attribute edits drive writes.

A plot type with no tracked inputs (`consumed_inputs` empty — Surface / unknown) gets
no node and is built once per screen via the M1 `to_ovrtx_object` path.
"""
function register_ovrtx_robj!(screen, scene, plot)
    inputs = consumed_inputs(plot)
    if isempty(inputs)
        h = to_ovrtx_object(screen, scene, plot)
        h === nothing && return nothing
        path = plot_prim_path(screen.scene2scope, scene, plot)
        robj = OvrtxRObj(path, h)
        if is_materialized(plot)
            # M3.5: a materialized no-diff-node plot (Surface) gets its PRE-AUTHORED OmniPBR
            # material BOUND here (its `to_ovrtx_object` already emitted geometry WITHOUT
            # `displayColor`).  No diff node ⇒ STATIC material (no live edit) — fine for M3.5.
            OV.bind_material!(screen.renderer, path, material_prim_path(plot))
            robj.material_shader = material_prim_path(plot) * "/Shader"
        end
        screen.plot2robj[objectid(plot)] = robj
        return robj
    end

    attr = plot.attributes
    # Per-screen build context: the diff node consumes this; pointing it at a new
    # screen marks it dirty → rebuild on that screen's stage (see docstring).
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
                # `scene` is captured from register_ovrtx_robj!'s arg; on a rebuild for
                # a NEW screen, scr.scene2scope (same objectids) yields the same nested
                # path — so the reference re-nests identically on the fresh stage.
                robj = author_usd_prim!(scr, scene, plot, args)      # (RE)BUILD on the active screen
                # M2.4: create the persistent hot-path bindings ONCE on the fresh
                # reference (a NEW screen rebinds against its own renderer's stage).
                robj === nothing || bind_hot_attributes!(scr, robj, plot, args)
            else
                robj = last.ovrtx_renderobject
                if robj !== nothing
                    for name in keys(args)
                        name === :ovrtx_screen && continue
                        changed[name] || continue                    # minimal-delta gate
                        push_to_ovrtx!(scr, robj, name, args[name])
                    end
                end
            end
            scr === nothing || (scr.requires_update = true)
            return (robj,)
        end
    end

    built = attr[:ovrtx_renderobject][]                              # force resolve → (re)build on `screen`
    built === nothing || (screen.plot2robj[objectid(plot)] = built)
    return built
end

# ------------------------------------------------------------------
# pull_ovrtx_nodes! — colorbuffer per-frame node resolution
# ------------------------------------------------------------------

# Resolve one plot's `:ovrtx_renderobject` node (recursing composites).  A clean node
# is a no-op; a dirty one fires `push_to_ovrtx!` for each changed input and flips
# `screen.requires_update`.  A failed resolve is marked resolved so it doesn't re-throw
# every frame (the frame still renders).
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

Resolve every plot's `:ovrtx_renderobject` diff node (scene + child scenes) before a
render.  No-op for a static graph; any geometry change flips `screen.requires_update`
(via the node callback) so `colorbuffer` issues one `OV.reset!`.
"""
function pull_ovrtx_nodes!(screen, scene)
    for plot in scene.plots
        _pull_plot_node!(screen, plot)
    end
    foreach(child -> pull_ovrtx_nodes!(screen, child), scene.children)
    return
end
