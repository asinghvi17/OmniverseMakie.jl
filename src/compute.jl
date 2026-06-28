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
"""
mutable struct OvrtxRObj
    prim_path::String
    usd_handle::UInt64
    bindings::Dict{Symbol,Any}
end

OvrtxRObj(prim_path::AbstractString, usd_handle::Integer) =
    OvrtxRObj(String(prim_path), UInt64(usd_handle), Dict{Symbol,Any}())

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
consumed_inputs(::Makie.Mesh)         = [:positions_transformed_f32c, :model_f32c, :faces, :normals, :scaled_color, :visible]
consumed_inputs(::Makie.Scatter)      = [:positions_transformed_f32c, :model_f32c, :scaled_color, :visible]
consumed_inputs(::Makie.MeshScatter)  = [:positions_transformed_f32c, :model_f32c, :scaled_color, :visible]
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
        return OvrtxRObj(path, h)
    end

    # Non-materialized: the M1 USD-native `displayColor` path, byte-unchanged.
    values, interp = _displaycolor_from_scaled(args[:scaled_color], length(points))
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
    values, interp = _displaycolor_from_scaled(args[:scaled_color], n)
    proto_color     = interp == "constant" ? values  : nothing
    instancer_color = interp == "constant" ? nothing : values
    usda = _usda_pointinstancer(pos, scales, nothing, instancer_color,
                                _sphere_proto_body(proto_color); model = args[:model_f32c])
    path = plot_prim_path(screen.scene2scope, scene, plot)
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
    values, interp = _displaycolor_from_scaled(args[:scaled_color], n)
    proto_color     = interp == "constant" ? values  : nothing
    instancer_color = interp == "constant" ? nothing : values
    proto = _mesh_proto_body(mpts, mfaces, mnrm, proto_color)
    usda  = _usda_pointinstancer(pos, scales, orientations, instancer_color, proto;
                                 model = args[:model_f32c])
    path = plot_prim_path(screen.scene2scope, scene, plot)
    h = OV.add_usd_reference!(screen.renderer, usda, path)
    return OvrtxRObj(path, h)
end

function author_usd_prim!(screen, scene, plot::Makie.Lines, args)
    pts = args[:positions_transformed_f32c]
    n   = length(pts)
    n < 2 && return nothing
    values, interp = _displaycolor_from_scaled(args[:scaled_color], n)
    width = _curve_width(pts, plot.linewidth[])
    usda  = _usda_basiscurves(pts, [n], width, values, interp; model = args[:model_f32c])
    path = plot_prim_path(screen.scene2scope, scene, plot)
    h = OV.add_usd_reference!(screen.renderer, usda, path)
    return OvrtxRObj(path, h)
end

function author_usd_prim!(screen, scene, plot::Makie.LineSegments, args)
    pts  = args[:positions_transformed_f32c]
    nseg = length(pts) ÷ 2
    nseg < 1 && return nothing
    pts2 = pts[1:2*nseg]
    values, interp = _displaycolor_from_scaled(args[:scaled_color], length(pts2))
    width = _curve_width(pts, plot.linewidth[])
    usda  = _usda_basiscurves(pts2, fill(2, nseg), width, values, interp; model = args[:model_f32c])
    path = plot_prim_path(screen.scene2scope, scene, plot)
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

"""
    push_to_ovrtx!(screen, robj, name::Symbol, value) -> Bool

Route ONE changed compute output to its minimal in-place USD write on the referenced
prim `robj.prim_path` (no re-author):

- `:model_f32c`                 → `OV.write_xform!` (`omni:xform`, composed world transform)
- `:positions_transformed_f32c` → `points` array write
- `:normals`                    → `normals` array write
- `:faces`                      → `faceVertexIndices` + `faceVertexCounts`
- `:scaled_color`               → `primvars:displayColor`
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
        _push_displaycolor!(r, prim, value)
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
        robj = OvrtxRObj(plot_prim_path(screen.scene2scope, scene, plot), h)
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
