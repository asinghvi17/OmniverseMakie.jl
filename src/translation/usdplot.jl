# usdplot recipe + bind_usd! — place an external USD file into a Makie scene
# as a first-class atomic plot rendered through ovrtx, and tie Julia
# `Observable`s to prims/attributes inside that file for live updates.
#
# Included after compute.jl + screen.jl: adds methods to `author_usd_prim!`,
# `consumed_inputs`, `bind_hot_attributes!`, and `_usdplot_model` (compute.jl),
# and reads `Screen.pending_usd_writes` / `OvrtxRObj` (screen is duck-typed).
#
# Four load-bearing composition rules this file encodes:
#  1. A reference pulls in the file's defaultPrim subtree only → bind paths
#     are relative to it (prefixed with the plot's prim path at write time).
#  2. Files with relative sub-assets compose from file (directory anchor).
#  3. Transforms route through `omni:xform` only; `xformOp:*` is refused.
#  4. Makie owns the asset's root transform; interior prims are untouched.

# ============================================================================
# The recipe
# ============================================================================

"""
    usdplot(path; bbox, up, kwargs...)
    usdplot!(ax_or_scene, path; bbox, up, kwargs...)

Place the external USD file at `path` (`.usda` or `.usdc`) into a Makie scene
as an atomic plot rendered through the OmniverseMakie (ovrtx) backend,
alongside ordinary Makie plots.

The file is NOT parsed: pass `bbox` (a `Rect3f`) for camera framing, and `up`
(`:z`, the default, or `:y` for a typical DCC export — `:y` folds a +90° X
rotation in so the asset stands upright). Ordinary
`translate!`/`scale!`/`rotate!` drive the asset's root transform, and
`p.visible[]` toggles it. Tie observables to prims/attributes inside the
file with [`bind_usd!`](@ref).

Renders offscreen (`save`/`colorbuffer`), in `interactive_display`, and
inside a `replace_scene!` panel. In a plain GLMakie window (no ovrtx
backend) it renders nothing.

```julia
p = usdplot!(ax, "assets/robot.usdc";
             bbox = Rect3f(Point3f(-1), Vec3f(2)), up = :y)
translate!(p, 0, 0, 1)
bind_usd!(p, "/Arm", Observable(Makie.rotationmatrix_z(0.4f0)))
bind_usd!(p, "/Arm/Geo.primvars:displayColor", Observable([RGBf(1,0,0)]))
```
"""
@recipe USDPlot (path,) begin
    "Axis-aligned bounds used only for camera framing (the file is not parsed)."
    bbox = Rect3f(Point3f(-1), Vec3f(2))
    """Source up-axis: `:z` (default) or `:y` (typical DCC export); `:y`
    folds a +90° X rotation in so the asset stands upright (Z-up scene)."""
    up = :z
    Makie.mixin_generic_plot_attributes()...
end

# No child plots → the backend's `isempty(plot.plots)` walker treats this as
# an atomic plot: `register_ovrtx_robj!` → `author_usd_prim!(::USDPlot)`. A
# path-only recipe never calls `register_positions_transformed_f32c!` (what
# normally emits `:model_f32c`), so register `:model_f32c` here — the diff
# node then resolves it like any other plot's and live `translate!` drives it.
function Makie.plot!(p::USDPlot)
    haskey(p.attributes, :model_f32c) || Makie.register_model_f32c!(p.attributes)
    return p
end

# Absolutize immediately: the cwd may change between construction and display.
Makie.convert_arguments(::Type{<:USDPlot}, path::AbstractString) = (abspath(String(path)),)

# The file is not parsed, so the user-supplied `bbox` frames the axis.
Makie.data_limits(p::USDPlot) = Rect3d(p.bbox[])

# The diff node tracks the composed world transform (→ `omni:xform` on the
# plot prim) and visibility; geometry/materials live in the referenced file
# and are never re-authored.
consumed_inputs(::USDPlot) = [:model_f32c, :visible]

# A USDPlot's transform is a one-shot `omni:xform` write (user-rate), so it
# takes no persistent hot-path binding — same choice as Volume.
bind_hot_attributes!(screen, robj::OvrtxRObj, ::USDPlot, args) = robj

# ============================================================================
# up-axis fold
# ============================================================================

# +90° about X: maps the asset's local +Y onto the scene's +Z, so a Y-up DCC
# export stands upright in OmniverseMakie's Z-up world.
const ROT_X_90 = Makie.rotationmatrix_x(Float32(π / 2))

# Up-axis fold (specializes the generic in compute.jl, applied both in
# push_to_ovrtx!'s `:model_f32c` branch and at author time so the initial and
# every live transform agree): `:y` post-multiplies ROT_X_90 (asset-local
# first; the user's transforms compose outside), `:z` passes through.
_usdplot_model(plot::USDPlot, model) = plot.up[] === :y ? model * ROT_X_90 : model

# ============================================================================
# bind_usd! — tie an Observable to a prim/attribute inside the referenced file
# ============================================================================

# One declared binding. `target` is the user-facing string (re-bind/unbind
# identity); `prim` is the subprim path relative to the file's defaultPrim,
# prefixed with the plot's prim path at write time; `attr` is the attribute
# name, or `nothing` for a prim (transform) binding; `obs` drives it.
struct USDBinding
    target::String
    prim::String
    attr::Union{Nothing,String}
    obs::Observable
end

# Declared bindings per plot (a plot may be bound before display and shown on
# several screens); a WeakKeyDict so a GC'd plot drops its bindings. Source
# of truth: each screen wires its own listeners (into that screen's
# `pending_usd_writes`) at author time.
const _USD_BINDINGS = WeakKeyDict{Makie.AbstractPlot,Vector{USDBinding}}()

# --- target parsing / validation --------------------------------------------

# Split a bind target at the first `.` (USD prim names cannot contain `.`):
# before = the prim path, after = the attribute name (none → a prim/transform
# binding). Paths are relative to the file's defaultPrim; must start with `/`.
function _parse_usd_target(target::AbstractString)
    t = String(target)
    startswith(t, "/") ||
        throw(ArgumentError("bind_usd! target must start with `/` (a path relative to the " *
                            "file's defaultPrim), got $(repr(t))."))
    dot  = findfirst('.', t)
    prim = dot === nothing ? t : t[1:prevind(t, dot)]
    attr = dot === nothing ? nothing : t[nextind(t, dot):end]
    _validate_usd_prim_path(prim, t)
    attr === nothing || _validate_usd_attr_name(attr, t)
    return (prim, attr)
end

# Validate "/A/B/C": at least one segment, each a legal USD identifier.
# `prim` starts with `/`, so `split` leaves a leading "" we drop; any other
# empty segment (a trailing/double `/`, or a `/.` from a bad target) is
# rejected so it cannot silently normalize to a valid-looking path.
function _validate_usd_prim_path(prim::AbstractString, target::AbstractString)
    segs = split(prim, '/'; keepempty = true)[2:end]
    (isempty(segs) || any(isempty, segs)) &&
        throw(ArgumentError("bind_usd! target $(repr(target)) has an empty prim-path segment " *
                            "(a leading/trailing/double `/`, or a `.` right after `/`); give a " *
                            "clean path like \"/Arm/Geo\"."))
    for s in segs
        _usd_identifier(String(s); what = "bind_usd! prim-path segment")
    end
    return prim
end

# Validate an attribute name, e.g. "primvars:displayColor": each
# `:`-separated component is a legal USD identifier. Refuse `xformOp:*` /
# `xformOpOrder` up front (baked at load, not live-writable; bind the prim
# itself with a matrix observable instead).
function _validate_usd_attr_name(attr::AbstractString, target::AbstractString)
    isempty(attr) &&
        throw(ArgumentError("bind_usd! target $(repr(target)) has an empty attribute name after " *
                            "the `.`."))
    (startswith(attr, "xformOp:") || attr == "xformOpOrder") &&
        throw(ArgumentError("bind_usd! cannot drive `$(attr)` (target $(repr(target))): USD " *
                            "`xformOp:*` transforms are baked by ovrtx at load and are not " *
                            "live-writable.  Bind the PRIM itself (e.g. \"/Arm\") with a 4×4 " *
                            "matrix Observable — it drives the prim's `omni:xform`."))
    for c in split(attr, ':'; keepempty = false)
        _usd_identifier(String(c); what = "bind_usd! attribute-name component")
    end
    return attr
end

# --- value → USD write ------------------------------------------------------

# A single 3-vector value (Vec3/Point3/NTuple{3}); a plain `Vector{<:Real}`
# is deliberately not one (that is the array-attribute case), removing the
# length-3 ambiguity.
_is_vec3(::Any) = false
_is_vec3(::NTuple{3,<:Real}) = true
_is_vec3(::GeometryBasics.Vec{3,<:Real}) = true
_is_vec3(::GeometryBasics.Point{3,<:Real}) = true

function _flat_rgb(cs::AbstractVector{<:Colorant})
    flat = Vector{Float32}(undef, 3 * length(cs))
    @inbounds for (i, col) in enumerate(cs)
        c = convert(RGBf, col)
        flat[3i-2] = red(c); flat[3i-1] = green(c); flat[3i] = blue(c)
    end
    return flat
end

function _flat_vec3(vs::AbstractVector)
    flat = Vector{Float32}(undef, 3 * length(vs))
    @inbounds for (i, v) in enumerate(vs)
        flat[3i-2] = Float32(v[1]); flat[3i-1] = Float32(v[2]); flat[3i] = Float32(v[3])
    end
    return flat
end

# Write one attribute binding value on the full composed prim path.
# `prim_mode` lets the wire-time probe use MUST_EXIST (fail-fast) while the
# per-frame flush uses EXISTING_ONLY. An `is_array` mismatch against the
# attribute's declared type surfaces as a clear ovrtx error under MUST_EXIST,
# so array-typed attrs like `primvars:displayColor` take a Vector value.
function _write_usd_attr!(r, prim::AbstractString, attr::AbstractString, value; prim_mode)
    F32   = UInt8(LibOVRTX.kDLFloat)
    color3 = LibOVRTX.DLDataType(F32, UInt8(32), UInt16(3))
    if value isa Real
        OV._write_attribute!(r, prim, attr, LibOVRTX.DLDataType(F32, UInt8(32), UInt16(1)), false,
            LibOVRTX.OVRTX_SEMANTIC_NONE, Float32[value], Int64[1]; prim_mode)
    elseif value isa Colorant
        c = convert(RGBf, value)
        OV._write_attribute!(r, prim, attr, color3, false, LibOVRTX.OVRTX_SEMANTIC_NONE,
            Float32[red(c), green(c), blue(c)], Int64[1]; prim_mode)
    elseif value isa AbstractVector{<:Colorant}
        OV._write_attribute!(r, prim, attr, color3, true, LibOVRTX.OVRTX_SEMANTIC_NONE,
            _flat_rgb(value), Int64[length(value)]; prim_mode)
    elseif _is_vec3(value)
        OV._write_attribute!(r, prim, attr, color3, false, LibOVRTX.OVRTX_SEMANTIC_NONE,
            Float32[value[1], value[2], value[3]], Int64[1]; prim_mode)
    elseif value isa AbstractVector && !isempty(value) && _is_vec3(first(value))
        OV._write_attribute!(r, prim, attr, color3, true, LibOVRTX.OVRTX_SEMANTIC_NONE,
            _flat_vec3(value), Int64[length(value)]; prim_mode)
    else
        throw(ArgumentError("bind_usd!: unsupported value type $(typeof(value)) for attribute " *
                            "`$(attr)`.  Supported: `Real`; a 3-vector (Vec3/Point3/NTuple{3}) " *
                            "or RGB `Colorant`; or a `Vector` of 3-vectors / Colorants (for an " *
                            "array attribute like `primvars:displayColor`)."))
    end
    return nothing
end

# A prim (transform) binding expects a 4×4 matrix (Makie column-vector
# convention).
function _to_mat4(value)
    (value isa AbstractMatrix && size(value) == (4, 4)) ||
        throw(ArgumentError("bind_usd! prim (transform) binding expects a 4×4 matrix Observable " *
                            "(Makie column-vector convention, e.g. Makie.translationmatrix / " *
                            "rotationmatrix_z), got $(typeof(value))."))
    return value
end

# Apply a binding's value directly on `full_prim` (initial state; fail-fast
# under MUST_EXIST). Prim binding → `omni:xform`; attribute → typed write.
function _apply_binding!(r, full_prim::AbstractString, binding::USDBinding, value; prim_mode)
    if binding.attr === nothing
        OV.write_xform!(r, full_prim, _model_to_usd_xform(_to_mat4(value)); prim_mode)
    else
        _write_usd_attr!(r, full_prim, binding.attr, value; prim_mode)
    end
    return nothing
end

# --- wiring / registry ------------------------------------------------------

# The live, open screen + robj for an already-authored USDPlot, or
# (nothing, nothing). `register_ovrtx_robj!` stores the owning screen on the
# plot's `:ovrtx_screen` compute input; the robj is in `plot2robj`. A plot's
# compute graph outlives a Screen, so guard with `isopen` — a closed screen
# counts as not displayed (bindings stash and wire at next authoring).
function _live_screen_robj(p::USDPlot)
    attr = p.attributes
    haskey(attr, :ovrtx_screen) || return (nothing, nothing)
    scr = attr[:ovrtx_screen][]
    (scr === nothing || !isopen(scr)) && return (nothing, nothing)
    robj = get(scr.plot2robj, objectid(p), nothing)
    return robj === nothing ? (nothing, nothing) : (scr, robj)
end

# Wire one binding to a live screen: apply the current value (`probe=true`
# uses MUST_EXIST so a bogus prim/attr throws, naming it), then attach a
# listener that enqueues coalesced writes into `screen.pending_usd_writes`
# (flushed in `_sync_and_needs_reset!`). Idempotent per target: a prior
# listener for the same target on this robj is detached first (re-bind
# replaces); the ObserverFunction lands in `robj.meta[:usd_binding_obsfuncs]`.
function _wire_binding!(screen, robj::OvrtxRObj, binding::USDBinding; probe::Bool)
    r         = screen.renderer
    full_prim = robj.prim_path * binding.prim
    key       = (full_prim, binding.attr)
    prim_mode = probe ? LibOVRTX.OVRTX_BINDING_PRIM_MODE_MUST_EXIST :
                        LibOVRTX.OVRTX_BINDING_PRIM_MODE_EXISTING_ONLY
    _apply_binding!(r, full_prim, binding, binding.obs[]; prim_mode)

    ofs = get!(() -> Dict{String,Any}(), robj.meta, :usd_binding_obsfuncs)::Dict{String,Any}
    old = get(ofs, binding.target, nothing)
    old === nothing || Makie.Observables.off(old)  # re-bind replaces
    of = Makie.Observables.on(binding.obs) do v
        screen.pending_usd_writes[key] = (binding, v)  # latest value wins
        screen.requires_update = true                  # signal a re-render
        return
    end
    ofs[binding.target] = of
    return nothing
end

# Apply + wire every registered binding for `plot` onto a freshly-authored
# robj (called from `author_usd_prim!`). A probe failure degrades to a loud
# `@warn` + skip so one bad binding can't abort the whole figure (a bind on
# an already-displayed plot fails fast instead).
function _wire_registered_bindings!(screen, robj::OvrtxRObj, plot::USDPlot)
    for binding in get(_USD_BINDINGS, plot, USDBinding[])
        try
            _wire_binding!(screen, robj, binding; probe = true)
        catch e
            @warn "OmniverseMakie: usdplot bind_usd!($(repr(binding.target))) failed to apply — \
                   skipped (the prim/attribute may not exist in the referenced file)." exception=e
        end
    end
    return robj
end

"""
    bind_usd!(p::USDPlot, target::AbstractString, obs) -> Observable

Tie `obs` (an `Observable`, or a plain value that is wrapped in one) to a
prim or attribute inside the USD file `p` references, so updating it
live-updates the render. `target` is a path relative to the file's
defaultPrim, split at the first `.`:

  - `"/Arm"` (no dot) → prim binding: the value is a 4×4 Makie-convention
    matrix → the prim's `omni:xform`. This REPLACES the prim's local
    transform, so to keep the prim's authored placement compose it in,
    e.g. `T · R(θ) · L`.
  - `"/Arm/Geo.primvars:displayColor"` → attribute binding: the value is
    written by type (`Real` → float; a 3-vector / RGB `Colorant` → color3f;
    a `Vector` of those → the array form used by array attributes like
    `primvars:displayColor`).

Validated up front (target starts with `/`, each prim segment / attribute
component is a legal USD identifier, `xformOp:*` refused). If `p` is already
displayed the binding is applied immediately and a bogus prim/attribute
throws `OVRTXError` naming it (fail-fast); otherwise it is stashed and
applied at display time (a failure there warns + skips). Re-binding the same
target replaces it. Returns the (possibly newly wrapped) `Observable`.
"""
function bind_usd!(p::USDPlot, target::AbstractString, obs)
    observable = obs isa Observable ? obs : Observable(obs)
    prim, attr = _parse_usd_target(target)  # throws on a bad target
    binding    = USDBinding(String(target), prim, attr, observable)
    # Wire before registering: on a displayed plot `_wire_binding!` probes
    # with MUST_EXIST and throws on a bogus prim/attribute; registering only
    # after success leaves the registry and any existing same-target binding
    # untouched on failure.
    scr, robj = _live_screen_robj(p)
    scr === nothing || _wire_binding!(scr, robj, binding; probe = true)
    bindings = get!(() -> USDBinding[], _USD_BINDINGS, p)
    filter!(b -> b.target != binding.target, bindings)  # re-bind replaces
    push!(bindings, binding)
    return observable
end

"""
    unbind_usd!(p::USDPlot, target::AbstractString) -> p

Remove the binding for `target`: detach its listener on any live screen
(leaving the last-written value in place) and drop it from the registry.
A no-op if `target` is not bound.
"""
function unbind_usd!(p::USDPlot, target::AbstractString)
    t = String(target)
    bindings = get(_USD_BINDINGS, p, nothing)
    if bindings !== nothing
        filter!(b -> b.target != t, bindings)
        isempty(bindings) && delete!(_USD_BINDINGS, p)
    end
    _scr, robj = _live_screen_robj(p)
    if robj !== nothing
        ofs = get(robj.meta, :usd_binding_obsfuncs, nothing)
        if ofs isa AbstractDict && haskey(ofs, t)
            Makie.Observables.off(ofs[t]); delete!(ofs, t)
        end
    end
    return p
end

# ============================================================================
# authoring + per-frame flush
# ============================================================================

"""
    author_usd_prim!(screen, scene, plot::USDPlot, args) -> OvrtxRObj

Compose the external USD file into `screen`'s open stage as a reference under
the plot's prim path, write the initial `omni:xform` (Makie owns the asset's
root transform; `up = :y` folds a +90° X rotation in), and apply+wire any
bindings registered by pre-display `bind_usd!` calls. Rides the normal robj
lifecycle (model/visibility diff, teardown, accumulate-mode structural
resets).
"""
function author_usd_prim!(screen, scene, plot::USDPlot, args)
    up = plot.up[]
    up in (:y, :z) || throw(ArgumentError("usdplot `up` must be :y or :z, got $(repr(up))."))
    path = plot[1][]  # abspath (convert_arguments)
    isfile(path) || throw(ArgumentError("usdplot: USD file not found: $(path)"))
    prim = plot_prim_path(screen.scene2scope, scene, plot)
    h    = OV.add_usd_reference_from_file!(screen.renderer, path, prim)
    robj = OvrtxRObj(prim, h)
    # Makie owns the root transform: write the initial omni:xform (with the
    # up-fold) so the asset picks up translate!/scale!/rotate! immediately.
    # Live changes ride the :model_f32c diff, which applies the same fold.
    OV.write_xform!(screen.renderer, prim,
                    _model_to_usd_xform(_usdplot_model(plot, args[:model_f32c])))
    _wire_registered_bindings!(screen, robj, plot)
    return robj
end

# Flush the per-screen coalesced usdplot binding writes (one EXISTING_ONLY
# write per target, latest value wins). Called from `_sync_and_needs_reset!`
# before the accumulate gate; returns whether any write was issued so the
# caller ORs it into `need_reset` — default mode reconverges, accumulate mode
# ignores it (bound writes are non-structural; RT2 reprojection absorbs them).
function _flush_pending_usd_writes!(screen)
    pending = screen.pending_usd_writes
    isempty(pending) && return false
    r = screen.renderer
    for ((full_prim, _attr), (binding, v)) in pending
        try
            _apply_binding!(r, full_prim, binding, v;
                            prim_mode = LibOVRTX.OVRTX_BINDING_PRIM_MODE_EXISTING_ONLY)
        catch e
            @warn "OmniverseMakie: usdplot bound write to $(repr(binding.target)) failed — \
                   skipped." exception=e maxlog=1
        end
    end
    empty!(pending)
    return true
end
