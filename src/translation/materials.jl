# Materials translation for OmniverseMakie.
#
# Provides:
#   displaycolor_for(plot, npoints) — Makie plot.color → USD primvars:displayColor
#
# M1.5 emits USD-NATIVE displayColor (NOT an RPR/MaterialX material): either a
# single CONSTANT colour, or one colour PER VERTEX.  The `plot.material=` escape
# hatch (MaterialX / OmniPBR) is reserved for M4.
#
# NOTE: included inside the OmniverseMakie module, after usd.jl.
#       Makie, Colors, ColorTypes (red/green/blue) are all in scope.

"""
    displaycolor_for(plot, npoints::Int) -> (values, interpolation::String)

Translate a plot's `color` attribute into USD `primvars:displayColor` data.

Returns `(values, interpolation)`:
- `values` is either a single `(r, g, b)` `Float32` tuple (constant colour) or a
  `Vector` of `(r, g, b)` tuples (one per vertex).
- `interpolation` is `"constant"` or `"vertex"`.

Supported `plot.color[]` forms:
- scalar `Colorant` / `Symbol` / colour-like → one CONSTANT colour
  (e.g. `:red` → `((1,0,0), "constant")`).
- `AbstractVector{<:Colorant}` → per-VERTEX colours (`"vertex"`); the vector
  length should equal `npoints`.
- `AbstractVector{<:Real}` + `colormap`/`colorrange` → per-VERTEX colours mapped
  through the colormap (`"vertex"`).

`plot.material` (MaterialX / OmniPBR) is reserved for M4 and ignored here.
"""
function displaycolor_for(plot, npoints::Int)
    return _displaycolor(plot.color[], plot, npoints)
end

# --- per-vertex colours -------------------------------------------------------
function _displaycolor(colors::AbstractVector{<:Colorant}, plot, npoints::Int)
    if length(colors) != npoints
        @warn "OmniverseMakie: per-vertex colour count $(length(colors)) != vertex count \
               $(npoints); displayColor may be misaligned"
    end
    return ([_rgb(c) for c in colors], "vertex")
end

# --- per-vertex numeric values mapped through the colormap --------------------
# Mapped manually via to_colormap + interpolated_getindex; Makie's
# numbers_to_colors errors when called outside the plot's compute pipeline.
function _displaycolor(values::AbstractVector{<:Real}, plot, npoints::Int)
    cmap  = Makie.to_colormap(plot.colormap[])
    range = _colorrange(plot, values)
    return ([_rgb(Makie.interpolated_getindex(cmap, Float32(v), range)) for v in values], "vertex")
end

# --- scalar / fallback → one constant colour ----------------------------------
function _displaycolor(color, plot, npoints::Int)
    return (_rgb(Makie.to_color(color)), "constant")
end

# Resolve a numeric colorrange, falling back to data extrema when `automatic`.
function _colorrange(plot, values)
    cr = plot.colorrange[]
    if (cr isa Tuple || cr isa AbstractVector) && length(cr) == 2
        return (Float32(cr[1]), Float32(cr[2]))
    end
    lo, hi = extrema(values)
    return (Float32(lo), Float32(hi))
end

# (r, g, b) Float32 tuple from any Colorant (handles RGB, RGBA, HSV, … uniformly).
_rgb(c) = (Float32(red(c)), Float32(green(c)), Float32(blue(c)))

# ==================================================================
# M3.1 — OmniPBR material authoring
#
# An OmniPBR `UsdShade Material` is a `def Material` holding a single `def Shader`
# bound to the bundled `OmniPBR.mdl` MDL asset (`info:implementationSource =
# "sourceAsset"`, `info:mdl:sourceAsset = @OmniPBR.mdl@`, subIdentifier "OmniPBR").
# The bare `@OmniPBR.mdl@` resolves through ovrtx's MDL search path.  The USDA mirrors
# the proven radar example (`references/ovrtx/.../radar_example.usda:137-156`); USD is
# order-insensitive, so iterating `inputs` (a `Dict`) in non-deterministic order is fine.
# ==================================================================

"""
    material_prim_path(plot) -> String

The canonical USD prim path of `plot`'s OmniPBR material: `/World/Looks/Mat_<id>`,
where `<id>` is `objectid(plot)` (one material per atomic plot, no dedup — M3).  The
parent `/World/Looks` scope is authored into the root by `looks_scope_usda`
(`author_root_from_scene!`); the material is PRE-AUTHORED into that scope's body at
open-time (M3.1-validated — a runtime `add_usd_reference!` of a Material is not
bindable) and bound to its geometry prim at runtime with `OV.bind_material!`.
"""
material_prim_path(plot) = "/World/Looks/Mat_$(objectid(plot))"

"""
    usda_omnipbr_material(name::AbstractString, inputs::AbstractDict) -> String

Emit the USDA fragment for ONE OmniPBR `UsdShade Material` named `name` (e.g.
`Mat_<objectid>`), to be composed at `/World/Looks/<name>`.  `inputs` maps an OmniPBR
shader-input name (e.g. `"metallic_constant"`, `"diffuse_color_constant"`,
`"reflection_roughness_constant"`, `"diffuse_texture"`) to its value, typed by the
USD attribute kind emitted:

- `NTuple{3}` (`(r,g,b)` Float32) → `color3f inputs:<name> = (r, g, b)`
- `Bool` (e.g. `project_uvw`)     → `bool inputs:<name> = <0|1>`
- `AbstractString`               → `asset inputs:<name> = @<value>@` (a texture asset)
- otherwise (a real)             → `float inputs:<name> = <Float32(value)>`

The fragment is indented to nest as a child of `/World/Looks` (8-space `def Material`).
`outputs:mdl:surface.connect` targets `</World/Looks/<name>/Shader.outputs:out>` — the
absolute path the material composes to.

⚠️ M3.1 VALIDATED CONSTRAINT: this fragment must be composed INLINE into the
`/World/Looks` scope body at root-author time (`looks_scope_usda(materials)`), i.e.
PRE-AUTHORED into the stage passed to `open_usd_string!`.  Adding it to the OPEN stage
at runtime via `OV.add_usd_reference!` does NOT make the material bindable in our ovrtx
build (a silent no-op for `material:binding`).  `OV.bind_material!` then binds it at
runtime once the stage is open.
"""
usda_omnipbr_material(name::AbstractString, inputs::AbstractDict) =
    _usda_mdl_material(name, "OmniPBR.mdl", "OmniPBR", inputs)

"""
    usda_glass_material(name, inputs) -> String

Emit a TRUE-GLASS `UsdShade Material` backed by the bundled `OmniGlass.mdl` (subIdentifier
"OmniGlass") — refraction via `glass_ior`, tint via `glass_color`, optional
`frosting_roughness`/`thin_walled`.  Selected when a plot sets `material=(; glass = true, …)`
(M4 follow-up); composed into `/World/Looks` and bound exactly like the OmniPBR materials.
Unlike the OmniPBR `opacity` (a flat alpha cut-out), OmniGlass actually refracts.
"""
usda_glass_material(name::AbstractString, inputs::AbstractDict) =
    _usda_mdl_material(name, "OmniGlass.mdl", "OmniGlass", inputs)

# Author ONE MDL-backed `UsdShade Material` named `name`, sourced from `mdl_asset`
# (subIdentifier `subid`), with `inputs` typed by value (NTuple{3}→color3f, Bool→bool,
# AbstractString→asset, else→float).  Shared by the OmniPBR + OmniGlass authoring; the OmniPBR
# output is byte-identical to the pre-M4 `usda_omnipbr_material`.
function _usda_mdl_material(name::AbstractString, mdl_asset::AbstractString,
                           subid::AbstractString, inputs::AbstractDict)
    lines = String[]
    for (k, v) in inputs
        if v isa NTuple{3}                     # color3f
            push!(lines, "                color3f inputs:$k = ($(v[1]), $(v[2]), $(v[3]))")
        elseif v isa Bool                      # bool (enable_emission/_opacity, project_uvw, thin_walled)
            push!(lines, "                bool inputs:$k = $(v ? 1 : 0)")   # — must precede Real/float
        elseif v isa AbstractString            # asset (texture)
            push!(lines, "                asset inputs:$k = @$(v)@")
        else                                   # float
            push!(lines, "                float inputs:$k = $(Float32(v))")
        end
    end
    return """
        def Material "$(name)"
        {
            token outputs:mdl:surface.connect = </World/Looks/$(name)/Shader.outputs:out>
            def Shader "Shader"
            {
                uniform token info:implementationSource = "sourceAsset"
                uniform asset info:mdl:sourceAsset = @$(mdl_asset)@
                uniform token info:mdl:sourceAsset:subIdentifier = "$(subid)"
$(join(lines, "\n"))
                token outputs:out
            }
        }
"""
end

# ==================================================================
# M3.2 — the `material=` escape hatch + `color`→base composition
#
# A plot is MATERIALIZED (gets an OmniPBR material instead of a USD-native
# `displayColor`) when EITHER the user set the `material=` escape hatch OR `color`
# is an image (`Matrix{<:Colorant}` — a texture; texture mapping itself is M3.3).
# `material_inputs_from` composes BOTH into one `Dict` of OmniPBR shader-input names,
# applying precedence (`material=(; base_color=…)` overrides `color`).
# ==================================================================

"""
    _OMNIPBR_KEY_MAP :: Dict{Symbol,String}

The thin DOCUMENTED map from a user-facing `material=` escape-hatch key to its OmniPBR
shader-input name.  `base_color`, `emissive`, and `opacity` are handled specially in
`material_inputs_from` (they emit companion inputs, e.g. `enable_emission`); the entries
here give their direct rename.  `*_texture` keys map to OmniPBR texture inputs (their
VALUE — an asset path, or an image written to a temp PNG — is resolved by
`_texture_asset_for` in `_merge_material_input!`, M3.3).  An unknown key is `@warn`ed
and skipped.
"""
const _OMNIPBR_KEY_MAP = Dict{Symbol,String}(
    :metallic           => "metallic_constant",
    :roughness          => "reflection_roughness_constant",
    :opacity            => "opacity_constant",
    :base_color         => "diffuse_color_constant",
    :emissive           => "emissive_color",
    # texture inputs — M3.3 populates the VALUES (asset paths); mapping defined now.
    :base_color_texture => "diffuse_texture",
    :normal_texture     => "normalmap_texture",
    :roughness_texture  => "reflectionroughness_texture",
    :metallic_texture   => "metallic_texture",
)

# Default OmniPBR `emissive_intensity` authored alongside an `emissive=` material key
# (M3.5).  OmniPBR's MDL default is 40.0 (soft-range 0–1000); we author a higher constant
# so an emission-only material reads clearly emissive-coloured through auto-exposure
# (the M3.5 emissive-red Lines render test), rather than near-black.
const _EMISSIVE_INTENSITY = 5000.0f0

"""
    _enable_material_attribute!() -> Nothing

Register `:material` as a Makie GENERIC attribute (M3.5) so the `material=` escape hatch
is accepted on EVERY plot type — not just the recipes that document it natively
(`mesh`/`meshscatter`/`surface`).  Makie's `validate_attribute_keys` rejects an undocumented
keyword UNLESS it is in `Makie.attribute_name_allowlist()` (the universal-attribute list:
`:model`, `:transformation`, `:cycle`, …); without this, `lines!(…; material=…)` /
`scatter!(…; material=…)` throw `InvalidAttributeError` (`Lines`/`Scatter`/`LineSegments`
have no `material` attribute).  We append `:material` to that allowlist at `__init__`
(idempotently, capturing the existing list so we never drop Makie's own entries), making
materials a backend-universal attribute.  Plots that don't set it are unaffected
(`is_materialized` reads it through a `haskey` guard → `false` for a plain plot, keeping
the `displayColor` path byte-unchanged).
"""
function _enable_material_attribute!()
    base = Makie.attribute_name_allowlist()
    :material in base && return nothing
    newlist = (base..., :material)
    @eval Makie attribute_name_allowlist() = $newlist
    return nothing
end

# ==================================================================
# M3.3 — the temp-texture writer + `_texture_asset_for`
#
# An image `color` (a `Matrix{<:Colorant}`) is resolved to an on-disk asset PATH at
# OPEN-time (inside the pre-authoring walk), referenced by the OmniPBR `diffuse_texture`
# input.  The texture is written as a real PNG with `PNGFiles.save` — a small, focused
# libpng wrapper (the USER-APPROVED imaging dep, added via Pkg; it's what ImageIO uses for
# PNG under the hood).  ovrtx loads PNG natively (the bundled test data uses
# `@checkerboard.png@`), so this is the robust route.  The file lives in a STABLE session
# temp dir (not a deleted-on-return tempname) so it persists until ovrtx reads it during
# RT2.
# ==================================================================

# The stable per-process session dir holding written textures.  Created lazily on the
# first write and NOT auto-cleaned on return, so the asset persists until ovrtx reads it
# during the render (it is reaped by the OS / at process exit).
const _TEXTURE_DIR = Ref{String}("")
function _texture_dir()
    if isempty(_TEXTURE_DIR[]) || !isdir(_TEXTURE_DIR[])
        _TEXTURE_DIR[] = mktempdir(; prefix = "omniversemakie_tex_", cleanup = false)
    end
    return _TEXTURE_DIR[]
end

"""
    _texture_asset_for(img_or_path, plot) -> String

Resolve an image `color` (or an explicit `*_texture` value) to an on-disk asset PATH for
an OmniPBR texture input, at OPEN-time:

- an `AbstractString` is a path the user already supplied → returned AS-IS (no write).
- a `Matrix{<:Colorant}` (an in-memory image) is written to a stable session-temp PNG with
  `PNGFiles.save` (converted to `RGBA{N0f8}`; PNGFiles maps `img[row, col]` to the pixel
  at that row from the TOP / that column, so the matrix orientation is preserved) named by
  `objectid(plot)` (one material per plot, so the name is unique) → its ABSOLUTE path is
  returned (USD resolves absolute `@…@` asset paths directly; the root stage is an
  in-memory string with no anchor for a relative path).
"""
_texture_asset_for(path::AbstractString, plot) = String(path)
function _texture_asset_for(img::AbstractMatrix{<:Colorant}, plot)
    path = joinpath(_texture_dir(), "tex_$(objectid(plot)).png")
    PNGFiles.save(path, convert.(RGBA{N0f8}, img))
    return path
end

# Read a plot's raw `material` escape-hatch value SAFELY.  Most non-Mesh plots have a
# built-in `material` attribute defaulting to `nothing`; passing `material=(; …)` to
# `mesh!` round-trips as a Makie `Attributes` (NamedTuple) or a `Dict` (Makie 0.24.12 —
# MCP-verified).  Returns `nothing` when the plot has no `material` attribute or it is
# unset.  Guarded with `haskey` so a plot lacking the attribute never errors.
_plot_material(plot) = haskey(plot, :material) ? plot.material[] : nothing

# Read a plot's resolved `color` SAFELY (`nothing` if the plot has no `color`).
_plot_color(plot) = haskey(plot, :color) ? plot.color[] : nothing

# Whether a materialized plot's OmniPBR material samples a texture (so the mesh must
# author the `st` UV primvar — M3.3): an image `color`, OR any `*_texture` material key.
function _needs_texcoords(plot)
    _plot_color(plot) isa AbstractMatrix && return true
    mat = _plot_material(plot)
    mat === nothing && return false
    for k in keys(mat)
        endswith(String(k), "_texture") && return true
    end
    return false
end

"""
    is_materialized(plot) -> Bool

Whether `plot` should be rendered with an OmniPBR `UsdShade Material` (the M3 material
path) rather than a USD-native `primvars:displayColor`.  `true` when the `material=`
escape hatch is set (`plot.material[] !== nothing`) OR `color` is an image
(`plot.color[] isa AbstractMatrix` — a texture).  Reads both attributes through `haskey`
guards, so it returns `false` cleanly for an ordinary `mesh!(…; color=:red)` and never
errors on a plot lacking either attribute.
"""
function is_materialized(plot)
    _plot_material(plot) === nothing || return true
    return _plot_color(plot) isa AbstractMatrix
end

"""
    material_inputs_from(plot) -> Dict{String,Any}

Compose a materialized plot's `color` (base colour) and `material=` escape-hatch
parameters into ONE `Dict` keyed by OmniPBR shader-input name, ready for
`usda_omnipbr_material`.  Scalar values are kept in their RAW numeric type (the USDA
emitter `Float32`-converts at write time); colours are `(r,g,b)` `Float32` tuples.

Composition:
- `color` → a scalar `Colorant` becomes `"diffuse_color_constant"` (the base colour); a
  per-vertex colour collapses to a constant AVERAGE base (+ `@warn`, the spec's stretch
  fallback — OmniPBR has no per-vertex base); an IMAGE colour (`Matrix{<:Colorant}`)
  becomes a `"diffuse_texture"` asset (written at OPEN-time by `_texture_asset_for`) plus
  `"project_uvw" => false` so OmniPBR samples the mesh's `st` UV primvar (M3.3).
- `material=` keys are merged via `_OMNIPBR_KEY_MAP`; `*_texture` keys resolve their value
  to an asset path (`_texture_asset_for`); `emissive`/`opacity` also emit their `enable_*`
  companion; an unknown key is `@warn`ed and skipped.
- Precedence: `material=(; base_color=…)` OVERRIDES `color` (`@warn`); an explicit
  `base_color_texture` OVERRIDES an image `color` texture (`@warn`).
"""
function material_inputs_from(plot)
    inputs = Dict{String,Any}()

    color = _plot_color(plot)
    have_base    = false
    have_texture = false
    if color isa AbstractMatrix
        # IMAGE colour → an OmniPBR `diffuse_texture` (M3.3).  Resolve the asset PATH at
        # OPEN-time (write the temp PNG) and use the mesh's `st` UV primvar (NOT world-space
        # triplanar) — `project_uvw = false` — so the texture follows the geometry's UVs.
        inputs["diffuse_texture"] = _texture_asset_for(color, plot)
        inputs["project_uvw"]     = false
        have_texture = true
    elseif color isa AbstractVector
        # Per-vertex colour + a material → OmniPBR has no per-vertex base, so collapse to
        # a constant average (the spec's documented stretch fallback).
        @warn "OmniverseMakie: per-vertex `color` with `material=` → using a constant \
               average base colour for the OmniPBR material."
        inputs["diffuse_color_constant"] = _average_rgb(plot, color)
        have_base = true
    elseif color !== nothing
        inputs["diffuse_color_constant"] = _rgb(Makie.to_color(color))
        have_base = true
    end

    mat = _plot_material(plot)
    if mat !== nothing
        for k in keys(mat)
            _merge_material_input!(inputs, Symbol(k), Makie.to_value(mat[k]),
                                   have_base, have_texture)
        end
    end
    return inputs
end

# Merge ONE `material=` key/value into `inputs` (mapped to its OmniPBR input name).
function _merge_material_input!(inputs::AbstractDict, key::Symbol, value,
                                have_base::Bool, have_texture::Bool = false)
    if key === :base_color
        have_base && @warn "OmniverseMakie: `material=(; base_color=…)` overrides the \
                            plot `color` for this material."
        inputs["diffuse_color_constant"] = _rgb(Makie.to_color(value))
    elseif key === :base_color_texture
        # Explicit base-colour texture path OVERRIDES an image `color` (M3.3 precedence).
        have_texture && @warn "OmniverseMakie: `material=(; base_color_texture=…)` \
                              overrides the image `color` texture for this material."
        inputs["diffuse_texture"] = _texture_asset_for(value, nothing)
    elseif key in (:normal_texture, :roughness_texture, :metallic_texture)
        # Other `*_texture` maps — paths used AS-IS (no temp write; already asset paths).
        inputs[_OMNIPBR_KEY_MAP[key]] = _texture_asset_for(value, nothing)
    elseif key === :emissive
        inputs["emissive_color"]     = _rgb(Makie.to_color(value))
        # OmniPBR's `enable_emission` / `enable_opacity` are MDL `uniform bool` gates;
        # they MUST be authored as USD `bool` (a `float`/`int` against a bool MDL param
        # fails to bind → emission/opacity SILENTLY off).  `true` (a `Bool`) hits
        # `usda_omnipbr_material`'s `Bool` branch (ordered BEFORE the `Real` float branch,
        # since `Bool <: Real`) → `bool inputs:enable_emission = 1`.
        inputs["enable_emission"]    = true            # companion bool gate (OmniPBR)
        # `emissive_intensity` (OmniPBR float, default 40.0, soft-range 0–1000) scales the
        # emission; author it explicitly so emission is clearly VISIBLE (a curve/mesh with
        # only `emissive=` set reads emissive-coloured rather than near-black).
        inputs["emissive_intensity"] = _EMISSIVE_INTENSITY
    elseif key === :opacity
        inputs["opacity_constant"] = value
        inputs["enable_opacity"]   = true              # companion bool gate (OmniPBR)
    elseif haskey(_OMNIPBR_KEY_MAP, key)
        inputs[_OMNIPBR_KEY_MAP[key]] = value   # metallic, roughness, …
    else
        @warn "OmniverseMakie: unknown `material=` key `$(key)` — skipped (not an OmniPBR \
               escape-hatch input)."
    end
    return inputs
end

# Constant average `(r,g,b)` Float32 base colour from a per-vertex `color` (Colorant
# vector OR numeric vector mapped through the colormap) — reuses `_displaycolor`.
function _average_rgb(plot, color::AbstractVector)
    vals, _ = _displaycolor(color, plot, length(color))
    if vals isa AbstractVector && !isempty(vals)
        n = length(vals)
        return (Float32(sum(v[1] for v in vals) / n),
                Float32(sum(v[2] for v in vals) / n),
                Float32(sum(v[3] for v in vals) / n))
    end
    return vals  # `_displaycolor` already collapsed to a constant tuple
end

# M4 follow-up — material KIND.  A plot whose `material=(; glass = true, …)` is set wants TRUE
# refractive glass (OmniGlass) instead of the OmniPBR `opacity` alpha cut-out.
function _material_kind(plot)
    mat = _plot_material(plot)
    mat === nothing && return :omnipbr
    return (haskey(mat, :glass) && Makie.to_value(mat[:glass]) === true) ? :glass : :omnipbr
end

"""
    _glass_inputs_from(plot) -> Dict{String,Any}

Compose an OmniGlass input Dict for a `material=(; glass = true, …)` plot: `glass_color` ←
`color` / `material=(; base_color=…)` (default OmniGlass white), `glass_ior` ←
`material=(; ior=…)` (default 1.491), `frosting_roughness` ← `material=(; roughness=…)`,
`thin_walled` ← `material=(; thin_walled=…)`.  The `glass` flag itself and any unmapped key
are skipped (with a `@warn`).
"""
function _glass_inputs_from(plot)
    inputs = Dict{String,Any}()
    color  = _plot_color(plot)
    if color !== nothing && !(color isa AbstractArray)          # a scalar base tint (not an image/per-vertex)
        inputs["glass_color"] = _rgb(Makie.to_color(color))
    end
    mat = _plot_material(plot)
    if mat !== nothing
        for k in keys(mat)
            kk = Symbol(k)
            kk === :glass && continue
            v = Makie.to_value(mat[k])
            if kk === :base_color || kk === :color
                inputs["glass_color"] = _rgb(Makie.to_color(v))
            elseif kk === :ior
                inputs["glass_ior"] = v
            elseif kk === :roughness
                inputs["frosting_roughness"] = v
            elseif kk === :thin_walled
                inputs["thin_walled"] = v === true
            else
                @warn "OmniverseMakie: glass `material=` ignores key `$(kk)` (OmniGlass takes \
                       glass_color / ior / roughness / thin_walled)."
            end
        end
    end
    get!(inputs, "glass_ior", 1.491f0)
    return inputs
end

"""
    materialized_looks_usda(root_scene) -> String

Compose the `/World/Looks` scope body for `root_scene`: walk every ATOMIC plot in the
scene tree (mirroring `pull_ovrtx_nodes!`), and for each `is_materialized` plot emit its
`usda_omnipbr_material("Mat_<objectid(plot)>", material_inputs_from(plot))` fragment.
The joined fragments are wrapped by `looks_scope_usda`.  Called by
`author_root_from_scene!` so every materialized plot's OmniPBR material is PRE-AUTHORED
into the stage at open-time (M3.1-validated: a Material added to the OPEN stage is not
bindable); the per-plot build branch then binds at runtime with `OV.bind_material!`.
"""
function materialized_looks_usda(root_scene)
    frags = String[]
    _collect_materialized!(frags, root_scene)
    return looks_scope_usda(join(frags))
end

function _collect_materialized!(frags::Vector{String}, scene)
    for plot in scene.plots
        _collect_materialized_plot!(frags, plot)
    end
    foreach(child -> _collect_materialized!(frags, child), scene.children)
    return
end

function _collect_materialized_plot!(frags::Vector{String}, plot)
    if isempty(plot.plots)                       # atomic plot
        if is_materialized(plot)
            name = "Mat_$(objectid(plot))"
            frag = _material_kind(plot) === :glass ?
                usda_glass_material(name, _glass_inputs_from(plot)) :
                usda_omnipbr_material(name, material_inputs_from(plot))
            push!(frags, frag)
        end
    else                                         # composite → recurse into children
        foreach(p -> _collect_materialized_plot!(frags, p), plot.plots)
    end
    return
end
