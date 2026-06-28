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
function usda_omnipbr_material(name::AbstractString, inputs::AbstractDict)
    lines = String[]
    for (k, v) in inputs
        if v isa NTuple{3}                     # color3f
            push!(lines, "                color3f inputs:$k = ($(v[1]), $(v[2]), $(v[3]))")
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
                uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@
                uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"
$(join(lines, "\n"))
                token outputs:out
            }
        }
"""
end
