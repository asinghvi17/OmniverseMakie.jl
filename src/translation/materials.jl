# Materials translation: Makie plot.color → USD-native primvars:displayColor
# (constant or per-vertex; NOT a MaterialX/RPR material), plus OmniPBR/OmniGlass
# MDL material authoring (the `material=` escape hatch).
# Included in the OmniverseMakie module after usd.jl; Makie, Colors, ColorTypes
# (red/green/blue) are in scope.

"""
    displaycolor_for(plot, npoints::Int) -> (values, interpolation::String)

Translate `plot.color` into USD `primvars:displayColor`. Returns `(values,
interpolation)`:
- scalar `Colorant`/`Symbol` → one `(r,g,b)` Float32 tuple, `"constant"`
  (e.g. `:red` → `((1,0,0), "constant")`).
- `AbstractVector{<:Colorant}` → one tuple per vertex, `"vertex"` (length should
  equal `npoints`).
- `AbstractVector{<:Real}` + `colormap`/`colorrange` → per-vertex via the
  colormap, `"vertex"`.

Does NOT handle `plot.material` (the OmniPBR material path); authored separately.
"""
function displaycolor_for(plot, npoints::Int)
    return _displaycolor(plot.color[], plot, npoints)
end

# Per-vertex colours.
function _displaycolor(colors::AbstractVector{<:Colorant}, plot, npoints::Int)
    if length(colors) != npoints
        @warn "OmniverseMakie: per-vertex colour count $(length(colors)) != vertex count \
               $(npoints); displayColor may be misaligned"
    end
    return ([_rgb(c) for c in colors], "vertex")
end

# Per-vertex numeric values → colours via the colormap. Mapped manually
# (to_colormap + interpolated_getindex) because Makie's numbers_to_colors errors
# when called outside the plot's compute pipeline.
function _displaycolor(values::AbstractVector{<:Real}, plot, npoints::Int)
    cmap   = Makie.to_colormap(plot.colormap[])
    crange = _colorrange(plot, values)
    return ([_rgb(Makie.interpolated_getindex(cmap, Float32(v), crange)) for v in values], "vertex")
end

# Scalar / fallback → one constant colour.
function _displaycolor(color, plot, npoints::Int)
    return (_rgb(Makie.to_color(color)), "constant")
end

# Resolve a numeric colorrange, falling back to data extrema when `automatic`.
function _colorrange(plot, values)
    crange = plot.colorrange[]
    if (crange isa Tuple || crange isa AbstractVector) && length(crange) == 2
        return (Float32(crange[1]), Float32(crange[2]))
    end
    lo, hi = extrema(values)
    return (Float32(lo), Float32(hi))
end

# (r,g,b) Float32 tuple from any Colorant (RGB, RGBA, HSV, … all handled).
_rgb(c) = (Float32(red(c)), Float32(green(c)), Float32(blue(c)))

# === M3.1 — OmniPBR material authoring ===
# An OmniPBR material is a `def Material` + one `def Shader` bound to the
# bundled OmniPBR.mdl (implementationSource "sourceAsset", info:mdl:sourceAsset
# @OmniPBR.mdl@, subIdentifier "OmniPBR"); the bare @OmniPBR.mdl@ resolves via
# ovrtx's MDL search path. USD is order-insensitive, so iterating `inputs`
# (a Dict) in non-deterministic order is fine. Mirrors radar_example.usda.

"""
    material_prim_path(plot) -> String

Canonical prim path of `plot`'s material: `/World/Looks/Mat_<objectid(plot)>`
(one material per atomic plot, no dedup). The material must be PRE-AUTHORED into
the `/World/Looks` scope at stage open-time (a runtime `add_usd_reference!` of a
Material is not bindable); `OV.bind_material!` binds it to the geometry at runtime.
"""
material_prim_path(plot) = "/World/Looks/Mat_$(objectid(plot))"

"""
    usda_omnipbr_material(name::AbstractString, inputs::AbstractDict) -> String

USDA fragment for ONE OmniPBR `UsdShade Material` named `name`, composed at
`/World/Looks/<name>` (8-space indent to nest under `/World/Looks`). `inputs`
maps an OmniPBR shader-input name to a value, typed by the USD kind emitted:
`NTuple{3}`→`color3f (r,g,b)`, `Bool`→`bool 0|1`, `AbstractString`→`asset @…@`,
else→`float Float32(value)`. `outputs:mdl:surface.connect` targets the material's
absolute `Shader.outputs:out`.

CONSTRAINT (M3.1-validated): must be composed INLINE at root-author time, i.e.
PRE-AUTHORED into the stage passed to `open_usd_string!`. Adding it to the OPEN
stage via `OV.add_usd_reference!` does NOT make the material bindable in our ovrtx
build (silent no-op for `material:binding`); `OV.bind_material!` binds once open.
"""
usda_omnipbr_material(name::AbstractString, inputs::AbstractDict) =
    _usda_mdl_material(name, "OmniPBR.mdl", "OmniPBR", inputs)

"""
    usda_glass_material(name, inputs) -> String

USDA fragment for a `UsdShade Material` backed by the bundled OmniGlass.mdl
(subIdentifier "OmniGlass"): refraction via `glass_ior`, tint via `glass_color`,
optional `frosting_roughness`/`thin_walled`. Selected by `material=(; glass=true)`;
composed/bound like the OmniPBR materials. Unlike OmniPBR `opacity` (a flat alpha
cut-out), OmniGlass actually refracts.
"""
usda_glass_material(name::AbstractString, inputs::AbstractDict) =
    _usda_mdl_material(name, "OmniGlass.mdl", "OmniGlass", inputs)

# Author ONE MDL-backed `UsdShade Material` `name` from `mdl_asset`
# (subIdentifier `subid`); `inputs` typed by value: NTuple{3}→color3f,
# Bool→bool, AbstractString→asset, else→float. Shared by OmniPBR + OmniGlass
# authoring; OmniPBR output is byte-identical to pre-M4 usda_omnipbr_material.
function _usda_mdl_material(name::AbstractString, mdl_asset::AbstractString,
                           subid::AbstractString, inputs::AbstractDict)
    lines = String[]
    for (input_name, input_value) in inputs
        if input_value isa NTuple{3}                     # color3f
            push!(lines, "                color3f inputs:$(input_name) = ($(input_value[1]), $(input_value[2]), $(input_value[3]))")
        elseif input_value isa Bool                      # bool; MUST precede float (Bool <: Real)
            push!(lines, "                bool inputs:$(input_name) = $(input_value ? 1 : 0)")
        elseif input_value isa AbstractString            # asset (texture)
            push!(lines, "                asset inputs:$(input_name) = @$(input_value)@")
        else                                             # float
            push!(lines, "                float inputs:$(input_name) = $(Float32(input_value))")
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

# === M3.2 — `material=` escape hatch + `color`→base composition ===
# A plot is MATERIALIZED (OmniPBR material instead of USD-native displayColor)
# when `material=` is set OR `color` is an image (Matrix{<:Colorant} texture).
# material_inputs_from composes both into one Dict of OmniPBR input names;
# `material=(; base_color=…)` overrides `color`.

"""
    _OMNIPBR_KEY_MAP :: Dict{Symbol,String}

Map from a user-facing `material=` key to its OmniPBR shader-input name.
`base_color`/`emissive`/`opacity` are handled specially in `material_inputs_from`
(they also emit companion inputs, e.g. `enable_emission`); `*_texture` values are
resolved to asset paths by `_texture_asset_for`. Unknown keys are warned/skipped.
"""
const _OMNIPBR_KEY_MAP = Dict{Symbol,String}(
    :metallic           => "metallic_constant",
    :roughness          => "reflection_roughness_constant",
    :opacity            => "opacity_constant",
    :base_color         => "diffuse_color_constant",
    :emissive           => "emissive_color",
    # *_texture → OmniPBR texture input names (values resolved to asset paths).
    :base_color_texture => "diffuse_texture",
    :normal_texture     => "normalmap_texture",
    :roughness_texture  => "reflectionroughness_texture",
    :metallic_texture   => "metallic_texture",
)

# Default OmniPBR `emissive_intensity` authored with an `emissive=` key.
# OmniPBR's MDL default is 40.0 (soft-range 0–1000); we author higher so an
# emission-only material reads clearly emissive under auto-exposure, not black.
const _EMISSIVE_INTENSITY = 5000.0f0

"""
    _enable_material_attribute!() -> Nothing

Append `:material` to `Makie.attribute_name_allowlist()` so the `material=` escape
hatch is accepted on EVERY plot type (else `validate_attribute_keys` throws
`InvalidAttributeError` on plots like `Lines`/`Scatter` that lack it natively).
Idempotent, and captures the existing list so Makie's own entries are preserved.
Called at `__init__`. Plots that don't set `material=` are unaffected.
"""
function _enable_material_attribute!()
    base = Makie.attribute_name_allowlist()
    :material in base && return nothing
    newlist = (base..., :material)
    @eval Makie attribute_name_allowlist() = $newlist
    return nothing
end

# === M3.3 — temp-texture writer + `_texture_asset_for` ===
# An image `color` (Matrix{<:Colorant}) is written to a PNG at open-time
# (via PNGFiles.save; ovrtx loads PNG natively) and referenced by the OmniPBR
# `diffuse_texture` input. The file lives in a STABLE session temp dir (not a
# deleted-on-return tempname) so it survives until ovrtx reads it during render.

# Stable per-process temp dir for textures; created lazily, cleanup=false
# so assets persist until ovrtx reads them (reaped by the OS at process exit).
const _TEXTURE_DIR = Ref{String}("")
function _texture_dir()
    if isempty(_TEXTURE_DIR[]) || !isdir(_TEXTURE_DIR[])
        _TEXTURE_DIR[] = mktempdir(; prefix = "omniversemakie_tex_", cleanup = false)
    end
    return _TEXTURE_DIR[]
end

"""
    _texture_asset_for(img_or_path, plot) -> String

Resolve an image `color` or `*_texture` value to an on-disk asset PATH for an
OmniPBR texture input, at open-time:
- `AbstractString` → returned AS-IS (an existing path, no write).
- `Matrix{<:Colorant}` → written to a stable session-temp PNG (`PNGFiles.save`,
  converted to `RGBA{N0f8}`), named by `objectid(plot)`; its ABSOLUTE path is
  returned (USD needs it absolute — the in-memory root stage has no anchor for a
  relative `@…@` path).
"""
_texture_asset_for(path::AbstractString, plot) = String(path)
function _texture_asset_for(img::AbstractMatrix{<:Colorant}, plot)
    path = joinpath(_texture_dir(), "tex_$(objectid(plot)).png")
    PNGFiles.save(path, convert.(RGBA{N0f8}, img))
    return path
end

# Read a plot's raw `material` escape-hatch value safely (haskey-guarded, so a
# plot lacking the attribute never errors). Returns `nothing` when absent/unset.
# `material=(; …)` round-trips as a Makie Attributes/NamedTuple or a Dict.
_plot_material(plot) = haskey(plot, :material) ? plot.material[] : nothing

# Read a plot's resolved `color` SAFELY (`nothing` if the plot has no `color`).
_plot_color(plot) = haskey(plot, :color) ? plot.color[] : nothing

# True when a materialized plot's material samples a texture (image `color`, or
# any `*_texture` key), so the mesh must author the `st` UV primvar.
function _needs_texcoords(plot)
    _plot_color(plot) isa AbstractMatrix && return true
    mat = _plot_material(plot)
    mat === nothing && return false
    for key in keys(mat)
        endswith(String(key), "_texture") && return true
    end
    return false
end

"""
    is_materialized(plot) -> Bool

`true` when `plot` should render with an OmniPBR material rather than USD-native
`displayColor`: `material=` is set OR `color` is an image (`AbstractMatrix`).
`haskey`-guarded, so returns `false` cleanly for a plain `mesh!(…; color=:red)`.
"""
function is_materialized(plot)
    _plot_material(plot) === nothing || return true
    return _plot_color(plot) isa AbstractMatrix
end

"""
    material_inputs_from(plot) -> Dict{String,Any}

Compose a materialized plot's `color` and `material=` params into ONE Dict keyed by
OmniPBR shader-input name, for `usda_omnipbr_material`. Scalars are kept RAW (the
emitter Float32-converts); colours are `(r,g,b)` Float32 tuples.

- `color`: scalar → `diffuse_color_constant`; per-vertex → constant AVERAGE base
  (+`@warn`; OmniPBR has no per-vertex base); image `Matrix{<:Colorant}` →
  `diffuse_texture` asset (written at open-time) + `project_uvw=false` so OmniPBR
  samples the mesh's `st` UV primvar.
- `material=` keys merged via `_OMNIPBR_KEY_MAP`; `*_texture` values → asset paths;
  `emissive`/`opacity` also emit their `enable_*` companion; unknown keys warned.
- Precedence: `material=(; base_color=…)` overrides `color`; `base_color_texture`
  overrides an image `color` texture.
"""
function material_inputs_from(plot)
    inputs = Dict{String,Any}()

    color = _plot_color(plot)
    have_base    = false
    have_texture = false
    if color isa AbstractMatrix
        # Image → diffuse_texture (resolved at open-time); project_uvw=false so
        # OmniPBR samples the mesh's `st` UVs, not world-space triplanar.
        inputs["diffuse_texture"] = _texture_asset_for(color, plot)
        inputs["project_uvw"]     = false
        have_texture = true
    elseif color isa AbstractVector
        # Per-vertex colour + material → collapse to a constant average
        # (OmniPBR has no per-vertex base).
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
        for key in keys(mat)
            _merge_material_input!(inputs, Symbol(key), Makie.to_value(mat[key]),
                                   have_base, have_texture)
        end
    end
    return inputs
end

# Merge ONE `material=` key/value into `inputs`, mapped to its OmniPBR name.
function _merge_material_input!(inputs::AbstractDict, key::Symbol, value,
                                have_base::Bool, have_texture::Bool = false)
    if key === :base_color
        have_base && @warn "OmniverseMakie: `material=(; base_color=…)` overrides the \
                            plot `color` for this material."
        inputs["diffuse_color_constant"] = _rgb(Makie.to_color(value))
    elseif key === :base_color_texture
        # Explicit texture path overrides an image `color` (precedence).
        have_texture && @warn "OmniverseMakie: `material=(; base_color_texture=…)` \
                              overrides the image `color` texture for this material."
        inputs["diffuse_texture"] = _texture_asset_for(value, nothing)
    elseif key in (:normal_texture, :roughness_texture, :metallic_texture)
        # Other `*_texture` paths used AS-IS (asset paths, no temp write).
        inputs[_OMNIPBR_KEY_MAP[key]] = _texture_asset_for(value, nothing)
    elseif key === :emissive
        inputs["emissive_color"]     = _rgb(Makie.to_color(value))
        # enable_emission/_opacity are MDL `uniform bool` gates. MUST be
        # authored as USD `bool` (a float/int vs a bool MDL param fails
        # to bind → the gate is SILENTLY off). `true` (Bool <: Real)
        # hits the emitter's Bool branch first.
        inputs["enable_emission"]    = true
        inputs["emissive_intensity"] = _EMISSIVE_INTENSITY   # explicit so emission is visible
    elseif key === :opacity
        inputs["opacity_constant"] = value
        inputs["enable_opacity"]   = true              # companion bool gate
    elseif haskey(_OMNIPBR_KEY_MAP, key)
        inputs[_OMNIPBR_KEY_MAP[key]] = value   # metallic, roughness, …
    else
        @warn "OmniverseMakie: unknown `material=` key `$(key)` — skipped (not an OmniPBR \
               escape-hatch input)."
    end
    return inputs
end

# Constant average `(r,g,b)` Float32 base colour from a per-vertex `color`
# (Colorant vector, or numeric vector mapped through the colormap); reuses
# `_displaycolor`.
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

# Material KIND: `material=(; glass=true)` → :glass (true refractive OmniGlass),
# else :omnipbr (whose `opacity` is only a flat alpha cut-out).
function _material_kind(plot)
    mat = _plot_material(plot)
    mat === nothing && return :omnipbr
    return (haskey(mat, :glass) && Makie.to_value(mat[:glass]) === true) ? :glass : :omnipbr
end

# Merge ONE glass `material=` key/value into `inputs` (OmniGlass input names);
# the OmniGlass counterpart of `_merge_material_input!`, shared by the author
# path and the live `:material` push so a glass edit writes OmniGlass names,
# not OmniPBR ones.
function _merge_glass_input!(inputs::AbstractDict, key::Symbol, value)
    if key === :base_color || key === :color
        inputs["glass_color"] = _rgb(Makie.to_color(value))
    elseif key === :ior
        inputs["glass_ior"] = value
    elseif key === :roughness
        inputs["frosting_roughness"] = value
    elseif key === :thin_walled
        inputs["thin_walled"] = value === true
    elseif key === :glass
        # the material-KIND flag, not itself an OmniGlass input.
    else
        @warn "OmniverseMakie: glass `material=` ignores key `$(key)` (OmniGlass takes \
               glass_color / ior / roughness / thin_walled)."
    end
    return inputs
end

# Base-colour shader input for a material kind (OmniGlass `glass_color` vs
# OmniPBR `diffuse_color_constant`); used by the live colour push to rewrite it.
_base_color_input(kind::Symbol) = kind === :glass ? "glass_color" : "diffuse_color_constant"

"""
    _glass_inputs_from(plot) -> Dict{String,Any}

Compose the OmniGlass input Dict for a `material=(; glass=true, …)` plot:
`glass_color` ← `color`/`base_color` (default white), `glass_ior` ← `ior`
(default 1.491), `frosting_roughness` ← `roughness`, `thin_walled` ← `thin_walled`.
The `glass` flag and any unmapped key are skipped.
"""
function _glass_inputs_from(plot)
    inputs = Dict{String,Any}()
    color  = _plot_color(plot)
    if color !== nothing && !(color isa AbstractArray)          # a scalar base tint (not an image/per-vertex)
        inputs["glass_color"] = _rgb(Makie.to_color(color))
    end
    mat = _plot_material(plot)
    if mat !== nothing
        for key in keys(mat)
            _merge_glass_input!(inputs, Symbol(key), Makie.to_value(mat[key]))
        end
    end
    get!(inputs, "glass_ior", 1.491f0)
    return inputs
end

"""
    materialized_looks_usda(root_scene) -> String

Compose the `/World/Looks` scope body: walk every atomic plot in `root_scene` and
emit a material fragment (`usda_omnipbr_material`/`usda_glass_material`) for each
`is_materialized` plot, wrapped by `looks_scope_usda`. Called by
`author_root_from_scene!` so materials are PRE-AUTHORED at open-time (a Material
added to the OPEN stage is not bindable); bound later by `OV.bind_material!`.
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
        foreach(subplot -> _collect_materialized_plot!(frags, subplot), plot.plots)
    end
    return
end
