# Materials translation: Makie plot.color → USD-native primvars:displayColor
# (constant or per-vertex), plus OmniPBR/OmniGlass MDL material authoring
# (the `material=` escape hatch). Included after usd.jl; Makie, Colors,
# ColorTypes (red/green/blue) in scope.

"""
    displaycolor_for(plot, npoints::Int) -> (values, interpolation::String)

Translate `plot.color` into USD `primvars:displayColor`. Returns
`(values, interpolation)`:
- scalar `Colorant`/`Symbol` → one `(r,g,b)` Float32 tuple, `"constant"`
  (e.g. `:red` → `((1,0,0), "constant")`).
- `AbstractVector{<:Colorant}` → one tuple per vertex, `"vertex"` (length
  should equal `npoints`).
- `AbstractVector{<:Real}` + `colormap`/`colorrange` → per-vertex via the
  colormap, `"vertex"`.

Does NOT handle `plot.material` (the OmniPBR path); authored separately.
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

# Per-vertex numeric values → colours via `_map_through_colormap` (the shared,
# NaN-safe colormap + colorrange mapper).
function _displaycolor(values::AbstractVector{<:Real}, plot, npoints::Int)
    return (_map_through_colormap(plot, values), "vertex")
end

# Scalar / fallback → one constant colour.
function _displaycolor(color, plot, npoints::Int)
    return (_rgb(Makie.to_color(color)), "constant")
end

"""
    _map_through_colormap(plot, values) -> Vector{NTuple{3,Float32}}

Map per-vertex numeric `values` through `plot.colormap` over
`_resolve_colorrange(plot, values)`, done manually (`to_colormap` +
`interpolated_getindex`) because Makie's `numbers_to_colors` errors outside
the plot's compute pipeline. NaN-safe: a non-finite value maps to
`plot.nan_color` (haskey-guarded, default transparent). RGB-only is fine —
a non-finite surface vertex belongs to no emitted face. The sole
numeric-colour mapper (scatter/lines/mesh + surface).
"""
function _map_through_colormap(plot, values)
    cmap    = Makie.to_colormap(plot.colormap[])
    crange  = _resolve_colorrange(plot, values)
    nan_rgb = _rgb(Makie.to_color(haskey(plot, :nan_color) ? plot.nan_color[] : :transparent))
    return NTuple{3,Float32}[
        isfinite(v) ? _rgb(Makie.interpolated_getindex(cmap, Float32(v), crange)) : nan_rgb
        for v in values]
end

"""
    _resolve_colorrange(plot, values) -> NTuple{2,Float32}

Resolve `plot.colorrange` to a concrete `(lo, hi)`: an explicit 2-element
tuple/vector is honored verbatim; `automatic` derives extrema over the finite
`values` only (raw `extrema` would give `(NaN, NaN)` on masked data); with no
finite values it falls back to `(0f0, 1f0)`. Shared by every colour-mapped
site (`_volume_colorrange` delegates here for its automatic branch).
"""
function _resolve_colorrange(plot, values)
    crange = plot.colorrange[]
    if (crange isa Tuple || crange isa AbstractVector) && length(crange) == 2
        return (Float32(crange[1]), Float32(crange[2]))
    end
    finite = filter(isfinite, values)
    isempty(finite) && return (0.0f0, 1.0f0)
    lo, hi = extrema(finite)
    return (Float32(lo), Float32(hi))
end

# (r,g,b) Float32 tuple from any Colorant (RGB, RGBA, HSV, … all handled).
_rgb(c) = (Float32(red(c)), Float32(green(c)), Float32(blue(c)))

# === OmniPBR material authoring ===
# An OmniPBR material is a `def Material` + one `def Shader` bound to the
# bundled OmniPBR.mdl; the bare @OmniPBR.mdl@ resolves via ovrtx's MDL search
# path. USD is order-insensitive, so Dict iteration order is fine.

"""
    material_prim_path(plot) -> String

Canonical prim path of `plot`'s material: `/World/Looks/Mat_<objectid(plot)>`
(one material per atomic plot, no dedup). The material must be pre-authored
into the `/World/Looks` scope at stage open-time (a runtime
`add_usd_reference!` of a Material is not bindable); `OV.bind_material!`
binds it to the geometry at runtime.
"""
material_prim_path(plot) = "/World/Looks/Mat_$(objectid(plot))"

"""
    usda_omnipbr_material(name::AbstractString, inputs::AbstractDict) -> String

USDA fragment for one OmniPBR `UsdShade Material` named `name`, composed at
`/World/Looks/<name>` (8-space indent to nest under `/World/Looks`). `inputs`
maps an OmniPBR shader-input name to a value, typed by the USD kind emitted:
`NTuple{3}`→`color3f (r,g,b)`, `NTuple{2}`→`float2 (x,y)`, `Bool`→`bool 0|1`,
`AbstractString`→`asset @…@`, else→`float Float32(value)`.
`outputs:mdl:surface.connect` targets the material's absolute
`Shader.outputs:out`.

Constraint: must be composed inline at root-author time, i.e. pre-authored
into the stage passed to `open_usd_string!`. Adding it to the open stage via
`OV.add_usd_reference!` does not make the material bindable (silent no-op
for `material:binding`); `OV.bind_material!` binds once open.
"""
usda_omnipbr_material(name::AbstractString, inputs::AbstractDict) =
    _usda_mdl_material(name, "OmniPBR.mdl", "OmniPBR", inputs)

"""
    usda_glass_material(name, inputs) -> String

USDA fragment for a `UsdShade Material` backed by the bundled OmniGlass.mdl
(subIdentifier "OmniGlass"): refraction via `glass_ior`, tint via
`glass_color`, optional `frosting_roughness`/`thin_walled`. Selected by
`material=(; glass=true)`; composed/bound like the OmniPBR materials. Unlike
OmniPBR `opacity` (a flat alpha cut-out), OmniGlass actually refracts.
"""
usda_glass_material(name::AbstractString, inputs::AbstractDict) =
    _usda_mdl_material(name, "OmniGlass.mdl", "OmniGlass", inputs)

# Author one MDL-backed `UsdShade Material` `name` from `mdl_asset`
# (subIdentifier `subid`); `inputs` typed by value: NTuple{3}→color3f,
# NTuple{2}→float2, Bool→bool, AbstractString→asset, else→float. Shared by
# OmniPBR + OmniGlass authoring.
function _usda_mdl_material(name::AbstractString, mdl_asset::AbstractString,
                           subid::AbstractString, inputs::AbstractDict)
    lines = String[]
    for (input_name, input_value) in inputs
        if input_value isa NTuple{3}                     # color3f
            push!(lines, "                color3f inputs:$(input_name) = ($(Float32(input_value[1])), $(Float32(input_value[2])), $(Float32(input_value[3])))")
        elseif input_value isa NTuple{2}                 # float2 (UV tiling)
            push!(lines, "                float2 inputs:$(input_name) = ($(Float32(input_value[1])), $(Float32(input_value[2])))")
        elseif input_value isa Bool                      # bool (Bool <: Real)
            push!(lines, "                bool inputs:$(input_name) = $(input_value ? 1 : 0)")
        elseif input_value isa AbstractString            # asset (texture)
            push!(lines, "                asset inputs:$(input_name) = $(_usd_asset_path(input_value; what = "texture asset for input `$(input_name)`"))")
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
                uniform asset info:mdl:sourceAsset = $(_usd_asset_path(mdl_asset; what = "MDL source asset"))
                uniform token info:mdl:sourceAsset:subIdentifier = "$(subid)"
$(join(lines, "\n"))
                token outputs:out
            }
        }
"""
end

# === `material=` escape hatch + `color`→base composition ===
# A plot is materialized (OmniPBR material, not displayColor) when `material=`
# is set or `color` is an image. `material_inputs_from` composes both into one
# Dict of OmniPBR input names; `material=(; base_color=…)` overrides `color`.

"""
    _OMNIPBR_KEY_MAP :: Dict{Symbol,String}

Map from a user-facing `material=` key to its OmniPBR shader-input name, for
the keys `_merge_material_input!` maps generically. `base_color`/`emissive`/
`opacity`/`base_color_texture` are NOT here — dedicated `_merge_material_input!`
branches handle them first (they also emit companion inputs, e.g.
`enable_emission`, or override precedence). `*_texture` values are resolved to
asset paths by `_texture_asset_for`. Unknown keys are warned/skipped.
"""
const _OMNIPBR_KEY_MAP = Dict{Symbol,String}(
    :metallic           => "metallic_constant",
    :roughness          => "reflection_roughness_constant",
    # *_texture → OmniPBR texture input names (values resolved to asset paths).
    :normal_texture     => "normalmap_texture",
    :roughness_texture  => "reflectionroughness_texture",
    :metallic_texture   => "metallic_texture",
    # UVW projection / tiling (OmniPBR.mdl "UV" group): `project_uvw = true`
    # derives UVs by projection instead of the mesh's authored `st`;
    # `world_or_object` picks the space (true = world); `texture_scale`/
    # `_translate` (2-tuples → float2) and `texture_rotate` (degrees)
    # transform the projected UVs. OmniPBR gates all four on `project_uvw`.
    :project_uvw        => "project_uvw",
    :world_or_object    => "world_or_object",
    :texture_scale      => "texture_scale",
    :texture_translate  => "texture_translate",
    :texture_rotate     => "texture_rotate",
)

# Default OmniPBR `emissive_intensity` authored with an `emissive=` key.
# OmniPBR's MDL default is 40.0 (soft-range 0–1000); we author higher so an
# emission-only material reads clearly emissive under auto-exposure, not black.
const _EMISSIVE_INTENSITY = 5000.0f0

"""
    _enable_material_attribute!() -> Nothing

Append `:material` to `Makie.attribute_name_allowlist()` so the `material=`
escape hatch is accepted on every plot type (else `validate_attribute_keys`
throws `InvalidAttributeError` on plots like `Lines`/`Scatter` that lack it
natively). Idempotent; captures the existing list so Makie's own entries are
preserved. Called at `__init__`.
"""
function _enable_material_attribute!()
    base = Makie.attribute_name_allowlist()
    :material in base && return nothing
    newlist = (base..., :material)
    @eval Makie attribute_name_allowlist() = $newlist
    return nothing
end

# === Temp-texture writer + `_texture_asset_for` ===
# An image `color` (Matrix{<:Colorant}) is written to a PNG at open-time
# (ovrtx loads PNG natively) and referenced by the OmniPBR `diffuse_texture`
# input. The file lives in a stable session temp dir so it survives until
# ovrtx reads it during render.

# Stable per-process temp dir for textures; created lazily, cleanup=false
# because ovrtx may read assets lazily during render. Files are never deleted.
const _TEXTURE_DIR = Ref{String}("")
function _texture_dir()
    if isempty(_TEXTURE_DIR[]) || !isdir(_TEXTURE_DIR[])
        _TEXTURE_DIR[] = mktempdir(; prefix = "omniversemakie_tex_", cleanup = false)
    end
    return _TEXTURE_DIR[]
end

# Monotonic suffix so every image write lands at a fresh path.
const _TEX_WRITE_SEQ = Ref{UInt}(0)

"""
    _texture_asset_for(img_or_path, plot, key) -> String

Resolve an image `color` or `*_texture` value to an on-disk asset path for an
OmniPBR texture input, at open-time:
- `AbstractString` → absolutized (USD needs an absolute path; the in-memory
  root stage has no anchor for a relative one) and validated to exist —
  a missing file throws `ArgumentError` (a dangling asset ref renders
  silently untextured). `plot`/`key` unused.
- `Matrix{<:Colorant}` → written to a fresh session-temp PNG
  (`PNGFiles.save`, converted to `RGBA{N0f8}`), named
  `tex_<objectid(plot)>_<key>_<seq>.png` with a monotonic `<seq>` so no write
  ever overwrites an earlier one; its absolute path is returned (USD needs it
  absolute — the in-memory root stage has no anchor for a relative path).

`key` (a Symbol/String naming the input, e.g. `:color`,
`:base_color_texture`) keeps the `<oid>_<key>` prefix distinct per input per
plot; without it two image inputs would collide on one shared temp file and
silently overwrite each other.
"""
function _texture_asset_for(path::AbstractString, plot, key)
    abs_path = abspath(String(path))
    isfile(abs_path) || throw(ArgumentError(
        "OmniverseMakie: texture asset \"$(path)\" (resolved to \"$(abs_path)\") does " *
        "not exist — a dangling asset reference would render silently untextured."))
    return abs_path
end
function _texture_asset_for(img::AbstractMatrix{<:Colorant}, plot, key)
    # Fresh unique filename on every write — never overwrite a path ovrtx may
    # already have loaded: re-writing the same file makes ovrtx flag it a
    # changed "video texture" and disable it, and races its async PNG read
    # ("Corrupt PNG"). A per-write path is read exactly once.
    n = (_TEX_WRITE_SEQ[] += 1)
    path = joinpath(_texture_dir(), "tex_$(objectid(plot))_$(key)_$(n).png")
    PNGFiles.save(path, convert.(RGBA{N0f8}, img))
    return path
end

# Read a plot's raw `material` escape-hatch value safely (haskey-guarded so a
# plot lacking the attribute never errors); `nothing` when absent/unset.
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

`true` when `plot` should render with an OmniPBR material rather than
USD-native `displayColor`: `material=` is set OR `color` is an image
(`AbstractMatrix`). `haskey`-guarded, so returns `false` cleanly for a plain
`mesh!(…; color=:red)`.
"""
function is_materialized(plot)
    _plot_material(plot) === nothing || return true
    return _plot_color(plot) isa AbstractMatrix
end

"""
    material_inputs_from(plot) -> Dict{String,Any}

Compose a materialized plot's `color` and `material=` params into one Dict
keyed by OmniPBR shader-input name, for `usda_omnipbr_material`. Scalars are
kept raw (the emitter Float32-converts); colours are `(r,g,b)` Float32
tuples.

- `color`: scalar → `diffuse_color_constant`; per-vertex → constant average
  base (+`@warn`; OmniPBR has no per-vertex base); image `Matrix{<:Colorant}`
  → `diffuse_texture` asset (written at open-time) + `project_uvw=false` so
  OmniPBR samples the mesh's `st` UV primvar.
- `material=` keys merged via `_OMNIPBR_KEY_MAP`; `*_texture` values → asset
  paths; `emissive`/`opacity` also emit their `enable_*` companion; unknown
  keys warned.
- Precedence: `material=(; base_color=…)` overrides `color`;
  `base_color_texture` overrides an image `color` texture.
"""
function material_inputs_from(plot)
    inputs = Dict{String,Any}()

    color = _plot_color(plot)
    have_base    = false
    have_texture = false
    if color isa AbstractMatrix
        # Image → diffuse_texture (resolved at open-time); project_uvw=false
        # so OmniPBR samples the mesh's `st` UVs, not world-space triplanar.
        # Key `:color` keeps this temp PNG distinct from any `*_texture` one.
        inputs["diffuse_texture"] = _texture_asset_for(color, plot, :color)
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
                                   plot, have_base, have_texture)
        end
    end
    return inputs
end

# Merge one `material=` key/value into `inputs`, mapped to its OmniPBR name.
# `plot` threads through to `_texture_asset_for` so an image `*_texture` value
# is written to a temp PNG keyed by this plot + this input `key`.
function _merge_material_input!(inputs::AbstractDict, key::Symbol, value, plot,
                                have_base::Bool, have_texture::Bool = false)
    if key === :base_color
        have_base && @warn "OmniverseMakie: `material=(; base_color=…)` overrides the \
                            plot `color` for this material."
        inputs["diffuse_color_constant"] = _rgb(Makie.to_color(value))
    elseif key === :base_color_texture
        # Explicit texture path overrides an image `color` (precedence).
        have_texture && @warn "OmniverseMakie: `material=(; base_color_texture=…)` \
                              overrides the image `color` texture for this material."
        inputs["diffuse_texture"] = _texture_asset_for(value, plot, key)
    elseif key in (:normal_texture, :roughness_texture, :metallic_texture)
        # Other `*_texture` values: a path used as-is, or an image written to
        # a temp PNG keyed by (plot, key) so it can't collide with another.
        inputs[_OMNIPBR_KEY_MAP[key]] = _texture_asset_for(value, plot, key)
    elseif key === :emissive
        inputs["emissive_color"]     = _rgb(Makie.to_color(value))
        # enable_emission/_opacity are MDL `uniform bool` gates: they must be
        # authored as USD `bool` (a float/int fails to bind → the gate is
        # silently off). `true` hits the emitter's Bool branch first.
        inputs["enable_emission"]    = true
        inputs["emissive_intensity"] = _EMISSIVE_INTENSITY
    elseif key === :opacity
        inputs["opacity_constant"] = value
        inputs["enable_opacity"]   = true              # companion bool gate
    elseif haskey(_OMNIPBR_KEY_MAP, key)
        # Normalize a 2-element static vector (Vec2f/Point2f) to the NTuple{2}
        # the float2 emitter/live-push branches dispatch on, so
        # `texture_scale = Vec2f(4, 4)` works too.
        v = value isa Makie.VecTypes{2} ? (Float32(value[1]), Float32(value[2])) : value
        inputs[_OMNIPBR_KEY_MAP[key]] = v       # metallic, roughness, tiling, …
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

# Material kind: `material=(; glass=true)` → :glass (refractive OmniGlass),
# else :omnipbr (whose `opacity` is only a flat alpha cut-out).
function _material_kind(plot)
    mat = _plot_material(plot)
    mat === nothing && return :omnipbr
    return (haskey(mat, :glass) && Makie.to_value(mat[:glass]) === true) ? :glass : :omnipbr
end

# Merge one glass `material=` key/value into `inputs` (OmniGlass input
# names). Shared by the author path and the live `:material` push so a glass
# edit writes OmniGlass names, not OmniPBR ones.
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
        # the material-kind flag, not itself an OmniGlass input.
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
(default 1.491), `frosting_roughness` ← `roughness`, `thin_walled` ←
`thin_walled`. The `glass` flag and any unmapped key are skipped.
"""
function _glass_inputs_from(plot)
    inputs = Dict{String,Any}()
    color  = _plot_color(plot)
    if color !== nothing && !(color isa AbstractArray)  # scalar base tint only
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

Compose the `/World/Looks` scope body: walk every atomic plot in `root_scene`
and emit a material fragment (`usda_omnipbr_material`/`usda_glass_material`)
for each `is_materialized` plot, wrapped by `looks_scope_usda`. Called by
`author_root_from_scene!` so materials are pre-authored at open-time (a
Material added to the open stage is not bindable); bound later by
`OV.bind_material!`.
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
    else                                         # composite → recurse
        foreach(subplot -> _collect_materialized_plot!(frags, subplot), plot.plots)
    end
    return
end
