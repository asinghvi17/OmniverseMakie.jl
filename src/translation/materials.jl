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
