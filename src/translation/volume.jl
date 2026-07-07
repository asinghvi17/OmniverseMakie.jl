# UsdVol volume authoring: UsdVolVolume (+ OpenVDBAsset + an IndeX
# `nvindex:volume` Colormap material) from an on-disk .vdb/.nvdb into a
# Screen's open stage. Renders via NVIDIA IndeX Direct, which
# OV._ensure_index() must have enabled at Screen creation.
#
# ovrtx constraints:
#  1. Internal rel/connect targets must be layer-relative to the defaultPrim
#     (</Volume/density>); USD remaps them under the reference prim_path on
#     composition. An absolute target dangles → field unbound → renders black.
#  2. IndeX Direct ignores Colormap transfer-function colours and renders its
#     default grayscale TF; colours need a composite-capable Kit runtime. The
#     Colormap material is still authored — it is what such a build activates.

# Parallel `rgbaPoints` (float4 RGBA) + `xPoints` (positions in [0,1]) for an
# IndeX Colormap TF from a Makie colormap. Opacity ramps with position (low
# density → more transparent). Samples ≤ `npoints` colours evenly.
function _colormap_points(colormap; npoints::Int = 16)
    cs = Makie.to_colormap(colormap)
    n  = length(cs)
    idxs = n <= npoints ? collect(1:n) : round.(Int, range(1, n; length = npoints))
    m = length(idxs)
    rgba = String[]
    xs   = String[]
    for (k, i) in enumerate(idxs)
        t = m == 1 ? 0.0f0 : Float32((k - 1) / (m - 1))
        c = cs[i]
        push!(rgba, "($(Float32(c.r)), $(Float32(c.g)), $(Float32(c.b)), $(t))")
        push!(xs, string(t))
    end
    return (join(rgba, ", "), join(xs, ", "))
end

"""
    _vdb_volume_usda(vdb_path; prim_path="/World/Volume", field="density",
                     field_dtype="float", colormap=:viridis,
                     colorrange=nothing) -> String

Pure (no renderer) self-contained USDA layer for a `.vdb`/`.nvdb` volume: a
`def Volume "Volume"` (the `defaultPrim`) with a `field:<field>` rel to a
child `OpenVDBAsset`, plus an IndeX `nvindex:volume` `Colormap` material from
colormap/colorrange.

`field` names the rel, the `OpenVDBAsset` prim, AND its `fieldName` (the grid
name inside the VDB, e.g. `"torus_fog"` for `torus.vdb`).

Internal rel/connect targets are layer-relative (`</Volume/…>`) — USD remaps
them under the reference `prim_path` on composition (absolute would dangle →
black). `prim_path` itself is accepted only for signature symmetry with
`author_vdb_volume!`; it does not appear in the emitted string.
"""
function _vdb_volume_usda(vdb_path; prim_path = "/World/Volume", field = "density",
                          field_dtype = "float", colormap = :viridis, colorrange = nothing)
    # `field` is interpolated four ways below (rel name, target, prim name,
    # `fieldName` value), so it must be a legal USD identifier — validate
    # once here; a bad grid name would author a corrupt Volume prim.
    field = _usd_identifier(string(field); what = "volume `field` name")
    rgba_str, x_str = _colormap_points(colormap)
    lo, hi = something(colorrange, (0.0, 1.0))
    return """#usda 1.0
(
    defaultPrim = "Volume"
)
def Volume "Volume" (
    prepend apiSchemas = ["MaterialBindingAPI"]
)
{
    rel field:$(field) = </Volume/$(field)>
    rel material:binding = </Volume/Material>

    def OpenVDBAsset "$(field)"
    {
        token fieldName = "$(field)"
        token fieldDataType = "$(field_dtype)"
        asset filePath = $(_usd_asset_path(vdb_path; what = "VDB `filePath`"))
    }

    def Material "Material"
    {
        token outputs:nvindex:volume.connect = </Volume/Material/VolumeShader.outputs:volume>

        def Colormap "Colormap"
        {
            custom token colormapSource = "rgbaPoints"
            custom float2 domain = ($(Float64(lo)), $(Float64(hi)))
            uniform token domainBoundaryMode = "clampToTransparent"
            custom token outputs:colormap
            custom float4[] rgbaPoints = [$(rgba_str)]
            custom float[] xPoints = [$(x_str)]
        }

        def Shader "VolumeShader"
        {
            token inputs:colormap.connect = </Volume/Material/Colormap.outputs:colormap>
            token outputs:volume
        }
    }
}
"""
end

"""
    author_vdb_volume!(screen, scene, vdb_path; prim_path="/World/Volume",
                       field="density", field_dtype="float",
                       colormap=:viridis, colorrange=nothing) -> String

Reference a `UsdVolVolume` (from on-disk `vdb_path`) into `screen`'s
already-open stage (call after the camera/lights/render-product are set up);
returns `prim_path`. Renders through NVIDIA IndeX Direct, which must have
been enabled BEFORE `screen` was created (`OV._ensure_index`); otherwise this
errors clearly instead of authoring a prim that would silently render black.

`scene` is accepted only for signature symmetry with `volume!`; it is not
read. See `_vdb_volume_usda` for field/colormap notes and the constraint
that colormap colours need a composite-capable runtime (rendering uses IndeX
Direct's default grayscale TF).
"""
function author_vdb_volume!(screen, scene, vdb_path; prim_path = "/World/Volume", field = "density",
                            field_dtype = "float", colormap = :viridis, colorrange = nothing)
    OV._index_enabled() || error(
        "author_vdb_volume!: volume rendering requires NVIDIA IndeX, which is not enabled.  Set " *
        "OMNIVERSEMAKIE_INDEX_LIBS (or OMNIVERSEMAKIE_OVRTX_CONFIG) BEFORE creating the Screen, " *
        "then re-create it.")
    isfile(vdb_path) || error("author_vdb_volume!: VDB file not found: $(vdb_path)")
    usda = _vdb_volume_usda(vdb_path; prim_path, field, field_dtype, colormap, colorrange)
    OV.add_usd_reference!(screen.renderer, usda, prim_path)
    return prim_path
end

"""
    reload_volume_data!(screen, robj, plot, scalars) -> Nothing

Live volume data edit: write the plot's density field to a fresh temp `.nvdb`
and reload it into the open stage so NVIDIA IndeX shows the new data. Called
by `push_to_ovrtx!`'s `:volume` branch when `plot[4][] = new_array` fires
`changed[:volume]`.

The reload is a remove + re-reference, NOT a `filePath` write: IndeX loads
the grid into its own memory and never re-reads the source file, so a
`filePath` write does not update the render. A fresh temp path per edit also
dodges the carb asset cache serving the stale file; `robj.usd_handle`
becomes the new layer's handle.

Temps stay bounded: the prior temp is deleted here (one live grid on disk);
`destroy_bindings!` removes the last on close/delete. colormap/colorrange
are re-read from the plot, so a data edit keeps the same placement + TF.
"""
function reload_volume_data!(screen, robj, plot, scalars)
    r     = screen.renderer
    vprim = get(robj.meta, :volume_prim, robj.prim_path)
    data  = scalars isa Array{Float32,3} ? scalars : Float32.(scalars)
    xr = Makie.to_value(plot[1]); yr = Makie.to_value(plot[2]); zr = Makie.to_value(plot[3])
    origin = GeometryBasics.Point3f(first(xr), first(yr), first(zr))
    extent = GeometryBasics.Vec3f(last(xr) - first(xr), last(yr) - first(yr), last(zr) - first(zr))
    newtmp   = tempname() * ".nvdb"
    reloaded = false
    # Teardown-safe remove-then-add (do not reorder): the prim must disappear
    # and reappear for IndeX to re-read; remove_usd! does not clear the grid
    # from IndeX memory, so if the add throws the last grid keeps rendering.
    # On add failure, robj.meta[:usd_handle_valid]=false stops the stale
    # handle being re-removed next edit (that can throw and wedge the plot);
    # the next edit's fresh add recovers, and teardown reads the same flag.
    # newtmp is written first; try/finally rm's it on failure (no temp leak).
    try
        NanoVDBWriter.save_nanovdb(newtmp, data, origin, extent)
        usda = _vdb_volume_usda(newtmp; prim_path = vprim, field = "density",
                                colormap = Makie.to_value(plot.colormap),
                                colorrange = _volume_colorrange(plot, data))
        if get(robj.meta, :usd_handle_valid, true)
            OV.remove_usd!(r, robj.usd_handle)      # drop the current layer
            robj.meta[:usd_handle_valid] = false  # dangling until add succeeds
        end
        _note_composition_change!(screen)  # composition swap → structural reset
        robj.usd_handle = OV.add_usd_reference!(r, usda, vprim)
        robj.meta[:usd_handle_valid] = true
        reloaded = true
    finally
        if reloaded
            old = get(robj.meta, :vdb_tmp, nothing)  # GC the prior temp
            old isa AbstractString && old != newtmp && rm(old; force = true)
            robj.meta[:vdb_tmp] = newtmp
        else
            rm(newtmp; force = true)  # reload failed → GC the orphan
        end
    end
    # No inline OV.reset!: the diff-node callback sets requires_update=true;
    # colorbuffer's _sync_and_needs_reset! then issues the per-frame reset,
    # same as every push route.
    return nothing
end
