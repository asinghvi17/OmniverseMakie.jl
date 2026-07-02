# UsdVol volume authoring (Volumes M1).  Authors UsdVolVolume (+ OpenVDBAsset +
# an IndeX `nvindex:volume` Colormap material) from an on-disk .vdb/.nvdb into a
# Screen's OPEN stage.  Renders via IndeX Direct, which OV._ensure_index()
# must have enabled at Screen creation.  Included in the OmniverseMakie module
# after OV.jl / usd.jl (calls OV.add_usd_reference! / OV._index_enabled; screen
# is duck-typed).
#
# VERIFIED ovrtx constraints
# (ground truth: .superpowers/sdd/m6b/volume-spike-report.md):
#  1. add_usd_reference! of a Volume into the open stage RENDERS (torus.vdb ->
#     ~9.5k px).
#  2. Internal rel/connect targets MUST be LAYER-RELATIVE to the defaultPrim
#     (</Volume/density>) — USD remaps them under the reference prim_path on
#     composition.  An ABSOLUTE target (</World/Volume/density>) dangles ->
#     field unbound -> BLACK (absolute: 0 px; layer-relative: 9486 px).
#  3. Colormap transfer-function COLOURS ignored by IndeX Direct: it renders
#     with its DEFAULT grayscale-density TF (viridis gave byte-identical
#     gray).  Colours need the IndeX COMPOSITE path, ARCHITECTURALLY ABSENT from
#     this standalone ovrtx: composite flags (nvindex:composite, omni:rtx:skip,
#     rtx:index:compositeEnabled) are honored ONLY by Kit's
#     omni.rtx.index_composite + omni.hydra.rtx (no ovrtx .so uses them).
#     Enabling via a carb setting or customLayerData had ZERO colour effect;
#     omni:rtx:skip removed the volume from the only working path (Direct)
#     -> BLACK.  So COLOURS DEFER to a composite-capable Kit build; M2 ships
#     grayscale-Direct.  We still AUTHOR the Colormap material: non-black today,
#     and exactly what a composite build would activate, so author_vdb_volume!
#     stays the reusable authoring primitive.

# Parallel `rgbaPoints` (float4 RGBA) + `xPoints` (positions in [0,1]) for an
# IndeX Colormap TF from a Makie colormap.  Opacity ramps with position (low
# density -> more transparent).  Samples <= `npoints` colours evenly, keeping
# the emitted USDA compact.
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
        push!(rgba, "($(Float32(c.r)), $(Float32(c.g)), $(Float32(c.b)), $(t))")  # RGBA; α ramps with t
        push!(xs, string(t))
    end
    return (join(rgba, ", "), join(xs, ", "))
end

"""
    _vdb_volume_usda(vdb_path; prim_path="/World/Volume", field="density", field_dtype="float",
                     colormap=:viridis, colorrange=nothing) -> String

PURE (no renderer) self-contained USDA layer for a `.vdb`/`.nvdb` volume: a
`def Volume "Volume"` (the `defaultPrim`) with a `field:<field>` rel to a child
`OpenVDBAsset`, plus an IndeX `nvindex:volume` `Colormap` material from colormap/colorrange.

`field` names the rel, the `OpenVDBAsset` prim, AND its `fieldName` (the grid name inside
the VDB, e.g. `"torus_fog"` for `torus.vdb`).

Internal rel/connect targets are LAYER-RELATIVE (`</Volume/…>`) — USD remaps them under the
reference `prim_path` on composition (absolute would dangle -> BLACK).  `prim_path` itself is
accepted only for signature symmetry with `author_vdb_volume!`; it does NOT appear in the
emitted string.  Metadata is multi-line (the RenderProduct-prim gotcha — see usd.jl).
"""
function _vdb_volume_usda(vdb_path; prim_path = "/World/Volume", field = "density",
                          field_dtype = "float", colormap = :viridis, colorrange = nothing)
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
        asset filePath = @$(vdb_path)@
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
    author_vdb_volume!(screen, scene, vdb_path; prim_path="/World/Volume", field="density",
                       field_dtype="float", colormap=:viridis, colorrange=nothing) -> String

Reference a `UsdVolVolume` (from on-disk `vdb_path`) into `screen`'s already-open stage
(call after the camera/lights/render-product are set up); returns `prim_path`.  Renders
through NVIDIA IndeX Direct, which must have been enabled BEFORE `screen` was created
(`OV._ensure_index`); otherwise this errors clearly instead of authoring a prim that would
silently render black.

`scene` is accepted only for signature symmetry with M2's `volume!`; M1 does not read it.
See `_vdb_volume_usda` for field/colormap notes and the constraint that the colormap COLOURS
need the deferred composite path (M1 renders via IndeX Direct's default grayscale TF).
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

LIVE volume DATA edit: write the plot's density field to a FRESH temp `.nvdb` and RELOAD it
into the open stage so NVIDIA IndeX shows the new data.  Called by `push_to_ovrtx!`'s
`:volume` branch when `plot[4][] = new_array` fires `changed[:volume]`.

The reload is a REMOVE + RE-REFERENCE, NOT a `filePath` write: a `filePath` write does NOT
update the render — IndeX loads the grid into its own memory and evicts ("orphans") the
source file, so the on-disk change is never re-read (spike: filePath write -> moved ~0.01 px;
remove_usd! + add_usd_reference! of a fresh layer at a FRESH temp path -> moved ~26 px).  A
fresh temp path per edit also dodges the carb asset-cache serving the stale file;
`robj.usd_handle` becomes the new layer's handle.

Temps stay BOUNDED: the prior temp is deleted here (one live grid on disk), `destroy_bindings!`
removes the last on close/delete.  colormap/colorrange are re-read from the plot, so a data
edit keeps the same placement + (grayscale-Direct) TF.
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
    # Teardown-safe remove-THEN-add (order matters — do NOT reorder to
    # add-before-remove):
    #  • LAST-GOOD-FRAME: remove_usd! does NOT clear the volume (IndeX keeps the
    #    grid in its own memory — the same self-managed behavior that makes a
    #    filePath write a no-op), so if the add throws the last grid keeps
    #    rendering.  A reorder would keep the prim CONTINUOUSLY referenced and
    #    DEFEAT the reload: the prim must disappear+reappear for IndeX
    #    to re-read; remove-then-add is required, giving last-good-frame free.
    #  • RECOVERY/NO-WEDGE: add fails -> set robj.meta[:usd_handle_valid]=false
    #    so the stale handle is NOT re-removed next edit (remove_usd! of an
    #    already-removed handle throws on some ovrtx builds -> would wedge the
    #    plot); the next edit does a fresh add, replacing the retained grid and
    #    RECOVERING.  Teardown reads the same flag (screen.jl) so a stale handle
    #    can't abort delete!/empty!/close.
    #  • BOUNDED TEMPS: newtmp is written first; try/finally rm's it on ANY
    #    failure, so repeated fails do NOT leak (one grid on disk at a time).
    try
        NanoVDBWriter.save_nanovdb(newtmp, data, origin, extent)
        usda = _vdb_volume_usda(newtmp; prim_path = vprim, field = "density",
                                colormap = Makie.to_value(plot.colormap),
                                colorrange = _volume_colorrange(plot, data))
        if get(robj.meta, :usd_handle_valid, true)
            OV.remove_usd!(r, robj.usd_handle)               # drop the current layer (prim disappears)
            robj.meta[:usd_handle_valid] = false             # handle now dangling until the add succeeds
        end
        _invalidate_path_resolver!(screen)                   # remove+re-reference swaps composition → drop cached resolver
        robj.usd_handle = OV.add_usd_reference!(r, usda, vprim)  # fresh layer → fresh temp; new handle
        robj.meta[:usd_handle_valid] = true
        reloaded = true
    finally
        if reloaded
            old = get(robj.meta, :vdb_tmp, nothing)          # GC the prior temp (keep the count bounded)
            old isa AbstractString && old != newtmp && rm(old; force = true)
            robj.meta[:vdb_tmp] = newtmp
        else
            rm(newtmp; force = true)                         # threw before the reload completed → GC the orphan
        end
    end
    # No inline OV.reset!: the diff-node callback sets scr.requires_update=true;
    # colorbuffer's _sync_and_needs_reset! (screen.jl) then issues the per-frame
    # reset — same as every push route (verified: the edit re-renders the grid).
    return nothing
end
