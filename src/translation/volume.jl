# UsdVol volume authoring for OmniverseMakie (Volumes M1).
#
# Authors a `UsdVolVolume` (+ `UsdVolOpenVDBAsset` + an IndeX `nvindex:volume` Colormap material)
# from an on-disk `.vdb`/`.nvdb` file into a Screen's OPEN stage.  The render routes to NVIDIA
# IndeX Direct, which OV._ensure_index() (Task 1) must have enabled at Screen-creation time.
#
# NOTE: this file is included inside the OmniverseMakie module (after OV.jl / usd.jl), so it can
# call OV.add_usd_reference! / OV._index_enabled and use Makie (`screen` is duck-typed).
#
# ── VERIFIED ovrtx constraints (Volumes M1 plan Task 2 Step 1; ground truth in
#    .superpowers/sdd/m6b/volume-spike-report.md) ────────────────────────────────────────────
#  1. add_usd_reference! of a `Volume` into the open stage RENDERS (torus.vdb → ~9.5k px @ 256²).
#  2. INTERNAL rel/connect targets MUST be LAYER-RELATIVE to the layer's defaultPrim
#     (`</Volume/density>`).  USD remaps them under the reference target `prim_path` on
#     composition.  An ABSOLUTE target (`</World/Volume/density>`) authored in the layer is
#     dangling → the field never binds → the volume renders BLACK.  (Verified: absolute → 0 px,
#     layer-relative → 9486 px.)
#  3. The Colormap TRANSFER-FUNCTION COLOURS do NOT apply via IndeX Direct: a `nvindex:volume`
#     material added by reference is rendered by Direct with its DEFAULT (grayscale-density)
#     transfer function — a deliberately colourful viridis colormap produced byte-identical gray
#     output to the bare volume.  Applying the authored colours needs the IndeX COMPOSITE path
#     (Volume-prim `nvindex:composite`+`omni:rtx:skip` flags + a `rtx:index:compositeEnabled`
#     render setting).  ── VOLUMES M2 TASK 3 SPIKE FINDING (.superpowers/sdd/task-3-report.md):
#     that composite path is ARCHITECTURALLY ABSENT from this standalone ovrtx runtime.  Those
#     flags/settings are interpreted ONLY by the Kit `omni.rtx.index_composite` extension (pure
#     Python; ships NO `.so`) + `omni.hydra.rtx` — NO bundled ovrtx binary references any of the
#     strings.  Enabling `compositeEnabled` via a carb setting (mechanism a) OR via the root
#     layer's `customLayerData.renderSettings` (mechanism b) had ZERO colour effect, and turning
#     on `omni:rtx:skip` merely removed the volume from the ONLY working path (Direct) → BLACK
#     (0 lit px).  So the Colormap COLOURS DEFER to a composite-capable ovrtx build (a Kit runtime
#     carrying omni.rtx.index_composite); M2 ships grayscale-Direct.  We still AUTHOR the Colormap
#     material here: it renders non-black today and is the exact structure that composite build
#     would activate — so `author_vdb_volume!` is the reusable authoring primitive, unchanged.

# Build the parallel `rgbaPoints` (float4 RGBA) + `xPoints` (float positions in [0,1]) arrays for
# an IndeX `Colormap` transfer function from a Makie colormap.  Opacity ramps with position so low
# density is transparent (matching the reference torus-volume-with-geometry.usda form).  Samples
# at most `npoints` colours evenly so the emitted USDA stays compact.
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

Build a self-contained USDA layer (PURE — no renderer) for a `.vdb`/`.nvdb` volume: a
`def Volume "Volume"` (the layer's `defaultPrim`) with a `field:<field>` rel to a child
`OpenVDBAsset`, plus an IndeX `nvindex:volume` `Colormap` material built from `colormap`/`colorrange`.

`field` is the single knob for the grid: it names the `field:<field>` rel, the `OpenVDBAsset`
prim, AND its `fieldName` (the grid name inside the VDB — e.g. `"torus_fog"` for `torus.vdb`).

Internal rel/connect targets are LAYER-RELATIVE (`</Volume/…>`); USD remaps them under the
reference target when `author_vdb_volume!` adds the layer at `prim_path`.  `prim_path` is accepted
for signature symmetry with `author_vdb_volume!` (which references the layer there); the layer
itself is self-contained, so `prim_path` does not appear in the emitted string.

Metadata is newline-separated / multi-line (the RenderProduct-prim gotcha — see usd.jl).
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

Author a `UsdVolVolume` from the on-disk `vdb_path` into `screen`'s already-open stage (call after
`author_root_from_scene!`/`_author_screen!` set up the camera/lights/render-product), returning
`prim_path`.  The volume renders through NVIDIA IndeX Direct — which requires IndeX to have been
enabled BEFORE `screen` was created (`OV._ensure_index`, driven by `OMNIVERSEMAKIE_INDEX_LIBS` /
`OMNIVERSEMAKIE_OVRTX_CONFIG`); otherwise this errors clearly rather than authoring a prim that
would silently render black.

`scene` is accepted for signature symmetry with M2's `volume!` (which will derive `colorrange`
from scene data limits); M1 does not read it.  See `_vdb_volume_usda` for `field`/colormap notes,
and for the verified M1 constraint that the colormap's transfer-function COLOURS require the
deferred composite path (M1 renders the volume via IndeX Direct's default grayscale TF).
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

Volumes M2 (Task 5) — LIVE volume DATA edit.  Re-write the plot's density field to a FRESH temp
`.nvdb` and RELOAD it into the open stage so NVIDIA IndeX shows the new data.  Called by
`push_to_ovrtx!`'s `:volume` branch when `plot[4][] = new_array` fires `changed[:volume]`.

The reload is a REMOVE + RE-REFERENCE, NOT a `filePath` write: the Task-5 verify-or-degrade spike
proved that writing a new `filePath` on the `OpenVDBAsset` does NOT update the render — IndeX loads
the grid into its own memory and evicts the source file ("orphaned"), so the on-disk change is never
re-read (spike: filePath write → centroid MOVED ≈ 0.01 px; remove_usd! + add_usd_reference! of a fresh
layer pointing at a FRESH temp path → MOVED ≈ 26 px).  A fresh temp path each edit also side-steps the
carb asset-cache serving the stale file.  `robj.usd_handle` is updated to the new layer's handle.

Temp files stay BOUNDED: the PRIOR temp `.nvdb` is deleted here (one live grid on disk at a time), and
`destroy_bindings!` removes the last one on close/delete.  Ranges/colormap/colorrange are re-read from
the plot (identical to the build), so a data edit keeps the same placement + (grayscale-Direct) TF.
"""
function reload_volume_data!(screen, robj, plot, scalars)
    r     = screen.renderer
    vprim = get(robj.meta, :volume_prim, robj.prim_path)
    data  = scalars isa Array{Float32,3} ? scalars : Float32.(scalars)
    xr = Makie.to_value(plot[1]); yr = Makie.to_value(plot[2]); zr = Makie.to_value(plot[3])
    origin = GeometryBasics.Point3f(first(xr), first(yr), first(zr))
    extent = GeometryBasics.Vec3f(last(xr) - first(xr), last(yr) - first(yr), last(zr) - first(zr))
    newtmp = tempname() * ".nvdb"
    NanoVDBWriter.save_nanovdb(newtmp, data, origin, extent)
    usda = _vdb_volume_usda(newtmp; prim_path = vprim, field = "density",
                            colormap = Makie.to_value(plot.colormap),
                            colorrange = _volume_colorrange(plot, data))
    OV.remove_usd!(r, robj.usd_handle)                        # drop the stale layer
    robj.usd_handle = OV.add_usd_reference!(r, usda, vprim)   # fresh layer → fresh temp; update handle
    old = get(robj.meta, :vdb_tmp, nothing)                  # GC the prior temp (keep the count bounded)
    old isa AbstractString && old != newtmp && rm(old; force = true)
    robj.meta[:vdb_tmp] = newtmp
    OV.reset!(r)                                              # restart RT2 accumulation for the new grid
    return nothing
end
