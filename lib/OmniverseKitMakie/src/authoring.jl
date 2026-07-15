# Makie Scene -> self-contained USDA stage for the Kit composite runtime.
#
# Reuses OmniverseMakie's emitters wholesale: `_vdb_volume_usda` (Volume prim
# + Colormap material from the Makie colormap), `_volume_colorrange`,
# `usda_light`, and NanoVDBWriter for the density payload.  What this file
# adds is only what standalone ovrtx has no notion of: the composite
# enablement (root-layer renderSettings + per-prim `nvindex:composite` /
# `omni:rtx:skip`) and a bound USD Camera authored from the Makie camera.

# Column-vector (GL/Makie) 4×4 -> USD row-vector matrix4d text (translation
# in the last row) — same convention as OmniverseMakie's _model_to_usd_xform.
function _usd_matrix4d(m::AbstractMatrix)
    t = permutedims(Float64.(Matrix(m)))
    row(i) = "(" * join((string(t[i, j]) for j in 1:4), ", ") * ")"
    return "( " * join((row(i) for i in 1:4), ", ") * " )"
end

# Camera-to-world from the scene camera.  Makie's view matrix is the GL
# world->eye transform (eye looks down -Z, +Y up) — the same convention as a
# USD camera, so inv(view) IS the USD camera transform.
_camera_to_world(scene::Makie.Scene) =
    inv(Matrix{Float64}(Makie.camera(scene).view[]))

function _camera_fov(scene::Makie.Scene)
    ctrl = Makie.cameracontrols(scene)
    hasproperty(ctrl, :fov) ? Float64(Makie.to_value(ctrl.fov)) : 45.0
end

# 35mm-ish default aperture; focal length from the vertical fov so the USD
# camera frames what the Makie camera saw.
const _VERTICAL_APERTURE = 15.2908

function _camera_usda(scene::Makie.Scene)
    focal = _VERTICAL_APERTURE / (2 * tan(deg2rad(_camera_fov(scene)) / 2))
    return """
    def Camera "Camera"
    {
        float2 clippingRange = (0.01, 10000000)
        float focalLength = $(Float32(focal))
        float horizontalAperture = 20.955
        float verticalAperture = $(_VERTICAL_APERTURE)
        token projection = "perspective"
        matrix4d xformOp:transform = $(_usd_matrix4d(_camera_to_world(scene)))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
"""
end

# OmniverseMakie's whole-scene emitter handles the compute-graph lights,
# camera-relative directions, and the empty→default-Sun fallback; it takes
# the camera-to-world in row-vector layout.
_lights_usda(scene::Makie.Scene) =
    OM.lights_usda(scene; cam_to_world = permutedims(_camera_to_world(scene)))

# Depth-first over the plot tree (recipes nest); volumes collected, every
# other ATOMIC plot recorded so the caller can warn once.
function _collect_plots!(volumes, skipped, plots)
    for p in plots
        if p isa Makie.Volume
            push!(volumes, p)
        elseif isempty(p.plots)
            push!(skipped, nameof(typeof(p)))
        else
            _collect_plots!(volumes, skipped, p.plots)
        end
    end
end

# Default volume-payload writer (pure Julia): NanoVDBWriter .nvdb — what the
# standalone backend uses, and enough for text-level authoring tests.  A LIVE
# KitScreen must NOT use this: Kit's IndeX composite importer fails to fetch
# NanoVDB data (ours AND Warp's own samples) while classic OpenVDB works, so
# screen.jl supplies an RPC-backed writer that converts server-side to .vdb.
function _nvdb_volume_writer(workdir::AbstractString)
    return function (i, scalars, origin, extent)
        p = joinpath(workdir, "volume_$(i).nvdb")
        NanoVDBWriter.save_nanovdb(p, scalars, origin, extent)
        return p
    end
end

# One volume plot -> payload (via `volume_writer`) + fragment layer on disk
# (in `workdir`), returning the stage-side prim block: a reference to the
# fragment with the two composite markers (and the plot's model transform
# when non-identity) layered on the referencing prim.  Mirrors the data
# conversions of OmniverseMakie's `author_usd_prim!(::Makie.Volume)` exactly.
function _volume_prim_usda(plot::Makie.Volume, i::Int, workdir::AbstractString,
                           volume_writer)
    scalars = Float32.(Makie.to_value(plot[4]))
    if all(iszero, scalars)
        @warn "KitScreen: volume plot $i is all-zero (IndeX renders it fully transparent); skipping"
        return ""
    end
    xr = Makie.to_value(plot[1]); yr = Makie.to_value(plot[2]); zr = Makie.to_value(plot[3])
    origin = Makie.Point3f(first(xr), first(yr), first(zr))
    extent = Makie.Vec3f(last(xr) - first(xr), last(yr) - first(yr), last(zr) - first(zr))
    vol_path = volume_writer(i, scalars, origin, extent)
    frag = OM._vdb_volume_usda(vol_path;
        colormap = Makie.to_value(plot.colormap),
        colorrange = OM._volume_colorrange(plot, scalars),
        bounds = (origin, origin .+ extent))
    fragpath = joinpath(workdir, "volume_$(i).usda")
    write(fragpath, frag)
    model = Matrix(Makie.to_value(plot.model))
    xform = isapprox(model, Matrix{Float64}(I, 4, 4)) ? "" :
        "        matrix4d xformOp:transform = $(_usd_matrix4d(model))\n" *
        "        uniform token[] xformOpOrder = [\"xformOp:transform\"]\n"
    return """
    def "Volume$(i)" (
        prepend references = @$(fragpath)@
    )
    {
        custom bool nvindex:composite = 1
        custom bool omni:rtx:skip = 1
$(xform)    }
"""
end

"""
    stage_usda(scene::Makie.Scene; workdir=mktempdir(...), volume_writer=...) -> String

Author `scene` as a self-contained USDA stage for the Kit composite runtime:
every `volume!` plot becomes a volume payload (via `volume_writer(i, scalars,
origin, extent) -> path`; default: NanoVDBWriter `.nvdb`) + referenced Volume
fragment (written into `workdir`) marked `nvindex:composite`/`omni:rtx:skip`,
lights and the camera come from the scene, and the root layer carries the
composite renderSettings + `boundCamera`.  Non-volume atomic plots are
skipped with one warning — they are the standalone-ovrtx backend's job (v1
scope).  NOTE: a live `KitScreen` overrides `volume_writer` to produce `.vdb`
server-side — Kit's composite importer cannot read NanoVDB payloads.
"""
function stage_usda(scene::Makie.Scene;
        workdir::AbstractString = mktempdir(; prefix = "omk_stage_", cleanup = false),
        volume_writer = _nvdb_volume_writer(workdir))
    mkpath(workdir)
    volumes = Makie.Volume[]
    skipped = Symbol[]
    _collect_plots!(volumes, skipped, scene.plots)
    isempty(skipped) ||
        @warn "KitScreen v1 renders volume plots only; skipping" plot_types = unique(skipped)
    isempty(volumes) &&
        @warn "stage_usda: no volume plots found in the scene — the stage will be empty"
    vol_blocks = join((_volume_prim_usda(p, i, workdir, volume_writer)
                       for (i, p) in enumerate(volumes)))
    return """#usda 1.0
(
    customLayerData = {
        dictionary cameraSettings = {
            string boundCamera = "/World/Camera"
        }
        dictionary renderSettings = {
            int "rtx:index:compositeDepthMode" = 3
            bool "rtx:index:compositeEnabled" = 1
        }
    }
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{
$(vol_blocks)$(_lights_usda(scene))$(_camera_usda(scene))}
"""
end
