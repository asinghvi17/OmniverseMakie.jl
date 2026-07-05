# Sensor simulation — `lidar!` / `radar!` recipes rendered through ovrtx's native RTX sensor
# pipeline (OmniLidar/OmniRadar prims + PointCloud RenderProducts), and `step_sensors!` to
# advance sensor simulation time and read the resulting point clouds.
#
# Included AFTER compute.jl + screen.jl + usdplot.jl: adds methods to `author_usd_prim!`,
# `consumed_inputs`, `bind_hot_attributes!`, `_usdplot_model`, and `_is_sensor_plot` (generics
# defined in compute.jl / screen.jl).
#
# Spike-validated (docs/superpowers/specs/2026-07-05-sensors-lidar-radar-design.md, SPIKE
# RESULTS):
#  • A live-referenced layer carrying sensor + RenderProduct + RenderVar WORKS (no
#    author-before-display restriction) — one reference per sensor plot, the usdplot pattern.
#  • ovrtx DISCARDS a render product's accumulated history when a step's product set omits it,
#    so `step_sensors!` steps {camera + all sensors} together (a sensors-only set makes the next
#    camera frame ~8× noisier — measured).
#  • NO warmup needed: the FIRST sensor step returns a full scan.  `instantLidar = true` (our
#    default) emits a FULL scan per step at ANY dt; default (non-instant) mode emits clean
#    dt-proportional partial scans.
#  • Returns work WITHOUT any material bindings (nonvisual `omni:simready` tokens are a fidelity
#    upgrade, not a requirement).
#  • Motion BVH is required for correct MOVING-object returns only (static scenes work without);
#    the Screen auto-enables it when the displayed scene contains sensors, and authoring a
#    sensor on a non-BVH renderer warns once.
#  • Output is SENSOR-frame by default (+X forward); `Coordinates` arrives `[3,N]` row-major →
#    a Julia `(N,3)` Matrix{Float32} via `OV.read_pointcloud`'s reversed-dims copy.

# =============================================================================================
# The recipes
# =============================================================================================

"""
    lidar([origin::Point3f]; frame_rate, channels, instant, output_frame, usd_attributes, kwargs...)
    lidar!(ax_or_scene, [origin]; kwargs...)

Attach a simulated spinning LiDAR to the scene, traced by ovrtx's native RTX sensor pipeline
against the SAME stage the camera renders.  The sensor draws nothing; its measured point cloud
lands in the plain `Observable` returned by [`sensor_returns`](@ref) every time
[`step_sensors!`](@ref) advances the simulation.

`origin` is the sensor position in data space (default `Point3f(0)`); the plot's
`translate!`/`rotate!` compose on top and move the sensor live.  The sensor's forward axis is
**+X** (ovrtx convention).  Point coordinates are SENSOR-frame by default
(`output_frame = :sensor`); pass `:world` for stage-world coordinates.

**Units: a scene containing sensors is authored as a METER stage (1 data unit = 1 meter)** —
ovrtx's sensor engine works in physical meters (default lidar range 0.3–200 m, ~±4° elevation
fan), so data units and sensor physics coincide: a cube 10 units away returns points at
x ≈ 10.  Sensor-free scenes keep the default centimeter stage unchanged.

```julia
sensor = lidar!(ax, Point3f(0, 0, 1.5); channels = [:coordinates, :intensity])
display(fig)                                  # motion BVH auto-enabled (sensor in scene)
scatter(fig[1, 2], lift(r -> r.points, sensor_returns(sensor)))
for t in timeline
    # ... move scene objects ...
    step_sensors!(screen, 1/10)               # one full scan → sensor_returns updates
end
```

Requires the OmniverseMakie backend (offscreen, `interactive_display`, or a `replace_scene!`
panel); in a plain GLMakie window the sensor renders nothing and never updates.
"""
@recipe Lidar (origin,) begin
    "Scan rate in Hz (`omni:sensor:frameRate`); the natural `step_sensors!` dt is `1/frame_rate`."
    frame_rate = 10.0
    "Requested PointCloud channels: `:coordinates`, `:intensity`, `:time_offset_ns`, `:flags` (`Counts` is always delivered)."
    channels = [:coordinates, :intensity]
    "One FULL scan per sensor step regardless of dt (`omni:sensor:Core:instantLidar`). `false` = time-resolved dt-proportional partial scans (advanced; interacts with image-step history discard and accumulation resets)."
    instant = true
    "Point-coordinate frame: `:sensor` (default, +X forward) or `:world` (stage coordinates)."
    output_frame = :sensor
    "Raw `omni:sensor:*` schema attributes for tuned sensor models, e.g. `Dict(\"omni:sensor:Core:farRangeM\" => 400.0f0)`.  Author-time only."
    usd_attributes = Dict{String,Any}()
    Makie.mixin_generic_plot_attributes()...
end

"""
    radar([origin::Point3f]; frame_rate, channels, output_frame, usd_attributes, kwargs...)
    radar!(ax_or_scene, [origin]; kwargs...)

Attach a simulated radar (`OmniRadar` + the generic WpmDmat wave-propagation model) to the
scene.  Same lifecycle as [`lidar`](@ref): pose from `origin` + plot transforms (+X forward),
detections land in [`sensor_returns`](@ref) on every [`step_sensors!`](@ref).  Radar channels
add `:rcs` (radar cross-section) and `:radial_velocity` (m/s, negative = approaching).
"""
@recipe Radar (origin,) begin
    "Scan rate in Hz (`omni:sensor:frameRate`); the natural `step_sensors!` dt is `1/frame_rate`."
    frame_rate = 10.0
    "Requested PointCloud channels: `:coordinates`, `:rcs`, `:radial_velocity`, `:flags` (`Counts` is always delivered)."
    channels = [:coordinates, :rcs, :radial_velocity]
    "Detection-coordinate frame: `:sensor` (default, +X forward) or `:world` (stage coordinates)."
    output_frame = :sensor
    "Raw `omni:sensor:*` schema attributes for tuned sensor models.  Author-time only."
    usd_attributes = Dict{String,Any}()
    Makie.mixin_generic_plot_attributes()...
end

const SensorPlot = Union{Lidar,Radar}

# Makie rejects zero-positional-arg plot calls, so `origin` is the one argument; these
# convenience methods default it to the data-space origin.
lidar!(ax; kwargs...) = lidar!(ax, Point3f(0); kwargs...)
radar!(ax; kwargs...) = radar!(ax, Point3f(0); kwargs...)

Makie.convert_arguments(::Type{<:SensorPlot}, p::Makie.VecTypes{3}) = (Point3f(p),)

# NO child plots → atomic via the backend walker (the USDPlot pattern).  An origin-only recipe
# never emits `:model_f32c` on its own, so register it here (the usdplot lesson) — then
# translate!/rotate! drive the sensor through the ordinary diff pipeline.
function Makie.plot!(p::SensorPlot)
    haskey(p.attributes, :model_f32c) || Makie.register_model_f32c!(p.attributes)
    return p
end

# Sensors have no visual extent; a small box at the origin keeps axis limits sane without
# inflating them (the sensor's REACH is not data).
Makie.data_limits(p::SensorPlot) = Rect3d(Point3d(Makie.to_value(p[1])) .- 0.05, Vec3d(0.1))

# The diff node tracks the composed world transform (→ `omni:xform` on the plot prim) and
# visibility; the sensor model itself is author-time-only (NVIDIA: output-defining attributes
# are not runtime-mutable).
consumed_inputs(::SensorPlot) = [:model_f32c, :visible]

# A sensor's transform is a one-shot `omni:xform` write (user-rate) — no persistent hot-path
# binding, same choice as Volume/USDPlot.
bind_hot_attributes!(screen, robj::OvrtxRObj, ::SensorPlot, args) = robj

# The OmniLidar/OmniRadar models natively spin about the sensor's +Y axis; NVIDIA's Z-up
# example scenes mount every sensor with `xformOp:rotateXYZ = (90, 0, -90)` so the scan
# sweeps HORIZONTALLY and the model's canonical output frame (x forward, z up) aligns with
# the stage axes.  Without this mount rotation the scan is a vertical fan that misses the
# scene (measured: 757 stray points vs 203k on the same scene with it).  Rz(-90°)·Rx(90°)
# reproduces the USD rotateXYZ order (X applied first).
const _SENSOR_BASE_ROT = Makie.rotationmatrix_z(-Float32(π) / 2) *
                         Makie.rotationmatrix_x(Float32(π) / 2)

# The sensor MOUNT transform: plot model ∘ `origin` translation.  Because the model's output
# frame aligns with the mount frame (spike-verified: an unrotated mount reads the ground at
# z = −height), this is exactly the sensor→data-space map reported as `pose` in
# `sensor_returns`.
_sensor_mount(plot::SensorPlot, model) =
    model * Makie.translationmatrix(Vec3f(Makie.to_value(plot[1])))

# What gets WRITTEN as `omni:xform`: mount ∘ base rotation (innermost, sensor-local) — the
# same hook usdplot uses for its up-axis fold, so author time and live pushes agree.
# `origin` itself is not diffed: move a live sensor with translate!/rotate!, not by mutating
# the argument (documented).
_usdplot_model(plot::SensorPlot, model) = _sensor_mount(plot, model) * _SENSOR_BASE_ROT

# Screen's motion-BVH auto-detection hook (generic `_is_sensor_plot(::Any) = false` in screen.jl).
_is_sensor_plot(::SensorPlot) = true

# =============================================================================================
# Channel maps
# =============================================================================================

# kwarg Symbol → (USD PointCloud channel, `sensor_returns` field).  `:coordinates` becomes
# `points::Vector{Point3f}`; every other channel keeps its kwarg name as the returns field.
const _LIDAR_CHANNELS = Dict(
    :coordinates    => "Coordinates",
    :intensity      => "Intensity",
    :time_offset_ns => "TimeOffsetNs",
    :flags          => "Flags",
)
const _RADAR_CHANNELS = Dict(
    :coordinates     => "Coordinates",
    :rcs             => "RCS",
    :radial_velocity => "RadialVelocityMs",
    :flags           => "Flags",
)

_channel_map(::Lidar) = _LIDAR_CHANNELS
_channel_map(::Radar) = _RADAR_CHANNELS
_sensor_kind(::Lidar) = "lidar"
_sensor_kind(::Radar) = "radar"

function _validated_channels(plot::SensorPlot)
    chmap = _channel_map(plot)
    chans = collect(Symbol.(Makie.to_value(plot.channels)))
    isempty(chans) && throw(ArgumentError(
        "$(_sensor_kind(plot))!: `channels` must not be empty (at minimum [:coordinates])."))
    for ch in chans
        haskey(chmap, ch) || throw(ArgumentError(
            "$(_sensor_kind(plot))!: unknown channel $(repr(ch)) — valid channels are " *
            "$(join(sort!(collect(keys(chmap))), ", "))."))
    end
    return chans
end

# =============================================================================================
# USDA layer emission (one self-contained reference per sensor plot)
# =============================================================================================

# `usd_attributes` passthrough: raw `omni:sensor:*` names, values typed by Julia type.  Names and
# string values are validated (the layer is authored from a string — no injection through a
# stray quote/newline).
function _sensor_attr_line(name::AbstractString, value)
    occursin(r"^[A-Za-z0-9_:.]+$", name) && startswith(name, "omni:sensor:") ||
        throw(ArgumentError("sensor `usd_attributes` keys must be omni:sensor:* attribute names, got $(repr(name))."))
    if value isa Bool
        return "        bool $(name) = $(value)"
    elseif value isa Integer
        return "        int $(name) = $(value)"
    elseif value isa Real
        return "        float $(name) = $(Float32(value))"
    elseif value isa NTuple{2,Real}
        return "        double2 $(name) = ($(Float64(value[1])), $(Float64(value[2])))"
    elseif value isa AbstractString
        occursin(r"^[A-Za-z0-9_]+$", value) ||
            throw(ArgumentError("sensor `usd_attributes` string values must be plain tokens, got $(repr(value)) for $(name)."))
        return "        token $(name) = \"$(value)\""
    end
    throw(ArgumentError("sensor `usd_attributes` values must be Bool, Integer, Real, NTuple{2,Real}, " *
                        "or a token String; got $(typeof(value)) for $(name)."))
end

_sensor_prim_type(::Lidar)  = "OmniLidar"
_sensor_prim_type(::Radar)  = "OmniRadar"
_sensor_api(::Lidar)        = "OmniSensorGenericLidarCoreAPI"
_sensor_api(::Radar)        = "OmniSensorGenericRadarWpmDmatAPI"
_sensor_attr_prefix(::Lidar) = "omni:sensor:Core"
_sensor_attr_prefix(::Radar) = "omni:sensor:WpmDmat"

# The relative product path under the plot prim (the layer composes its defaultPrim subtree AT
# the plot prim path, so `<plot_prim>/Products/Product` is the step!/read path).
const _SENSOR_PRODUCT_SUFFIX = "/Products/Product"

"""
    _sensor_layer_usda(plot::SensorPlot) -> String

Emit the sensor's self-contained reference layer: the sensor prim (+ API schema, frame rate,
coordinate/frame attributes, `usd_attributes` passthrough), a RenderProduct whose `camera` rel
targets the sensor, and a PointCloud RenderVar with the requested channels.  Rel targets are
layer-absolute (repo convention) and get remapped to the plot prim path on composition.
"""
function _sensor_layer_usda(plot::SensorPlot)
    chans  = _validated_channels(plot)
    chmap  = _channel_map(plot)
    prefix = _sensor_attr_prefix(plot)

    rate = Float64(Makie.to_value(plot.frame_rate))
    rate > 0 || throw(ArgumentError("$(_sensor_kind(plot))!: `frame_rate` must be positive, got $(rate)."))

    frame = Makie.to_value(plot.output_frame)
    frame in (:sensor, :world) || throw(ArgumentError(
        "$(_sensor_kind(plot))!: `output_frame` must be :sensor or :world, got $(repr(frame))."))

    lines = String[]
    push!(lines, "        token $(prefix):elementsCoordsType = \"CARTESIAN\"")
    push!(lines, "        double2 omni:sensor:frameRate = ($(rate), 1)")
    # SENSOR is the schema default; author the token only for :world (keeps the default
    # emission minimal and golden-stable).
    frame === :world &&
        push!(lines, "        token $(prefix):outputFrameOfReference = \"WORLD\"")
    if plot isa Lidar && Makie.to_value(plot.instant)
        push!(lines, "        bool omni:sensor:Core:instantLidar = true")
    end
    extra = Makie.to_value(plot.usd_attributes)
    for name in sort!(collect(keys(extra)))          # deterministic order (goldens)
        push!(lines, _sensor_attr_line(String(name), extra[name]))
    end

    channel_list = join(("\"$(chmap[ch])\"" for ch in chans), ", ")

    return """
#usda 1.0
(
    defaultPrim = "Sensor"
)

def Xform "Sensor"
{
    def $(_sensor_prim_type(plot)) "Sensor" (
        prepend apiSchemas = ["$(_sensor_api(plot))"]
    )
    {
$(join(lines, "\n"))
    }

    def "Products"
    {
        def RenderProduct "Product"
        {
            rel camera = </Sensor/Sensor>
            rel orderedVars = [</Sensor/Vars/PointCloud>]
        }
    }

    def "Vars"
    {
        def RenderVar "PointCloud"
        {
            uniform string sourceName = "PointCloud"
            string[] channels = [$(channel_list)]
        }
    }
}
"""
end

# =============================================================================================
# Authoring
# =============================================================================================

"""
    author_usd_prim!(screen, scene, plot::SensorPlot, args) -> OvrtxRObj

Compose the sensor's self-contained layer (sensor prim + RenderProduct + PointCloud RenderVar)
as a reference at the plot's prim path, and write the initial `omni:xform` (plot model ∘
`origin` translation).  Live `translate!`/`rotate!` ride the ordinary `:model_f32c` diff.
Structural add/remove flows through the normal robj lifecycle (accumulate-mode resets,
teardown, `delete!`).
"""
function author_usd_prim!(screen, scene, plot::SensorPlot, args)
    prim = plot_prim_path(screen.scene2scope, scene, plot)
    usda = _sensor_layer_usda(plot)
    h    = OV.add_usd_reference!(screen.renderer, usda, prim)
    robj = OvrtxRObj(prim, h)
    OV.write_xform!(screen.renderer, prim,
                    _model_to_usd_xform(_usdplot_model(plot, args[:model_f32c])))
    if !screen.renderer.motion_bvh
        @warn "OmniverseMakie: a $(_sensor_kind(plot)) sensor was added to a Screen created \
               without sensor support — its stage is the centimeter stage (sensor physics \
               work in METERS: default ranges 0.3–200 m = 30–20000 data units, so returns \
               will be missing or mis-scaled) and motion BVH is off (moving-object returns \
               incorrect). Add the sensor before `display`, or pass `sensors = true` to \
               `activate!`/`Screen`." maxlog = 1
    end
    return robj
end

# =============================================================================================
# sensor_returns — the measurement Observable
# =============================================================================================

# Plain Observables keyed weakly by plot (the `_USD_BINDINGS` pattern): plot attributes are
# ComputePipeline `Computed`s, and a plain Observable keeps user-side `lift`/`on` free of
# framework coupling.  Created lazily with an EMPTY NamedTuple shaped by the plot's channels.
const _SENSOR_RETURNS = WeakKeyDict{Makie.AbstractPlot,Observable}()

_empty_channel(ch::Symbol) =
    ch === :time_offset_ns ? Int64[] :
    ch === :flags          ? UInt32[] :
    Float32[]                                        # intensity / rcs / radial_velocity

function _empty_returns(plot::SensorPlot)
    fields = Symbol[]; vals = Any[]
    for ch in _validated_channels(plot)
        if ch === :coordinates
            push!(fields, :points); push!(vals, Point3f[])
        else
            push!(fields, ch); push!(vals, _empty_channel(ch))
        end
    end
    push!(fields, :counts); push!(vals, 0)
    push!(fields, :pose);   push!(vals, Mat4f(LinearAlgebra.I))
    return NamedTuple{Tuple(fields)}(Tuple(vals))
end

"""
    sensor_returns(plot::Union{Lidar,Radar}) -> Observable{<:NamedTuple}

The sensor's measurement observable, updated by every [`step_sensors!`](@ref).  Fields follow
the plot's `channels` (`:coordinates` → `points::Vector{Point3f}`, others keep their kwarg
name), plus `counts::Int` (valid returns this scan) and `pose::Mat4f` (the sensor's world
transform at the step — use it to place `:sensor`-frame points in world/data space).  Safe to
`lift`/`on` before display; holds an empty scan until the first step.
"""
sensor_returns(plot::SensorPlot) =
    get!(() -> Observable{Any}(_empty_returns(plot)), _SENSOR_RETURNS, plot)

# Build the returns NamedTuple from the step's (possibly several, partial-scan) PointCloud
# frames: per-channel vcat across frames, `:coordinates` → Vector{Point3f}.
function _returns_from_frames(plot::SensorPlot, frames::Vector{<:NamedTuple})
    fields = Symbol[]; vals = Any[]
    chmap = _channel_map(plot)
    total = sum((haskey(f, :Counts) && !isempty(f.Counts)) ? Int(first(f.Counts)) : 0
                for f in frames; init = 0)
    for ch in _validated_channels(plot)
        usd = Symbol(chmap[ch])
        arrs = [getfield(f, usd) for f in frames if haskey(f, usd)]
        if ch === :coordinates
            pts = Point3f[]
            for A in arrs
                (A isa AbstractMatrix && size(A, 2) == 3) || continue
                append!(pts, Point3f.(view(A, :, 1), view(A, :, 2), view(A, :, 3)))
            end
            push!(fields, :points); push!(vals, pts)
        else
            push!(fields, ch)
            push!(vals, isempty(arrs) ? _empty_channel(ch) : reduce(vcat, map(vec, arrs)))
        end
    end
    push!(fields, :counts); push!(vals, total)
    push!(fields, :pose)
    push!(vals, Mat4f(_sensor_mount(plot, Makie.to_value(plot.model_f32c))))
    return NamedTuple{Tuple(fields)}(Tuple(vals))
end

# =============================================================================================
# step_sensors! — advance sensor simulation time
# =============================================================================================

# Live sensors = SensorPlots in the scene tree that hold an authored robj (the scene tree is
# the source of truth — no registry to go stale on delete!/empty!).  Walked from the ROOT
# scene: an embedded Screen's `.scene` is the camera scene while its plots live in the target
# scene; `plot2robj` membership already scopes the walk to THIS screen's sensors.
function _live_sensors(screen)
    out = Tuple{Makie.AbstractPlot,String}[]
    screen.scene === nothing && return out
    _walk_sensors!(out, screen, Makie.root(screen.scene))
    return out
end

function _walk_sensors!(out, screen, scene::Makie.Scene)
    for p in scene.plots
        p isa SensorPlot || continue
        robj = get(screen.plot2robj, objectid(p), nothing)
        robj === nothing && continue
        push!(out, (p, robj.prim_path * _SENSOR_PRODUCT_SUFFIX))
    end
    foreach(child -> _walk_sensors!(out, screen, child), scene.children)
    return out
end

"""
    step_sensors!(screen::Screen, dt::Real) -> nothing

Advance sensor simulation by `dt` seconds: push any pending scene edits (same sync as
`colorbuffer`), run ONE `OV.step!` over `{camera product, every live sensor product}` at
physical `dt`, then read each sensor's PointCloud and update its [`sensor_returns`](@ref)
observable.  With the default `instant = true` each call yields one FULL scan of the current
scene regardless of `dt`; with `instant = false` a step covers `dt` of scan time
(dt-proportional partial output).

The camera product is included deliberately: ovrtx discards the accumulated history of every
product a step omits (measured: a sensors-only step makes the next camera frame ~8× noisier).

Errors if the screen has no live sensors, or if a registered sensor yields no PointCloud var
(an authoring bug must fail loudly — pixel/tensor evidence is the only ovrtx write oracle).
"""
function step_sensors!(screen::Screen, dt::Real)
    scene = screen.scene
    scene === nothing && error("step_sensors!: screen has no scene")
    cam_scene = something(_scene_for_camera(scene), scene)
    screen.authored || _author_screen!(screen, cam_scene, scene)
    sensors = _live_sensors(screen)
    isempty(sensors) && error("step_sensors!: no lidar!/radar! sensors in this screen's scene")

    # Same edit-sync + reset contract as colorbuffer, so a scan taken without an intervening
    # render still sees the current scene state.  (OV.reset! rewinds sensor sim TIME too —
    # irrelevant under instant mode, documented for instant=false.)
    need_reset = _sync_and_needs_reset!(screen, cam_scene)
    need_reset && OV.reset!(screen.renderer)

    products = String[screen.product]
    for (_, prod) in sensors
        push!(products, prod)
    end
    sr = OV.step!(screen.renderer, products; dt = Float64(dt))
    try
        for (plot, prod) in sensors
            frames = OV.read_pointcloud(sr, prod)
            if isempty(frames) && plot isa Lidar && Makie.to_value(plot.instant)
                # An instant-mode lidar step ALWAYS yields a frame (spike-proven) — absence
                # means broken product/var wiring, and ovrtx won't say so itself (the
                # silent-ignore hazard: tensor evidence is the only write oracle).
                error("step_sensors!: sensor product $(prod) produced no PointCloud frame — " *
                      "authoring bug (the sensor/product/var wiring is broken).")
            end
            # Radar / non-instant lidar: zero frames is a valid partial-scan outcome → an
            # empty-scan update, not an error.
            plot_returns = sensor_returns(plot)
            plot_returns[] = _returns_from_frames(plot, frames)   # assignment, never notify()
        end
    finally
        close(sr)
    end
    return nothing
end
