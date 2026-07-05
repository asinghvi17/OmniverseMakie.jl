# Sensor simulation (lidar!/radar!) — pure recipe/emission contract + end-to-end GPU point
# clouds through ovrtx's native RTX sensor pipeline.
#
# Pure tier: layer emission goldens (prim/API/frameRate/instant/channels/attribute passthrough),
# channel + kwarg validation, returns-observable shape, origin fold, motion-BVH scene detection,
# ScreenConfig contract.
# GPU tier (one subprocess, 2 renderer creations): a lidar + radar on a known scene → full-scan
# counts, sensor-frame range bands (ground at −h, cube at ~10 m), observable updates, pose move
# shifts the scan, delete! drops the sensor from stepping, steady-state step_sensors! fires no
# image reset, motion-BVH auto-detection, and the post-display (no-BVH) insert path with its
# one-time warn.

using Test
include(joinpath(@__DIR__, "..", "helpers.jl"))

import OmniverseMakie as OM
using OmniverseMakie
using Makie

@testset "sensor recipes — pure contract" begin
    fig = Figure()
    ls  = LScene(fig[1, 1])

    @testset "lidar layer emission" begin
        p = lidar!(ls, Point3f(0, 0, 1.5);
                   channels = [:coordinates, :intensity],
                   usd_attributes = Dict{String,Any}("omni:sensor:Core:farRangeM" => 400.0f0,
                                                     "omni:sensor:Core:scanRateBaseHz" => 20))
        u = OM._sensor_layer_usda(p)
        @test occursin("def OmniLidar \"Sensor\"", u)
        @test occursin("prepend apiSchemas = [\"OmniSensorGenericLidarCoreAPI\"]", u)
        @test occursin("token omni:sensor:Core:elementsCoordsType = \"CARTESIAN\"", u)
        @test occursin("double2 omni:sensor:frameRate = (10.0, 1)", u)
        @test occursin("bool omni:sensor:Core:instantLidar = true", u)
        # usd_attributes passthrough, typed + deterministically ordered (farRangeM < scanRate…)
        @test occursin("float omni:sensor:Core:farRangeM = 400.0", u)
        @test occursin("int omni:sensor:Core:scanRateBaseHz = 20", u)
        @test findfirst("farRangeM", u).start < findfirst("scanRateBaseHz", u).start
        @test occursin("rel camera = </Sensor/Sensor>", u)
        @test occursin("rel orderedVars = [</Sensor/Vars/PointCloud>]", u)
        @test occursin("uniform string sourceName = \"PointCloud\"", u)
        @test occursin("string[] channels = [\"Coordinates\", \"Intensity\"]", u)
        # :sensor output frame is the schema default — NOT authored (golden-minimal emission)
        @test !occursin("outputFrameOfReference", u)
    end

    @testset "lidar emission variants" begin
        u_noinstant = OM._sensor_layer_usda(lidar!(ls; instant = false))
        @test !occursin("instantLidar", u_noinstant)
        u_world = OM._sensor_layer_usda(lidar!(ls; output_frame = :world))
        @test occursin("token omni:sensor:Core:outputFrameOfReference = \"WORLD\"", u_world)
        u_rate = OM._sensor_layer_usda(lidar!(ls; frame_rate = 20))
        @test occursin("double2 omni:sensor:frameRate = (20.0, 1)", u_rate)
    end

    @testset "radar layer emission" begin
        r = radar!(ls, Point3f(0, 0, 1.5))
        u = OM._sensor_layer_usda(r)
        @test occursin("def OmniRadar \"Sensor\"", u)
        @test occursin("prepend apiSchemas = [\"OmniSensorGenericRadarWpmDmatAPI\"]", u)
        @test occursin("token omni:sensor:WpmDmat:elementsCoordsType = \"CARTESIAN\"", u)
        @test occursin("string[] channels = [\"Coordinates\", \"RCS\", \"RadialVelocityMs\"]", u)
        @test !occursin("instantLidar", u)                     # lidar-only attribute
        u_world = OM._sensor_layer_usda(radar!(ls; output_frame = :world))
        @test occursin("token omni:sensor:WpmDmat:outputFrameOfReference = \"WORLD\"", u_world)
    end

    @testset "validation fails loud" begin
        @test_throws ArgumentError OM._sensor_layer_usda(lidar!(ls; channels = [:bogus]))
        @test_throws ArgumentError OM._sensor_layer_usda(lidar!(ls; channels = [:rcs]))       # radar-only
        @test_throws ArgumentError OM._sensor_layer_usda(radar!(ls; channels = [:intensity])) # lidar-only
        @test_throws ArgumentError OM._sensor_layer_usda(lidar!(ls; channels = Symbol[]))
        @test_throws ArgumentError OM._sensor_layer_usda(lidar!(ls; frame_rate = 0.0))
        @test_throws ArgumentError OM._sensor_layer_usda(lidar!(ls; output_frame = :screen))
        # usd_attributes hygiene: non-sensor keys, injection-y strings, unsupported types
        @test_throws ArgumentError OM._sensor_layer_usda(
            lidar!(ls; usd_attributes = Dict{String,Any}("omni:rtx:rendermode" => "X")))
        @test_throws ArgumentError OM._sensor_layer_usda(
            lidar!(ls; usd_attributes = Dict{String,Any}("omni:sensor:Core:x\"y" => 1)))
        @test_throws ArgumentError OM._sensor_layer_usda(
            lidar!(ls; usd_attributes = Dict{String,Any}("omni:sensor:Core:t" => "a\"b")))
        @test_throws ArgumentError OM._sensor_layer_usda(
            lidar!(ls; usd_attributes = Dict{String,Any}("omni:sensor:Core:t" => [1, 2])))
    end

    @testset "attribute value typing" begin
        @test OM._sensor_attr_line("omni:sensor:Core:b", true)         == "        bool omni:sensor:Core:b = true"
        @test OM._sensor_attr_line("omni:sensor:Core:i", 7)            == "        int omni:sensor:Core:i = 7"
        @test OM._sensor_attr_line("omni:sensor:Core:f", 2.5)          == "        float omni:sensor:Core:f = 2.5"
        @test OM._sensor_attr_line("omni:sensor:Core:d2", (10, 1))     == "        double2 omni:sensor:Core:d2 = (10.0, 1.0)"
        @test OM._sensor_attr_line("omni:sensor:Core:t", "SPHERICAL")  == "        token omni:sensor:Core:t = \"SPHERICAL\""
    end

    @testset "returns observable + recipe mechanics" begin
        p = lidar!(ls, Point3f(0, 0, 1.5); channels = [:coordinates, :intensity, :flags])
        r = sensor_returns(p)
        @test r === sensor_returns(p)                          # identity-stable accessor
        @test keys(r[]) == (:points, :intensity, :flags, :counts, :pose)
        @test r[].points isa Vector{Point3f} && isempty(r[].points)
        @test r[].counts == 0
        @test r[].pose == Makie.Mat4f(Makie.LinearAlgebra.I)

        rad = radar!(ls)
        @test keys(sensor_returns(rad)[]) == (:points, :rcs, :radial_velocity, :counts, :pose)

        # origin folds into the omni:xform model (author + live writes agree)
        m = OM._usdplot_model(p, Makie.Mat4f(Makie.LinearAlgebra.I))
        @test m[Vec(1, 2, 3), 4] == Vec3f(0, 0, 1.5)
        # the WRITTEN xform carries the Y-spin mount rotation; the sensor→data `pose` does NOT
        # (the model's output frame aligns with the mount frame)
        @test m[Vec(1, 2, 3), Vec(1, 2, 3)] ≈ OM._SENSOR_BASE_ROT[Vec(1, 2, 3), Vec(1, 2, 3)]
        mount = OM._sensor_mount(p, Makie.Mat4f(Makie.LinearAlgebra.I))
        @test mount[Vec(1, 2, 3), Vec(1, 2, 3)] ≈ Makie.Mat3f(Makie.LinearAlgebra.I)
        @test mount[Vec(1, 2, 3), 4] == Vec3f(0, 0, 1.5)

        @test OM.consumed_inputs(p) == [:model_f32c, :visible]
        @test haskey(p.attributes, :model_f32c)                # registered by plot! (usdplot lesson)
        bb = Makie.data_limits(p)
        @test all(Makie.widths(bb) .≈ 0.1)                     # sensors don't inflate axis limits

        @test OM._is_sensor_plot(p) && OM._is_sensor_plot(rad)
        @test !OM._is_sensor_plot(mesh!(ls, Rect3f(Point3f(0), Vec3f(1))))
        @test OM._scene_contains_sensors(Makie.get_scene(ls))
        fig2 = Figure(); ls2 = LScene(fig2[1, 1])
        mesh!(ls2, Rect3f(Point3f(0), Vec3f(1)))
        @test !OM._scene_contains_sensors(Makie.get_scene(ls2))
    end

    @testset "ScreenConfig contract" begin
        @test fieldnames(OM.ScreenConfig)[end] == :sensors     # appended (positional-ctor contract)
        @test fieldtype(OM.ScreenConfig, :sensors) == Bool
        theme = Makie.CURRENT_DEFAULT_THEME[:OmniverseMakie]
        @test Makie.to_value(theme[:sensors]) == false         # no BVH cost for sensor-free scenes
    end
end

# =============================================================================================
# GPU tier — one subprocess, 2 renderer creations
# =============================================================================================

const _SENSOR_PROG = """
using OmniverseMakie, ColorTypes, FixedPointNumbers
import OmniverseMakie as OM
using OmniverseMakie: OV
using Test: @test_logs

# Known geometry (data space, Z-up): ground top at z=0, a 1.5 m cube centred at x = 10.
# Sensor height h = 1.5 → sensor-frame ground plane at z = −1.5.
function build_scene()
    scene = Scene(size = (320, 320); lights = AbstractLight[
        DirectionalLight(RGBf(1, 1, 1), Vec3f(-1, -1, -0.4), false)])
    cam3d!(scene)
    mesh!(scene, Rect3f(Point3f(-50, -50, -0.1), Vec3f(100, 100, 0.1)); color = :gray)
    mesh!(scene, Rect3f(Point3f(9.25, -0.75, 0), Vec3f(1.5)); color = :red)
    update_cam!(scene, Vec3f(6, -6, 4), Vec3f(5, 0, 0), Vec3f(0, 0, 1))
    return scene
end

scene  = build_scene()
sensor = lidar!(scene, Point3f(0, 0, 1.5); channels = [:coordinates, :intensity])
rad    = radar!(scene, Point3f(0, 0, 1.5))
screen = OM.Screen(scene)
println("MOTION_BVH_AUTO=", screen.renderer.motion_bvh)      # scene has sensors → auto-on

Makie.colorbuffer(screen)                                     # authors the stage + first render
println("SENSOR_PROG_READY")                                  # early ready marker (retry loop)

fired = Ref(0)
on(_ -> fired[] += 1, sensor_returns(sensor))

resets = Ref(0)
OV._RESET_OBSERVER[] = () -> (resets[] += 1)
step_sensors!(screen, 0.1)                                    # steady state: nothing changed
println("STEADY_STEP_RESETS=", resets[])
OV._RESET_OBSERVER[] = nothing

r = sensor_returns(sensor)[]
println("LIDAR_COUNTS=", r.counts)
println("LIDAR_FIRED=", fired[])
println("LIDAR_NPOINTS=", length(r.points))
println("LIDAR_INTENSITY_LEN=", length(r.intensity))
println("LIDAR_INTENSITY_POS=", count(>(0), r.intensity))
zs = [p[3] for p in r.points]
println("LIDAR_MINZ=", minimum(zs))                           # sensor frame: ground ⇒ ≈ −1.5
cube_hits = count(p -> 9.0 < p[1] < 11.0 && abs(p[2]) < 1.5 && p[3] > -1.4, r.points)
println("LIDAR_CUBE_HITS=", cube_hits)
println("LIDAR_TON_ELTYPE=", eltype(r.points))

# Pose move: slide the sensor 3 m toward the cube (live omni:xform diff).  Lateral moves are
# fan-geometry-proof oracles: the flat ground stays at sensor-frame z = −h while the cube
# cluster shifts from x ≈ 10 to x ≈ 7.  (Raising the sensor instead pushes the ground out of
# the default model's elevation-fan × ~16 m range envelope — measured, not a bug.)
translate!(sensor, 3, 0, 0)
step_sensors!(screen, 0.1)
r2 = sensor_returns(sensor)[]
println("LIDAR_MINZ_MOVED=", minimum(p[3] for p in r2.points))
cube_hits_moved = count(p -> 6.0 < p[1] < 8.0 && abs(p[2]) < 1.5 && p[3] > -1.4, r2.points)
println("LIDAR_CUBE_HITS_MOVED=", cube_hits_moved)
println("POSE_X=", sensor_returns(sensor)[].pose[1, 4])

# Radar: same scene, WpmDmat model.
step_sensors!(screen, 0.1)
rr = sensor_returns(rad)[]
println("RADAR_COUNTS=", rr.counts)
println("RADAR_KEYS_OK=", keys(rr) == (:points, :rcs, :radial_velocity, :counts, :pose))
println("RADAR_RCS_FINITE=", all(isfinite, rr.rcs))
println("RADAR_RV_FINITE=", all(isfinite, rr.radial_velocity))

# delete!: the lidar leaves the stepping set; stepping the remaining radar still works.
# (Bare-Scene test convention: screen-level delete!/insert! are called explicitly.)
delete!(screen, scene, sensor)
step_sensors!(screen, 0.1)
rr2 = sensor_returns(rad)[]
println("RADAR_COUNTS_AFTER_DELETE=", rr2.counts)
close(screen)

# Post-display insert on a sensor-less screen: no motion BVH → one-time warn, static scene works.
scene2  = build_scene()
screen2 = OM.Screen(scene2)
Makie.colorbuffer(screen2)
println("MOTION_BVH_SENSORLESS=", screen2.renderer.motion_bvh)
late = @test_logs (:warn, r"motion BVH") match_mode = :any begin
    p = lidar!(scene2, Point3f(0, 0, 1.5))
    insert!(screen2, scene2, p)       # authored live (screen already authored)
    p
end
step_sensors!(screen2, 0.1)
println("LATE_LIDAR_COUNTS=", sensor_returns(late)[].counts)
close(screen2)

println("OK_SENSORS")
"""

@testset "lidar/radar end-to-end point clouds (subprocess)" begin
    _, out = run_ovrtx_subprocess(_SENSOR_PROG; timeout = 600, retries = 4,
                                  ready_marker = "SENSOR_PROG_READY")
    contains(out, "OK_SENSORS") || @info "sensor prog output" out
    @test contains(out, "OK_SENSORS")

    @test contains(out, "MOTION_BVH_AUTO=true")            # sensors in scene → BVH auto-enabled
    @test contains(out, "STEADY_STEP_RESETS=0")            # step_sensors! alone never resets RT2

    m = match(r"LIDAR_COUNTS=(\d+)", out)
    @test m !== nothing && parse(Int, m.captures[1]) > 10_000     # a full instant scan (~200k on the spike scene)
    @test contains(out, "LIDAR_FIRED=1")                   # observable fired exactly once per step
    counts = parse(Int, m.captures[1])
    mnp = match(r"LIDAR_NPOINTS=(\d+)", out)
    @test mnp !== nothing && parse(Int, mnp.captures[1]) == counts   # validity-sliced to Counts
    mil = match(r"LIDAR_INTENSITY_LEN=(\d+)", out)
    @test mil !== nothing && parse(Int, mil.captures[1]) == counts
    mip = match(r"LIDAR_INTENSITY_POS=(\d+)", out)
    @test mip !== nothing && parse(Int, mip.captures[1]) > 0

    # Sensor-frame geometry oracles: flat ground at −h (invariant under the lateral move),
    # cube cluster at x ≈ 10, then x ≈ 7 after sliding the sensor 3 m toward it.
    mz = match(r"LIDAR_MINZ=([-\d.]+)", out)
    @test mz !== nothing && -1.6 < parse(Float64, mz.captures[1]) < -1.4
    mch = match(r"LIDAR_CUBE_HITS=(\d+)", out)
    @test mch !== nothing && parse(Int, mch.captures[1]) > 50     # the 10 m cube returns
    mzm = match(r"LIDAR_MINZ_MOVED=([-\d.]+)", out)
    @test mzm !== nothing && -1.6 < parse(Float64, mzm.captures[1]) < -1.4
    mchm = match(r"LIDAR_CUBE_HITS_MOVED=(\d+)", out)
    @test mchm !== nothing && parse(Int, mchm.captures[1]) > 50   # scan followed the pose move
    mpx = match(r"POSE_X=([-\d.]+)", out)
    @test mpx !== nothing && 2.9 < parse(Float64, mpx.captures[1]) < 3.1   # pose carries the move

    mrc = match(r"RADAR_COUNTS=(\d+)", out)
    @test mrc !== nothing && parse(Int, mrc.captures[1]) > 0
    @test contains(out, "RADAR_KEYS_OK=true")
    @test contains(out, "RADAR_RCS_FINITE=true")
    @test contains(out, "RADAR_RV_FINITE=true")
    mrd = match(r"RADAR_COUNTS_AFTER_DELETE=(\d+)", out)
    @test mrd !== nothing && parse(Int, mrd.captures[1]) > 0      # stepping survives the delete!

    @test contains(out, "MOTION_BVH_SENSORLESS=false")     # no silent BVH cost without sensors
    # Post-display insert on a sensor-less screen is the documented DEGRADED path (cm stage,
    # no BVH — the child's @test_logs pins the warn): the oracle is that authoring + stepping
    # WORK mechanically, not scan quality (sensor physics are 100× off in data units there).
    mlc = match(r"LATE_LIDAR_COUNTS=(\d+)", out)
    @test mlc !== nothing && parse(Int, mlc.captures[1]) >= 0
end
