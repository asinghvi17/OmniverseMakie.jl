# Environment-light image (IBL) + background source + OmniPBR UV tiling — pure + GPU.
#
# Pure: ScreenConfig `background` plumbing (tokens, :default = no token, bad symbol throws),
# `_env_dome_usda` content, `_env_texture_file` source resolution (fresh temps, file passthrough,
# HDR clamp), the retired root-baked EnvironmentLight block, and the tiling key map + float2
# emitter.
#
# GPU (subprocess, pixel-verified — ovrtx silently ignores unknown attrs, so pixels are the only
# oracle):
#   • an `EnvironmentLight(1, green)` with NO other lights illuminates the scene green;
#   • `push_environment_image!` live-swaps green → red (remove+re-reference; temp GC'd on close);
#   • a PRE-display push is stashed and applied at author time;
#   • `background = :domelight` shows the env map as the visible background;
#   • `background = :sky` TRIPWIRE: standalone ovrtx does NOT implement the procedural sky —
#     the background stays black (this test FLIPS when a future ovrtx renders it: then remove
#     the :sky warn in settings.jl and update this expectation);
#   • `material = (; project_uvw, world_or_object, texture_scale)` changes a textured plane's
#     rendering (the tiling inputs are honored, not silently ignored).

using Test
include("helpers.jl")
using OmniverseMakie
const OM = OmniverseMakie
import Makie
using Makie: RGBf, Vec2f, EnvironmentLight

# =============================================================================================
# Pure
# =============================================================================================

@testset "envlight pure: config / dome usda / texture sources / tiling keys" begin
    @testset "ScreenConfig background plumbing" begin
        @test fieldnames(OM.ScreenConfig)[end] === :background   # trailing (positional ctor)
        mk(bg) = OM.ScreenConfig(:rt2, 512, 64, 4, false, false, 40, bg)
        @test occursin("omni:rtx:background:source:type = \"sky\"", OM.rtx_settings_usda(mk(:sky)))
        @test occursin("omni:rtx:background:source:type = \"domeLight\"", OM.rtx_settings_usda(mk(:domelight)))
        @test !occursin("background", OM.rtx_settings_usda(mk(:default)))   # byte-identical default
        @test_throws ArgumentError OM.rtx_settings_usda(mk(:nope))
        # theme default resolves to :default
        d = OM.Makie.merge_screen_config(OM.ScreenConfig, Dict{Symbol,Any}())
        @test d.background === :default
    end

    @testset "_env_dome_usda" begin
        d = OM._env_dome_usda("/tmp/x.png"; intensity = 1.5)
        @test occursin("defaultPrim = \"EnvLight\"", d)          # self-contained reference layer
        @test occursin("float inputs:intensity = 1500.0", d)     # Makie 1.0 → USD 1000 scale
        @test occursin("asset inputs:texture:file = @/tmp/x.png@", d)
        @test occursin("token inputs:texture:format = \"latlong\"", d)
        @test occursin("inputs:texture:format = \"angular\"",
                       OM._env_dome_usda("/tmp/x.png"; format = "angular"))
    end

    @testset "_env_texture_file: matrix → fresh temp PNG; path passthrough" begin
        img = fill(RGBf(0, 1, 0), 4, 8)
        p1, t1 = OM._env_texture_file(img)
        p2, t2 = OM._env_texture_file(img)
        @test isfile(p1) && isfile(p2)
        @test p1 != p2                                           # FRESH path per call (asset-read-once)
        @test t1 && t2
        rm(p1); rm(p2)
        # file passthrough: absolutized, not a temp
        path = tempname() * ".exr"; write(path, "stub")
        pf, tf = OM._env_texture_file(path)
        @test pf == abspath(path) && !tf
        rm(path)
        @test_throws ArgumentError OM._env_texture_file("/no/such/env.exr")
        # HDR components clamp (N0f8 conversion of >1 would throw) — warns, does not throw
        p3, _ = @test_logs (:warn, r"clamped") OM._env_texture_file(fill(RGBf(2, 0.5, 0.5), 2, 4))
        @test isfile(p3); rm(p3)
    end

    @testset "EnvironmentLight no longer root-baked" begin
        # The dome is authored via the removable reference (envlight.jl) so it can be live-swapped;
        # the old texture-less root bake is retired.
        @test OM.usda_light(EnvironmentLight(1f0, fill(RGBf(0, 1, 0), 2, 2)), 0) == ""
    end

    @testset "OmniPBR UV tiling keys + float2 emitter" begin
        inputs = Dict{String,Any}()
        for (k, v) in pairs((; project_uvw = true, world_or_object = true,
                              texture_scale = (4.0, 4.0), texture_translate = Vec2f(0.1, 0.2),
                              texture_rotate = 45))
            OM._merge_material_input!(inputs, k, v, nothing, false, false)
        end
        @test sort(collect(keys(inputs))) ==
              ["project_uvw", "texture_rotate", "texture_scale", "texture_translate", "world_or_object"]
        usda = OM.usda_omnipbr_material("M", inputs)
        @test occursin("bool inputs:project_uvw = 1", usda)
        @test occursin("bool inputs:world_or_object = 1", usda)
        @test occursin("float2 inputs:texture_scale = (4.0, 4.0)", usda)       # NTuple{2} → float2
        @test occursin("float2 inputs:texture_translate = (0.1, 0.2)", usda)   # Vec2f normalized
        @test occursin("float inputs:texture_rotate = 45.0", usda)
        # unknown keys still warn/skip (no regression)
        n0 = length(inputs)
        @test_logs (:warn, r"unknown `material=` key") OM._merge_material_input!(
            inputs, :bogus_key, 1.0, nothing, false, false)
        @test length(inputs) == n0
    end
end

# =============================================================================================
# GPU subprocess
# =============================================================================================

const _ENVLIGHT_PROG = raw"""
using OmniverseMakie, ColorTypes, FixedPointNumbers
const OM = OmniverseMakie
using Makie: Scene, Vec3f, Point3f, Rect3f, RGBf, AbstractLight, EnvironmentLight,
             cam3d!, update_cam!

OM.activate!(warmup = 40, samples = 128)

lum(c) = Float32(red(c)) + Float32(green(c)) + Float32(blue(c))
nonblack(img) = count(c -> lum(c) > 0.04f0, img)
greendom(img) = count(c -> Float32(green(c)) > Float32(red(c)) + 0.05f0 && Float32(green(c)) > Float32(blue(c)) + 0.05f0, img)
reddom(img)   = count(c -> Float32(red(c)) > Float32(green(c)) + 0.05f0 && Float32(red(c)) > Float32(blue(c)) + 0.05f0, img)
corner(img)   = img[8, 8]   # top-left = background region (the test sphere is centered)

function white_ball_scene(; lights = AbstractLight[], background = :default)
    scene = Scene(size = (200, 200); lights = lights)
    cam3d!(scene)
    update_cam!(scene, Vec3f(0, 0, 5), Vec3f(0, 0, 0), Vec3f(0, 1, 0))
    mesh!(scene, Makie.Sphere(Point3f(0), 1f0); color = :white)
    return scene, OM.Screen(scene; background = background)
end

green_env = fill(RGBf(0, 1, 0), 32, 64)
red_env   = fill(RGBf(1, 0, 0), 32, 64)

# 1) EnvironmentLight illuminates (green), no other lights
scene1, scr1 = white_ball_scene(lights = AbstractLight[EnvironmentLight(1.0f0, green_env)])
img1 = Makie.colorbuffer(scr1)
println("ENV_GREEN=", greendom(img1))

# 2) live push swaps green → red on the SAME screen; temp GC'd on close
push_environment_image!(scr1, red_env)
img2 = Makie.colorbuffer(scr1)
println("SWAP_RED=", reddom(img2), " SWAP_GREEN=", greendom(img2))
tmp_after = scr1.env_light.tmp
close(scr1)
println("TMP_CLEANED=", !isfile(tmp_after))

# 3) background=:domelight shows the env map as the visible background
scene3, scr3 = white_ball_scene(lights = AbstractLight[EnvironmentLight(1.0f0, green_env)],
                                background = :domelight)
c3 = corner(Makie.colorbuffer(scr3))
println("DOME_BG_CORNER_GREEN=", Float32(green(c3)) > 0.1f0 && Float32(green(c3)) > Float32(red(c3)))
close(scr3)

# 4) :sky TRIPWIRE — standalone ovrtx does not render the procedural sky (background stays black).
#    Camera looks HORIZONTALLY so the frame top is above the horizon (where a sky would be).
scene4 = Scene(size = (200, 200); lights = AbstractLight[])
cam3d!(scene4)
update_cam!(scene4, Vec3f(5, 0, 0.5), Vec3f(0, 0, 0.5), Vec3f(0, 0, 1))
mesh!(scene4, Makie.Sphere(Point3f(0, 0, 0.5), 0.5f0); color = :white)
scr4 = OM.Screen(scene4; background = :sky)
img4 = Makie.colorbuffer(scr4)
println("SKY_TOP_LIT=", count(c -> lum(c) > 0.05f0, img4[1:60, :]))
close(scr4)

# 5) a PRE-display push is stashed and applied at author time
scene5, scr5 = white_ball_scene()                 # no lights at all
push_environment_image!(scr5, green_env)          # before the first colorbuffer → stashed
img5 = Makie.colorbuffer(scr5)
println("STASH_GREEN=", greendom(img5))
close(scr5)

# 6) UV tiling is honored: project_uvw + texture_scale changes a textured plane's rendering
checker = [isodd(i ÷ 4 + j ÷ 4) ? RGBf(1, 1, 1) : RGBf(0, 0, 0) for i in 0:31, j in 0:31]
function tiled_plane(scale)
    scene = Scene(size = (200, 200); lights = AbstractLight[Makie.AmbientLight(RGBf(1, 1, 1))])
    cam3d!(scene)
    update_cam!(scene, Vec3f(0, 0, 4), Vec3f(0, 0, 0), Vec3f(0, 1, 0))
    mesh!(scene, Rect3f(Point3f(-1.5, -1.5, 0), Vec3f(3, 3, 0.05)); color = checker,
          material = (; project_uvw = true, world_or_object = true, texture_scale = scale))
    scr = OM.Screen(scene)
    img = Makie.colorbuffer(scr)
    close(scr)
    return img
end
imgA = tiled_plane((1.0, 1.0))
imgB = tiled_plane((8.0, 8.0))
println("TILING_NB=", nonblack(imgA),
        " TILING_DIFF=", count(k -> abs(lum(imgA[k]) - lum(imgB[k])) > 0.15f0, eachindex(imgA)))

println("OK_ENVLIGHT")
"""

@testset "envlight GPU: illuminate / live swap / backgrounds / stash / tiling (subprocess)" begin
    exitcode, out = run_ovrtx_subprocess(_ENVLIGHT_PROG; timeout = 900, retries = 4,
                                         ready_marker = "OK_ENVLIGHT")
    contains(out, "OK_ENVLIGHT") || @info "envlight output" out
    @test exitcode == 0
    @test contains(out, "OK_ENVLIGHT")

    getint(tag) = (m = match(Regex("$(tag)=(-?\\d+)"), out); m === nothing ? nothing : parse(Int, m.captures[1]))

    @test getint("ENV_GREEN")  !== nothing && getint("ENV_GREEN")  > 5000   # env light illuminates
    @test getint("SWAP_RED")   !== nothing && getint("SWAP_RED")   > 5000   # live swap took
    @test getint("SWAP_GREEN") !== nothing && getint("SWAP_GREEN") < 100    # ...and green is gone
    @test contains(out, "TMP_CLEANED=true")                                 # bounded temps
    @test contains(out, "DOME_BG_CORNER_GREEN=true")                        # :domelight background
    @test getint("SKY_TOP_LIT") !== nothing && getint("SKY_TOP_LIT") == 0   # :sky tripwire (see header)
    @test getint("STASH_GREEN") !== nothing && getint("STASH_GREEN") > 5000 # pre-display push applied
    @test getint("TILING_NB")   !== nothing && getint("TILING_NB")   > 5000 # textured plane rendered
    @test getint("TILING_DIFF") !== nothing && getint("TILING_DIFF") > 1000 # tiling inputs honored
end
