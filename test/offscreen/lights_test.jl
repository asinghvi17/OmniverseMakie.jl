using Test
import OmniverseMakie   # bind the module name so this file runs standalone too
include(joinpath(@__DIR__, "..", "helpers.jl"))

# ---------------------------------------------------------------------------
# UsdLux lights affect the rendered image.
# Unit: usda_light dispatches to the right USD prim types; lights_usda falls
# back to a default Sun for an empty lights vector.
# Subprocess: render a grey cube with vs without a bright DirectionalLight
# and assert the lit scene's mean luminance is >= 1.2x the ambient-only one.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Unit tests — pure Julia, no subprocess, no renderer required
# ---------------------------------------------------------------------------

@testset "M1.4 usda_light type dispatch" begin
    # Use OmniverseMakie-qualified names: RGBf, Vec3f, Vec2f are re-exported
    # from Makie.
    # DirectionalLight → DistantLight prim
    dl = OmniverseMakie.DirectionalLight(OmniverseMakie.RGBf(1f0, 1f0, 1f0), OmniverseMakie.Vec3f(-1f0, -1f0, -1f0), false)
    s  = OmniverseMakie.usda_light(dl, 0)
    @test contains(s, "DistantLight")
    @test contains(s, "DirectionalLight_0")
    @test contains(s, "inputs:intensity")
    @test contains(s, "inputs:color")
    @test contains(s, "xformOp:transform")

    # PointLight → SphereLight prim
    pl = OmniverseMakie.PointLight(OmniverseMakie.RGBf(1f0, 0f0, 0f0), OmniverseMakie.Vec3f(0f0, 0f0, 100f0), OmniverseMakie.Vec2f(0f0, 0f0))
    s2 = OmniverseMakie.usda_light(pl, 0)
    @test contains(s2, "SphereLight")
    @test contains(s2, "PointLight_0")
    @test contains(s2, "inputs:radius")

    # AmbientLight → DomeLight prim
    al = OmniverseMakie.AmbientLight(OmniverseMakie.RGBf(0.5f0, 0.5f0, 0.5f0))
    s3 = OmniverseMakie.usda_light(al, 0)
    @test contains(s3, "DomeLight")
    @test contains(s3, "AmbientLight_0")

    # Intensity scaling: max-channel × scale (750 × 0.5 = 375.0)
    dl_dim = OmniverseMakie.DirectionalLight(OmniverseMakie.RGBf(0.5f0, 0.5f0, 0.5f0), OmniverseMakie.Vec3f(-1f0, 0f0, 0f0), false)
    s4 = OmniverseMakie.usda_light(dl_dim, 1)
    @test contains(s4, "375.0")

    # Multiple lights of same type → unique names via index
    al2 = OmniverseMakie.AmbientLight(OmniverseMakie.RGBf(0.2f0, 0.2f0, 0.2f0))
    @test contains(OmniverseMakie.usda_light(al, 0), "AmbientLight_0")
    @test contains(OmniverseMakie.usda_light(al2, 1), "AmbientLight_1")
end

@testset "M1.4 lights_usda fallback and DEFAULT_LIGHTS_STR" begin
    # _DEFAULT_LIGHTS_STR is the fallback Sun block
    @test contains(OmniverseMakie._DEFAULT_LIGHTS_STR, "DistantLight")
    @test contains(OmniverseMakie._DEFAULT_LIGHTS_STR, "\"Sun\"")
    @test contains(OmniverseMakie._DEFAULT_LIGHTS_STR, "3000")

    # (a) A genuinely light-less scene falls back to the default Sun so the
    # render stays lit. lights_usda reads scene.compute[:lights][].
    empty_scene = OmniverseMakie.Scene(lights = OmniverseMakie.AbstractLight[])
    @test contains(OmniverseMakie.lights_usda(empty_scene), "\"Sun\"")

    # (b) A scene lit ONLY by an EnvironmentLight must NOT get the Sun: the env
    # dome is authored separately (envlight.jl) and lights it. usda_light emits
    # "" for it, so before the fix isempty(result) wrongly injected the Sun.
    env = OmniverseMakie.EnvironmentLight(1f0, fill(OmniverseMakie.RGBf(0f0, 1f0, 0f0), 4, 4))
    env_scene = OmniverseMakie.Scene(lights = OmniverseMakie.AbstractLight[env])
    env_usda  = OmniverseMakie.lights_usda(env_scene)
    @test !contains(env_usda, "\"Sun\"")   # no Sun injected over the env dome
    @test isempty(env_usda)                 # env dome authored elsewhere → ""

    # A real DirectionalLight still emits its own block (no Sun needed).
    dl_scene = OmniverseMakie.Scene(lights = OmniverseMakie.AbstractLight[
        OmniverseMakie.DirectionalLight(OmniverseMakie.RGBf(1f0, 1f0, 1f0),
                                        OmniverseMakie.Vec3f(-1f0, -1f0, -1f0), false)])
    dl_usda = OmniverseMakie.lights_usda(dl_scene)
    @test contains(dl_usda, "DirectionalLight_0")
    @test !contains(dl_usda, "\"Sun\"")
end

# ---------------------------------------------------------------------------
# Subprocess test — render with/without a DirectionalLight; compare luminance
# ---------------------------------------------------------------------------

const _M14_LIGHTS_PROG = """
using OmniverseMakie, ColorTypes

const W, H = 400, 400

# ---- Cube geometry ----
cube_pts = [
    (-100f0,-100f0,-100f0), ( 100f0,-100f0,-100f0),
    ( 100f0, 100f0,-100f0), (-100f0, 100f0,-100f0),
    (-100f0,-100f0, 100f0), ( 100f0,-100f0, 100f0),
    ( 100f0, 100f0, 100f0), (-100f0, 100f0, 100f0),
]
cube_faces = [
    [0,3,2,1], [4,5,6,7], [0,1,5,4],
    [3,7,6,2], [0,4,7,3], [1,2,6,5],
]
cube_nrm = vcat(
    fill((  0f0,  0f0, -1f0), 4),
    fill((  0f0,  0f0,  1f0), 4),
    fill((  0f0, -1f0,  0f0), 4),
    fill((  0f0,  1f0,  0f0), 4),
    fill(( -1f0,  0f0,  0f0), 4),
    fill((  1f0,  0f0,  0f0), 4),
)
# Grey cube so colour doesn't interfere with luminance comparison.
cube_usda = OmniverseMakie.usda_mesh(cube_pts, OmniverseMakie._flat_faces(cube_faces)..., cube_nrm, (0.8f0, 0.8f0, 0.8f0))

# ---- Luminance helper: mean over non-background pixels ----
function mean_lum_nonblack(img)
    tot = 0.0; cnt = 0
    for c in img
        r = Float32(red(c)); g = Float32(green(c)); b = Float32(blue(c))
        if r + g + b > 0.01f0
            tot += 0.2126*r + 0.7152*g + 0.0722*b
            cnt += 1
        end
    end
    cnt == 0 ? 0.0 : tot / cnt
end

# ============================================================
# Scene A: bright DirectionalLight (world-space, toward cube)
#          + faint AmbientLight
# ============================================================
lights_A = AbstractLight[
    DirectionalLight(RGBf(1f0, 1f0, 1f0), Vec3f(-1f0, -1f0, -1f0), false),
    AmbientLight(RGBf(0.1f0, 0.1f0, 0.1f0)),
]
scene_A = Scene(size = (W, H); lights = lights_A)
cam3d!(scene_A)
update_cam!(scene_A, Vec3d(500.0, 500.0, 500.0), Vec3d(0.0, 0.0, 0.0), Vec3d(0.0, 0.0, 1.0))

screen_A = OmniverseMakie.Screen(scene_A)
OmniverseMakie.author_root_from_scene!(screen_A, scene_A)
OmniverseMakie.OV.add_usd_reference!(screen_A.renderer, cube_usda, "/World/cube")
OmniverseMakie.OV.reset!(screen_A.renderer)
img_A      = OmniverseMakie.OV.render_to_matrix(screen_A.renderer, screen_A.product; warmup = 48)
lum_A      = mean_lum_nonblack(img_A)
nonblack_A = count(c -> (red(c) + green(c) + blue(c)) > 0.01f0, img_A)
println("LUM_A=", lum_A, " NONBLACK_A=", nonblack_A)
@assert nonblack_A > 100 "scene A rendered black: nonblack=\$nonblack_A"

# ============================================================
# Scene B: faint AmbientLight only (no DirectionalLight)
# ============================================================
lights_B = AbstractLight[
    AmbientLight(RGBf(0.1f0, 0.1f0, 0.1f0)),
]
scene_B = Scene(size = (W, H); lights = lights_B)
cam3d!(scene_B)
update_cam!(scene_B, Vec3d(500.0, 500.0, 500.0), Vec3d(0.0, 0.0, 0.0), Vec3d(0.0, 0.0, 1.0))

screen_B = OmniverseMakie.Screen(scene_B)
OmniverseMakie.author_root_from_scene!(screen_B, scene_B)
OmniverseMakie.OV.add_usd_reference!(screen_B.renderer, cube_usda, "/World/cube")
OmniverseMakie.OV.reset!(screen_B.renderer)
img_B      = OmniverseMakie.OV.render_to_matrix(screen_B.renderer, screen_B.product; warmup = 48)
lum_B      = mean_lum_nonblack(img_B)
nonblack_B = count(c -> (red(c) + green(c) + blue(c)) > 0.01f0, img_B)
println("LUM_B=", lum_B, " NONBLACK_B=", nonblack_B)
@assert nonblack_B > 100 "scene B rendered black: nonblack=\$nonblack_B"

# ============================================================
# Assert the lit scene is meaningfully brighter (≥ 1.2×)
# ============================================================
ratio = lum_A / max(lum_B, 1e-6)
println("RATIO=", ratio)
@assert lum_A >= lum_B * 1.2 "lit scene not meaningfully brighter: lum_A=\$lum_A, lum_B=\$lum_B, ratio=\$ratio"

close(screen_A)
close(screen_B)
println("OK_LIGHTS")
"""

@testset "M1.4 lights affect rendered image (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M14_LIGHTS_PROG; timeout = 600, retries = 2, ready_marker = "LUM_A=")
    @info "M1.4 subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_LIGHTS")
    m = match(r"RATIO=([0-9.eE+\-]+)", output)
    if m !== nothing
        @test parse(Float64, m.captures[1]) >= 1.2
    else
        @test false   # RATIO line missing
    end
end
