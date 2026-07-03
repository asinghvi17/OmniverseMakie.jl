using Test
include(joinpath(@__DIR__, "..", "helpers.jl"))

# ---------------------------------------------------------------------------
# M2.1 Step 4b — live render-config on ONE open stage (authored ONCE):
#   (i)   orbit the camera 180° → content reframes (red centroid shifts ≥20px);
#         round-trip back → returns within RT2 noise.
#   (ii)  scale a light's intensity → mean luminance shifts clearly and the
#         change round-trips to baseline.  NOTE: the M1.2 camera carries
#         OmniRtxCameraAutoExposureAPI, which fully compensates a uniform
#         *brightening* (the rendered mean is exposure-normalised), so the
#         decisive, auto-exposure-immune signal is *dimming* one of two lights:
#         removing illumination drops luminance (you cannot expose-amplify
#         photons that aren't there).  We verify dim (luminance drops vs
#         baseline), bright (image moves clearly away from the dim state), and a
#         clean round-trip.
#   (iii) swap the lights' color white→red → blue content collapses (a hue shift
#         auto-exposure cannot mask); round-trip → returns to baseline.
#   Throughout, the stage is opened exactly ONCE — camera + light *attribute*
#   changes are live writes (write_xform! / inputs:intensity / inputs:color),
#   NOT re-authors.
#
# Two DirectionalLights (one dimmed for the intensity test; AmbientLight is
# excluded from scene.compute[:lights] by Makie).  Two distinctly-coloured boxes
# at ±x make a 180° orbit swap their image sides and make white→red collapse the
# blue box.
# ---------------------------------------------------------------------------

const _M21_RENDERCFG_PROG = """
using OmniverseMakie, ColorTypes, FixedPointNumbers
const OM = OmniverseMakie

OM.activate!(warmup = 40)

scene = Scene(size = (400, 400); lights = AbstractLight[
    DirectionalLight(RGBf(1, 1, 1), Vec3f(-1, -1, -0.4), false),
    DirectionalLight(RGBf(1, 1, 1), Vec3f( 1,  1, -0.4), false),
])
cam3d!(scene)
cam = Makie.cameracontrols(scene)
cam.lookat[]      = Vec3f(0, 0, 0)
cam.eyeposition[] = Vec3f(8, 8, 5)
cam.upvector[]    = Vec3f(0, 0, 1)
mesh!(scene, Rect3f(Point3f( 1.0, -0.5, -0.5), Vec3f(1)); color = :red)
mesh!(scene, Rect3f(Point3f(-2.0, -0.5, -0.5), Vec3f(1)); color = :blue)

screen = OM.Screen(scene)
N = 400 * 400

# ---- pixel-analysis helpers ----
function red_centroid(img)
    H, W = size(img); sr = 0.0; sc = 0.0; n = 0
    for h in 1:H, w in 1:W
        c = img[h, w]; r = Float32(red(c)); g = Float32(green(c)); b = Float32(blue(c))
        if r > g && r > b && r > 0.10f0
            sr += h; sc += w; n += 1
        end
    end
    return n > 0 ? (sr / n, sc / n, n) : (Float64(H) / 2, Float64(W) / 2, 0)
end
mean_lum(img) = (s = 0.0; for c in img
        s += 0.2126Float32(red(c)) + 0.7152Float32(green(c)) + 0.0722Float32(blue(c))
    end; s / length(img))
function mean_blue_nonblack(img)
    tb = 0.0; cnt = 0
    for c in img
        r = Float32(red(c)); g = Float32(green(c)); b = Float32(blue(c))
        if r + g + b > 0.01f0
            tb += b; cnt += 1
        end
    end
    return cnt == 0 ? 0.0 : tb / cnt
end
function changed(x, y; thr = 0.15)
    n = 0
    for i in eachindex(x)
        d = abs(Float32(red(x[i]))   - Float32(red(y[i]))) +
            abs(Float32(green(x[i])) - Float32(green(y[i]))) +
            abs(Float32(blue(x[i]))  - Float32(blue(y[i])))
        d > thr && (n += 1)
    end
    return n
end

# ======================= baseline =======================
img0 = Makie.colorbuffer(screen)
c0   = red_centroid(img0)
lum0 = mean_lum(img0)
blu0 = mean_blue_nonblack(img0)
println("OPENS_AFTER_BASELINE=\$(OM._ROOT_OPEN_COUNT[])")
println("BASE_CENTROID=\$(c0[1:2]) BASE_LUM=\$(round(lum0; digits=5)) BASE_BLUE=\$(round(blu0; digits=4))")
@assert c0[3] > 50 "baseline: too few red pixels (\$(c0[3]))"

# ======================= (i) CAMERA ORBIT 180° =======================
e0 = cam.eyeposition[]
cam.eyeposition[] = Vec3f(-e0[1], -e0[2], e0[3])
imgCam = Makie.colorbuffer(screen)
cCam   = red_centroid(imgCam)
shift  = sqrt((c0[1] - cCam[1])^2 + (c0[2] - cCam[2])^2)
println("CAM_SHIFT=\$(round(shift; digits=2))")
@assert cCam[3] > 50 "orbit: too few red pixels (\$(cCam[3]))"
@assert shift >= 20.0 "camera orbit did not reframe: red centroid shift \$(shift)px < 20"

cam.eyeposition[] = e0
imgCamRT = Makie.colorbuffer(screen)
ch_cam_rt = changed(img0, imgCamRT)
println("CAM_ROUNDTRIP_CHANGED=\$(ch_cam_rt) / \$(N)")
@assert ch_cam_rt < 0.03 * N "camera round-trip did not return to baseline (\$(ch_cam_rt)/\$(N) px)"

# ======================= (ii) LIGHT INTENSITY =======================
# DIM light 1 (auto-exposure-immune): luminance drops clearly vs baseline.
Makie.set_light!(scene, 1; color = RGBf(0.05, 0.05, 0.05))   # ~0.017× of original
imgDim = Makie.colorbuffer(screen)
lumDim = mean_lum(imgDim)
ch_dim = changed(img0, imgDim)
println("LUM_DIM=\$(round(lumDim; digits=5)) RATIO_DIM=\$(round(lumDim/max(lum0,1e-6); digits=3)) CHANGED_DIM=\$(ch_dim)")
@assert lumDim < lum0 * 0.85 "dimming light 1 did not reduce luminance (ratio \$(lumDim/max(lum0,1e-6)))"
@assert ch_dim > 0.005 * N "dim intensity change too small (\$(ch_dim)/\$(N))"

# BRIGHTEN light 1 5× — the rendered mean is auto-exposure-capped, so verify the
# write is honored by the image moving CLEARLY away from the dim state.
Makie.set_light!(scene, 1; color = RGBf(5, 5, 5))
imgBright = Makie.colorbuffer(screen)
lumBright = mean_lum(imgBright)
ch_bright_vs_dim = changed(imgDim, imgBright)
println("LUM_BRIGHT=\$(round(lumBright; digits=5)) CHANGED_BRIGHT_vs_DIM=\$(ch_bright_vs_dim)")
@assert lumBright > lumDim * 1.05 "5× intensity not brighter than dim state"
@assert ch_bright_vs_dim > 0.005 * N "5× intensity write not honored (\$(ch_bright_vs_dim)/\$(N) vs dim)"

# round-trip intensity to 1×
Makie.set_light!(scene, 1; color = RGBf(1, 1, 1))
imgLumRT = Makie.colorbuffer(screen)
ch_lum_rt = changed(img0, imgLumRT)
println("INT_ROUNDTRIP_CHANGED=\$(ch_lum_rt) / \$(N)")
@assert ch_lum_rt < 0.03 * N "intensity round-trip did not return to baseline (\$(ch_lum_rt)/\$(N))"

# ======================= (iii) LIGHT COLOR white→red =======================
Makie.set_light!(scene, 1; color = RGBf(1, 0, 0))
Makie.set_light!(scene, 2; color = RGBf(1, 0, 0))
imgRed = Makie.colorbuffer(screen)
bluRed = mean_blue_nonblack(imgRed)
ch_red = changed(img0, imgRed)
println("BLUE_BASE=\$(round(blu0; digits=4)) BLUE_RED=\$(round(bluRed; digits=4)) CHANGED_RED=\$(ch_red)")
@assert bluRed < blu0 * 0.40 "red light did not collapse blue content (blue \$(blu0) → \$(bluRed))"
@assert ch_red > 0.005 * N "color swap barely changed the image (\$(ch_red)/\$(N))"

# round-trip color to white
Makie.set_light!(scene, 1; color = RGBf(1, 1, 1))
Makie.set_light!(scene, 2; color = RGBf(1, 1, 1))
imgColRT = Makie.colorbuffer(screen)
ch_col_rt = changed(img0, imgColRT)
println("COLOR_ROUNDTRIP_CHANGED=\$(ch_col_rt) / \$(N)")
@assert ch_col_rt < 0.03 * N "color round-trip did not return to baseline (\$(ch_col_rt)/\$(N))"

# ======================= stage authored ONCE =======================
opens = OM._ROOT_OPEN_COUNT[]
println("ROOT_OPENS=\$(opens)")
@assert opens == 1 "stage opened \$(opens)× — live camera/light writes must NOT re-author"

close(screen)
println("OK_RENDERCFG")
"""

@testset "M2.1 live render-config (camera + lights) on open stage (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M21_RENDERCFG_PROG; timeout = 1800, retries = 2, ready_marker = "OK_RENDERCFG")
    @info "M2.1 render-config subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_RENDERCFG")

    # (i) camera orbit reframes ≥20px.
    msh = match(r"CAM_SHIFT=([0-9.]+)", output)
    @test msh !== nothing && parse(Float64, msh.captures[1]) >= 20.0
    # camera round-trip returns within noise.
    mcr = match(r"CAM_ROUNDTRIP_CHANGED=(\d+)", output)
    @test mcr !== nothing && parse(Int, mcr.captures[1]) < 0.03 * 160000

    # (ii) intensity: dimming a light drops luminance clearly; round-trip returns.
    mrd = match(r"RATIO_DIM=([0-9.]+)", output)
    @test mrd !== nothing && parse(Float64, mrd.captures[1]) < 0.85
    mir = match(r"INT_ROUNDTRIP_CHANGED=(\d+)", output)
    @test mir !== nothing && parse(Int, mir.captures[1]) < 0.03 * 160000

    # (iii) red light collapses blue content; round-trip returns.
    mbb = match(r"BLUE_BASE=([0-9.]+) BLUE_RED=([0-9.]+)", output)
    @test mbb !== nothing && parse(Float64, mbb.captures[2]) < parse(Float64, mbb.captures[1]) * 0.40
    mcc = match(r"COLOR_ROUNDTRIP_CHANGED=(\d+)", output)
    @test mcc !== nothing && parse(Int, mcc.captures[1]) < 0.03 * 160000

    # Stage authored exactly once across all live writes.
    mo = match(r"ROOT_OPENS=(\d+)", output)
    @test mo !== nothing && parse(Int, mo.captures[1]) == 1
end
