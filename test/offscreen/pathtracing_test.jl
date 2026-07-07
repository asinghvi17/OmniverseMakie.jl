using Test
include(joinpath(@__DIR__, "..", "helpers.jl"))
import OmniverseMakie as OM

# ---------------------------------------------------------------------------
# ScreenConfig.mode / .samples wiring: settings emission + GPU pixel proof.
# Pure: rtx_settings_usda emits PathTracing + omni:rtx:pt sample settings for
# :pathtracing, no pt: lines for :rt2; :minimal / unknown symbols throw.
# GPU: one subprocess, <=3 renderers (SyncScopeIds budget) renders the same
# scene via :rt2 and :pathtracing at high/low SPP — PT fills the background
# non-black and a low-SPP still differs measurably from a high-SPP one.
# ---------------------------------------------------------------------------

@testset "pathtracing pure: rendermode + samples emission" begin
    # 9-field positional ctor (mode, samples, warmup, max_bounces,
    # selection_outline, accumulate_across_frames, accumulation_preroll,
    # background, sensors).
    mk(mode; samples = 512, warmup = 64, mb = 4) =
        OM.ScreenConfig(mode, samples, warmup, mb, false, false, 40, :default, false)

    # :rt2 — RealTimePathTracing, NO pt: lines (default path stays
    # byte-identical).
    rt2 = OM.rtx_settings_usda(mk(:rt2))
    @test occursin("token omni:rtx:rendermode = \"RealTimePathTracing\"", rt2)
    @test !occursin("omni:rtx:pt:", rt2)
    @test occursin("int omni:rtx:rtpt:maxBounces = 4", rt2)

    # :pathtracing — PathTracing + SPP cap + per-iteration
    # (cld(samples,warmup)) + pt maxBounces.
    pt = OM.rtx_settings_usda(mk(:pathtracing; samples = 512, warmup = 64, mb = 6))
    @test occursin("token omni:rtx:rendermode = \"PathTracing\"", pt)
    @test occursin("int omni:rtx:pt:samplesPerPixel = 512", pt)
    # cld(512, 64) = 8
    @test occursin("int omni:rtx:pt:samplesPerIteration = 8", pt)
    # PT ignores rtpt:maxBounces; rtpt kept (harmless), = max_bounces
    @test occursin("int omni:rtx:pt:maxBounces = 6", pt)
    @test occursin("int omni:rtx:rtpt:maxBounces = 6", pt)

    # samplesPerIteration ceils so warmup×spi ≥ samples (the cap is reachable
    # via the warmup loop).
    pt2 = OM.rtx_settings_usda(mk(:pathtracing; samples = 100, warmup = 64))
    @test occursin("int omni:rtx:pt:samplesPerPixel = 100", pt2)
    # cld(100, 64) = 2
    @test occursin("int omni:rtx:pt:samplesPerIteration = 2", pt2)

    # :minimal is NOT selectable via USD rendermode (exact RT2 fallback);
    # unknown symbols throw too.
    @test_throws ArgumentError OM.rtx_settings_usda(mk(:minimal))
    @test_throws ArgumentError OM.rtx_settings_usda(mk(:bogus_mode))

    # :pathtracing divides samples into per-iteration SPP (cld(samples,warmup)):
    # warmup = 0 would be a DivideError, and non-positive samples are
    # meaningless — both throw a clear ArgumentError instead.
    @test_throws ArgumentError OM.rtx_settings_usda(mk(:pathtracing; warmup = 0))
    @test_throws ArgumentError OM.rtx_settings_usda(mk(:pathtracing; samples = 0))
    @test_throws ArgumentError OM.rtx_settings_usda(mk(:pathtracing; samples = -5))
    # :rt2 ignores samples/warmup for SPP, so warmup = 0 is fine there.
    @test occursin("RealTimePathTracing", OM.rtx_settings_usda(mk(:rt2; warmup = 0)))
end

const _PATHTRACING_PROG = """
import OmniverseMakie as OM
using OmniverseMakie   # re-exports Makie names (Scene, cam3d!, mesh!, ...)

$(PROG_PIXEL_HELPERS)

OM.activate!()   # register backend; per-Screen kwargs select the mode below.

# Identical scene per Screen: a red cube on a grey slab lit by a PointLight
# (SphereLight area source: soft shadows) plus a low ambient. Penumbra + GI
# carry real Monte-Carlo variance, so a low-SPP still differs measurably from
# a converged one. No dome light, so RT2 leaves the background black.
function build_scene()
    scene = Scene(size = (320, 320); lights = AbstractLight[
        # close+bright ⇒ wide soft penumbra
        PointLight(RGBf(3, 3, 3), Vec3f(1.6, 1.6, 2.6)),
        # near-zero ⇒ deep, high-contrast shadow
        AmbientLight(RGBf(0.03, 0.03, 0.03)),
    ])
    cam3d!(scene)
    cam = Makie.cameracontrols(scene)
    cam.lookat[]      = Vec3f(0, 0, 0)
    cam.eyeposition[] = Vec3f(6, 6, 4)
    cam.upvector[]    = Vec3f(0, 0, 1)
    # ground, then cube
    mesh!(scene, Rect3f(Point3f(-2.5, -2.5, -0.6), Vec3f(5, 5, 0.1)); color = :gray70)
    mesh!(scene, Rect3f(Point3f(-0.5, -0.5, -0.5), Vec3f(1)); color = :red)
    return scene
end

# Per-pixel 3-channel abs-diff metrics: full-frame mean, mean over a CONTENT
# mask (pixels lit in the RT2 image, where variance lives — the PT background
# is uniform/near-noiseless), and counts of pixels changing by > 0.05 / > 0.2
# (dilution-proof: a count, not an average).
function diff_metrics(a, b, mask)
    H, W = size(a); s_all = 0.0; s_msk = 0.0; nmsk = 0; c05 = 0; c20 = 0
    for h in 1:H, w in 1:W
        p = a[h, w]; q = b[h, w]
        d = abs(Float32(p.r) - Float32(q.r)) + abs(Float32(p.g) - Float32(q.g)) +
            abs(Float32(p.b) - Float32(q.b))
        s_all += d
        d > 0.05f0 && (c05 += 1); d > 0.2f0 && (c20 += 1)
        mask[h, w] && (s_msk += d; nmsk += 1)
    end
    return (mean_all = s_all / (H * W), mean_msk = nmsk == 0 ? 0.0 : s_msk / nmsk,
            c05 = c05, c20 = c20, nmask = nmsk)
end

# (a) :rt2 baseline — background black ⇒ only the lit content (slab + cube)
# is non-black.
s_rt2   = OM.Screen(build_scene(); mode = :rt2)
img_rt2 = Makie.colorbuffer(s_rt2)
TOTAL   = size(img_rt2, 1) * size(img_rt2, 2)
rt2_nb  = nonblack(img_rt2)
# ready_marker: first print after 1st render
println("RT2_NONBLACK=", rt2_nb, " TOTAL=", TOTAL)
mask    = [lum(img_rt2[h, w]) > LUM_MIN for h in 1:size(img_rt2, 1), w in 1:size(img_rt2, 2)]
close(s_rt2)

# (a) :pathtracing (samples=512 default) — the offline path tracer fills the
# WHOLE frame (non-black background) AND is the high-SPP reference for the
# samples test.
# samples = 512 (theme default)
s_pt   = OM.Screen(build_scene(); mode = :pathtracing)
img_pt = Makie.colorbuffer(s_pt)
pt_nb  = nonblack(img_pt)
println("PT512_NONBLACK=", pt_nb)
close(s_pt)

# (b) :pathtracing with samples=1 (spi=1, one accumulated SPP) is heavily
# under-sampled in the penumbra + GI, so it differs sharply from the 512-SPP
# one — samples is honored, not inert. PT is deterministic per reset, so the
# diff is reproducible, not run-to-run noise.
s_pt1   = OM.Screen(build_scene(); mode = :pathtracing, samples = 1)
img_pt1 = Makie.colorbuffer(s_pt1)
pt1_nb  = nonblack(img_pt1)
m       = diff_metrics(img_pt, img_pt1, mask)
println("PT1_NONBLACK=", pt1_nb, " MASK_PX=", m.nmask)
println("DIFF_MEAN_ALL=", round(m.mean_all; digits = 5))
println("DIFF_MEAN_MASK=", round(m.mean_msk; digits = 5))
println("DIFF_C05=", m.c05, " DIFF_C20=", m.c20)
close(s_pt1)

println("OK_PT")
"""

@testset "pathtracing GPU: mode discriminator + samples honored (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_PATHTRACING_PROG;
        timeout = 1800, retries = 2, ready_marker = "RT2_NONBLACK=")
    @info "pathtracing subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_PT")

    m_total = match(r"RT2_NONBLACK=(\d+) TOTAL=(\d+)", output)
    m_pt    = match(r"PT512_NONBLACK=(\d+)", output)
    m_mask  = match(r"DIFF_MEAN_MASK=([0-9.]+)", output)
    m_c05   = match(r"DIFF_C05=(\d+)", output)
    @test m_total !== nothing && m_pt !== nothing && m_mask !== nothing && m_c05 !== nothing

    if m_total !== nothing && m_pt !== nothing && m_mask !== nothing && m_c05 !== nothing
        rt2_nb   = parse(Int, m_total.captures[1])
        total    = parse(Int, m_total.captures[2])
        pt_nb    = parse(Int, m_pt.captures[1])
        diff_msk = parse(Float64, m_mask.captures[1])
        c05      = parse(Int, m_c05.captures[1])

        # (a) PathTracing renders the background non-black: nonblack covers
        # nearly the full frame and sits well above the :rt2 render.
        @test pt_nb >= 0.95 * total
        @test pt_nb > rt2_nb + 10_000

        # (b) samples honored: a 1-SPP PT differs sharply from the 512-SPP
        # one; PT is deterministic per reset, so these floors are stable.
        @test diff_msk > 0.006
        @test c05 > 600
    end
end
