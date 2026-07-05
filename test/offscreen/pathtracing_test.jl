using Test
include(joinpath(@__DIR__, "..", "helpers.jl"))
import OmniverseMakie as OM

# ---------------------------------------------------------------------------
# `mode = :pathtracing` / `samples` wiring — pure settings emission + GPU pixel proof.
#
# The ScreenConfig.mode / .samples knobs used to be DEAD (rtx_settings_usda hardcoded
# RealTimePathTracing; samples was read nowhere).  This pins the now-live wiring:
#   PURE  — rtx_settings_usda emits "PathTracing" + omni:rtx:pt:samplesPerPixel/
#           samplesPerIteration (= cld(samples, warmup)) / maxBounces for :pathtracing, and stays
#           byte-identical (no pt: lines) for :rt2; :minimal + unknown symbols throw ArgumentError.
#   GPU   — one subprocess, ≤3 renderers (SyncScopeIds budget): render the SAME scene through
#           :rt2, :pathtracing (samples 512), :pathtracing (samples 4), and assert the two
#           PROBE-PROVEN facts — PathTracing renders the background non-black (nonblack ≈ full
#           frame, well above :rt2's lit-content-only nonblack), and `samples` is honored (a
#           low-SPP PT still differs measurably from a high-SPP one).  Parent judges printed
#           values (no noise-marginal in-child asserts); ready_marker is the first post-render
#           print, so only a startup crash is retried.
# ---------------------------------------------------------------------------

@testset "pathtracing pure: rendermode + samples emission" begin
    # 9-field positional ctor (mode, samples, warmup, max_bounces, selection_outline,
    # accumulate_across_frames, accumulation_preroll, background, sensors).
    mk(mode; samples = 512, warmup = 64, mb = 4) =
        OM.ScreenConfig(mode, samples, warmup, mb, false, false, 40, :default, false)

    # :rt2 — RealTimePathTracing, NO pt: lines (default path stays byte-identical).
    rt2 = OM.rtx_settings_usda(mk(:rt2))
    @test occursin("token omni:rtx:rendermode = \"RealTimePathTracing\"", rt2)
    @test !occursin("omni:rtx:pt:", rt2)
    @test occursin("int omni:rtx:rtpt:maxBounces = 4", rt2)

    # :pathtracing — PathTracing + SPP cap + per-iteration (cld(samples,warmup)) + pt maxBounces.
    pt = OM.rtx_settings_usda(mk(:pathtracing; samples = 512, warmup = 64, mb = 6))
    @test occursin("token omni:rtx:rendermode = \"PathTracing\"", pt)
    @test occursin("int omni:rtx:pt:samplesPerPixel = 512", pt)
    @test occursin("int omni:rtx:pt:samplesPerIteration = 8", pt)      # cld(512, 64) = 8
    @test occursin("int omni:rtx:pt:maxBounces = 6", pt)               # PT ignores rtpt:maxBounces
    @test occursin("int omni:rtx:rtpt:maxBounces = 6", pt)             # kept (harmless), = max_bounces

    # samplesPerIteration ceils so warmup×spi ≥ samples (the cap is reachable via the warmup loop).
    pt2 = OM.rtx_settings_usda(mk(:pathtracing; samples = 100, warmup = 64))
    @test occursin("int omni:rtx:pt:samplesPerPixel = 100", pt2)
    @test occursin("int omni:rtx:pt:samplesPerIteration = 2", pt2)     # cld(100, 64) = 2

    # :minimal is NOT selectable via USD rendermode (exact RT2 fallback); unknown symbols throw too.
    @test_throws ArgumentError OM.rtx_settings_usda(mk(:minimal))
    @test_throws ArgumentError OM.rtx_settings_usda(mk(:bogus_mode))
end

const _PATHTRACING_PROG = """
import OmniverseMakie as OM
using OmniverseMakie          # re-exports Makie names (Scene, cam3d!, mesh!, Rect3f, ...)

$(PROG_PIXEL_HELPERS)

OM.activate!()   # register the backend; per-Screen kwargs select the mode below.

# Identical scene per Screen: a red cube on a grey ground slab lit by a PointLight (a SphereLight
# area source in USD → SOFT shadows) with a low ambient.  The soft shadow penumbra + plane↔cube
# GI carry real Monte-Carlo variance, so a low-SPP path-traced still differs measurably from a
# converged one (a single directly-lit cube is nearly variance-free — samples would look inert).
# No dome/EnvironmentLight ⇒ RT2 leaves the background BLACK (the PT-fills-the-frame discriminator).
function build_scene()
    scene = Scene(size = (320, 320); lights = AbstractLight[
        PointLight(RGBf(3, 3, 3), Vec3f(1.6, 1.6, 2.6)),   # close+bright ⇒ wide soft penumbra
        AmbientLight(RGBf(0.03, 0.03, 0.03)),              # near-zero ⇒ deep, high-contrast shadow
    ])
    cam3d!(scene)
    cam = Makie.cameracontrols(scene)
    cam.lookat[]      = Vec3f(0, 0, 0)
    cam.eyeposition[] = Vec3f(6, 6, 4)
    cam.upvector[]    = Vec3f(0, 0, 1)
    mesh!(scene, Rect3f(Point3f(-2.5, -2.5, -0.6), Vec3f(5, 5, 0.1)); color = :gray70)  # ground
    mesh!(scene, Rect3f(Point3f(-0.5, -0.5, -0.5), Vec3f(1)); color = :red)             # cube
    return scene
end

# Per-pixel 3-channel abs-diff metrics: full-frame mean, mean over a CONTENT mask (pixels lit in
# the RT2 image, where variance lives — the PT background is uniform/near-noiseless), and counts of
# pixels changing by > 0.05 / > 0.2 (dilution-proof: a count, not an average).
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

# (a) :rt2 baseline — background black ⇒ only the lit content (slab + cube) is non-black.
s_rt2   = OM.Screen(build_scene(); mode = :rt2)
img_rt2 = Makie.colorbuffer(s_rt2)
TOTAL   = size(img_rt2, 1) * size(img_rt2, 2)
rt2_nb  = nonblack(img_rt2)
println("RT2_NONBLACK=", rt2_nb, " TOTAL=", TOTAL)   # ready_marker: first print after 1st render
mask    = [lum(img_rt2[h, w]) > LUM_MIN for h in 1:size(img_rt2, 1), w in 1:size(img_rt2, 2)]
close(s_rt2)

# (a) :pathtracing (samples=512 default) — the offline path tracer fills the WHOLE frame
# (non-black background) AND is the high-SPP reference for the samples test.
s_pt   = OM.Screen(build_scene(); mode = :pathtracing)   # samples = 512 (theme default)
img_pt = Makie.colorbuffer(s_pt)
pt_nb  = nonblack(img_pt)
println("PT512_NONBLACK=", pt_nb)
close(s_pt)

# (b) :pathtracing with samples=1 (warmup as-is ⇒ spi=1 ⇒ 1 accumulated SPP) — the probe's own
# spp-1-vs-spp-512 pair (measured diff 4.37).  A 1-SPP still is heavily under-sampled in the soft
# shadow penumbra + GI, so it differs sharply from the 512-SPP one — proving `samples` is honored,
# not inert.  PT is deterministic per reset, so this diff is reproducible, not run-to-run noise.
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

        # (a) PathTracing renders the background non-black: nonblack ≈ full frame, and WELL above
        # the :rt2 render of the same scene (probe: 65536/65536 vs RT2 27721 @256²).
        @test pt_nb >= 0.95 * total
        @test pt_nb > rt2_nb + 10_000

        # (b) samples honored: a 1-SPP PT still differs sharply from the 512-SPP one.  Deterministic
        # per reset, so these floors sit ≥3× below the measured values (masked-mean 0.0207 → 0.006;
        # >0.05-changed pixel count 2092 → 600).
        @test diff_msk > 0.006
        @test c05 > 600
    end
end
