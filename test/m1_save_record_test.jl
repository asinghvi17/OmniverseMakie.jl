using Test

# ---------------------------------------------------------------------------
# M1.6 — save/record/offscreen plumbing
#
# RED checks (before M1.6 fixes):
#   - `Base.showable(MIME"image/jpeg", fig) == false` (RED without backend_showable(jpeg)=true)
#   - Re-author centroid: calling colorbuffer twice on the same Screen with different camera
#     positions → centroid of red pixels MUST SHIFT.  Without the re-author fix the stage
#     is stale (camera not updated in USD) → centroid stays put → ASSERT FAILS → RED.
#
# GREEN checks (after M1.6 fixes):
#   1. backend_showable(jpeg)=true  → showable check passes
#   2. re-author-per-call fix: always call setup_scene!(screen) + empty!(plot2usd) first
#      → each colorbuffer re-authors the stage → centroid shifts when camera moves → GREEN
#
# FileIO.save bypasses backend_showable (goes directly getscreen → backend_show) so
# save(fig, "x.jpg") already worked in M1.5 via the FileIO/JPEG encode path.  The
# backend_showable(jpeg) addition is still correct for Base.showable / Jupyter display.
#
# Per-frame change strategy (camera orbit):
#   `cam.eyeposition[]` is read directly by `author_root_from_scene!`, so setting it
#   before each colorbuffer / record frame is the guaranteed re-author trigger.
#   Scene-transform composition (`scene.transformation.model[]`) is a M2 forward-carry.
# ---------------------------------------------------------------------------

const _M16_SAVE_RECORD_PROG = """
using OmniverseMakie, ColorTypes, FixedPointNumbers

# Lower warmup for speed (4+ renders × 32 steps ≈ manageable).
OmniverseMakie.activate!(warmup = 32)

tmp      = mktempdir()
png_path = joinpath(tmp, "x.png")
jpg_path = joinpath(tmp, "x.jpg")
mp4_path = joinpath(tmp, "x.mp4")

fig = Figure()
ax  = LScene(fig[1, 1])
mesh!(ax, Rect3f(Point3f(0), Vec3f(1)); color = :red)

# -----------------------------------------------------------------------
# Test 1: save PNG (sanity — already worked in M1.5)
# -----------------------------------------------------------------------
save(png_path, fig)
@assert isfile(png_path) "save(png) wrote no file"
png_bytes = read(png_path)
println("PNG_BYTES=\$(length(png_bytes))")
@assert length(png_bytes) > 1000 "PNG too small: \$(length(png_bytes)) bytes"
@assert png_bytes[1:8] == UInt8[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] "not a valid PNG header"
println("SAVE_PNG_OK")

# -----------------------------------------------------------------------
# Test 2: save JPEG via FileIO (works via FileIO path regardless of backend_showable)
# -----------------------------------------------------------------------
save(jpg_path, fig)
@assert isfile(jpg_path) "save(jpg) wrote no file"
jpg_bytes = read(jpg_path)
println("JPG_BYTES=\$(length(jpg_bytes))")
@assert length(jpg_bytes) > 1000 "JPEG too small: \$(length(jpg_bytes)) bytes"
@assert jpg_bytes[1:3] == UInt8[0xFF, 0xD8, 0xFF] "not a valid JPEG header: \$(jpg_bytes[1:3])"
println("SAVE_JPG_OK")

# -----------------------------------------------------------------------
# Test 3: Base.showable(MIME"image/jpeg", fig)
# RED without backend_showable(jpeg)=true: showable returns false.
# GREEN with fix: returns true, so Jupyter / Base.show dispatch works.
# -----------------------------------------------------------------------
jpeg_showable = Base.showable(MIME("image/jpeg"), fig)
println("JPEG_SHOWABLE=\$(jpeg_showable)")
@assert jpeg_showable "backend_showable(jpeg)=true not set: Base.showable returns false"
println("JPEG_SHOWABLE_OK")

# -----------------------------------------------------------------------
# Test 4: re-author per call (centroid-based, robust against RT2 noise)
# Create ONE Screen, call colorbuffer twice with different camera positions.
# Without re-author fix: USD stage is stale (camera not updated) → cube appears at
# the SAME image position in both renders → centroid diff ≈ 0 → ASSERT FAILS.
# With fix: stage re-authored each call → cube shifts in image → centroid diff > 30px.
#
# Camera change: 180° horizontal orbit ((x,y,z) → (-x,-y,z)) guarantees a large
# shift (cube appears on opposite side of image).  expected ≥30px in practice; threshold is 20px.
# -----------------------------------------------------------------------
screen = OmniverseMakie.Screen(ax.scene)
cam    = Makie.cameracontrols(ax.scene)

imgA = Makie.colorbuffer(screen)
println("DEFAULT_EYE=\$(cam.eyeposition[])")

# 180° horizontal orbit
eye0 = cam.eyeposition[]
cam.eyeposition[] = Vec3f(-eye0[1], -eye0[2], eye0[3])
println("NEW_EYE=\$(cam.eyeposition[])")

imgB = Makie.colorbuffer(screen)

# Centroid of red-dominant pixels (function scope to avoid soft-scope for-loop issue)
function red_centroid(img)
    H, W = size(img)
    sr = 0.0; sc = 0.0; n = 0
    for h in 1:H, w in 1:W
        c = img[h, w]
        r = Float32(red(c)); g = Float32(green(c)); b = Float32(blue(c))
        if r > g && r > b && r > 0.1f0
            sr += h; sc += w; n += 1
        end
    end
    return n > 0 ? (sr / n, sc / n, n) : (Float64(H)/2, Float64(W)/2, 0)
end

cA = red_centroid(imgA)
cB = red_centroid(imgB)
println("CENTROID_A=\$(cA[1:2])  RED_N_A=\$(cA[3])")
println("CENTROID_B=\$(cB[1:2])  RED_N_B=\$(cB[3])")

centroid_diff = sqrt((cA[1]-cB[1])^2 + (cA[2]-cB[2])^2)
println("CENTROID_DIFF=\$(centroid_diff)")
@assert cA[3] > 50 "too few red pixels in frame A (n=\$(cA[3])) — cube not rendered"
@assert cB[3] > 50 "too few red pixels in frame B (n=\$(cB[3])) — cube not rendered (re-author failed?)"
@assert centroid_diff > 20.0 "Re-author fix failed: red centroid did not shift after camera change (diff=\$(centroid_diff)px); re-author per call not active"

Base.close(screen)
println("REAUTHOR_OK")

# -----------------------------------------------------------------------
# Test 5: Makie.record → MP4
# Three-frame orbit.  Each frame: record callback → Makie calls colorbuffer(io.screen)
# → with fix, re-authors per frame so each reflects the current camera.
# -----------------------------------------------------------------------
cam_r = Makie.cameracontrols(ax.scene)
Makie.record(fig, mp4_path, 1:3; framerate = 1) do i
    angle = Float32(i - 1) * (2.0f0 * Float32(pi) / 3.0f0)
    r     = 3.0f0
    cam_r.eyeposition[] = Vec3f(r * cos(angle), r * sin(angle), 2.0f0)
    println("RECORD_FRAME=\$(i)  eye=\$(cam_r.eyeposition[])")
end
@assert isfile(mp4_path) "record wrote no mp4 file"
mp4_size = filesize(mp4_path)
println("MP4_BYTES=\$(mp4_size)")
@assert mp4_size > 5000 "MP4 too small: \$(mp4_size) bytes (expected > 5 KB)"
println("RECORD_OK")

println("OK_SAVE_RECORD")
"""

@testset "M1.6 save/record/offscreen plumbing (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M16_SAVE_RECORD_PROG; timeout = 1800)
    @info "M1.6 subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_SAVE_RECORD")

    # --- PNG ---
    mpng = match(r"PNG_BYTES=(\d+)", output)
    if mpng !== nothing
        @test parse(Int, mpng.captures[1]) > 1000
    else
        @test false   # PNG_BYTES line missing
    end
    @test contains(output, "SAVE_PNG_OK")

    # --- JPEG (FileIO path) ---
    mjpg = match(r"JPG_BYTES=(\d+)", output)
    if mjpg !== nothing
        @test parse(Int, mjpg.captures[1]) > 1000
    else
        @test false   # JPG_BYTES line missing
    end
    @test contains(output, "SAVE_JPG_OK")

    # --- Base.showable(jpeg) ---
    @test contains(output, "JPEG_SHOWABLE=true")
    @test contains(output, "JPEG_SHOWABLE_OK")

    # --- Re-author per call: centroid must shift ---
    @test contains(output, "REAUTHOR_OK")
    mcd = match(r"CENTROID_DIFF=([0-9.eE+\-]+)", output)
    if mcd !== nothing
        @test parse(Float64, mcd.captures[1]) > 20.0
    else
        @test false   # CENTROID_DIFF line missing
    end

    # --- MP4 record ---
    mmp4 = match(r"MP4_BYTES=(\d+)", output)
    if mmp4 !== nothing
        @test parse(Int, mmp4.captures[1]) > 5000
    else
        @test false   # MP4_BYTES line missing
    end
    @test contains(output, "RECORD_OK")

    # All 3 record frames executed
    @test contains(output, "RECORD_FRAME=1")
    @test contains(output, "RECORD_FRAME=2")
    @test contains(output, "RECORD_FRAME=3")
end
