using Test
import OmniverseMakie   # bind the module name so this file runs standalone too
include(joinpath(@__DIR__, "..", "helpers.jl"))

# ---------------------------------------------------------------------------
# author_camera! drives the rendered viewpoint: render a cube from two camera
# poses and assert >= 20_000 pixels change between the images.
# Camera prims ignore omni:xform; author_camera! bakes the pose into the root
# USDA and re-opens the stage, so USD references must be re-added after each
# call.
# ---------------------------------------------------------------------------

const _M13_CAMERA_PROG = """
using OmniverseMakie, ColorTypes

const W, H = 400, 400
scene = Scene(size = (W, H))
cam3d!(scene)   # attach Camera3D controller

cube_pts = [
    (-100f0,-100f0,-100f0), ( 100f0,-100f0,-100f0),
    ( 100f0, 100f0,-100f0), (-100f0, 100f0,-100f0),
    (-100f0,-100f0, 100f0), ( 100f0,-100f0, 100f0),
    ( 100f0, 100f0, 100f0), (-100f0, 100f0, 100f0),
]
# USD 0-based indices
cube_faces = [
    [0,3,2,1], [4,5,6,7], [0,1,5,4],
    [3,7,6,2], [0,4,7,3], [1,2,6,5],
]
# 24 face-varying normals (4 per face × 6 faces)
cube_nrm = vcat(
    fill((  0f0,  0f0, -1f0), 4),   # bottom (-Z)
    fill((  0f0,  0f0,  1f0), 4),   # top    (+Z)
    fill((  0f0, -1f0,  0f0), 4),   # front  (-Y)
    fill((  0f0,  1f0,  0f0), 4),   # back   (+Y)
    fill(( -1f0,  0f0,  0f0), 4),   # left   (-X)
    fill((  1f0,  0f0,  0f0), 4),   # right  (+X)
)
cube_usda = OmniverseMakie.usda_mesh(cube_pts, OmniverseMakie._flat_faces(cube_faces)..., cube_nrm, (1f0, 0.5f0, 0f0))

screen = OmniverseMakie.Screen(scene)

# Pose A: far away, cube appears small in frame. author_camera! bakes the
# pose into the root USDA and re-opens the stage.
update_cam!(scene,
    Vec3d(500.0, 500.0, 500.0),
    Vec3d(  0.0,   0.0,   0.0),
    Vec3d(  0.0,   0.0,   1.0))
OmniverseMakie.author_camera!(screen, scene)   # opens stage with camera A
OmniverseMakie.OV.add_usd_reference!(screen.renderer, cube_usda, "/World/cube")
OmniverseMakie.OV.reset!(screen.renderer)
img1 = OmniverseMakie.OV.render_to_matrix(screen.renderer, screen.product; warmup=48)
nonblack1 = count(c -> (red(c) + green(c) + blue(c)) > 0.0f0, img1)
println("SIZE=", size(img1), " NONBLACK_A=", nonblack1)
@assert nonblack1 > 200 "pose A rendered black: nonblack=\$nonblack1"

# Pose B: same view diagonal (same lit faces) but 3.3x closer; the cube fills
# ~91% of the frame vs ~28% at pose A, so well over 20_000 pixels change.
# Re-opening the stage drops the cube; re-add it.
update_cam!(scene,
    Vec3d(150.0, 150.0, 150.0),
    Vec3d(  0.0,   0.0,   0.0),
    Vec3d(  0.0,   0.0,   1.0))
OmniverseMakie.author_camera!(screen, scene)   # re-opens stage with camera B
OmniverseMakie.OV.add_usd_reference!(screen.renderer, cube_usda, "/World/cube")
OmniverseMakie.OV.reset!(screen.renderer)
img2 = OmniverseMakie.OV.render_to_matrix(screen.renderer, screen.product; warmup=48)
nonblack2 = count(c -> (red(c) + green(c) + blue(c)) > 0.0f0, img2)
println("NONBLACK_B=", nonblack2)
@assert nonblack2 > 500 "pose B rendered black: nonblack=\$nonblack2"
# Pose B (close) should show significantly more non-black pixels than
# pose A (far).
@assert nonblack2 > nonblack1 "close camera did not show more of the cube: nonblack2=\$nonblack2 vs nonblack1=\$nonblack1"

# Differential pixel count: per-channel threshold of 8/255.
Himg, Wimg = size(img1)
threshold = 8       # out of 255
changed = let c = 0
    for h in 1:Himg, w in 1:Wimg
        c1 = img1[h, w];  c2 = img2[h, w]
        dr = abs(Float32(red(c1))   - Float32(red(c2)))
        dg = abs(Float32(green(c1)) - Float32(green(c2)))
        db = abs(Float32(blue(c1))  - Float32(blue(c2)))
        if dr*255 >= threshold || dg*255 >= threshold || db*255 >= threshold
            c += 1
        end
    end
    c
end
println("CHANGED_PIXELS=", changed, " / ", Himg * Wimg)
@assert changed >= 20_000 "camera pose change did not reframe: changed=\$changed < 20_000"

close(screen)
println("OK_CAMERA")
"""

@testset "camera_intrinsics unit" begin
    # Wider FOV → shorter focal length (inverse relationship).
    @test OmniverseMakie.camera_intrinsics(90, 400, 400).focal_length <
          OmniverseMakie.camera_intrinsics(30, 400, 400).focal_length
    # Square image → h_aperture == v_aperture (aspect ratio = 1).
    intr_sq = OmniverseMakie.camera_intrinsics(45, 400, 400)
    @test intr_sq.h_aperture ≈ intr_sq.v_aperture
    # Landscape 2:1 image → h_aperture == 2 * v_aperture.
    intr_wide = OmniverseMakie.camera_intrinsics(45, 800, 400)
    @test intr_wide.h_aperture ≈ 2 * intr_wide.v_aperture
end

@testset "author_camera! drives viewpoint (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M13_CAMERA_PROG; timeout=600, retries=2, ready_marker="SIZE=")
    @info "M1.3 subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_CAMERA")
    # Parse and validate CHANGED_PIXELS >= 20_000
    m = match(r"CHANGED_PIXELS=(\d+)", output)
    if m !== nothing
        @test parse(Int, m.captures[1]) >= 20_000
    else
        @test false   # CHANGED_PIXELS line missing
    end
end
