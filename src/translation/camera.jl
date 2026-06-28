# Camera translation for OmniverseMakie.
#
# Provides:
#   camera_to_world(eye, target, up)    — 4×4 USD row-vector camera-to-world matrix
#   _usda_row_vector_matrix(M)          — format a row-vector matrix as USD matrix4d text
#   _validate_camera_path(path)         — guard: must be /World/<name> (closes M1.2 Minor #2)
#   author_camera!(screen, scene)       — bake Makie 3-D camera pose into the root stage
#   sync_camera!(screen, scene)         — live: write_xform!("/World/Camera", …) on a pose change
#
# Camera update mechanism (CORRECTED in M2.1 — supersedes the M1.3 diagnostic):
#   The M1.3 note claimed `write_xform!`/`omni:xform` is "ignored on camera prims"
#   (it saw only ~7177/160000 px change and read that as RT2 noise).  That finding is
#   DISPROVEN.  An open-stage spike drove the camera purely with
#       OV.write_xform!(r, "/World/Camera", camera_to_world(eye, target, up)); OV.reset!(r)
#   and an A→B→A round-trip returned to the original within 8/270000 px — a clean
#   reframe, exactly how the NVIDIA vulkan-interop C example drives its camera.  So in
#   M2.1 the camera is BAKED once at author time (author_render_root!) and thereafter
#   UPDATED LIVE via `sync_camera!`'s `write_xform!` — NO stage re-open on a pose change.
#   (A rare FOV change is a `float focalLength` scalar write; orbit/pan/zoom is the xform.)
#
# NOTE: included inside OmniverseMakie module, after usd.jl.
#       OV, LibOVRTX, Makie, LinearAlgebra, author_render_root! are all in scope.

import LinearAlgebra: cross, normalize, norm

# ------------------------------------------------------------------
# camera_to_world — 4×4 USD row-vector camera-to-world matrix
# ------------------------------------------------------------------

"""
    camera_to_world(eye, target, up) -> Matrix{Float64}

Build a 4×4 camera-to-world transform in **USD row-vector convention**.

Rows 1–3 are the camera-space basis vectors (x/right, y/up, z/back-toward-camera).
Row 4 is the eye position with homogeneous weight 1.  Translation is in the last row,
matching the convention used by `write_xform!` and the USD `matrix4d` type.

USD uses a right-handed coordinate system:
- Camera +Z points **away** from `target` (toward the viewer).
- Camera +X is the right axis: `normalize(cross(up, z))`.
- Camera +Y is the reorthogonalised up: `cross(z, x)`.

Numeric check — reproduces the M1.2 placeholder camera exactly:

    camera_to_world((500,500,500),(0,0,0),(0,0,1))
    ≈ [ -0.70711  0.70711  0.0      0.0
        -0.40825 -0.40825  0.81650  0.0
         0.57735  0.57735  0.57735  0.0
         500.0    500.0    500.0    1.0 ]
"""
function camera_to_world(eye, target, up)
    e = Float64[eye[1],    eye[2],    eye[3]]
    t = Float64[target[1], target[2], target[3]]
    u = Float64[up[1],     up[2],     up[3]]
    z = normalize(e .- t)        # camera +Z: away from target
    x = normalize(cross(u, z))   # camera +X: right
    y = cross(z, x)              # camera +Y: up (reorthogonalised)
    return Float64[ x[1] x[2] x[3] 0.0
                    y[1] y[2] y[3] 0.0
                    z[1] z[2] z[3] 0.0
                    e[1] e[2] e[3] 1.0 ]
end

# ------------------------------------------------------------------
# _usda_row_vector_matrix — format for the USDA bake path
# ------------------------------------------------------------------

"""
    _usda_row_vector_matrix(M) -> String

Format a 4×4 row-vector matrix (basis vectors in rows 1–3, translation in row 4)
as a USD `matrix4d` literal string.  No transposition is applied; rows are emitted
as-is.  Used by `author_camera!` to bake the camera pose into the root USDA.

Contrast with `usda_matrix4d`, which starts from a Makie column-vector matrix
and transposes it before formatting.
"""
function _usda_row_vector_matrix(M::AbstractMatrix)
    M64 = Float64.(collect(M))
    rows = ntuple(i -> "($(join(M64[i,:], ", ")))", 4)
    return "( $(join(rows, ", ")) )"
end

# ------------------------------------------------------------------
# _validate_camera_path — close M1.2 Minor #2
# ------------------------------------------------------------------

"""
    _validate_camera_path(camera_path) -> camera_path

Validate that `camera_path` is a direct `/World/<name>` child
(e.g. `"/World/Camera"`).  Throws an error otherwise.

This enforces consistency with `author_render_root!`, which always creates the
camera prim as a direct child of `/World` using the last path segment.
"""
function _validate_camera_path(camera_path::String)
    parts = split(camera_path, "/")
    # "/World/Camera" splits to ["", "World", "Camera"] — length 3,
    # first part empty (leading /), second "World", third non-empty name.
    if length(parts) == 3 && parts[1] == "" && parts[2] == "World" && !isempty(parts[3])
        return camera_path
    end
    error("camera_path must be a direct /World/<name> child " *
          "(e.g. \"/World/Camera\"); got: \"$(camera_path)\"")
end

# ------------------------------------------------------------------
# camera_intrinsics — derive USD focalLength / aperture from Makie FOV
# ------------------------------------------------------------------

"""
    camera_intrinsics(fov_deg, W, H) -> NamedTuple

Compute USD camera intrinsics from a vertical field-of-view (degrees) and
image dimensions (W × H pixels).

Returns a NamedTuple with fields `focal_length`, `h_aperture`, `v_aperture`.

# Formulas
USD vertical FOV definition: `vfov = 2·atan(v_aperture / (2·focal_length))`
Inverting: `focal_length = v_aperture / (2·tand(fov_deg/2))`
Horizontal aperture matches the image aspect: `h_aperture = v_aperture · (W/H)`

# Reference vertical aperture
`v_aperture = 15.2908` (matches the M1.2 spike-proven stage).
"""
function camera_intrinsics(fov_deg, W, H)
    v_aperture   = 15.2908
    focal_length = v_aperture / (2 * tand(fov_deg / 2))
    h_aperture   = v_aperture * (W / H)
    return (focal_length = focal_length, h_aperture = h_aperture, v_aperture = v_aperture)
end

# ------------------------------------------------------------------
# author_camera! — bake Makie 3-D camera pose into the root USD stage
# ------------------------------------------------------------------

"""
    author_camera!(screen, scene; camera_path="/World/Camera") -> Nothing

Bake `scene`'s 3-D camera pose (and lights) into the USD render-root and re-open
the stage.

Delegates to `author_root_from_scene!` (usd.jl), which reads both the scene camera
AND `scene.compute[:lights][]` and opens the stage once with both baked in.  This
is the INITIAL bake; after it, live pose changes go through `sync_camera!`
(`write_xform!`), NOT a re-bake.

# Side effect: stage re-open
Every call re-opens the stage.  All previously added USD references are lost and
must be re-added by the caller.

# Precondition
`scene` must have a 3-D camera controller (`cam3d!(scene)` or `LScene`).

# camera_path
Must be a direct `/World/<name>` child (default `"/World/Camera"`).

# After this call
Call `OV.reset!(screen.renderer)` before rendering to restart RT2 accumulation.
"""
function author_camera!(screen, scene; camera_path::String = "/World/Camera")
    # Delegate to the canonical combined authorer (usd.jl).
    # camera_path validation happens inside author_root_from_scene!.
    return author_root_from_scene!(screen, scene; camera_path = camera_path)
end

# ------------------------------------------------------------------
# sync_camera! — live: push a camera pose / FOV change to the open stage
# ------------------------------------------------------------------

# Snapshot of the scene's 3-D camera pose + FOV, or `nothing` if the scene has no
# 3-D camera controller (a 2-D / pixel camera has no eyeposition/lookat/up/fov).
function _camera_snapshot(scene)
    cam = Makie.cameracontrols(scene)
    cam isa Makie.Camera3D || return nothing
    return (eye = cam.eyeposition[], target = cam.lookat[],
            up = cam.upvector[], fov = cam.fov[])
end

# Scalar Float32 write of `focalLength` on a camera prim (rare FOV-change path).
function _write_camera_focal_length!(r, prim, fl::Real)
    dtype = LibOVRTX.DLDataType(UInt8(LibOVRTX.kDLFloat), UInt8(32), UInt16(1))
    OV._write_attribute!(r, prim, "focalLength", dtype, false, LibOVRTX.OVRTX_SEMANTIC_NONE,
                         Float32[fl], Int64[1])
    return nothing
end

"""
    sync_camera!(screen, scene; camera_path="/World/Camera") -> Bool

Compare the scene's current 3-D camera pose/FOV to `screen.last_camera` (the
snapshot last written/baked).  On a pose change (`eyeposition`/`lookat`/`upvector`)
push `OV.write_xform!(camera_path, camera_to_world(eye, target, up))`; on a FOV
change push a `focalLength` scalar write.  Updates `screen.last_camera` and returns
`true` iff anything was written (so `colorbuffer` knows to `OV.reset!`).

Returns `false` (no-op) when the scene has no 3-D camera or nothing changed — a
static scene keeps accumulating without a reset.
"""
function sync_camera!(screen, scene; camera_path::String = "/World/Camera")
    snap = _camera_snapshot(scene)
    snap === nothing && return false
    old = screen.last_camera
    r   = screen.renderer
    changed = false

    if old === nothing || snap.eye != old.eye || snap.target != old.target || snap.up != old.up
        OV.write_xform!(r, camera_path, camera_to_world(snap.eye, snap.target, snap.up))
        changed = true
    end
    if old === nothing || snap.fov != old.fov
        W, H = screen.fb_size
        fl = camera_intrinsics(snap.fov, W, H).focal_length
        _write_camera_focal_length!(r, camera_path, fl)
        changed = true
    end

    screen.last_camera = snap
    return changed
end
