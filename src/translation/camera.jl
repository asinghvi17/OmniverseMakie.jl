# Camera translation.  Provides:
#   camera_to_world(eye, target, up)  — USD row-vector camera-to-world matrix
#   _usda_row_vector_matrix(M)        — that matrix as USD matrix4d text
#   _validate_camera_path(path)       — guard: path must be /World/<name>
#   author_camera!(screen, scene)     — bake the 3-D camera pose into the root
#   sync_camera!(screen, scene)       — live write_xform! on a pose change
#
# Update mechanism (M2.1, supersedes the M1.3 diagnostic): write_xform! DOES
# drive a camera prim — the M1.3 "ignored on camera prims" claim is DISPROVEN
# (an A->B->A spike round-tripped to within 8/270000 px).  So camera is BAKED
# once at author time, then UPDATED LIVE via sync_camera!'s write_xform! — NO
# stage re-open on a pose change.  (A rare FOV change is a `float focalLength`
# scalar write; orbit/pan/zoom is the xform.)
#
# Included in the OmniverseMakie module after usd.jl.  In scope: OV, LibOVRTX,
# Makie, LinearAlgebra, author_render_root!.

import LinearAlgebra: cross, normalize, norm

# ------------------------------------------------------------------
# camera_to_world — 4×4 USD row-vector camera-to-world matrix
# ------------------------------------------------------------------

"""
    camera_to_world(eye, target, up) -> Matrix{Float64}

4x4 camera-to-world transform in USD row-vector convention: rows 1-3 are the camera
basis (x/right, y/up, z/back-toward-camera), row 4 is the eye position (weight 1) —
the layout `write_xform!` and USD `matrix4d` expect.

Right-handed: +Z = normalize(eye-target) (toward viewer), +X = normalize(cross(up,Z)),
+Y = cross(Z,X).  Reproduces the M1.2 placeholder camera for
(eye,target,up)=((500,500,500),(0,0,0),(0,0,1)).
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

Format a 4x4 row-vector matrix (basis in rows 1-3, translation in row 4) as a USD
`matrix4d` literal, rows emitted as-is (NO transpose).  Contrast `usda_matrix4d`,
which transposes a Makie column-vector matrix first.
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

Return `camera_path` if it is a direct `/World/<name>` child (e.g. `"/World/Camera"`);
else throw.  Enforces consistency with `author_render_root!`, which creates the camera
prim as a direct `/World` child from the last path segment.
"""
function _validate_camera_path(camera_path::String)
    parts = split(camera_path, "/")
    # "/World/Camera" splits to ["", "World", "Camera"] — length 3,
    # first part empty (leading /), second "World", third non-empty name.
    if length(parts) == 3 && parts[1] == "" && parts[2] == "World" && !isempty(parts[3])
        # Depth alone is NOT enough: "/World/My Camera" has the right shape but the
        # segment "My Camera" (space) is not a legal USD identifier — it would
        # author a broken `def Camera "My Camera"` prim. Each segment (the "World"
        # scope and the camera name, both prim identifiers) must validate.
        _usd_identifier(parts[2]; what = "camera_path segment")
        _usd_identifier(parts[3]; what = "camera_path segment")
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

USD camera intrinsics from a vertical FOV (degrees) and image size (W x H px).  Returns
`(focal_length, h_aperture, v_aperture)`, from the USD FOV definition
`vfov = 2*atan(v_aperture / (2*focal_length))`:
  focal_length = v_aperture / (2*tand(fov_deg/2));  h_aperture = v_aperture*(W/H).
`v_aperture = 15.2908` (the M1.2 spike-proven value).
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

INITIAL bake of `scene`'s 3-D camera pose + lights into the USD render-root, delegating
to `author_root_from_scene!` (usd.jl, reads the camera and `scene.compute[:lights][]`).
Later pose changes go through `sync_camera!` (write_xform!), NOT a re-bake.

Re-opens the stage on every call: all previously added USD references are LOST and must
be re-added, then call `OV.reset!(screen.renderer)` before rendering.  Precondition:
`scene` has a 3-D camera controller.  `camera_path` must be a direct `/World/<name>`
child (default `/World/Camera`).
"""
function author_camera!(screen, scene; camera_path::String = "/World/Camera")
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

Diff the scene's 3-D camera pose/FOV against `screen.last_camera`.  On a pose change
push `write_xform!(camera_path, camera_to_world(...))`; on a FOV change push a
`focalLength` scalar write.  Updates `last_camera`; returns `true` iff anything was
written (so `colorbuffer` knows to `OV.reset!`).  Returns `false` when the scene has no
3-D camera or nothing changed (a static scene keeps accumulating without a reset).
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
