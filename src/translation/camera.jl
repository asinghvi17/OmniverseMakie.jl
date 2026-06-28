# Camera translation for OmniverseMakie.
#
# Provides:
#   camera_to_world(eye, target, up)    — 4×4 USD row-vector camera-to-world matrix
#   _usda_row_vector_matrix(M)          — format a row-vector matrix as USD matrix4d text
#   _validate_camera_path(path)         — guard: must be /World/<name> (closes M1.2 Minor #2)
#   author_camera!(screen, scene)       — bake Makie 3-D camera pose into the root stage
#
# Design (M1.3 diagnostic):
#   PRIMARY mechanism (write_xform! on the UsdGeomCamera prim) was tested first.
#   Result: only ~7177/160000 pixels changed — consistent with RT2 noise, not a reframe.
#   Conclusion: write_xform! / omni:xform is ignored on camera prims.
#
#   FALLBACK (used here): bake the camera pose into the root USDA string via
#   author_render_root!(screen; camera_xform_str=...).  This re-opens the stage,
#   so callers must re-add any USD references after each author_camera! call.
#   The differential test (pose A vs B) uses this path and achieves ≥20_000 changed px.
#
# NOTE: included inside OmniverseMakie module, after usd.jl.
#       OV, Makie, LinearAlgebra, author_render_root! are all in scope.

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
# author_camera! — bake Makie 3-D camera pose into the root USD stage
# ------------------------------------------------------------------

"""
    author_camera!(screen, scene; camera_path="/World/Camera") -> Nothing

Bake `scene`'s 3-D camera pose into the USD render-root and re-open the stage.

Reads `eyeposition`, `lookat`, and `upvector` from the `Camera3D` attached to
`scene` (via `Makie.cameracontrols`), computes the camera-to-world transform with
`camera_to_world`, and re-authors the entire root stage via `author_render_root!`
with the new camera xform embedded in the USDA string.

# Implementation note (fallback path)
The primary mechanism — writing `omni:xform` to the camera prim via
`OV.write_xform!` — was tested first but produced only ~7177/160000 changed pixels
(indistinguishable from RT2 accumulation noise), confirming that camera prims ignore
the `omni:xform` live-update attribute.  The fallback bakes the camera transform
directly into the USDA root string, which re-opens the stage.

# Side effect: stage re-open
Because `author_render_root!` is called internally, the stage is **re-opened** on
every `author_camera!` call.  All previously added USD references
(`OV.add_usd_reference!`) are lost and must be re-added by the caller.

# Precondition
`scene` must have a 3-D camera controller (`cam3d!(scene)` or `LScene`).

# camera_path
Must be a direct `/World/<name>` child (default `"/World/Camera"`).  An error is
thrown for any other path (closes M1.2 Minor #2).

# After this call
Call `OV.reset!(screen.renderer)` before rendering to restart RT2 accumulation
from the new viewpoint.
"""
function author_camera!(screen, scene; camera_path::String = "/World/Camera")
    _validate_camera_path(camera_path)

    # Read 3-D camera controls from the Makie scene.
    cam    = Makie.cameracontrols(scene)
    eye    = cam.eyeposition[]
    target = cam.lookat[]
    up     = cam.upvector[]

    # Compute camera-to-world (USD row-vector convention).
    M = camera_to_world(eye, target, up)

    # Format the row-vector matrix as a USD matrix4d literal (no transpose).
    xform_str = _usda_row_vector_matrix(M)

    # Bake camera into root (fallback path: re-opens stage).
    author_render_root!(screen;
        resolution       = screen.fb_size,
        camera_path      = camera_path,
        camera_xform_str = xform_str)
    return nothing
end
