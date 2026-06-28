# Lights translation for OmniverseMakie.
#
# Provides:
#   lights_usda(scene; cam_to_world)       — all scene lights as UsdLux USDA prim blocks
#   usda_light(l, index)                   — per-type USDA prim block
#   _direction_to_xform(dir)               — orientation matrix string for directional lights
#   author_root_from_scene!(screen, scene) — canonical bake: camera + lights in ONE open
#   author_lights!(screen, scene)          — intent-named delegator to author_root_from_scene!
#
# Type mapping (fully tested: DirectionalLight, PointLight, AmbientLight;
# best-effort: RectLight, SpotLight, EnvironmentLight):
#   DirectionalLight → UsdLuxDistantLight  (intensity = 3000 × max-channel)
#   PointLight       → UsdLuxSphereLight   (intensity = 10000 × max-channel, radius = 1)
#   AmbientLight     → UsdLuxDomeLight     (intensity = 1000 × max-channel)
#   RectLight        → UsdLuxRectLight     [best-effort; orient + translate from fields]
#   SpotLight        → UsdLuxSphereLight   + ShapingAPI [best-effort]
#   EnvironmentLight → UsdLuxDomeLight     [best-effort; texture image deferred to M4]
#
# Composition architecture (M1.4):
#   `author_root_from_scene!` is the canonical single-pass root authorer.
#   Both `author_camera!` (camera.jl) and `author_lights!` (this file) delegate to it.
#   They are intent-named entry points over the same combined authoring — callers that
#   conceptually update just the camera or just the lights both re-bake the full root.
#   M2 will specialize live diffing here if needed.
#
# NOTE: included inside OmniverseMakie module, after camera.jl.
#       All of: OV, Makie, LinearAlgebra, camera_to_world, camera_intrinsics,
#       _usda_row_vector_matrix, _validate_camera_path, author_render_root! are in scope.

# ------------------------------------------------------------------
# _direction_to_xform — orientation matrix string for directional lights
# ------------------------------------------------------------------

"""
    _direction_to_xform(dir) -> String

Build a USD `matrix4d` orientation transform for a light whose emission axis is `dir`.

USD DistantLights (and RectLights) emit along local −Z.  The returned transform
rotates the light so that local −Z aligns with `dir` in world space, i.e. local +Z
points opposite to the emission direction.

Returns a USDA `matrix4d` literal (row-vector convention) via `_usda_row_vector_matrix`.
"""
function _direction_to_xform(dir)
    z = -normalize(Float64[dir[1], dir[2], dir[3]])   # local +Z = −emission direction
    a = abs(z[3]) < 0.99 ? Float64[0, 0, 1] : Float64[1, 0, 0]
    x = normalize(cross(a, z))
    y = cross(z, x)
    M = Float64[x[1] x[2] x[3] 0.0
                y[1] y[2] y[3] 0.0
                z[1] z[2] z[3] 0.0
                0.0  0.0  0.0  1.0]
    return _usda_row_vector_matrix(M)
end

# ------------------------------------------------------------------
# _intensity_and_color — factor magnitude out into intensity; normalize hue
# ------------------------------------------------------------------

# Returns (intensity::Float64, r::Float64, g::Float64, b::Float64).
# intensity = scale × max(r,g,b); color = (r,g,b) / max(r,g,b) so max-channel = 1.
#
# NOTE: In Makie 0.24.x, AmbientLight.color is an Observable{RGBf} while all other
# light types store a plain RGBf.  This function accepts both forms.
function _intensity_and_color(color_or_obs, scale::Float64)
    c = color_or_obs isa Observable ? color_or_obs[] : color_or_obs
    r  = Float64(c.r)
    g  = Float64(c.g)
    b  = Float64(c.b)
    mag = max(r, g, b, 1e-10)
    return scale * mag, r / mag, g / mag, b / mag
end

# ------------------------------------------------------------------
# usda_light — per-type USDA prim block
# ------------------------------------------------------------------

"""
    usda_light(l, index::Int) -> String

Emit a UsdLux prim block for a single Makie light.  `index` is used to produce a
unique prim name when multiple lights of the same type exist (name = `<Type>_<index>`).

Each returned string starts with 4-space indent (direct child of `/World`) and ends
with `}\\n\\n` so blocks can be concatenated and placed directly in the `World` Xform.

Type mapping summary:
  DirectionalLight → UsdLuxDistantLight   intensity=3000×max
  PointLight       → UsdLuxSphereLight    intensity=10000×max, radius=1
  AmbientLight     → UsdLuxDomeLight      intensity=1000×max
  RectLight        → UsdLuxRectLight      [best-effort]
  SpotLight        → UsdLuxSphereLight    + ShapingAPI cone [best-effort]
  EnvironmentLight → UsdLuxDomeLight      (texture image deferred to M4) [best-effort]
"""
function usda_light(l::Makie.DirectionalLight, index::Int)
    intensity, cr, cg, cb = _intensity_and_color(l.color, 3000.0)
    xform = _direction_to_xform(l.direction)
    return """    def DistantLight "DirectionalLight_$(index)" (
        prepend apiSchemas = ["ShapingAPI"]
    )
    {
        float inputs:angle = 1
        float inputs:intensity = $(intensity)
        color3f inputs:color = ($(cr), $(cg), $(cb))
        matrix4d xformOp:transform = $(xform)
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }

"""
end

function usda_light(l::Makie.PointLight, index::Int)
    intensity, cr, cg, cb = _intensity_and_color(l.color, 10000.0)
    p = l.position
    xform_str = _usda_row_vector_matrix(Float64[
        1.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
        0.0 0.0 1.0 0.0
        Float64(p[1]) Float64(p[2]) Float64(p[3]) 1.0
    ])
    return """    def SphereLight "PointLight_$(index)"
    {
        float inputs:radius = 1
        float inputs:intensity = $(intensity)
        color3f inputs:color = ($(cr), $(cg), $(cb))
        matrix4d xformOp:transform = $(xform_str)
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }

"""
end

function usda_light(l::Makie.AmbientLight, index::Int)
    intensity, cr, cg, cb = _intensity_and_color(l.color, 1000.0)
    return """    def DomeLight "AmbientLight_$(index)"
    {
        float inputs:intensity = $(intensity)
        color3f inputs:color = ($(cr), $(cg), $(cb))
    }

"""
end

function usda_light(l::Makie.RectLight, index::Int)
    # Best-effort: UsdLuxRectLight with width=norm(u1), height=norm(u2).
    # Orientation from direction; translation from position.
    intensity, cr, cg, cb = _intensity_and_color(l.color, 3000.0)
    w = norm(Float64[l.u1[1], l.u1[2], l.u1[3]])
    h = norm(Float64[l.u2[1], l.u2[2], l.u2[3]])
    # Build combined rotation+translation matrix.
    d = Float64[l.direction[1], l.direction[2], l.direction[3]]
    z = -normalize(d)
    a = abs(z[3]) < 0.99 ? Float64[0, 0, 1] : Float64[1, 0, 0]
    x_ax = normalize(cross(a, z))
    y_ax = cross(z, x_ax)
    p = l.position
    M = Float64[x_ax[1] x_ax[2] x_ax[3] 0.0
                y_ax[1] y_ax[2] y_ax[3] 0.0
                z[1]    z[2]    z[3]    0.0
                Float64(p[1]) Float64(p[2]) Float64(p[3]) 1.0]
    xform_str = _usda_row_vector_matrix(M)
    return """    def RectLight "RectLight_$(index)"
    {
        float inputs:width = $(w)
        float inputs:height = $(h)
        float inputs:intensity = $(intensity)
        color3f inputs:color = ($(cr), $(cg), $(cb))
        matrix4d xformOp:transform = $(xform_str)
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }

"""
end

function usda_light(l::Makie.SpotLight, index::Int)
    # Best-effort: UsdLuxSphereLight + ShapingAPI cone from l.angles.
    # l.angles = (inner_cutoff_radians, outer_cutoff_radians).
    intensity, cr, cg, cb = _intensity_and_color(l.color, 10000.0)
    p = l.position
    xform_str = _usda_row_vector_matrix(Float64[
        1.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
        0.0 0.0 1.0 0.0
        Float64(p[1]) Float64(p[2]) Float64(p[3]) 1.0
    ])
    inner_deg = Float64(l.angles[1]) * (180.0 / π)
    outer_deg = Float64(l.angles[2]) * (180.0 / π)
    softness  = max(0.0, (outer_deg - inner_deg) / max(outer_deg, 1e-6))
    return """    def SphereLight "SpotLight_$(index)" (
        prepend apiSchemas = ["ShapingAPI"]
    )
    {
        float inputs:radius = 1
        float inputs:intensity = $(intensity)
        color3f inputs:color = ($(cr), $(cg), $(cb))
        float inputs:shaping:cone:angle = $(outer_deg)
        float inputs:shaping:cone:softness = $(softness)
        matrix4d xformOp:transform = $(xform_str)
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }

"""
end

function usda_light(l::Makie.EnvironmentLight, index::Int)
    # Best-effort: DomeLight with l.intensity scaled to USD range.
    # Texture image (l.image) is deferred to M4 — HDR environment map loading.
    intensity = Float64(l.intensity) * 1000.0
    return """    def DomeLight "EnvironmentLight_$(index)"
    {
        float inputs:intensity = $(intensity)
    }

"""
end

# Fallback for any future unknown light types — emit nothing, warn.
function usda_light(l, index::Int)
    @warn "OmniverseMakie: unsupported light type $(typeof(l)) at index $index — skipped"
    return ""
end

# ------------------------------------------------------------------
# lights_usda — concatenate all scene lights into a USDA block
# ------------------------------------------------------------------

"""
    lights_usda(scene; cam_to_world = nothing) -> String

Convert `scene.compute[:lights][]` to a concatenated set of UsdLux prim blocks
for embedding inside the `/World` Xform in the root stage.

If the lights vector is empty, returns `_DEFAULT_LIGHTS_STR` (the M1.2 "Sun"
DistantLight) so renders stay lit.

`cam_to_world` (optional): the 4×4 camera-to-world matrix (USD row-vector
convention, from `camera_to_world`).  When provided, any `DirectionalLight` with
`camera_relative = true` has its direction transformed from camera space to world
space before emission.  If `nil`, camera-relative directions are treated as world-
space (a reasonable approximation when no camera matrix is available).
"""
function lights_usda(scene; cam_to_world = nothing)
    lights = scene.compute[:lights][]
    isempty(lights) && return _DEFAULT_LIGHTS_STR

    counts = Dict{DataType, Int}()
    buf    = IOBuffer()

    for l in lights
        # Camera-relative DirectionalLight: transform direction to world space.
        l_emit = if l isa Makie.DirectionalLight && l.camera_relative && cam_to_world !== nothing
            d   = Float64[l.direction[1], l.direction[2], l.direction[3]]
            M33 = cam_to_world[1:3, 1:3]
            # Camera-space direction → world space (row-vector convention):
            # d_world[j] = Σᵢ d_cam[i] * M[i,j]  ≡  M33' * d_cam  (column-vector multiply)
            wd  = M33' * d
            Makie.DirectionalLight(l.color, Vec3f(wd[1], wd[2], wd[3]), false)
        else
            l
        end

        T   = typeof(l)            # key on original type for counting
        idx = get(counts, T, 0)
        counts[T] = idx + 1
        write(buf, usda_light(l_emit, idx))
    end

    result = String(take!(buf))
    isempty(result) && return _DEFAULT_LIGHTS_STR   # all lights were unsupported types
    return result
end

# ------------------------------------------------------------------
# author_root_from_scene! — canonical bake: camera + lights in ONE open
# ------------------------------------------------------------------

"""
    author_root_from_scene!(screen, scene; resolution=screen.fb_size,
                             camera_path="/World/Camera") -> Nothing

Canonical single-pass root authorer: reads the scene's 3-D camera AND all lights
from `scene.compute[:lights][]`, then calls `author_render_root!` exactly ONCE
(one `open_usd_string!`) with both baked in.

This is the function M1.5's `display` will call, then add plot USD references after.

Design note (why combined):
- Re-opening the stage wipes all added USD references (M1.3 diagnostic finding).
- `omni:xform` live-writes are ignored on non-geometry prims (camera, lights).
- Therefore camera and lights must be baked together in the same root USDA.
- `author_camera!` and `author_lights!` both delegate here; they are intent-named
  entry points so callers read naturally.  M2 can specialize each for live diffing
  without breaking callers.

# Side effect
Calls `author_render_root!` → the stage is re-opened.  All previously added USD
references (`OV.add_usd_reference!`) are lost and must be re-added by the caller.

# Precondition
`scene` must have a 3-D camera controller (`cam3d!(scene)` or `LScene`).
"""
function author_root_from_scene!(screen, scene;
                                  resolution  = screen.fb_size,
                                  camera_path::String = "/World/Camera")
    _validate_camera_path(camera_path)

    # Read 3-D camera controls from the Makie scene.
    cam    = Makie.cameracontrols(scene)
    eye    = cam.eyeposition[]
    target = cam.lookat[]
    up     = cam.upvector[]
    fov    = cam.fov[]

    W, H = resolution

    # Compute camera-to-world (USD row-vector convention).
    M = camera_to_world(eye, target, up)
    xform_str = _usda_row_vector_matrix(M)

    # Derive intrinsics from vertical FOV and image aspect ratio.
    intr = camera_intrinsics(fov, W, H)

    # Translate scene lights to USDA, passing camera matrix for camera-relative lights.
    lstr = lights_usda(scene; cam_to_world = M)

    # Bake camera + lights into root in ONE open_usd_string! call.
    author_render_root!(screen;
        resolution       = resolution,
        camera_path      = camera_path,
        camera_xform_str = xform_str,
        focal_length     = intr.focal_length,
        h_aperture       = intr.h_aperture,
        v_aperture       = intr.v_aperture,
        lights_str       = lstr)
    return nothing
end

# ------------------------------------------------------------------
# author_lights! — intent-named delegator to author_root_from_scene!
# ------------------------------------------------------------------

"""
    author_lights!(screen, scene; camera_path="/World/Camera") -> Nothing

Bake `scene`'s lights (and camera) into the USD render-root.

Delegates to `author_root_from_scene!` — camera and lights are always authored
together in one `open_usd_string!`.  This is intentional: baking them separately
would require two stage re-opens, and the second would wipe the first.

This function is an intent-named entry point for callers that are conceptually
updating the lights.  M2 will specialize it for live diffing if needed.
"""
function author_lights!(screen, scene; camera_path::String = "/World/Camera")
    return author_root_from_scene!(screen, scene; camera_path = camera_path)
end
