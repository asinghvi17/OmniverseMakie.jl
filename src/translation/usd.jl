# USD stage authoring utilities for OmniverseMakie.
#
# Provides:
#   author_render_root!(screen; resolution, camera_path) — open a self-renderable stage
#   usda_mesh(points, faces, normals, displaycolor; model) — UsdGeomMesh USDA layer
#   usda_matrix4d(model) — USD matrix4d string (row-vector / row-major convention)
#
# NOTE: this file is included inside the OmniverseMakie module (after OV.jl and
# settings.jl, before screen.jl).  `screen` arguments are duck-typed so the
# Screen type does not need to be in scope at compile time.

# Number of times the render-root stage has been (re-)opened in THIS process.
# Incremented by `author_render_root!` (the sole `OV.open_usd_string!` for the root).
# Used by the M2.1 open-stage test to assert the stage is authored exactly ONCE
# across multiple `colorbuffer` calls (M1 re-authored per call → 2; M2.1 → 1).
const _ROOT_OPEN_COUNT = Ref(0)

# ------------------------------------------------------------------
# usda_matrix4d — convert a Makie model matrix to USD matrix4d text
# ------------------------------------------------------------------

"""
    usda_matrix4d(model) -> String

Convert a 4×4 model matrix to a USD `matrix4d` literal string.

USD uses a **row-vector** convention (v' = v·M), while Makie uses column-major
column-vector convention (v' = M·v).  Transposing Makie's matrix yields the
USD form where the translation lives in the last ROW (indices [4,1:3]).

The output format is `( (r00,r01,r02,r03), (r10,...), ..., (tx,ty,tz,1) )`.
"""
function usda_matrix4d(model)
    # Transpose: Makie col-major col-vector → USD row-major row-vector.
    M = Float64.(collect(model'))   # 4×4 Float64, row-major layout
    rows = ntuple(i -> "($(join(M[i,:], ", ")))", 4)
    return "( $(join(rows, ", ")) )"
end

# ------------------------------------------------------------------
# usda_mesh — emit a standalone UsdGeomMesh USDA layer (M1.5 uses)
# ------------------------------------------------------------------

"""
    usda_mesh(points, faces, normals, displaycolor; model = I₄,
              normal_interpolation = "faceVarying", color_interpolation = "constant") -> String

Emit a self-contained USDA layer containing a single `UsdGeomMesh` prim at the
`defaultPrim = "mesh"` path.  Intended for use with `OV.add_usd_reference!`.

# Arguments
- `points`       — iterable of 3-element point positions (Float32 in output)
- `faces`        — iterable of index iterables; any polygon arity is supported.
                   Indices must be **0-based** (USD convention).
- `normals`      — one `normal3f` per face-vertex (`faceVarying`) or per vertex
                   (`vertex`), per `normal_interpolation`.
- `displaycolor` — a single `(r, g, b)` colour (constant interpolation) **or** a
                   `Vector` of `(r, g, b)` colours (per-vertex interpolation),
                   per `color_interpolation`.
- `model`        — optional 4×4 transform matrix applied via `xformOp:transform`
                   (defaults to identity; Makie `Mat4f` accepted directly).

# Keyword arguments (backward-compatible defaults match M1.2)
- `normal_interpolation::AbstractString = "faceVarying"` — M1.5 meshes pass
  `"vertex"` (Makie supplies per-vertex normals).
- `color_interpolation::AbstractString = "constant"` — M1.5 passes `"constant"`
  (single colour) or `"vertex"` (per-vertex `displaycolor` vector).
"""
function usda_mesh(points, faces, normals, displaycolor;
                   model = Matrix{Float64}(LinearAlgebra.I, 4, 4),
                   normal_interpolation::AbstractString = "faceVarying",
                   color_interpolation::AbstractString  = "constant")
    # Format points
    pts_str = join(
        ["($(Float32(p[1])), $(Float32(p[2])), $(Float32(p[3])))" for p in points], ", ")

    # Face vertex counts + flattened indices
    face_counts  = [length(f) for f in faces]
    face_indices = [idx for f in faces for idx in f]
    fvc_str = join(string.(face_counts), ", ")
    fvi_str = join(string.(face_indices), ", ")

    # Normals (faceVarying or per-vertex, per normal_interpolation)
    nrm_str = join(
        ["($(Float32(n[1])), $(Float32(n[2])), $(Float32(n[3])))" for n in normals], ", ")

    # Display colour: single (r,g,b) → constant; Vector of (r,g,b) → per-vertex.
    col_str = _displaycolor_str(displaycolor)

    xform_str = usda_matrix4d(model)

    # NOTE: no upAxis in reference layers — the root stage's upAxis governs.
    # Including upAxis here caused ovrtx to render black (M1.2 diagnostic finding).
    return """#usda 1.0
( defaultPrim = "mesh" )
def Mesh "mesh"
{
    int[] faceVertexCounts = [$(fvc_str)]
    int[] faceVertexIndices = [$(fvi_str)]
    normal3f[] normals = [$(nrm_str)] (
        interpolation = "$(normal_interpolation)"
    )
    point3f[] points = [$(pts_str)]
    color3f[] primvars:displayColor = [$(col_str)] (
        interpolation = "$(color_interpolation)"
    )
    uniform token subdivisionScheme = "none"
    matrix4d xformOp:transform = $(xform_str)
    uniform token[] xformOpOrder = ["xformOp:transform"]
}
"""
end

# Format a displayColor payload as the inner text of `primvars:displayColor = [...]`.
# A single `(r, g, b)` tuple → one constant colour; a `Vector` of `(r, g, b)` → a
# comma-separated per-vertex list.  (A bare 3-tuple is NOT an AbstractVector, so it
# is correctly treated as one colour, preserving the M1.2/1.3/1.4 constant path.)
function _displaycolor_str(dc)
    if dc isa AbstractVector && !isempty(dc) && first(dc) isa Union{Tuple, AbstractVector}
        return join(
            ["($(Float32(c[1])), $(Float32(c[2])), $(Float32(c[3])))" for c in dc], ", ")
    else
        return "($(Float32(dc[1])), $(Float32(dc[2])), $(Float32(dc[3])))"
    end
end

# ------------------------------------------------------------------
# scene_scopes_usda — nested `def Scope` skeleton mirroring the scene tree
# ------------------------------------------------------------------

"""
    scene_scopes_usda(root_scene) -> (usda::String, scene2scope::Dict{UInt64,String})

Walk `root_scene` and its `.children` recursively and emit a nested `def Scope`
skeleton mirroring the Makie scene tree, for embedding inside the root layer's
`def Xform "World"` block (M2.3 subscene grouping — the open USD prim hierarchy then
mirrors the Makie `Scene` graph).

- `root_scene` maps to `/World` (the existing root Xform — NO extra `Scene_` scope).
- each non-root subscene becomes a `def Scope "Scene_<objectid(scene)>"` nested under
  its parent scene's scope, so the prim hierarchy mirrors `scene.children`.

Returns the USDA fragment (scope blocks, 4-space-indented as direct children of
`/World`; `""` when `root_scene` has no children) and `scene2scope`, mapping every
scene's `objectid` to its full scope path.

Purely organizational: scopes carry NO transform (plots keep their composed-world
`:model_f32c`) and the fragment OMITS `upAxis` (only the root governs — M1.2).  Paths
derive from stable `objectid`s, so they are identical across screens (consistent with
the M2.2 `:ovrtx_screen` rebuild, which re-authors on a new screen).
"""
function scene_scopes_usda(root_scene)
    scene2scope = Dict{UInt64,String}()
    scene2scope[objectid(root_scene)] = "/World"
    buf = IOBuffer()
    for child in root_scene.children
        _emit_scene_scope!(buf, scene2scope, child, "/World", 1)
    end
    return (String(take!(buf)), scene2scope)
end

# Emit one `def Scope "Scene_<id>" { … }` block (children recursively nested) and
# record its full path in `scene2scope`.  `depth` sets the indent (1 = direct child
# of /World → 4 spaces, matching the camera/light blocks).
function _emit_scene_scope!(buf::IO, scene2scope::Dict{UInt64,String}, scene,
                            parent_path::AbstractString, depth::Int)
    name = "Scene_$(objectid(scene))"
    path = "$(parent_path)/$(name)"
    scene2scope[objectid(scene)] = path
    indent = "    "^depth
    print(buf, indent, "def Scope \"", name, "\"\n", indent, "{\n")
    for child in scene.children
        _emit_scene_scope!(buf, scene2scope, child, path, depth + 1)
    end
    print(buf, indent, "}\n")
    return
end

# ------------------------------------------------------------------
# looks_scope_usda — `def Scope "Looks"` holding (pre-authored) OmniPBR materials (M3.1)
# ------------------------------------------------------------------

"""
    looks_scope_usda(materials::AbstractString = "") -> String

Emit a `def Scope "Looks"` block (4-space indented, a direct child of `/World`, like
the M2.3 scene scopes) to hold OmniPBR `UsdShade Material` prims.  `materials` is the
USDA body composed inside it (one or more `usda_omnipbr_material` fragments); the
default `""` yields an empty scope.  Authored into the root by `author_root_from_scene!`
alongside the scene scopes.

⚠️ M3.1 VALIDATED CONSTRAINT (load-bearing): an OmniPBR `Material` must be
PRE-AUTHORED into the stage at open-time (composed into this scope's body) to be
usable.  A `Material` added to the OPEN stage at runtime via `OV.add_usd_reference!`
is a SILENT NO-OP for `material:binding` in our ovrtx build — the MDL material never
becomes bindable, regardless of timing (before/after the first render).  `bind_material!`
itself works at runtime, but only on a material that was present when the stage opened.
So materials compose INTO this scope at author-time; they are NOT added by reference.

Purely organizational: carries no transform and OMITS `upAxis` (only the root
governs — M1.2).
"""
looks_scope_usda(materials::AbstractString = "") = """    def Scope "Looks"
    {
$(materials)    }
"""

# ------------------------------------------------------------------
# author_render_root! — open a self-renderable USD stage in the renderer
# ------------------------------------------------------------------

"""
    author_render_root!(screen; resolution = screen.fb_size,
                        camera_path = "/World/Camera") -> Nothing

Author a complete, self-renderable USD stage and open it in `screen.renderer`
via `OV.open_usd_string!`.  The stage contains:

- A `DistantLight "Sun"` under `/World` for scene illumination.
- A placeholder `Camera` at `camera_path` (M1.3 replaces it with the scene
  camera; M1.4 replaces the light).
- The full render-config hierarchy: `/Render/OVMakie/RenderProduct` with the
  `OmniRtxSettings*` / `OmniRtxPost*` apiSchemas proven by the M1.2 spike,
  `/Render/GlobalRenderSettings`, and `/Render/Vars/LdrColor`.
- `scopes_str` (M2.3): an optional nested `def Scope` skeleton (from
  `scene_scopes_usda`) embedded inside `def Xform "World"` as a sibling of the
  camera, so plot references can nest at `/World/Scene_<id>/plot_<id>`.  Defaults
  to `""` (no subscene scopes — the M1 flat layout).

The RenderProduct path is fixed at `/Render/OVMakie/RenderProduct` (matching
`Screen.product`).

# upAxis
The stage is authored with `upAxis = "Z"`.  The default camera matrix was
precomputed for eye=(500,500,500), target=origin, Z-up (row-vector convention).

# Notes
- Calling this function more than once replaces the stage (ovrtx re-opens).
"""
# Default camera-to-world xform string (eye=(500,500,500), lookat=origin, Z-up).
# Used as the placeholder when author_render_root! is called without a scene camera.
# Matches the M1.2 spike-verified matrix exactly.
const _DEFAULT_CAMERA_XFORM_STR = "( (-0.70711, 0.70711, 0.0, 0.0), (-0.40825, -0.40825, 0.81650, 0.0), (0.57735, 0.57735, 0.57735, 0.0), (500.0, 500.0, 500.0, 1.0) )"

# Default lights block — a single DistantLight "Sun" that matches the M1.2 spike stage.
# Used as the fallback when author_render_root! is called without a lights_str kwarg, and
# when lights_usda finds no lights in the scene.  The block is 4-space indented (World child)
# and ends with two newlines so it fits directly before the Camera prim in the template.
const _DEFAULT_LIGHTS_STR = """    def DistantLight "Sun" (
        prepend apiSchemas = ["ShapingAPI"]
    )
    {
        float inputs:angle = 1
        float inputs:intensity = 3000
        matrix4d xformOp:transform = ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }

"""

function author_render_root!(screen;
                              resolution = screen.fb_size,
                              camera_path::String = "/World/Camera",
                              camera_xform_str::String = _DEFAULT_CAMERA_XFORM_STR,
                              focal_length::Float64 = 18.147562,
                              h_aperture::Float64   = 20.955,
                              v_aperture::Float64   = 15.2908,
                              lights_str::String    = _DEFAULT_LIGHTS_STR,
                              scopes_str::String    = "")
    W, H       = resolution
    cam_name   = split(camera_path, "/")[end]   # last segment → prim name under /World
    rtx_lines  = rtx_settings_usda(screen.config)

    usda = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 0.01
    upAxis = "Z"
    startTimeCode = 0
    endTimeCode = 100
    timeCodesPerSecond = 60
)

def Xform "World"
{
$(lights_str)    def Camera "$(cam_name)" (
        prepend apiSchemas = ["OmniRtxCameraAutoExposureAPI_1", "OmniRtxCameraExposureAPI_1"]
    )
    {
        float2 clippingRange = (1, 10000000)
        float focalLength = $(focal_length)
        float focusDistance = 400
        float horizontalAperture = $(h_aperture)
        float verticalAperture = $(v_aperture)
        token projection = "perspective"
        matrix4d xformOp:transform = $(camera_xform_str)
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
$(scopes_str)}

def "Render" (
    hide_in_stage_window = true
    no_delete = true
)
{
    def "OVMakie"
    {
        def RenderProduct "RenderProduct" (
            prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1", "OmniRtxSettingsRtAdvancedAPI_1", "OmniRtxSettingsPtAdvancedAPI_1", "OmniRtxPostColorGradingAPI_1", "OmniRtxPostChromaticAberrationAPI_1", "OmniRtxPostBloomPhysicalAPI_1", "OmniRtxPostMatteObjectAPI_1", "OmniRtxPostCompositingAPI_1", "OmniRtxPostDofAPI_1", "OmniRtxPostMotionBlurAPI_1", "OmniRtxPostTvNoiseAPI_1", "OmniRtxPostTonemapIrayReinhardAPI_1", "OmniRtxPostDebugSettingsAPI_1", "OmniRtxDebugSettingsAPI_1"]
            hide_in_stage_window = true
            no_delete = true
        )
        {
            rel camera = <$(camera_path)>
$(rtx_lines)
            rel orderedVars = </Render/Vars/LdrColor>
            uniform int2 resolution = ($(W), $(H))
        }
    }

    def RenderSettings "GlobalRenderSettings" (
        prepend apiSchemas = ["OmniRtxSettingsGlobalRtAdvancedAPI_1", "OmniRtxSettingsGlobalPtAdvancedAPI_1"]
        no_delete = true
    )
    {
        rel products = </Render/OVMakie/RenderProduct>
    }

    def "Vars"
    {
        def RenderVar "LdrColor" (
            hide_in_stage_window = true
            no_delete = true
        )
        {
            uniform string sourceName = "LdrColor"
        }
    }
}
"""
    OV.open_usd_string!(screen.renderer, usda)
    _ROOT_OPEN_COUNT[] += 1
    return nothing
end

# ------------------------------------------------------------------
# author_root_from_scene! — canonical bake: camera + lights in ONE open
# (relocated from lights.jl in M2.1 — composition lives with USD authoring)
# ------------------------------------------------------------------

"""
    author_root_from_scene!(screen, scene; resolution=screen.fb_size,
                             camera_path="/World/Camera") -> Nothing

Canonical single-pass root authorer: reads the scene's 3-D camera AND all lights
from `scene.compute[:lights][]`, then calls `author_render_root!` exactly ONCE
(one `open_usd_string!`) with both baked in.

Open-stage model (M2.1): the root is baked ONCE on the first `colorbuffer`; later
camera/light *attribute* changes are pushed as live writes (`sync_camera!` /
`sync_lights!`), NOT a re-bake.  A re-open happens only on a structural change.

# Side effect
Calls `author_render_root!` → the stage is (re-)opened.  All previously added USD
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

    # M2.3: author a `def Scope` per Makie subscene, nested to mirror the scene tree,
    # so plot references nest at /World/Scene_<id>/plot_<id>.  The hierarchy is rooted
    # at the scene that `insertplots!` walks — `screen.scene` (the figure ROOT when
    # `save(fig)`, or the LScene scene when `colorbuffer(ax.scene)`), NOT the camera
    # scene (a descendant).  Store the map on the screen so `add_scene!` /
    # `plot_prim_path` resolve each plot's nested path before any reference is added.
    root_scene = something(screen.scene, scene)
    scopes_str, scene2scope = scene_scopes_usda(root_scene)
    screen.scene2scope = scene2scope

    # M3.1: author a `/World/Looks` scope alongside the scene scopes to hold OmniPBR
    # materials.  Empty here (no materialized plots in M3.1); the M3.2 build branch
    # composes each materialized plot's `usda_omnipbr_material` fragment INTO this
    # scope's body at author-time.  Materials MUST be pre-authored (open-time) — a
    # Material added to the OPEN stage via `add_usd_reference!` is not bindable
    # (M3.1-validated); `OV.bind_material!` then binds at runtime.
    scopes_str = scopes_str * looks_scope_usda()

    # Bake camera + lights + scope skeleton into root in ONE open_usd_string! call.
    author_render_root!(screen;
        resolution       = resolution,
        camera_path      = camera_path,
        camera_xform_str = xform_str,
        focal_length     = intr.focal_length,
        h_aperture       = intr.h_aperture,
        v_aperture       = intr.v_aperture,
        lights_str       = lstr,
        scopes_str       = scopes_str)
    return nothing
end
