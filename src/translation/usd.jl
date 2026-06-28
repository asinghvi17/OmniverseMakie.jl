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
    usda_mesh(points, faces, normals, displaycolor; model = I₄) -> String

Emit a self-contained USDA layer containing a single `UsdGeomMesh` prim at the
`defaultPrim = "mesh"` path.  Intended for use with `OV.add_usd_reference!`.

# Arguments
- `points`       — iterable of 3-element point positions (Float32 in output)
- `faces`        — iterable of index iterables; any polygon arity is supported
- `normals`      — one `normal3f` per face-vertex (faceVarying interpolation)
- `displaycolor` — single (r, g, b) colour (constant interpolation)
- `model`        — optional 4×4 transform matrix applied via `xformOp:transform`
                   (defaults to identity; Makie `Mat4f` accepted directly)
"""
function usda_mesh(points, faces, normals, displaycolor;
                   model = Matrix{Float64}(LinearAlgebra.I, 4, 4))
    # Format points
    pts_str = join(
        ["($(Float32(p[1])), $(Float32(p[2])), $(Float32(p[3])))" for p in points], ", ")

    # Face vertex counts + flattened indices
    face_counts  = [length(f) for f in faces]
    face_indices = [idx for f in faces for idx in f]
    fvc_str = join(string.(face_counts), ", ")
    fvi_str = join(string.(face_indices), ", ")

    # Normals (per face-vertex)
    nrm_str = join(
        ["($(Float32(n[1])), $(Float32(n[2])), $(Float32(n[3])))" for n in normals], ", ")

    # Constant display colour
    r = Float32(displaycolor[1]); g = Float32(displaycolor[2]); b = Float32(displaycolor[3])
    col_str = "($(r), $(g), $(b))"

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
        interpolation = "faceVarying"
    )
    point3f[] points = [$(pts_str)]
    color3f[] primvars:displayColor = [$(col_str)] (
        interpolation = "constant"
    )
    uniform token subdivisionScheme = "none"
    matrix4d xformOp:transform = $(xform_str)
    uniform token[] xformOpOrder = ["xformOp:transform"]
}
"""
end

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

The RenderProduct path is fixed at `/Render/OVMakie/RenderProduct` (matching
`Screen.product`).

# upAxis
The stage is authored with `upAxis = "Z"`.  The default camera matrix was
precomputed for eye=(500,500,500), target=origin, Z-up (row-vector convention).

# Notes
- Calling this function more than once replaces the stage (ovrtx re-opens).
- After this call, `screen.setup` is set to `true`.
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
                              lights_str::String    = _DEFAULT_LIGHTS_STR)
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
}

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
    screen.setup = true
    return nothing
end
