# USD stage authoring utilities for OmniverseMakie: author_render_root! (open a
# self-renderable stage), usda_mesh (UsdGeomMesh layer), usda_matrix4d (matrix4d
# string, row-vector convention).
# Included inside the OmniverseMakie module; `screen` args are duck-typed (Screen
# need not be in scope at compile time).

# Count of render-root (re-)opens this process; incremented by author_render_root!
# (sole open_usd_string! for the root). M2.1 test asserts the stage opens exactly
# ONCE across multiple colorbuffer calls (M1 re-authored per call).
const _ROOT_OPEN_COUNT = Ref(0)

# ------------------------------------------------------------------
# USD string hygiene — guard user-supplied strings that enter USDA (B6)
# ------------------------------------------------------------------
# User values reach the authored stage as prim/property IDENTIFIERS (a VDB
# `field` name, a camera-path segment) and as `@asset@` PATHS (texture files, the
# MDL source, a `.vdb`/`.nvdb` file). An identifier with a space/dash/dot or a
# path with a literal `@` authors a CORRUPT stage that renders black (or fails to
# open) with no clear cause. These two guards fail LOUD — naming the offender and
# where it came from — instead of emitting broken USDA. Clean inputs pass through
# byte-for-byte (the non-`@` asset wrap is exactly the old `@$(path)@`).

# A legal USD prim/property identifier: an ASCII letter or underscore, then ASCII
# letters/digits/underscores (compiled once). Explicit ASCII ranges — NOT `\w` —
# so non-ASCII letters (e.g. "café") are rejected, as USD identifiers require.
# End anchor is `\z` (true end of string), NOT `$`: PCRE `$` also matches BEFORE a
# trailing newline, so `$` let "density\n" through and authored a corrupt
# `def OpenVDBAsset "density\n"`. `\z` admits no trailing newline.
const _USD_IDENTIFIER_RX = r"^[A-Za-z_][A-Za-z0-9_]*\z"

"""
    _usd_identifier(name; what="USD identifier") -> String

Return `name` as a `String` if it is a legal USD identifier
(`[A-Za-z_][A-Za-z0-9_]*`), else `error` naming the offender and where it came
from (`what`). Guards every site that interpolates a user-supplied string into a
prim/property name (a VDB `field`, a camera-path segment). Rejects spaces,
dashes, dots, leading digits, non-ASCII, and the empty string.
"""
function _usd_identifier(name::AbstractString; what::AbstractString = "USD identifier")
    occursin(_USD_IDENTIFIER_RX, name) && return String(name)
    error("OmniverseMakie: $(what) \"$(name)\" is not a valid USD identifier — a USD " *
          "prim/property name must match [A-Za-z_][A-Za-z0-9_]* (an ASCII letter or " *
          "underscore, then ASCII letters/digits/underscores; no spaces, dashes, dots, " *
          "leading digits, or non-ASCII characters).")
end

"""
    _usd_asset_path(path; what="asset path") -> String

Wrap `path` as a USDA asset reference: `@path@` normally, but `@@@path@@@` when
`path` contains a literal `@` (USDA's escape for `@` inside an asset path). A
`path` that itself contains the `@@@` delimiter is unrepresentable → `error`
naming the offender and where it came from (`what`). A clean `path` (no `@`)
returns `@path@`, byte-identical to the prior hand-written `@…@` interpolation.
"""
function _usd_asset_path(path::AbstractString; what::AbstractString = "asset path")
    if occursin("@@@", path)
        error("OmniverseMakie: $(what) \"$(path)\" contains the reserved sequence \"@@@\", " *
              "which cannot be represented in a USDA `@…@` asset reference.")
    elseif occursin('@', path)
        return "@@@$(path)@@@"
    else
        return "@$(path)@"
    end
end

# ------------------------------------------------------------------
# usda_matrix4d — convert a Makie model matrix to USD matrix4d text
# ------------------------------------------------------------------

"""
    usda_matrix4d(model) -> String

4×4 model matrix → USD `matrix4d` literal `( (r00,..), .., (tx,ty,tz,1) )`.
USD is row-vector (v'=v·M), Makie column-vector (v'=M·v), so we transpose:
the translation ends up in the last ROW.
"""
function usda_matrix4d(model)
    # Transpose: Makie col-vector → USD row-vector convention.
    usd_matrix = Float64.(collect(model'))
    rows = ntuple(i -> "($(join(usd_matrix[i,:], ", ")))", 4)
    return "( $(join(rows, ", ")) )"
end

# ------------------------------------------------------------------
# Face flattening + non-allocating number streaming (B7 authoring-path allocations)
# ------------------------------------------------------------------

"""
    _flat_faces(faces) -> (counts::Vector{Int}, indices::Vector{Int})

Flatten an iterable of faces (each an iterable of vertex indices) ONCE into USD's
`(faceVertexCounts, faceVertexIndices)` pair. `indices` are 0-based: `GeometryBasics.raw`
yields the 0-based index from an `OffsetInteger` face vertex (Makie/GeometryBasics faces)
and is identity on a plain `Integer` (a surface re-mesh / merged-marker index is already
0-based). Replaces three identical per-face `Int[Int(GeometryBasics.raw(i)) for i in f]`
comprehensions that allocated one `Int[]` per face only for the emitter to re-flatten.
"""
function _flat_faces(faces)
    counts = Vector{Int}(undef, length(faces))
    total  = 0
    for (k, f) in enumerate(faces)
        c = length(f); counts[k] = c; total += c
    end
    indices = Vector{Int}(undef, total)
    t = 0
    for f in faces, i in f
        indices[t += 1] = Int(GeometryBasics.raw(i))
    end
    return counts, indices
end

# Julia 1.12's `print(io, ::Float32/::Int)` allocates a StringVector PER call (via `Ryu.show`
# / `string`), which alone caps the streaming win near ~1.6×. `_emit_f32!`/`_emit_int!` write
# the SAME bytes as `string(x)` into a REUSED `scratch` buffer instead — Ryu shortest for
# Float32 (verified byte-identical incl. Inf/NaN), plain decimal for Int — so a whole number
# list streams with no per-element allocation. `scratch` (64 B) exceeds the longest Float32/
# Int64 token.
@inline function _emit_f32!(io::IO, scratch::Vector{UInt8}, x::Float32)
    pos = Base.Ryu.writeshortest(scratch, 1, x)          # x's shortest form at scratch[1:pos-1]
    GC.@preserve scratch unsafe_write(io, pointer(scratch), pos - 1)
    return
end
@inline function _emit_int!(io::IO, scratch::Vector{UInt8}, n::Integer)
    x = Int(n)
    x < 0 && (write(io, 0x2d); x = -x)                   # '-'
    x == 0 && (write(io, 0x30); return)                  # '0'
    ndig = 0; t = x
    while t > 0; ndig += 1; t ÷= 10; end
    t = x
    @inbounds for j in ndig:-1:1
        scratch[j] = 0x30 + (t % 10) % UInt8; t ÷= 10
    end
    GC.@preserve scratch unsafe_write(io, pointer(scratch), ndig)
    return
end

# Stream `(x, y, z), (x, y, z), …` (each component emitted Float32) into `io` — byte-identical
# to `join(["($(Float32(p[1])), …)" for p in items], ", ")` without the per-element String.
function _emit_vec3_list!(io::IO, scratch::Vector{UInt8}, items)
    first = true
    for p in items
        first ? (first = false) : print(io, ", ")
        print(io, "(")
        _emit_f32!(io, scratch, Float32(p[1])); print(io, ", ")
        _emit_f32!(io, scratch, Float32(p[2])); print(io, ", ")
        _emit_f32!(io, scratch, Float32(p[3])); print(io, ")")
    end
    return
end

# Stream `a, b, c, …` (decimal Ints) into `io` — byte-identical to `join(string.(xs), ", ")`.
function _emit_int_list!(io::IO, scratch::Vector{UInt8}, xs)
    first = true
    for x in xs
        first ? (first = false) : print(io, ", ")
        _emit_int!(io, scratch, x)
    end
    return
end

# ------------------------------------------------------------------
# usda_mesh — emit a standalone UsdGeomMesh USDA layer (M1.5 uses)
# ------------------------------------------------------------------

"""
    usda_mesh(points, face_counts, face_indices, normals, displaycolor; model=I₄,
              normal_interpolation="faceVarying", color_interpolation="constant",
              texcoords=nothing, texcoord_interpolation="vertex") -> String

Self-contained USDA layer with one `UsdGeomMesh` at `defaultPrim="mesh"` (for
`OV.add_usd_reference!`). Returns the layer string.

- `points`: 3-element positions (emitted Float32).
- `face_counts` / `face_indices`: flat `faceVertexCounts` + 0-based `faceVertexIndices`
  (built ONCE by `_flat_faces`; the emitter no longer re-flattens per-face `Int[]`s).
- `normals`: one per face-vertex ("faceVarying") or per vertex ("vertex").
- `displaycolor`: one `(r,g,b)` (constant) OR a `Vector` of `(r,g,b)` (per-vertex);
  `nothing` OMITS `primvars:displayColor` so a bound OmniPBR material governs
  shading (M3.2).
- `model`: 4×4 transform via `xformOp:transform` (default identity; `Mat4f` ok).
- `texcoords`: per-vertex `(u,v)` UVs → `primvars:st` for an OmniPBR `*_texture`
  input (M3.3); `nothing` OMITS `st`. Interpolation kwargs match the payload.

Every number list streams through one reused `IOBuffer` (no per-element String/join); the
`nothing`/no-texcoords emit is byte-for-byte the earlier output (regression guard).
"""
function usda_mesh(points, face_counts, face_indices, normals, displaycolor;
                   model = Matrix{Float64}(LinearAlgebra.I, 4, 4),
                   normal_interpolation::AbstractString = "faceVarying",
                   color_interpolation::AbstractString  = "constant",
                   texcoords = nothing,
                   texcoord_interpolation::AbstractString = "vertex")
    # One IOBuffer + one scratch buffer, reused across every number list (take! resets the buffer).
    io = IOBuffer(); scratch = Vector{UInt8}(undef, 64)
    _emit_vec3_list!(io, scratch, points);      pts_str          = String(take!(io))
    _emit_vec3_list!(io, scratch, normals);     nrm_str          = String(take!(io))
    _emit_int_list!(io, scratch, face_counts);  face_counts_str  = String(take!(io))
    _emit_int_list!(io, scratch, face_indices); face_indices_str = String(take!(io))

    # displayColor block: single constant or per-vertex vector. `nothing` (a
    # MATERIALIZED plot, M3.2) OMITS it so the bound OmniPBR material governs shading.
    # Non-`nothing` branch is byte-for-byte the earlier emit (regression guard).
    col_block = displaycolor === nothing ? "" :
        "    color3f[] primvars:displayColor = [$(_displaycolor_str(displaycolor))] (\n" *
        "        interpolation = \"$(color_interpolation)\"\n" *
        "    )\n"

    # `st` UV block (M3.3): sampled by an OmniPBR `*_texture` input. `nothing` (every
    # non-textured mesh, the default) OMITS it, keeping the emit byte-for-byte.
    st_block = texcoords === nothing ? "" :
        "    texCoord2f[] primvars:st = [$(_texcoords_str(texcoords))] (\n" *
        "        interpolation = \"$(texcoord_interpolation)\"\n" *
        "    )\n"

    xform_str = usda_matrix4d(model)

    # No upAxis in reference layers — the root stage's upAxis governs; adding it here
    # renders BLACK in ovrtx (M1.2 finding).
    return """#usda 1.0
( defaultPrim = "mesh" )
def Mesh "mesh"
{
    int[] faceVertexCounts = [$(face_counts_str)]
    int[] faceVertexIndices = [$(face_indices_str)]
    normal3f[] normals = [$(nrm_str)] (
        interpolation = "$(normal_interpolation)"
    )
    point3f[] points = [$(pts_str)]
$(col_block)$(st_block)    uniform token subdivisionScheme = "none"
    matrix4d xformOp:transform = $(xform_str)
    uniform token[] xformOpOrder = ["xformOp:transform"]
}
"""
end

# Inner text of `primvars:displayColor = [...]`: a single `(r,g,b)` → one constant
# colour; a `Vector` of `(r,g,b)` → per-vertex list. A bare 3-tuple is not an
# AbstractVector, so it stays the constant path.
function _displaycolor_str(displaycolor)
    if displaycolor isa AbstractVector && !isempty(displaycolor) &&
       first(displaycolor) isa Union{Tuple, AbstractVector}
        return join(
            ["($(Float32(color[1])), $(Float32(color[2])), $(Float32(color[3])))" for color in displaycolor], ", ")
    else
        return "($(Float32(displaycolor[1])), $(Float32(displaycolor[2])), $(Float32(displaycolor[3])))"
    end
end

# Inner text of `primvars:st = [...]`: per-vertex `(u,v)` Float32 tuples/`Vec2f` (M3.3).
_texcoords_str(texcoords) = join(
    ["($(Float32(uv[1])), $(Float32(uv[2])))" for uv in texcoords], ", ")

# ------------------------------------------------------------------
# scene_scopes_usda — nested `def Scope` skeleton mirroring the scene tree
# ------------------------------------------------------------------

"""
    scene_scopes_usda(root_scene) -> (usda::String, scene2scope::Dict{UInt64,String})

Nested `def Scope` skeleton mirroring the Makie scene tree, for embedding in the
root layer's `def Xform "World"` (M2.3 subscene grouping).

- `root_scene` → `/World` (existing root Xform, no extra scope).
- each non-root subscene → `def Scope "Scene_<objectid>"` nested under its parent,
  mirroring `scene.children`.

Returns the fragment (4-space-indented; `""` if no children) and `scene2scope`
(every scene's `objectid` → full scope path). Organizational only: NO transform,
OMITS `upAxis` (root governs). `objectid`-derived paths are stable across screens.
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

# Emit one `def Scope "Scene_<id>" {...}` (children nested recursively) and record
# its path in `scene2scope`. `depth` sets indent (1 = /World child = 4 spaces).
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

`def Scope "Looks"` (4-space indented, a `/World` child) holding OmniPBR
`UsdShade Material` prims. `materials` is the composed body (default `""` = empty
scope); authored into the root by `author_root_from_scene!`.

⚠️ M3.1 CONSTRAINT (load-bearing): a Material must be PRE-AUTHORED at open-time
(composed into this body) to be bindable. Adding one to the OPEN stage via
`OV.add_usd_reference!` is a SILENT NO-OP for `material:binding` in our ovrtx build,
regardless of timing. `bind_material!` works at runtime only on a material present
when the stage opened. So materials compose in here, not by reference.
Organizational: no transform, OMITS `upAxis` (root governs).
"""
looks_scope_usda(materials::AbstractString = "") = """    def Scope "Looks"
    {
$(materials)    }
"""

# ------------------------------------------------------------------
# author_render_root! — open a self-renderable USD stage in the renderer
# ------------------------------------------------------------------

"""
    author_render_root!(screen; resolution=screen.fb_size,
                        camera_path="/World/Camera", ...) -> Nothing

Author a complete self-renderable USD stage and open it in `screen.renderer` via
`OV.open_usd_string!`. Contains: default `DistantLight "Sun"`, a `Camera` at
`camera_path`, and the render-config hierarchy `/Render/OVMakie/RenderProduct`
(fixed path, matching `Screen.product`) with the `OmniRtxSettings*`/`OmniRtxPost*`
apiSchemas, `/Render/GlobalRenderSettings`, and `/Render/Vars/LdrColor`.

`scopes_str` (M2.3, default `""`): optional nested `def Scope` skeleton embedded in
`def Xform "World"` so plot references nest at `/World/Scene_<id>/plot_<id>`.

Authored `upAxis="Z"`; the default camera matrix assumes eye=(500,500,500),
target=origin, Z-up (row-vector). Each call replaces the stage (ovrtx re-opens).
"""
# Default camera-to-world xform (eye=(500,500,500), lookat=origin, Z-up); placeholder
# when author_render_root! gets no scene camera. Matches the M1.2 spike matrix exactly.
const _DEFAULT_CAMERA_XFORM_STR = "( (-0.70711, 0.70711, 0.0, 0.0), (-0.40825, -0.40825, 0.81650, 0.0), (0.57735, 0.57735, 0.57735, 0.0), (500.0, 500.0, 500.0, 1.0) )"

# Default lights block: a single DistantLight "Sun" (M1.2 spike). Fallback when
# author_render_root! gets no lights_str, or lights_usda finds no scene lights.
# 4-space indented (World child), ends with two newlines to sit before the Camera prim.
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
    cam_name   = split(camera_path, "/")[end]   # last segment = prim name under /World
    # M6.B selection-outline (verified by rendering): the RTX outline pass is (1)
    # SUPPRESSED by the explicit post-process `apiSchemas` list and (2) drawn WHITE (not
    # the group colour) when `HdrColor` is an ordered AOV. So selection_outline=true
    # authors a LEANER RenderProduct: no post-process apiSchemas, no `omni:rtx:*` settings
    # (which need those schemas), LdrColor-only. The false branch is byte-for-byte the
    # pre-M6.B USDA, so non-selection renders are unchanged; outline just isn't combined
    # with HdrColor GPU-direct present! (falls back to the LdrColor blit).
    selection_outline = screen.config.selection_outline
    rtx_lines  = selection_outline ? "" : rtx_settings_usda(screen.config)
    rp_schemas = selection_outline ? "" :
        "            prepend apiSchemas = [\"OmniRtxSettingsCommonAdvancedAPI_1\", \"OmniRtxSettingsRtAdvancedAPI_1\", \"OmniRtxSettingsPtAdvancedAPI_1\", \"OmniRtxPostColorGradingAPI_1\", \"OmniRtxPostChromaticAberrationAPI_1\", \"OmniRtxPostBloomPhysicalAPI_1\", \"OmniRtxPostMatteObjectAPI_1\", \"OmniRtxPostCompositingAPI_1\", \"OmniRtxPostDofAPI_1\", \"OmniRtxPostMotionBlurAPI_1\", \"OmniRtxPostTvNoiseAPI_1\", \"OmniRtxPostTonemapIrayReinhardAPI_1\", \"OmniRtxPostDebugSettingsAPI_1\", \"OmniRtxDebugSettingsAPI_1\"]\n"
    ordered_vars = selection_outline ? "[</Render/Vars/LdrColor>]" :
                              "[</Render/Vars/LdrColor>, </Render/Vars/HdrColor>]"
    hdr_var    = selection_outline ? "" :
        "        def RenderVar \"HdrColor\" (\n            hide_in_stage_window = true\n            no_delete = true\n        )\n        {\n            uniform string sourceName = \"HdrColor\"\n        }\n"

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
$(rp_schemas)            hide_in_stage_window = true
            no_delete = true
        )
        {
            rel camera = <$(camera_path)>
$(rtx_lines)
            rel orderedVars = $(ordered_vars)
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
$(hdr_var)    }
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

Canonical single-pass root authorer: reads the scene's 3-D camera and all lights
(`scene.compute[:lights][]`) and calls `author_render_root!` exactly ONCE with both
baked in.

Open-stage model (M2.1): baked ONCE on first `colorbuffer`; later camera/light
attribute changes go through live writes (`sync_camera!`/`sync_lights!`), not a
re-bake. Re-open only on a structural change.

Side effect: (re-)opens the stage, so all prior `OV.add_usd_reference!` refs are
lost and must be re-added. Precondition: `scene` has a 3-D camera controller.
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

    # Camera-to-world (USD row-vector) + intrinsics from vertical FOV and aspect.
    cam_to_world = camera_to_world(eye, target, up)
    xform_str = _usda_row_vector_matrix(cam_to_world)
    intrinsics = camera_intrinsics(fov, W, H)

    # Scene lights → USDA (camera matrix orients camera-relative lights).
    lights_str = lights_usda(scene; cam_to_world = cam_to_world)

    # M2.3: nested `def Scope` per subscene so plot refs nest at
    # /World/Scene_<id>/plot_<id>. Rooted at the scene `insertplots!` walks —
    # `screen.scene` (figure root on save, or the LScene scene), NOT the camera scene.
    # Store the map so `add_scene!`/`plot_prim_path` resolve nested paths first.
    root_scene = something(screen.scene, scene)
    scopes_str, scene2scope = scene_scopes_usda(root_scene)
    screen.scene2scope = scene2scope

    # M3.2: append a `/World/Looks` scope PRE-AUTHORING one OmniPBR material per
    # materialized plot. Materials MUST be pre-authored at open-time — one added to the
    # OPEN stage via `add_usd_reference!` is not bindable (M3.1); the build branch binds
    # at runtime with `OV.bind_material!`. Empty when nothing is materialized.
    scopes_str = scopes_str * materialized_looks_usda(root_scene)

    # Bake camera + lights + scopes into the root in ONE open_usd_string! call.
    author_render_root!(screen;
        resolution       = resolution,
        camera_path      = camera_path,
        camera_xform_str = xform_str,
        focal_length     = intrinsics.focal_length,
        h_aperture       = intrinsics.h_aperture,
        v_aperture       = intrinsics.v_aperture,
        lights_str       = lights_str,
        scopes_str       = scopes_str)
    return nothing
end
