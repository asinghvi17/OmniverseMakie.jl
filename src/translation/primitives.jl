# Primitive translations: scatter / meshscatter / lines / surface.  Each emits a
# self-contained USDA reference layer (defaultPrim, no upAxis), added under the
# plot's scope path (/World/Scene_<id>/plot_<id>) via OV.add_usd_reference!.
#
# The USDA emitters are LIVE: author_usd_prim! (compute.jl) drives them from
# resolved compute outputs.  Surface (empty consumed_inputs) is the ONE type
# still built via to_ovrtx_object here; old build wrappers were dead + removed.
#
# Schemas (all non-black through ovrtx RT2; guarded by m1_primitives_test.jl):
#   Scatter/MeshScatter → PointInstancer + Sphere/Mesh prototype
#   Lines/LineSegments  → BasisCurves (uniform token type = "linear")
#   Surface             → Mesh (grid re-meshed; reuses usda_mesh)
# ovrtx honors both PointInstancer and BasisCurves, so the merged-mesh fallbacks
# (marker copies / tube mesh, both on the usda_mesh path) are NOT implemented.
# They are the documented switch if a build regresses a schema (test -> RED).
#
# Included in the OmniverseMakie module AFTER meshes.jl.  In scope: Makie,
# GeometryBasics, OV, LinearAlgebra (cross/norm/normalize), usda_mesh,
# usda_matrix4d, displaycolor_for, _rgb, _colorrange, _displaycolor_str.

# ------------------------------------------------------------------
# small formatting / data helpers
# ------------------------------------------------------------------

# (x, y, z) Float32 tuple from any 2- or 3-element point (Point2 → z = 0).
_point3f(p) = (Float32(p[1]), Float32(p[2]), Float32(length(p) >= 3 ? p[3] : 0.0f0))

# Join a sequence of points as USD `point3f[]` body text: "(x, y, z), (x, y, z), ...".
_point3f_list(pts) = join(["($(c[1]), $(c[2]), $(c[3]))" for c in (_point3f(p) for p in pts)], ", ")

# Per-instance (sx, sy, sz) scale from one markersize element (a scalar or a Vec).
_scale_tuple(s::Real) = (Float32(s), Float32(s), Float32(s))
function _scale_tuple(s)
    sx = Float32(s[1])
    sy = Float32(length(s) >= 2 ? s[2] : s[1])
    sz = Float32(length(s) >= 3 ? s[3] : s[1])
    return (sx, sy, sz)
end

# Resolve a markersize attribute into one (sx, sy, sz) per point.
# A single `Vec`/scalar broadcasts; a `Base.Array` of length `n` is per-point.
# (A single `Vec2f`/`Vec3f` is a StaticArray, NOT a `Base.Array`, so it broadcasts.)
function _scales_for(markersize, n)
    if markersize isa Base.Array && length(markersize) == n
        return [_scale_tuple(m) for m in markersize]
    end
    return fill(_scale_tuple(markersize), n)
end

# Makie's `Quaternion` stores `data = (x, y, z, w)`; USD `quath` wants (w, x, y, z).
_quat_wxyz(q) = (Float32(q[4]), Float32(q[1]), Float32(q[2]), Float32(q[3]))

# Resolve a rotation attribute into per-instance (w, x, y, z) orientations, or
# `nothing` when every rotation is (near) identity — in which case we omit the
# `orientations` attribute entirely (fewer quath tokens for ovrtx to parse).
function _orientations_for(rot, n)
    quats = rot isa Base.Array && length(rot) == n ? rot : fill(rot, n)
    out = NTuple{4,Float32}[]
    all_identity = true
    for q in quats
        wxyz = try
            _quat_wxyz(q)
        catch
            (1.0f0, 0.0f0, 0.0f0, 0.0f0)   # unknown rotation form → identity
        end
        (abs(wxyz[2]) + abs(wxyz[3]) + abs(wxyz[4]) > 1.0f-6) && (all_identity = false)
        push!(out, wxyz)
    end
    return all_identity ? nothing : out
end

# Axis-aligned bounding-box diagonal length of a point cloud (Float64).
function _bbox_diag(pts)
    isempty(pts) && return 1.0
    lo = [Float64(_point3f(first(pts))[i]) for i in 1:3]
    hi = copy(lo)
    for p in pts
        c = _point3f(p)
        for i in 1:3
            lo[i] = min(lo[i], Float64(c[i])); hi[i] = max(hi[i], Float64(c[i]))
        end
    end
    d = sqrt(sum((hi .- lo) .^ 2))
    return d < 1e-8 ? 1.0 : d
end

# ------------------------------------------------------------------
# UsdGeomPointInstancer authoring (Scatter / MeshScatter primary)
# ------------------------------------------------------------------

# `def Sphere "proto" { ... }` body (4-space indented, child of the instancer).
# `color_const` (an (r,g,b) tuple) is baked as a constant displayColor when given;
# `nothing` (a materialized instancer) OMITS displayColor.
function _sphere_proto_body(color_const)
    col = color_const === nothing ? "" : """
        color3f[] primvars:displayColor = [($(Float32(color_const[1])), $(Float32(color_const[2])), $(Float32(color_const[3])))] (
            interpolation = "constant"
        )
"""
    return """    def Sphere "proto"
    {
        double radius = 1
$(col)    }"""
end

# `def Mesh "proto" { ... }` body from an explicit mesh (4-space indented).
# `color_const === nothing` (a materialized instancer) OMITS displayColor.
function _mesh_proto_body(points, faces0, normals, color_const)
    counts_str  = join(string.([length(f) for f in faces0]), ", ")
    indices_str = join(string.([i for f in faces0 for i in f]), ", ")
    normals_str = join(["($(Float32(n[1])), $(Float32(n[2])), $(Float32(n[3])))" for n in normals], ", ")
    col = color_const === nothing ? "" : """
        color3f[] primvars:displayColor = [($(Float32(color_const[1])), $(Float32(color_const[2])), $(Float32(color_const[3])))] (
            interpolation = "constant"
        )
"""
    return """    def Mesh "proto"
    {
        int[] faceVertexCounts = [$(counts_str)]
        int[] faceVertexIndices = [$(indices_str)]
        normal3f[] normals = [$(normals_str)] (
            interpolation = "vertex"
        )
        point3f[] points = [$(_point3f_list(points))]
$(col)        uniform token subdivisionScheme = "none"
    }"""
end

# ------------------------------------------------------------------
# Merged-instances mesh — materialized Scatter / MeshScatter (M3.5).
# ovrtx does NOT honor a PointInstancer material:binding (instances render the
# MDL default wherever bound — verified).  So a materialized scatter/meshscatter
# becomes ONE merged UsdGeomMesh of marker copies (binds a material like Mesh).
# Non-materialized scatter stays on the PointInstancer path.
# ------------------------------------------------------------------

# Rotate a 3-vector by a quaternion (w, x, y, z) — `v + 2w(q×v) + 2(q×(q×v))`.
function _quat_rotate(q, v)
    w, x, y, z = Float32(q[1]), Float32(q[2]), Float32(q[3]), Float32(q[4])
    vx, vy, vz = Float32(v[1]), Float32(v[2]), Float32(v[3])
    tx = 2f0 * (y * vz - z * vy); ty = 2f0 * (z * vx - x * vz); tz = 2f0 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx))
end

# Concatenate the marker mesh (`mpts`/`mfaces` 0-based/`mnrm`) transformed to each
# instance (per-instance `scales`, optional `orientations` quaternions, `positions`)
# into ONE (points, 0-based faces, normals) mesh.
function _merged_instances_mesh(mpts, mfaces, mnrm, positions, scales, orientations)
    P = Point3f[]; N = Vec3f[]; F = Vector{Int}[]
    nmark = length(mpts)
    for i in eachindex(positions)
        s = scales[i]; pos = positions[i]
        q = orientations === nothing ? nothing : orientations[i]
        off = length(P)
        for k in 1:nmark
            marker_pt   = mpts[k]
            scaled_pt   = (Float32(marker_pt[1]) * s[1], Float32(marker_pt[2]) * s[2], Float32(marker_pt[3]) * s[3])
            rotated_pt  = q === nothing ? scaled_pt : _quat_rotate(q, scaled_pt)
            push!(P, Point3f(rotated_pt[1] + Float32(pos[1]), rotated_pt[2] + Float32(pos[2]), rotated_pt[3] + Float32(pos[3])))
            marker_nrm  = mnrm[k]
            rotated_nrm = q === nothing ? (Float32(marker_nrm[1]), Float32(marker_nrm[2]), Float32(marker_nrm[3])) : _quat_rotate(q, marker_nrm)
            push!(N, Vec3f(rotated_nrm[1], rotated_nrm[2], rotated_nrm[3]))
        end
        for f in mfaces
            push!(F, Int[off + idx for idx in f])
        end
    end
    return P, F, N
end

# Author a complete PointInstancer reference layer around `proto_body`.
# instancer_color: per-instance (r,g,b) -> displayColor interp "vertex" on the
# instancer; nothing -> prototype carries a constant colour instead.
function _usda_pointinstancer(positions, scales, orientations, instancer_color, proto_body; model)
    n        = length(positions)
    pos_str  = _point3f_list(positions)
    idx_str  = join(fill("0", n), ", ")
    scl_str  = join(["($(s[1]), $(s[2]), $(s[3]))" for s in scales], ", ")
    ori_block = orientations === nothing ? "" :
        "    quath[] orientations = [" *
        join(["($(o[1]), $(o[2]), $(o[3]), $(o[4]))" for o in orientations], ", ") * "]\n"
    col_block = instancer_color === nothing ? "" : """
    color3f[] primvars:displayColor = [$(_displaycolor_str(instancer_color))] (
        interpolation = "vertex"
    )
"""
    return """#usda 1.0
( defaultPrim = "inst" )
def PointInstancer "inst"
{
    point3f[] positions = [$(pos_str)]
    int[] protoIndices = [$(idx_str)]
    float3[] scales = [$(scl_str)]
$(ori_block)$(col_block)    rel prototypes = [</inst/proto>]
    matrix4d xformOp:transform = $(usda_matrix4d(model))
    uniform token[] xformOpOrder = ["xformOp:transform"]
$(proto_body)
}
"""
end

# Live Scatter/MeshScatter build = author_usd_prim!(::Scatter/::MeshScatter) in
# compute.jl (reuses the emitters above); old to_ovrtx_object methods removed.

# ------------------------------------------------------------------
# UsdGeomBasisCurves authoring (Lines / LineSegments primary)
# ------------------------------------------------------------------

# Visible data-unit curve width from a pixel-space linewidth.  Pixel-accurate
# sizing is a later refinement (M1 only needs a visible curve): scale a small
# fraction of the data extent by the linewidth.
function _curve_width(pts, linewidth)
    lw = linewidth isa AbstractVector ?
        (isempty(linewidth) ? 1.0 : Float64(sum(linewidth) / length(linewidth))) :
        Float64(linewidth)
    return Float32(max(0.01 * lw * _bbox_diag(pts), 1.0f-4))
end

function _usda_basiscurves(points, counts, width, color_values, color_interp; model)
    counts_str = join(string.(counts), ", ")
    # color_values === nothing (MATERIALIZED) OMITS primvars:displayColor, so
    # the bound OmniPBR material governs shading; the non-nothing branch is the
    # byte-for-byte M1 emit (regression guard).  Mirrors usda_mesh's col_block.
    col_block = color_values === nothing ? "" :
        """    color3f[] primvars:displayColor = [$(_displaycolor_str(color_values))] (
        interpolation = "$(color_interp)"
    )
"""
    return """#usda 1.0
( defaultPrim = "curve" )
def BasisCurves "curve"
{
    uniform token type = "linear"
    int[] curveVertexCounts = [$(counts_str)]
    point3f[] points = [$(_point3f_list(points))]
    float[] widths = [$(width)] (
        interpolation = "constant"
    )
$(col_block)    matrix4d xformOp:transform = $(usda_matrix4d(model))
    uniform token[] xformOpOrder = ["xformOp:transform"]
}
"""
end

# Live Lines/LineSegments build = author_usd_prim!(::Lines/::LineSegments) in
# compute.jl (reuses the emitters above); old to_ovrtx_object methods removed.

# ------------------------------------------------------------------
# Surface -> UsdGeomMesh (grid re-meshed; reuses usda_mesh).  Surface has NO
# compute outputs => empty consumed_inputs => the ONE plot type still built here
# (via register_ovrtx_robj!'s empty-inputs branch).  `scene` is threaded so its
# reference nests under the scene's def Scope like every other plot.
# ------------------------------------------------------------------

# Build (points, 0-based quad faces, per-vertex normals) for a grid surface.
# `zs` is an (nx, ny) matrix; `xs`/`ys` are vectors (xs[i], ys[j]) or matrices
# (xs[i,j], ys[i,j]).  Vertices are laid out i-major: linear index (i-1)*ny + j.
function _surface_mesh(xs, ys, zs)
    nx, ny = size(zs)
    xmat = xs isa AbstractMatrix
    ymat = ys isa AbstractMatrix
    getx = (i, j) -> Float32(xmat ? xs[i, j] : xs[i])
    gety = (i, j) -> Float32(ymat ? ys[i, j] : ys[j])

    P = Array{Point3f}(undef, nx, ny)
    for i in 1:nx, j in 1:ny
        P[i, j] = Point3f(getx(i, j), gety(i, j), Float32(zs[i, j]))
    end

    npts    = nx * ny
    points  = Vector{Point3f}(undef, npts)
    normals = Vector{Vec3f}(undef, npts)
    lin = (i, j) -> (i - 1) * ny + j          # 1-based linear index

    for i in 1:nx, j in 1:ny
        points[lin(i, j)] = P[i, j]
        ip = min(i + 1, nx); im = max(i - 1, 1)
        jp = min(j + 1, ny); jm = max(j - 1, 1)
        tx = Vec3f(P[ip, j] - P[im, j])       # tangent along x (i)
        ty = Vec3f(P[i, jp] - P[i, jm])       # tangent along y (j)
        nrm = cross(tx, ty)
        nl  = norm(nrm)
        normals[lin(i, j)] = nl < 1.0f-8 ? Vec3f(0, 0, 1) : Vec3f(nrm ./ nl)
    end

    faces0 = Vector{Int}[]
    for i in 1:nx-1, j in 1:ny-1
        v00 = (i - 1) * ny + (j - 1)          # 0-based corners (CCW from above)
        v10 = i * ny + (j - 1)
        v11 = i * ny + j
        v01 = (i - 1) * ny + j
        if all(isfinite, (P[i, j][3], P[i+1, j][3], P[i+1, j+1][3], P[i, j+1][3]))
            push!(faces0, [v00, v10, v11, v01])
        end
    end
    return points, faces0, normals
end

# Per-vertex surface colours, in the SAME i-major order as `_surface_mesh` points.
# Surface defaults `plot.color[] === nothing` → colour by `zs` through the colormap.
function _surface_colors(plot, zs)
    nx, ny = size(zs)
    c = plot.color[]
    if c === nothing
        cmap = Makie.to_colormap(plot.colormap[])
        zvec = Float32[zs[i, j] for i in 1:nx for j in 1:ny]
        rng  = _colorrange(plot, zvec)
        return ([_rgb(Makie.interpolated_getindex(cmap, v, rng)) for v in zvec], "vertex")
    end
    return displaycolor_for(plot, nx * ny)
end

# Per-vertex `st` UVs for a TEXTURED surface, i-major like _surface_mesh points.
# ORIENTATION gotcha (u/v wrong -> texture 90-rotates):
#   u (image cols) <- 2nd grid axis (j)
#   v (image rows) <- 1st grid axis (i), FLIPPED so i=1 -> st bottom-left origin
# VERIFIED vs GLMakie equirectangular-earth render; earlier u<-i,v<-j (no flip)
# rotated textured surfaces 90 (continents on their side).  Mesh spheres get
# Makie's own :texturecoordinates instead.  Without st the bound diffuse_texture
# samples nothing -> surface renders white.
function _surface_texcoords(nx, ny)
    st = Vector{Vec2f}(undef, nx * ny)
    for i in 1:nx, j in 1:ny
        u = ny == 1 ? 0.0f0 : Float32((j - 1) / (ny - 1))   # image cols <- j
        v = nx == 1 ? 0.0f0 : Float32((nx - i) / (nx - 1))  # image rows <- i (flipped)
        st[(i - 1) * ny + j] = Vec2f(u, v)
    end
    return st
end

"""
    to_ovrtx_object(screen, scene, plot::Makie.Surface) -> Union{UInt64,Nothing}

Re-mesh the grid surface (`plot[1..3][]` = xs, ys, zs) into a `UsdGeomMesh` of quad
cells with per-vertex finite-difference normals and per-vertex colours (z-driven
colormap by default), authored via `usda_mesh`.  Returns `nothing` for a degenerate
grid (<2x2 or no finite cells).
"""
function to_ovrtx_object(screen, scene, plot::Makie.Surface)
    xs = plot[1][]; ys = plot[2][]; zs = plot[3][]
    (isempty(zs) || size(zs, 1) < 2 || size(zs, 2) < 2) && return nothing

    points, faces0, normals = _surface_mesh(xs, ys, zs)
    isempty(faces0) && return nothing
    path = plot_prim_path(screen.scene2scope, scene, plot)

    if is_materialized(plot)
        # Materialized Surface = UsdGeomMesh: emit WITHOUT displayColor (the
        # `nothing` sentinel) so the pre-authored OmniPBR material (bound by
        # register_ovrtx_robj!) governs shading.  A TEXTURED surface needs the
        # grid's st UVs, else the diffuse_texture samples nothing -> white.
        texcoords = _needs_texcoords(plot) ? _surface_texcoords(size(zs)...) : nothing
        usda = usda_mesh(points, faces0, normals, nothing;
                         model                = plot.model[],
                         normal_interpolation = "vertex",
                         texcoords            = texcoords)
        return OV.add_usd_reference!(screen.renderer, usda, path)
    end

    values, interp = _surface_colors(plot, zs)
    usda = usda_mesh(points, faces0, normals, values;
                     model                = plot.model[],
                     normal_interpolation = "vertex",
                     color_interpolation  = interp)
    return OV.add_usd_reference!(screen.renderer, usda, path)
end
