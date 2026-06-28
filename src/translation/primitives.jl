# Primitive translations for OmniverseMakie (M1.7): scatter / meshscatter / lines / surface.
#
# Each `to_ovrtx_object(screen, scene, plot::T)` reads the plot's data, authors a
# self-contained USDA reference layer (defaultPrim, NO `upAxis` — M1.2 finding),
# adds it under `/World/plot_<objectid(plot)>` via `OV.add_usd_reference!`, and
# returns the `ovrtx_usd_handle_t` — exactly the M1.5 Mesh pattern.  `Base.insert!`
# dispatches here; unknown plot types hit the generic `to_ovrtx_object(...) = nothing`.
#
# USD schemas — all VALIDATED at render time by test/m1_primitives_test.jl (M1.7),
# each producing a non-black render through ovrtx RT2:
#   Scatter            → UsdGeomPointInstancer + UsdGeomSphere prototype  (assumption 4 ✓)
#   MeshScatter        → UsdGeomPointInstancer + UsdGeomMesh prototype    (assumption 4 ✓)
#   Lines/LineSegments → UsdGeomBasisCurves (uniform token type = "linear")  (BasisCurves ✓)
#   Surface            → UsdGeomMesh (grid re-meshed; reuses `usda_mesh`)
#
# ovrtx HONORS PointInstancer and BasisCurves, so the documented fallbacks in
# m1.7-context.md (a merged UsdGeomMesh of marker copies for PointInstancer; a merged
# tube mesh for BasisCurves) were NOT needed and are intentionally not implemented.
# If a future ovrtx build regresses one of these schemas, m1_primitives_test.jl turns
# RED and those merged-mesh fallbacks (both riding the proven `usda_mesh` path) are
# the documented switch.
#
# NOTE: included inside the OmniverseMakie module, AFTER meshes.jl.  In scope:
#   Makie, GeometryBasics, OV, LinearAlgebra (cross/norm/normalize),
#   usda_mesh, usda_matrix4d, displaycolor_for, _rgb, _colorrange, _displaycolor_str.

# ------------------------------------------------------------------
# small formatting / data helpers
# ------------------------------------------------------------------

# (x, y, z) Float32 tuple from any 2- or 3-element point (Point2 → z = 0).
_p3(p) = (Float32(p[1]), Float32(p[2]), Float32(length(p) >= 3 ? p[3] : 0.0f0))

# Join a sequence of points as USD `point3f[]` body text: "(x, y, z), (x, y, z), ...".
_pt3f_list(pts) = join(["($(c[1]), $(c[2]), $(c[3]))" for c in (_p3(p) for p in pts)], ", ")

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
    lo = [Float64(_p3(first(pts))[i]) for i in 1:3]
    hi = copy(lo)
    for p in pts
        c = _p3(p)
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
# `color_const` (an (r,g,b) tuple) is baked as a constant displayColor when given.
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
function _mesh_proto_body(points, faces0, normals, color_const)
    fvc = join(string.([length(f) for f in faces0]), ", ")
    fvi = join(string.([i for f in faces0 for i in f]), ", ")
    nrm = join(["($(Float32(n[1])), $(Float32(n[2])), $(Float32(n[3])))" for n in normals], ", ")
    col = color_const === nothing ? "" : """
        color3f[] primvars:displayColor = [($(Float32(color_const[1])), $(Float32(color_const[2])), $(Float32(color_const[3])))] (
            interpolation = "constant"
        )
"""
    return """    def Mesh "proto"
    {
        int[] faceVertexCounts = [$(fvc)]
        int[] faceVertexIndices = [$(fvi)]
        normal3f[] normals = [$(nrm)] (
            interpolation = "vertex"
        )
        point3f[] points = [$(_pt3f_list(points))]
$(col)        uniform token subdivisionScheme = "none"
    }"""
end

# Author a complete PointInstancer reference layer around `proto_body`.
# `instancer_color`: per-instance (r,g,b) Vector → emitted with interpolation
# "vertex" on the instancer; `nothing` → no instancer-level color (the prototype
# carries a constant colour instead).
function _usda_pointinstancer(positions, scales, orientations, instancer_color, proto_body; model)
    n        = length(positions)
    pos_str  = _pt3f_list(positions)
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

# ------------------------------------------------------------------
# Scatter → UsdGeomPointInstancer (UsdGeomSphere prototype)
# ------------------------------------------------------------------

"""
    to_ovrtx_object(screen, scene, plot::Makie.Scatter) -> UInt64

Author the scatter as a `UsdGeomPointInstancer` whose single prototype is a unit
`UsdGeomSphere`, one instance per `plot[1][]` position, scaled by `plot.markersize`
(treated as a data-unit radius for M1) and coloured via `displaycolor_for`.
A constant colour rides on the prototype; per-point colours ride on the instancer.
"""
function to_ovrtx_object(screen, scene, plot::Makie.Scatter)
    pos = plot[1][]
    n   = length(pos)
    n == 0 && return nothing

    scales = _scales_for(plot.markersize[], n)
    values, interp = displaycolor_for(plot, n)

    proto_color     = interp == "constant" ? values  : nothing
    instancer_color = interp == "constant" ? nothing : values

    usda = _usda_pointinstancer(pos, scales, nothing, instancer_color,
                                _sphere_proto_body(proto_color); model = plot.model[])
    return OV.add_usd_reference!(screen.renderer, usda, plot_prim_path(plot))
end

# ------------------------------------------------------------------
# MeshScatter → UsdGeomPointInstancer (UsdGeomMesh prototype = plot.marker)
# ------------------------------------------------------------------

"""
    to_ovrtx_object(screen, scene, plot::Makie.MeshScatter) -> UInt64

Author the meshscatter as a `UsdGeomPointInstancer` whose prototype is the marker
mesh (`plot.marker[]`, converted via `GeometryBasics.mesh` if needed), instanced at
each `plot[1][]` position, scaled by `plot.markersize` and (when non-identity)
oriented by `plot.rotation`.  Colour handling matches `Scatter`.
"""
function to_ovrtx_object(screen, scene, plot::Makie.MeshScatter)
    pos = plot[1][]
    n   = length(pos)
    n == 0 && return nothing

    marker = plot.marker[]
    gm     = marker isa GeometryBasics.Mesh ? marker : GeometryBasics.mesh(marker)
    mpts   = GeometryBasics.coordinates(gm)
    mnrm   = GeometryBasics.normals(gm)
    mfaces = [Int[Int(GeometryBasics.raw(i)) for i in f] for f in GeometryBasics.faces(gm)]

    scales       = _scales_for(plot.markersize[], n)
    orientations = _orientations_for(plot.rotation[], n)
    values, interp = displaycolor_for(plot, n)

    proto_color     = interp == "constant" ? values  : nothing
    instancer_color = interp == "constant" ? nothing : values

    proto = _mesh_proto_body(mpts, mfaces, mnrm, proto_color)
    usda  = _usda_pointinstancer(pos, scales, orientations, instancer_color, proto;
                                 model = plot.model[])
    return OV.add_usd_reference!(screen.renderer, usda, plot_prim_path(plot))
end

# ------------------------------------------------------------------
# UsdGeomBasisCurves authoring (Lines / LineSegments primary)
# ------------------------------------------------------------------

# A visible data-unit curve width from a (pixel-space) linewidth.  Pixel-accurate
# sizing is a later refinement (M1 gate only needs a visible, non-black curve), so
# we scale a small fraction of the data extent by the linewidth.
function _curve_width(pts, linewidth)
    lw = linewidth isa AbstractVector ?
        (isempty(linewidth) ? 1.0 : Float64(sum(linewidth) / length(linewidth))) :
        Float64(linewidth)
    return Float32(max(0.01 * lw * _bbox_diag(pts), 1.0f-4))
end

function _usda_basiscurves(points, counts, width, color_values, color_interp; model)
    counts_str = join(string.(counts), ", ")
    col_block = """    color3f[] primvars:displayColor = [$(_displaycolor_str(color_values))] (
        interpolation = "$(color_interp)"
    )
"""
    return """#usda 1.0
( defaultPrim = "curve" )
def BasisCurves "curve"
{
    uniform token type = "linear"
    int[] curveVertexCounts = [$(counts_str)]
    point3f[] points = [$(_pt3f_list(points))]
    float[] widths = [$(width)] (
        interpolation = "constant"
    )
$(col_block)    matrix4d xformOp:transform = $(usda_matrix4d(model))
    uniform token[] xformOpOrder = ["xformOp:transform"]
}
"""
end

"""
    to_ovrtx_object(screen, scene, plot::Makie.Lines) -> Union{UInt64,Nothing}

Author one polyline as a linear `UsdGeomBasisCurves` (single curve of `N` vertices).
"""
function to_ovrtx_object(screen, scene, plot::Makie.Lines)
    pts = plot[1][]
    n   = length(pts)
    n < 2 && return nothing
    values, interp = displaycolor_for(plot, n)
    width = _curve_width(pts, plot.linewidth[])
    usda  = _usda_basiscurves(pts, [n], width, values, interp; model = plot.model[])
    return OV.add_usd_reference!(screen.renderer, usda, plot_prim_path(plot))
end

"""
    to_ovrtx_object(screen, scene, plot::Makie.LineSegments) -> Union{UInt64,Nothing}

Author independent 2-vertex segments as a linear `UsdGeomBasisCurves`
(`curveVertexCounts = [2, 2, ...]`, one entry per consecutive point pair).
"""
function to_ovrtx_object(screen, scene, plot::Makie.LineSegments)
    pts  = plot[1][]
    nseg = length(pts) ÷ 2
    nseg < 1 && return nothing
    pts2 = pts[1:2*nseg]
    values, interp = displaycolor_for(plot, length(pts2))
    width  = _curve_width(pts, plot.linewidth[])
    usda   = _usda_basiscurves(pts2, fill(2, nseg), width, values, interp; model = plot.model[])
    return OV.add_usd_reference!(screen.renderer, usda, plot_prim_path(plot))
end

# ------------------------------------------------------------------
# Surface → UsdGeomMesh (grid re-meshed; reuses usda_mesh)
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

"""
    to_ovrtx_object(screen, scene, plot::Makie.Surface) -> Union{UInt64,Nothing}

Re-mesh the grid surface (`plot[1..3][]` = xs, ys, zs) into a `UsdGeomMesh` of
quad cells with per-vertex normals (finite-difference grid normals) and per-vertex
colours (z-driven colormap by default), authored through the proven `usda_mesh`.
"""
function to_ovrtx_object(screen, scene, plot::Makie.Surface)
    xs = plot[1][]; ys = plot[2][]; zs = plot[3][]
    (isempty(zs) || size(zs, 1) < 2 || size(zs, 2) < 2) && return nothing

    points, faces0, normals = _surface_mesh(xs, ys, zs)
    isempty(faces0) && return nothing
    values, interp = _surface_colors(plot, zs)

    usda = usda_mesh(points, faces0, normals, values;
                     model                = plot.model[],
                     normal_interpolation = "vertex",
                     color_interpolation  = interp)
    return OV.add_usd_reference!(screen.renderer, usda, plot_prim_path(plot))
end
