# Primitive translations: scatter / meshscatter / lines / surface. Each emits
# a self-contained USDA reference layer (defaultPrim, no upAxis) added under
# the plot's scope path (/World/Scene_<id>/plot_<id>). The emitters are driven
# live by author_usd_prim! (compute.jl); Surface alone builds here. Schemas:
#   Scatter/MeshScatter → PointInstancer + Sphere/Mesh prototype
#   Lines/LineSegments  → BasisCurves (type = "linear")
#   Surface             → Mesh (grid re-meshed; reuses usda_mesh)

# ------------------------------------------------------------------
# small formatting / data helpers
# ------------------------------------------------------------------

# (x, y, z) Float32 tuple from any 2- or 3-element point (Point2 → z = 0).
_point3f(p) = (Float32(p[1]), Float32(p[2]), Float32(length(p) >= 3 ? p[3] : 0.0f0))

# USD `point3f[]` body text `(x, y, z), (x, y, z), …` for a sequence of points
# (2-D → z=0), streamed through one reused IOBuffer (no per-element String).
function _point3f_list(pts)
    io = IOBuffer(); scratch = Vector{UInt8}(undef, 64); first = true
    for p in pts
        c = _point3f(p)
        first ? (first = false) : print(io, ", ")
        print(io, "(")
        _emit_f32!(io, scratch, c[1]); print(io, ", ")
        _emit_f32!(io, scratch, c[2]); print(io, ", ")
        _emit_f32!(io, scratch, c[3]); print(io, ")")
    end
    return String(take!(io))
end

# Per-instance (sx, sy, sz) scale from one markersize element (scalar or Vec).
_scale_tuple(s::Real) = (Float32(s), Float32(s), Float32(s))
function _scale_tuple(s)
    sx = Float32(s[1])
    sy = Float32(length(s) >= 2 ? s[2] : s[1])
    sz = Float32(length(s) >= 3 ? s[3] : s[1])
    return (sx, sy, sz)
end

# Resolve a markersize attribute into one (sx, sy, sz) per point. A single
# `Vec`/scalar broadcasts; a `Base.Array` of length `n` is per-point. (A
# `Vec2f`/`Vec3f` is a StaticArray, not a `Base.Array`, so it broadcasts.)
function _scales_for(markersize, n)
    if markersize isa Base.Array && length(markersize) == n
        return [_scale_tuple(m) for m in markersize]
    end
    return fill(_scale_tuple(markersize), n)
end

# Makie's `Quaternion` stores `data = (x, y, z, w)`; USD `quath` wants
# (w, x, y, z).
_quat_wxyz(q) = (Float32(q[4]), Float32(q[1]), Float32(q[2]), Float32(q[3]))

# Resolve a rotation attribute into per-instance (w, x, y, z) orientations, or
# `nothing` when every rotation is (near) identity — in which case we omit the
# `orientations` attribute entirely (fewer quath tokens for ovrtx to parse).
function _orientations_for(rot, n)
    if rot isa Base.Array
        # A per-instance rotation vector must match the instance count — a
        # wrong length is a caller error, not a broadcastable single rotation.
        length(rot) == n || throw(ArgumentError(
            "OmniverseMakie: rotation vector length $(length(rot)) != instance count $n."))
        quats = rot
    else
        quats = fill(rot, n)                # single rotation broadcasts
    end
    out = NTuple{4,Float32}[]
    all_identity = true
    for q in quats
        wxyz = try
            _quat_wxyz(q)
        catch err
            err isa InterruptException && rethrow()
            @warn "OmniverseMakie: could not convert rotation of type $(typeof(q)) to a \
                   quaternion — using identity." maxlog = 1
            (1.0f0, 0.0f0, 0.0f0, 0.0f0)
        end
        (abs(wxyz[2]) + abs(wxyz[3]) + abs(wxyz[4]) > 1.0f-6) && (all_identity = false)
        push!(out, wxyz)
    end
    return all_identity ? nothing : out
end

# Axis-aligned bounding-box diagonal of a point cloud (Float64). Non-finite
# points (NaN separators — Makie's broken-line idiom) are skipped so
# `_curve_width` stays finite; empty or no-finite-point clouds → 1.0.
function _bbox_diag(pts)
    lo = nothing; hi = nothing
    for p in pts
        all(isfinite, p) || continue
        c = _point3f(p)
        if lo === nothing
            lo = [Float64(c[i]) for i in 1:3]; hi = copy(lo)
        else
            for i in 1:3
                lo[i] = min(lo[i], Float64(c[i])); hi[i] = max(hi[i], Float64(c[i]))
            end
        end
    end
    lo === nothing && return 1.0            # empty or no finite point
    d = sqrt(sum((hi .- lo) .^ 2))
    return d < 1e-8 ? 1.0 : d
end

# Split a point sequence at non-finite points into contiguous finite runs —
# Makie's NaN-separator broken-line idiom. Returns (finite_pts, counts, keep):
# finite_pts concatenates every run of length ≥ 2 (each run = one BasisCurves
# curve); counts is the per-run vertex count (curveVertexCounts); keep is the
# per-input-vertex survivor mask so a parallel per-vertex array (colours)
# filters identically via colours[keep]. Length-1 runs are dropped; no runs
# (all non-finite / empty / lone point) → counts empty, author emits nothing.
function _split_nan_runs(pts)
    n = length(pts)
    counts = Int[]
    keep = falses(n)
    i = 1
    while i <= n
        if all(isfinite, pts[i])
            j = i
            while j <= n && all(isfinite, pts[j])
                j += 1
            end
            if j - i >= 2                   # run of ≥ 2 → one curve
                push!(counts, j - i)
                keep[i:j-1] .= true
            end
            i = j
        else
            i += 1
        end
    end
    return pts[keep], counts, keep
end

# LineSegments variant of `_split_nan_runs`: points are independent pairs
# (1,2),(3,4),…, so any segment with a non-finite endpoint is dropped (a
# trailing odd point is ignored). Returns (finite_pts, counts, keep) in the
# same shape; counts is all 2s, one per surviving segment.
function _finite_segments(pts)
    n = length(pts)
    counts = Int[]
    keep = falses(n)
    for s in 1:(n ÷ 2)
        a = 2s - 1; b = 2s
        if all(isfinite, pts[a]) && all(isfinite, pts[b])
            keep[a] = true; keep[b] = true
            push!(counts, 2)
        end
    end
    return pts[keep], counts, keep
end

# ------------------------------------------------------------------
# UsdGeomPointInstancer authoring (Scatter / MeshScatter primary)
# ------------------------------------------------------------------

# `def Sphere "proto" { ... }` body (4-space indented, instancer child).
# `color_const` (an (r,g,b) tuple) bakes a constant displayColor when given;
# `nothing` (a materialized instancer) omits displayColor.
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
# `face_counts`/`face_indices` are flat (from `_flat_faces`);
# `color_const === nothing` (a materialized instancer) omits displayColor.
function _mesh_proto_body(points, face_counts, face_indices, normals, color_const)
    io = IOBuffer(); scratch = Vector{UInt8}(undef, 64)
    _emit_int_list!(io, scratch, face_counts);  counts_str  = String(take!(io))
    _emit_int_list!(io, scratch, face_indices); indices_str = String(take!(io))
    _emit_vec3_list!(io, scratch, normals);     normals_str = String(take!(io))
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
# Merged-instances mesh — materialized Scatter / MeshScatter.
# ovrtx does not honor a PointInstancer material:binding (instances render
# the MDL default wherever bound), so a materialized scatter/meshscatter
# becomes one merged UsdGeomMesh of marker copies (binds like Mesh).
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

# Concatenate the marker mesh (`mpts`/`mnrm` + flat 0-based
# `mcounts`/`mindices`) transformed to each instance (`scales`, optional
# `orientations` quaternions, `positions`) into one mesh, returned as flat
# (points, face_counts, face_indices, normals) for `usda_mesh`. Every instance
# shares topology: `counts` repeats `mcounts`, `indices` is `mindices` shifted
# by the instance's base vertex.
function _merged_instances_mesh(mpts, mcounts, mindices, mnrm, positions, scales, orientations)
    nmark = length(mpts); ninst = length(positions)
    P = Point3f[]; N = Vec3f[]; counts = Int[]; indices = Int[]
    sizehint!(P, nmark * ninst);            sizehint!(N, nmark * ninst)
    sizehint!(counts, length(mcounts) * ninst); sizehint!(indices, length(mindices) * ninst)
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
        append!(counts, mcounts)
        for idx in mindices
            push!(indices, off + idx)
        end
    end
    return P, counts, indices, N
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

# Live Scatter/MeshScatter builds run through author_usd_prim! (compute.jl),
# which reuses the emitters above.

# ------------------------------------------------------------------
# UsdGeomBasisCurves authoring (Lines / LineSegments primary)
# ------------------------------------------------------------------

# Visible data-unit curve width from a pixel-space linewidth: a small
# fraction of the data extent scaled by the linewidth. Not pixel-accurate.
function _curve_width(pts, linewidth)
    lw = linewidth isa AbstractVector ?
        (isempty(linewidth) ? 1.0 : Float64(sum(linewidth) / length(linewidth))) :
        Float64(linewidth)
    return Float32(max(0.01 * lw * _bbox_diag(pts), 1.0f-4))
end

function _usda_basiscurves(points, counts, width, color_values, color_interp; model)
    counts_str = join(string.(counts), ", ")
    # color_values === nothing (materialized) omits primvars:displayColor so
    # the bound OmniPBR material governs shading. Mirrors usda_mesh's
    # col_block.
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

# Live Lines/LineSegments builds run through author_usd_prim! (compute.jl),
# which reuses the emitters above.

# ------------------------------------------------------------------
# Surface → UsdGeomMesh (grid re-meshed; reuses usda_mesh). Surface has no
# compute outputs, so empty consumed_inputs routes it through
# register_ovrtx_robj!'s empty-inputs branch — the one plot type built here.
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
        # A non-finite grid point belongs to no emitted face (the face filter
        # below drops any cell with a non-finite corner z); zero-fill it so no
        # "nan"/"inf" token reaches the authored `point3f[]`.
        pij = P[i, j]
        points[lin(i, j)] = all(isfinite, pij) ? pij : Point3f(0, 0, 0)
        ip = min(i + 1, nx); im = max(i - 1, 1)
        jp = min(j + 1, ny); jm = max(j - 1, 1)
        tx = Vec3f(P[ip, j] - P[im, j])       # tangent along x (i)
        ty = Vec3f(P[i, jp] - P[i, jm])       # tangent along y (j)
        nrm = cross(tx, ty)
        nl  = norm(nrm)
        # `isfinite(nl)` guards NaN/Inf normals: `NaN < 1f-8` is false, so a
        # bare `nl < 1f-8` check would let a non-finite normal through.
        normals[lin(i, j)] = (isfinite(nl) && nl >= 1.0f-8) ? Vec3f(nrm ./ nl) : Vec3f(0, 0, 1)
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

# Per-vertex surface colours, in the same i-major order as `_surface_mesh`
# points. Surface defaults `plot.color[] === nothing` → colour by `zs`
# through the colormap (NaN-safe; the mesh path drops non-finite cells).
function _surface_colors(plot, zs)
    nx, ny = size(zs)
    if plot.color[] === nothing
        zvec = Float32[zs[i, j] for i in 1:nx for j in 1:ny]
        return (_map_through_colormap(plot, zvec), "vertex")
    end
    return displaycolor_for(plot, nx * ny)
end

# Per-vertex `st` UVs for a textured surface, i-major like _surface_mesh
# points. Orientation (u/v swapped rotates the texture 90 degrees):
#   u (image cols) <- 2nd grid axis (j)
#   v (image rows) <- 1st grid axis (i), flipped so i=1 -> bottom-left origin
# Mesh spheres get Makie's own :texturecoordinates instead. Without st the
# bound diffuse_texture samples nothing -> the surface renders white.
function _surface_texcoords(nx, ny)
    st = Vector{Vec2f}(undef, nx * ny)
    for i in 1:nx, j in 1:ny
        u = ny == 1 ? 0.0f0 : Float32((j - 1) / (ny - 1))   # image cols <- j
        v = nx == 1 ? 0.0f0 : Float32((nx - i) / (nx - 1))  # image rows <- i
        st[(i - 1) * ny + j] = Vec2f(u, v)
    end
    return st
end

"""
    to_ovrtx_object(screen, scene, plot::Surface) -> Union{UInt64,Nothing}

Re-mesh the grid surface (`plot[1..3][]` = xs, ys, zs) into a `UsdGeomMesh`
of quad cells with per-vertex finite-difference normals and per-vertex
colours (z-driven colormap by default), authored via `usda_mesh`. Returns
`nothing` for a degenerate grid (<2x2 or no finite cells).
"""
function to_ovrtx_object(screen, scene, plot::Makie.Surface)
    xs = plot[1][]; ys = plot[2][]; zs = plot[3][]
    (isempty(zs) || size(zs, 1) < 2 || size(zs, 2) < 2) && return nothing

    points, faces0, normals = _surface_mesh(xs, ys, zs)
    isempty(faces0) && return nothing
    face_counts, face_indices = _flat_faces(faces0)   # flatten once
    path = plot_prim_path(screen.scene2scope, scene, plot)

    if is_materialized(plot)
        # Materialized Surface: emit without displayColor (the `nothing`
        # sentinel) so the pre-authored OmniPBR material governs shading. A
        # textured surface needs the grid's st UVs, else the diffuse_texture
        # samples nothing -> white.
        texcoords = _needs_texcoords(plot) ? _surface_texcoords(size(zs)...) : nothing
        usda = usda_mesh(points, face_counts, face_indices, normals, nothing;
                         model                = plot.model[],
                         normal_interpolation = "vertex",
                         texcoords            = texcoords)
        return OV.add_usd_reference!(screen.renderer, usda, path)
    end

    values, interp = _surface_colors(plot, zs)
    usda = usda_mesh(points, face_counts, face_indices, normals, values;
                     model                = plot.model[],
                     normal_interpolation = "vertex",
                     color_interpolation  = interp)
    return OV.add_usd_reference!(screen.renderer, usda, path)
end
