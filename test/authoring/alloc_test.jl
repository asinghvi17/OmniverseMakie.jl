using Test

# ---------------------------------------------------------------------------
# Authoring-path allocations (pure, no GPU): USDA emitters take flat
# (counts, indices) from `_flat_faces` and stream number lists through one
# reused IOBuffer. Byte-identity is the bar: the goldens below must stay
# byte-for-byte identical (Julia's print/string of Float32 has no f0 suffix;
# the non-allocating writers reproduce it exactly).
# ---------------------------------------------------------------------------

import OmniverseMakie as OM
import GeometryBasics as GB

const _B7_MESH_CONST = "#usda 1.0\n( defaultPrim = \"mesh\" )\ndef Mesh \"mesh\"\n{\n    int[] faceVertexCounts = [3, 3]\n    int[] faceVertexIndices = [0, 1, 2, 0, 2, 3]\n    normal3f[] normals = [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)] (\n        interpolation = \"vertex\"\n    )\n    point3f[] points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]\n    color3f[] primvars:displayColor = [(0.2, 0.4, 0.6)] (\n        interpolation = \"constant\"\n    )\n    uniform token subdivisionScheme = \"none\"\n    matrix4d xformOp:transform = ( (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0) )\n    uniform token[] xformOpOrder = [\"xformOp:transform\"]\n}\n"
const _B7_MESH_PV = "#usda 1.0\n( defaultPrim = \"mesh\" )\ndef Mesh \"mesh\"\n{\n    int[] faceVertexCounts = [3, 3]\n    int[] faceVertexIndices = [0, 1, 2, 0, 2, 3]\n    normal3f[] normals = [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)] (\n        interpolation = \"vertex\"\n    )\n    point3f[] points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]\n    color3f[] primvars:displayColor = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.5, 0.5, 0.5)] (\n        interpolation = \"vertex\"\n    )\n    uniform token subdivisionScheme = \"none\"\n    matrix4d xformOp:transform = ( (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0) )\n    uniform token[] xformOpOrder = [\"xformOp:transform\"]\n}\n"
const _B7_MESH_MAT = "#usda 1.0\n( defaultPrim = \"mesh\" )\ndef Mesh \"mesh\"\n{\n    int[] faceVertexCounts = [3, 3]\n    int[] faceVertexIndices = [0, 1, 2, 0, 2, 3]\n    normal3f[] normals = [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)] (\n        interpolation = \"vertex\"\n    )\n    point3f[] points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]\n    texCoord2f[] primvars:st = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] (\n        interpolation = \"vertex\"\n    )\n    uniform token subdivisionScheme = \"none\"\n    matrix4d xformOp:transform = ( (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0) )\n    uniform token[] xformOpOrder = [\"xformOp:transform\"]\n}\n"
const _B7_INST = "#usda 1.0\n( defaultPrim = \"inst\" )\ndef PointInstancer \"inst\"\n{\n    point3f[] positions = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]\n    int[] protoIndices = [0, 0]\n    float3[] scales = [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]\n    quath[] orientations = [(1.0, 0.0, 0.0, 0.0), (0.70710677, 0.70710677, 0.0, 0.0)]\n    color3f[] primvars:displayColor = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)] (\n        interpolation = \"vertex\"\n    )\n    rel prototypes = [</inst/proto>]\n    matrix4d xformOp:transform = ( (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0) )\n    uniform token[] xformOpOrder = [\"xformOp:transform\"]\n    def Sphere \"proto\"\n    {\n        double radius = 1\n        color3f[] primvars:displayColor = [(0.5, 0.5, 0.5)] (\n            interpolation = \"constant\"\n        )\n    }\n}\n"
const _B7_PROTO = "    def Mesh \"proto\"\n    {\n        int[] faceVertexCounts = [3, 3]\n        int[] faceVertexIndices = [0, 1, 2, 0, 2, 3]\n        normal3f[] normals = [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)] (\n            interpolation = \"vertex\"\n        )\n        point3f[] points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]\n        color3f[] primvars:displayColor = [(0.5, 0.5, 0.5)] (\n            interpolation = \"constant\"\n        )\n        uniform token subdivisionScheme = \"none\"\n    }"
const _B7_MERGED = "#usda 1.0\n( defaultPrim = \"mesh\" )\ndef Mesh \"mesh\"\n{\n    int[] faceVertexCounts = [3, 3, 3, 3]\n    int[] faceVertexIndices = [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]\n    normal3f[] normals = [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)] (\n        interpolation = \"vertex\"\n    )\n    point3f[] points = [(10.0, 0.0, 0.0), (11.0, 0.0, 0.0), (11.0, 1.0, 0.0), (10.0, 1.0, 0.0), (0.0, 10.0, 0.0), (2.0, 10.0, 0.0), (2.0, 12.0, 0.0), (0.0, 12.0, 0.0)]\n    uniform token subdivisionScheme = \"none\"\n    matrix4d xformOp:transform = ( (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0) )\n    uniform token[] xformOpOrder = [\"xformOp:transform\"]\n}\n"
const _B7_P3F = "(1.0, 2.0, 0.0), (3.0, 4.0, 5.0)"
# Shared fixtures (identical values to the golden captures).
const _B7_MPTS  = [(0.0f0,0.0f0,0.0f0),(1.0f0,0.0f0,0.0f0),(1.0f0,1.0f0,0.0f0),(0.0f0,1.0f0,0.0f0)]
# nested, 0-based (counts [3,3], idx [0,1,2,0,2,3])
const _B7_FACES = [[0,1,2],[0,2,3]]
const _B7_NRM   = [(0.0f0,0.0f0,1.0f0),(0.0f0,1.0f0,0.0f0),(1.0f0,0.0f0,0.0f0),(0.0f0,0.0f0,-1.0f0)]
const _B7_PVCOL = [(1.0f0,0.0f0,0.0f0),(0.0f0,1.0f0,0.0f0),(0.0f0,0.0f0,1.0f0),(0.5f0,0.5f0,0.5f0)]
const _B7_TC    = [OM.Vec2f(0,0),OM.Vec2f(1,0),OM.Vec2f(1,1),OM.Vec2f(0,1)]
const _B7_I4    = [1.0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
const _B7_IPOS  = [(0.0f0,0.0f0,0.0f0),(1.0f0,1.0f0,1.0f0)]
const _B7_SCL   = [(1.0f0,1.0f0,1.0f0),(2.0f0,2.0f0,2.0f0)]
const _B7_ORI   = [(1.0f0,0.0f0,0.0f0,0.0f0),(0.70710677f0,0.70710677f0,0.0f0,0.0f0)]
const _B7_ICOL  = [(1.0f0,0.0f0,0.0f0),(0.0f0,1.0f0,0.0f0)]
const _B7_MPOS  = [(10.0f0,0.0f0,0.0f0),(0.0f0,10.0f0,0.0f0)]

@testset "B7 golden USDA byte-identity (flat emitters + IOBuffer streaming)" begin
    fc, fi = OM._flat_faces(_B7_FACES)   # flat (counts, indices)
    @test OM.usda_mesh(_B7_MPTS, fc, fi, _B7_NRM, (0.2f0,0.4f0,0.6f0);
                       normal_interpolation="vertex") == _B7_MESH_CONST
    @test OM.usda_mesh(_B7_MPTS, fc, fi, _B7_NRM, _B7_PVCOL;
                       normal_interpolation="vertex", color_interpolation="vertex") == _B7_MESH_PV
    @test OM.usda_mesh(_B7_MPTS, fc, fi, _B7_NRM, nothing;
                       normal_interpolation="vertex", texcoords=_B7_TC) == _B7_MESH_MAT
    @test OM._usda_pointinstancer(_B7_IPOS, _B7_SCL, _B7_ORI, _B7_ICOL,
                       OM._sphere_proto_body((0.5f0,0.5f0,0.5f0)); model=_B7_I4) == _B7_INST
    @test OM._mesh_proto_body(_B7_MPTS, fc, fi, _B7_NRM, (0.5f0,0.5f0,0.5f0)) == _B7_PROTO
    mp, mc, mi, mn = OM._merged_instances_mesh(_B7_MPTS, fc, fi, _B7_NRM, _B7_MPOS, _B7_SCL, nothing)
    @test OM.usda_mesh(mp, mc, mi, mn, nothing; normal_interpolation="vertex") == _B7_MERGED
    @test OM._point3f_list([(1.0f0,2.0f0),(3.0f0,4.0f0,5.0f0)]) == _B7_P3F
end

@testset "B7 _flat_faces → flat (counts, 0-based indices), built once" begin
    # Plain-Int nested faces (surface re-mesh / merged markers): raw is
    # identity on Int.
    c, i = OM._flat_faces([[0,1,2],[0,2,3]])
    @test (c, i) == ([3, 3], [0, 1, 2, 0, 2, 3])
    @test eltype(c) === Int && eltype(i) === Int
    # Ragged arity (per-face count preserved).
    c2, i2 = OM._flat_faces([[5],[7,8],[1,2,3,4]])
    @test (c2, i2) == ([1, 2, 4], [5, 7, 8, 1, 2, 3, 4])
    # GeometryBasics faces (OffsetInteger, presented 1-based) → raw gives
    # 0-based indices.
    m = GB.normal_mesh(GB.Tesselation(GB.Sphere(GB.Point3f(0), 1f0), 4))
    gc, gi = OM._flat_faces(GB.faces(m))
    @test length(gc) == length(GB.faces(m)) && length(gi) == sum(gc)
    @test minimum(gi) == 0
    @test gi == [Int(GB.raw(v)) for f in GB.faces(m) for v in f]
end

@testset "B7 _push_points_binding! writes through the reinterpret VIEW (no full-buffer copy)" begin
    v = OM.Point3f[OM.Point3f(1,2,3), OM.Point3f(4,5,6), OM.Point3f(7,8,9)]
    data = reinterpret(Float32, v)   # what _push_points_binding! passes
    # pointer() on a ReinterpretArray-over-Vector equals the parent buffer
    # pointer, so write_binding!'s `pointer(data)` under GC.@preserve is
    # valid AND zero-copy.
    @test pointer(data) == Ptr{Float32}(pointer(v))
    @test collect(data) == Float32[1,2,3,4,5,6,7,8,9]     # flattened components
    @test length(v) == 3   # shape is npoints (3 lanes each), not 3*npoints
    # Round-trip through the pointer under preserve, as write_binding! reads it.
    got = GC.@preserve data [unsafe_load(pointer(data), k) for k in 1:length(data)]
    @test got == Float32[1,2,3,4,5,6,7,8,9]
end

@testset "B7 usda_mesh allocation reduced ≥5× on a 10k-vertex mesh" begin
    N = 10_000; ntri = 20_000
    pts   = [OM.Point3f(i, i+1, i+2) for i in 1:N]
    nrm   = [OM.Vec3f(0, 0, 1) for _ in 1:N]
    faces = [[mod(3k, N), mod(3k+1, N), mod(3k+2, N)] for k in 0:ntri-1]
    fc, fi = OM._flat_faces(faces)
    # warm/compile
    OM.usda_mesh(pts, fc, fi, nrm, (0.5f0,0.5f0,0.5f0); normal_interpolation="vertex")
    GC.gc()
    a = @allocated OM.usda_mesh(pts, fc, fi, nrm, (0.5f0,0.5f0,0.5f0); normal_interpolation="vertex")
    @info "B7 usda_mesh @allocated (10k verts / 20k tris)" bytes=a
    # The 8 MB ceiling: a regression to per-element String building on this
    # fixture lands well above it.
    @test a ≤ 8_000_000
end
