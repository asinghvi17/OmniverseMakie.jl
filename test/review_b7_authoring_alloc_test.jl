using Test

# ---------------------------------------------------------------------------
# Review Track B / Task B7 — authoring-path allocations.
#
# Mesh conversion allocated one `Int[]` per face in three identical comprehensions
# (author_usd_prim! for Mesh / materialized Scatter / materialized MeshScatter) only
# for the USDA emitters to re-flatten; the emitters then built one temporary String
# per vertex/index before `join`.  The refactor (a) flattens faces ONCE via
# `_flat_faces` and has emitters take flat `(counts, indices)`, and (b) streams every
# USDA number list through one reused IOBuffer instead of per-element Strings + join.
#
# BYTE-IDENTITY IS THE BAR.  These goldens were captured from the PRE-refactor emitters
# (this testset is GREEN against pre-refactor code — it is the regression anchor) and
# MUST stay byte-for-byte identical afterwards.  Julia's `print`/`string` of Float32 has
# no `f0` suffix (already relied on); the non-allocating writers reproduce it exactly.
#
# PURE (no GPU): USDA string emission + `_flat_faces` + the reinterpret-view push are all
# directly assertable.  The live binding write path (`_push_points_binding!` through
# `write_binding!`) is exercised by the B2/B4 GPU testsets.
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
const _B7_FACES = [[0,1,2],[0,2,3]]                     # nested, 0-based (counts [3,3], idx [0,1,2,0,2,3])
const _B7_NRM   = [(0.0f0,0.0f0,1.0f0),(0.0f0,1.0f0,0.0f0),(1.0f0,0.0f0,0.0f0),(0.0f0,0.0f0,-1.0f0)]
const _B7_PVCOL = [(1.0f0,0.0f0,0.0f0),(0.0f0,1.0f0,0.0f0),(0.0f0,0.0f0,1.0f0),(0.5f0,0.5f0,0.5f0)]
const _B7_TC    = [OM.Vec2f(0,0),OM.Vec2f(1,0),OM.Vec2f(1,1),OM.Vec2f(0,1)]
const _B7_I4    = [1.0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
const _B7_IPOS  = [(0.0f0,0.0f0,0.0f0),(1.0f0,1.0f0,1.0f0)]
const _B7_SCL   = [(1.0f0,1.0f0,1.0f0),(2.0f0,2.0f0,2.0f0)]
const _B7_ORI   = [(1.0f0,0.0f0,0.0f0,0.0f0),(0.70710677f0,0.70710677f0,0.0f0,0.0f0)]
const _B7_ICOL  = [(1.0f0,0.0f0,0.0f0),(0.0f0,1.0f0,0.0f0)]
const _B7_MPOS  = [(10.0f0,0.0f0,0.0f0),(0.0f0,10.0f0,0.0f0)]

@testset "B7 golden USDA byte-identity (pre-refactor anchor)" begin
    @test OM.usda_mesh(_B7_MPTS, _B7_FACES, _B7_NRM, (0.2f0,0.4f0,0.6f0);
                       normal_interpolation="vertex") == _B7_MESH_CONST
    @test OM.usda_mesh(_B7_MPTS, _B7_FACES, _B7_NRM, _B7_PVCOL;
                       normal_interpolation="vertex", color_interpolation="vertex") == _B7_MESH_PV
    @test OM.usda_mesh(_B7_MPTS, _B7_FACES, _B7_NRM, nothing;
                       normal_interpolation="vertex", texcoords=_B7_TC) == _B7_MESH_MAT
    @test OM._usda_pointinstancer(_B7_IPOS, _B7_SCL, _B7_ORI, _B7_ICOL,
                       OM._sphere_proto_body((0.5f0,0.5f0,0.5f0)); model=_B7_I4) == _B7_INST
    @test OM._mesh_proto_body(_B7_MPTS, _B7_FACES, _B7_NRM, (0.5f0,0.5f0,0.5f0)) == _B7_PROTO
    mp, mf, mn = OM._merged_instances_mesh(_B7_MPTS, _B7_FACES, _B7_NRM, _B7_MPOS, _B7_SCL, nothing)
    @test OM.usda_mesh(mp, mf, mn, nothing; normal_interpolation="vertex") == _B7_MERGED
    @test OM._point3f_list([(1.0f0,2.0f0),(3.0f0,4.0f0,5.0f0)]) == _B7_P3F
end
