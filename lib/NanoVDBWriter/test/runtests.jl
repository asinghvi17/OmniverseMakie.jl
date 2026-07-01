using Test, NanoVDBWriter
using GeometryBasics: Point3f, Vec3f
@testset "NanoVDBWriter round-trip" begin
    data = zeros(Float32, 8, 8, 8); data[2,3,4] = 1.5f0; data[7,7,7] = 0.25f0   # sparse, asymmetric
    path = tempname() * ".nvdb"
    save_nanovdb(path, data, Point3f(0,0,0), Vec3f(1,1,1))
    @test isfile(path) && filesize(path) > 0
    h = NanoVDBWriter.parse_nanovdb_header(path)
    @test h.magic == "NanoVDB0"
    @test h.version_major == 32
    @test h.voxel_count == 2                      # two non-background voxels
    rm(path)
end
