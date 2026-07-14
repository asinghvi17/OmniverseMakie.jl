using Test

# gpu_update_mesh! (OmniverseMakieCUDADirectExt): mesh points pushed straight
# from CUDA device arrays as kDLCUDA DLTensors.  Pixel oracle (ovrtx
# accepts-and-ignores is the classic failure mode):
#   1. a +0.5 z shift written from a CuArray MOVES the box;
#   2. the GPU-path frame matches a CPU-path write of the same data;
#   3. tripwire: normals writes stay pixel-inert (why there is no normals API);
#   4. wrong length / undisplayed plot / materialized plot all throw
#      ArgumentError host-side.
const _GPU_DIRECT_PROG = """
using OmniverseMakie, CUDA, GeometryBasics, ColorTypes, FixedPointNumbers
import OmniverseMakie as OM
using OmniverseMakie: OV

println("CUDA_FUNCTIONAL=", CUDA.functional())
OM.activate!(warmup = 24)

lights = Makie.AbstractLight[
    AmbientLight(RGBf(0.3, 0.3, 0.3)),
    DirectionalLight(RGBf(2.2, 2.1, 2.0), Vec3f(-0.3, -0.4, -1.0), false)]
scene = Scene(size = (300, 300); lights)
cam3d!(scene)
box = GeometryBasics.normal_mesh(Rect3f(Point3f(-0.5, -0.5, -0.5), Vec3f(1)))
p = mesh!(scene, box; color = :red)
# dim floor (below the lit threshold) so pixel counts isolate the box
mesh!(scene, Rect3f(Point3f(-4, -4, -1.2), Vec3f(8, 8, 0.05)); color = RGBf(0.10, 0.12, 0.20))
update_cam!(scene, Vec3f(0, -5, 1.2), Vec3f(0, 0, 0.4), Vec3f(0, 0, 1))

screen = OM.Screen(scene)
a = Makie.colorbuffer(screen)
println("READY_GPU_DIRECT")          # startup survived — retries stop re-rolling

lum(c) = Float32(red(c)) + Float32(green(c)) + Float32(blue(c))
ndiff(x, y) = count(k -> abs(lum(x[k]) - lum(y[k])) > 0.08f0, eachindex(x))

pts0 = Vector{Point3f}(p.positions_transformed_f32c[])
ns0  = Vector{Vec3f}(p.normals[])
npts = length(pts0)
println("NPTS=", npts)

# 1. device points write (Point3f eltype exercises the reinterpret path)
shifted = [Point3f(q[1], q[2], q[3] + 0.5f0) for q in pts0]
OM.gpu_update_mesh!(screen, p; points = CuArray(shifted))
b = Makie.colorbuffer(screen)
println("MOVED_PX=", ndiff(a, b))

# 2. same data through the proven CPU binding path -> frames agree
robj = screen.plot2robj[objectid(p)]
OV.write_binding!(robj.bindings[:positions_transformed_f32c],
                  collect(reinterpret(Float32, shifted)), Int64[npts])
c = Makie.colorbuffer(screen)
println("GPU_VS_CPU_PX=", ndiff(b, c))

# 3. CONSTRAINT TRIPWIRE: vertex-normal writes are pixel-inert in standalone
# ovrtx (the engine shades meshes from geometry-derived normals; verified
# for authored AND live normals, RT2 AND :pathtracing, 2026-07-14) — which
# is why gpu_update_mesh! has no normals path.  If this starts CHANGING
# pixels on a new build, normals matter now: add the device normals API.
OV.write_array_attribute!(screen.renderer, robj.prim_path, "normals",
                          [Vec3f(0, 0, 1) for _ in ns0])
d = Makie.colorbuffer(screen)
println("NORMALS_INERT_PX=", ndiff(c, d))

# 4. error paths (host-side ArgumentErrors, no render)
err(f) = try f(); "none" catch e; e isa ArgumentError ? "argerr" : string(typeof(e)) end
println("ERR_LEN=",    err(() -> OM.gpu_update_mesh!(screen, p; points = CUDA.zeros(Float32, 5))))
orphan = mesh!(Scene(), Rect3f(Point3f(0), Vec3f(1)))
println("ERR_ORPHAN=", err(() -> OM.gpu_update_mesh!(screen, orphan; points = CuArray(shifted))))
surf = surface!(scene, -1:1, -1:1, zeros(3, 3))
Makie.colorbuffer(screen)
println("ERR_SURF=",   err(() -> OM.gpu_update_mesh!(screen, surf; points = CuArray(shifted))))

close(screen)
println("OK_GPU_DIRECT")
"""

include(joinpath(@__DIR__, "..", "helpers.jl"))

@testset "gpu_update_mesh!: device points through the binding + normals tripwire (subprocess)" begin
    # Retry past ovrtx's intermittent pre-render startup crash.
    exitcode, out = run_ovrtx_subprocess(_GPU_DIRECT_PROG; timeout = 600, retries = 4,
                                         ready_marker = "READY_GPU_DIRECT")
    contains(out, "OK_GPU_DIRECT") || @info "gpu_direct output" out
    # surface the pixel-oracle numbers either way
    @info "gpu_direct" moved=match(r"MOVED_PX=(\d+)", out) gpu_vs_cpu=match(r"GPU_VS_CPU_PX=(\d+)", out) normals_inert=match(r"NORMALS_INERT_PX=(\d+)", out)
    @test exitcode == 0
    @test contains(out, "CUDA_FUNCTIONAL=true")
    @test contains(out, "OK_GPU_DIRECT")

    # 1. The device write moved the box.
    m = match(r"MOVED_PX=(\d+)", out)
    @test m !== nothing && parse(Int, m.captures[1]) > 300

    # 2. GPU-path frame == CPU-path frame for identical data (noise floor).
    m = match(r"GPU_VS_CPU_PX=(\d+)", out)
    @test m !== nothing && parse(Int, m.captures[1]) < 500

    # 3. Tripwire: normals writes stay pixel-inert (see prog comment).
    m = match(r"NORMALS_INERT_PX=(\d+)", out)
    @test m !== nothing && parse(Int, m.captures[1]) < 100

    # 4. Host-side validation errors.
    @test contains(out, "ERR_LEN=argerr")
    @test contains(out, "ERR_ORPHAN=argerr")
    @test contains(out, "ERR_SURF=argerr")
end
