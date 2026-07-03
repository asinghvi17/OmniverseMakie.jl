using Test
include(joinpath(@__DIR__, "..", "helpers.jl"))

# ---------------------------------------------------------------------------
# M2.4 — persistent hot-path bindings (map_attribute / bind_array_attribute).
#
# The per-frame plot writes for the HOT attributes must go through persistent
# ovrtx attribute bindings created ONCE (`bind_hot_attributes!`) and reused, NOT a
# fresh `write_attribute` each frame:
#   :model_f32c (omni:xform)            → map_attribute zero-copy binding (OPTIMIZE)
#   :positions_transformed_f32c (points)→ bind_array_attribute + write (mesh tier)
#
# The test animates a mesh's omni:xform for 100 frames through the MAPPED binding
# and asserts:
#   (a) each frame moves the rendered content,
#   (b) NO per-frame USDA authoring — the binding object identity is STABLE across
#       all 100 frames (created once, reused) AND `_ROOT_OPEN_COUNT` is unchanged,
#   plus bindings are destroyed on `close(screen)`.
#
# RED (M2.3, empty `robj.bindings`): the `haskey(robj.bindings, :model_f32c)`
#   assertion fails right after the first colorbuffer — no persistent binding exists.
# GREEN (M2.4): bind_hot_attributes! creates them; push_to_ovrtx! routes through them.
# ---------------------------------------------------------------------------

const _M24_BINDING_PROG = """
using OmniverseMakie, ColorTypes, FixedPointNumbers, GeometryBasics
const OM = OmniverseMakie

OM.activate!(warmup = 6)

scene = Scene(size = (320, 320))
cam3d!(scene)
update_cam!(scene, Vec3d(7, 7, 5), Vec3d(0, 0, 0), Vec3d(0, 0, 1))
m = mesh!(scene, Rect3f(Point3f(-1, -1, -1), Vec3f(2)); color = :red)

screen = OM.Screen(scene)
Makie.push_screen!(scene, screen)

function changed(x, y; thr = 0.15)
    H, W = size(x); n = 0
    for h in 1:H, w in 1:W
        cx = x[h, w]; cy = y[h, w]
        d = abs(Float32(red(cx)) - Float32(red(cy))) + abs(Float32(green(cx)) - Float32(green(cy))) +
            abs(Float32(blue(cx)) - Float32(blue(cy)))
        d > thr && (n += 1)
    end
    n
end
nonblack(img) = count(c -> (Float32(red(c)) + Float32(green(c)) + Float32(blue(c))) > 0.05, img)

# ---- Frame 1: author stage + build plot reference + create persistent bindings ----
img1 = Makie.colorbuffer(screen)
@assert nonblack(img1) > 200 "build frame (near) black"
robj = screen.plot2robj[objectid(m)]

# Persistent omni:xform binding created ONCE on the referenced prim (the hot tier).
@assert haskey(robj.bindings, :model_f32c) "no persistent :model_f32c binding (M2.4 hot path)"
xb = robj.bindings[:model_f32c]
@assert xb isa OM.OV.Binding "xform binding is not an OV.Binding: \$(typeof(xb))"
@assert xb.handle != 0 "xform binding handle is 0 (create_attribute_binding failed)"
@assert !xb.is_array "xform binding must be the fixed-size (map) tier"
bind_id1     = objectid(xb)
bind_handle1 = xb.handle
println("BIND_OK=true HANDLE=\$(xb.handle)")

# Mesh points array binding too (bind_array_attribute tier).
@assert haskey(robj.bindings, :positions_transformed_f32c) "no persistent points binding (mesh array tier)"
pb = robj.bindings[:positions_transformed_f32c]
@assert pb isa OM.OV.Binding "points binding is not an OV.Binding"
@assert pb.is_array "points binding must be the array tier"

opens_before = OM._ROOT_OPEN_COUNT[]
@assert opens_before == 1 "stage opened \$(opens_before)× before the loop (expected 1)"

# Per-name push counter: every routed write fires the observer regardless of tier.
counter = Dict{Symbol,Int}()
OM._PUSH_OBSERVER[] = name -> (counter[name] = get(counter, name, 0) + 1)

# ---- Animate omni:xform for 100 frames through the MAPPED binding ----
prev = img1
movers = 0
for k in 1:100
    z = isodd(k) ? 4.0 : 0.0            # toggle → a guaranteed move every frame
    translate!(m, 0, 0, z)
    img = Makie.colorbuffer(screen)
    (changed(prev, img) > 150) && (global movers += 1)
    global prev = img
end
OM._PUSH_OBSERVER[] = nothing

println("MOVERS=\$(movers) / 100")
println("XFORM_WRITES=\$(get(counter, :model_f32c, 0))")

# (a) each frame moved the content through the binding.
@assert movers >= 95 "frames did not move through the mapped binding (movers=\$(movers))"

# (b) NO re-author + binding identity STABLE across all 100 frames (created once, reused).
xb2 = screen.plot2robj[objectid(m)].bindings[:model_f32c]
@assert objectid(xb2) == bind_id1 "xform binding object identity changed across frames (re-created)"
@assert xb2.handle == bind_handle1 "xform binding handle changed across frames"
opens_after = OM._ROOT_OPEN_COUNT[]
println("OPENS_AFTER=\$(opens_after)")
@assert opens_after == 1 "stage re-opened during the 100-frame animation (opens=\$(opens_after)); must be live binding writes"
@assert get(counter, :model_f32c, 0) >= 95 "expected ~100 :model_f32c writes, got \$(get(counter, :model_f32c, 0))"

# ---- bindings destroyed on close(screen) ----
close(screen)
@assert !xb.alive "xform binding not destroyed on close(screen)"
@assert !pb.alive "points binding not destroyed on close(screen)"
@assert isempty(robj.bindings) "robj.bindings not cleared on close(screen)"
println("OK_BINDING")
"""

@testset "M2.4 persistent map/bind hot-path bindings (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M24_BINDING_PROG; timeout = 900, retries = 2, ready_marker = "OK_BINDING")
    @info "M2.4 binding subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_BINDING")

    # Persistent binding created once on the referenced prim (non-zero handle).
    mh = match(r"HANDLE=(\d+)", output)
    @test mh !== nothing && parse(Int, mh.captures[1]) != 0

    # Each frame moved through the mapped binding.
    mm = match(r"MOVERS=(\d+)", output)
    @test mm !== nothing && parse(Int, mm.captures[1]) >= 95

    # The omni:xform writes went through the diff path (~one per changed frame).
    mw = match(r"XFORM_WRITES=(\d+)", output)
    @test mw !== nothing && parse(Int, mw.captures[1]) >= 95

    # NO per-frame USDA authoring across the whole animation.
    mo = match(r"OPENS_AFTER=(\d+)", output)
    @test mo !== nothing && parse(Int, mo.captures[1]) == 1
end
