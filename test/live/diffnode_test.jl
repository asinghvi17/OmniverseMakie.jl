using Test
include(joinpath(@__DIR__, "..", "helpers.jl"))
using OmniverseMakie
import OmniverseMakie as OM

# ---------------------------------------------------------------------------
# Empty resolved-color guards (pure): an EMPTY per-vertex color must not be
# wrapped as `[values]` and read `c[1]` under @inbounds (UB), nor reduced with
# `sum` over an empty collection.  Both helpers early-return cleanly.
# ---------------------------------------------------------------------------
@testset "empty-color push guards (pure)" begin
    # empty per-vertex displayColor → no write, unrouted (false), no throw
    @test OM._push_displaycolor!(nothing, "/p", nothing, RGBf[]) === false
    # empty per-vertex materialized color → neutral white (no sum over ∅)
    @test OM._materialized_base_rgb(nothing, RGBf[]) === (1f0, 1f0, 1f0)
end

# ---------------------------------------------------------------------------
# The :ovrtx_renderobject diff node + push_to_ovrtx! (the diff driver).
#
# On an already-open stage, an attribute edit must push EXACTLY ONE minimal C
# write (instrumented via a per-name counter on push_to_ovrtx!) and produce a
# visible render change — WITHOUT re-authoring the stage.
#
#   color edit     (m.color = :blue)    → one :scaled_color write → mesh blue
#   transform edit (translate!(m,…))    → one :model_f32c write → geometry
#                                         MOVES (the REFERENCED-prim write)
#   transform back (translate!(m,0,0,0))→ one :model_f32c write → frame
#                                         RETURNS (replace semantics)
#   :model_f32c carries the COMPOSED world transform.
# ---------------------------------------------------------------------------

const _M22_DIFFNODE_PROG = """
using OmniverseMakie, ColorTypes, FixedPointNumbers, GeometryBasics
const OM = OmniverseMakie

OM.activate!(warmup = 48)

scene = Scene(size = (300, 300))
cam3d!(scene)
update_cam!(scene, Vec3d(7, 7, 5), Vec3d(0, 0, 0), Vec3d(0, 0, 1))
# Build from a concrete mesh (not a Rect3f) so the arg-1 Observable's element
# type is loose enough to later swap in a different-vertex-count mesh.
m = mesh!(scene, GeometryBasics.normal_mesh(Rect3f(Point3f(-1, -1, -1), Vec3f(2))); color = :red)

screen = OM.Screen(scene)
Makie.push_screen!(scene, screen)

function mean_rgb(img)
    tr = 0.0; tg = 0.0; tb = 0.0; cnt = 0
    for c in img
        r = Float32(red(c)); g = Float32(green(c)); b = Float32(blue(c))
        if r + g + b > 0.05
            tr += r; tg += g; tb += b; cnt += 1
        end
    end
    cnt == 0 ? (0.0, 0.0, 0.0) : (round(tr/cnt; digits=3), round(tg/cnt; digits=3), round(tb/cnt; digits=3))
end
red_centroid(img) = begin
    H, W = size(img); sr = 0.0; sc = 0.0; n = 0
    for h in 1:H, w in 1:W
        c = img[h, w]; r = Float32(red(c)); g = Float32(green(c)); b = Float32(blue(c))
        (r > g && r > b && r > 0.10) && (sr += h; sc += w; n += 1)
    end
    n > 0 ? (round(sr/n; digits=1), round(sc/n; digits=1)) : (NaN, NaN)
end
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

# ---- BUILD: first colorbuffer authors the open stage + the diff node ----
imgA = Makie.colorbuffer(screen)
@assert haskey(screen.plot2robj, objectid(m)) "mesh not registered as an OvrtxRObj"
rgbA = mean_rgb(imgA)
println("RGB_A=\$(rgbA)")
@assert nonblack(imgA) > 300 "build frame (near) black"
@assert rgbA[1] > rgbA[3] "build frame not red-dominant: \$(rgbA)"

# ---- install the per-name push counter ----
counter = Dict{Symbol,Int}()
OM._PUSH_OBSERVER[] = name -> (counter[name] = get(counter, name, 0) + 1)

# ---- COLOR EDIT: exactly one :scaled_color write, mesh turns blue ----
empty!(counter)
m.color = :blue
imgB = Makie.colorbuffer(screen)
rgbB = mean_rgb(imgB)
println("COUNTER_COLOR=\$(counter)")
println("RGB_B=\$(rgbB)")
@assert counter == Dict(:scaled_color => 1) "color edit fired \$(counter), expected one :scaled_color"
@assert rgbB[3] > rgbB[1] "color edit did not turn mesh blue: \$(rgbB)"

# ---- TRANSFORM EDIT: one :model_f32c write, geometry MOVES (ref prim) ----
empty!(counter)
translate!(m, 0, 0, 4)
imgC = Makie.colorbuffer(screen)
moved = changed(imgB, imgC)
println("COUNTER_XFORM=\$(counter)")
println("CENTROID_B=\$(red_centroid(imgB)) CENTROID_C=\$(red_centroid(imgC))")
println("CHANGED_B_vs_C=\$(moved)")
@assert counter == Dict(:model_f32c => 1) "transform edit fired \$(counter), expected one :model_f32c"
@assert moved > 1500 "transform edit did not move geometry on the referenced prim (changed=\$(moved))"

# ---- TRANSFORM ROUND-TRIP: one write, frame RETURNS (replace semantics) ----
empty!(counter)
translate!(m, 0, 0, 0)
imgD = Makie.colorbuffer(screen)
ret = changed(imgB, imgD)
println("COUNTER_XFORM_BACK=\$(counter)")
println("CHANGED_B_vs_D=\$(ret)")
@assert counter == Dict(:model_f32c => 1) "transform round-trip fired \$(counter), expected one :model_f32c"
@assert ret < moved ÷ 4 "transform round-trip did not return (ret=\$(ret) moved=\$(moved))"

# ---- COUNT-CHANGING MESH EDIT: swap the box for a sphere with a DIFFERENT
# vertex count.  The persistent points binding is sized for the box; a
# count-changing edit MUST take the one-shot resize path (frozen
# :mesh_npoints gate), not write through the wrong-sized binding — and the
# new geometry must render. ----
robj = screen.plot2robj[objectid(m)]
na   = robj.meta[:mesh_npoints]
sph  = GeometryBasics.normal_mesh(GeometryBasics.Tesselation(GeometryBasics.Sphere(Point3f(0), 2f0), 32))
nb_v = length(GeometryBasics.coordinates(sph))
empty!(counter)
m[1][] = sph
imgE = Makie.colorbuffer(screen)
nbE  = nonblack(imgE)
frozen = robj.meta[:mesh_npoints]   # gate size is author-frozen (never re-pushed)
println("MESH_NA=\$(na) MESH_NB=\$(nb_v) MESH_FROZEN=\$(frozen)")
println("COUNTER_MESHSWAP=\$(counter)")
println("NONBLACK_E=\$(nbE)")
@assert na != nb_v "mesh swap did not change vertex count (na=\$(na) nb=\$(nb_v)); test is a no-op"
@assert :positions_transformed_f32c in keys(counter) "count-changing mesh edit did not push positions"
@assert frozen == na "mesh_npoints gate was mutated on push (\$(frozen) != \$(na)); must stay frozen"
@assert nbE > 300 "count-changing mesh edit did not render (nbE=\$(nbE)); positions write dropped"

# ---- stage authored exactly once across all the live diffs ----
opens = OM._ROOT_OPEN_COUNT[]
println("ROOT_OPENS=\$(opens)")
@assert opens == 1 "stage re-opened during diffs (opens=\$(opens)); diffs must be live writes"

close(screen)
println("OK_DIFFNODE")
"""

@testset "M2.2 :ovrtx_renderobject diff node + push_to_ovrtx! (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M22_DIFFNODE_PROG; timeout = 900, retries = 2, ready_marker = "RGB_A=")
    @info "M2.2 diffnode subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_DIFFNODE")

    # Exactly one write per edit (per-name counter).
    @test contains(output, "COUNTER_COLOR=Dict(:scaled_color => 1)")
    @test contains(output, "COUNTER_XFORM=Dict(:model_f32c => 1)")
    @test contains(output, "COUNTER_XFORM_BACK=Dict(:model_f32c => 1)")

    # Visible change: red build → blue after the color edit.
    ma = match(r"RGB_A=\((.*?)\)", output)
    mb = match(r"RGB_B=\((.*?)\)", output)
    if ma !== nothing && mb !== nothing
        a = parse.(Float64, split(ma.captures[1], ", "))
        b = parse.(Float64, split(mb.captures[1], ", "))
        @test a[1] > a[3]    # build red-dominant
        @test b[3] > b[1]    # after edit blue-dominant
    else
        @test false
    end

    # Transform moved then returned.
    mc = match(r"CHANGED_B_vs_C=(\d+)", output)
    md = match(r"CHANGED_B_vs_D=(\d+)", output)
    if mc !== nothing && md !== nothing
        moved = parse(Int, mc.captures[1]); ret = parse(Int, md.captures[1])
        @test moved > 1500       # referenced-prim write moved geometry
        @test ret < moved ÷ 4    # round-trip returned (replace semantics)
    else
        @test false
    end

    # Count-changing mesh edit: vertex count genuinely changed, positions
    # pushed, and the new geometry rendered (one-shot resize path, not a
    # dropped/corrupt write through the author-time-sized binding).
    mn = match(r"MESH_NA=(\d+) MESH_NB=(\d+) MESH_FROZEN=(\d+)", output)
    @test mn !== nothing && parse(Int, mn.captures[1]) != parse(Int, mn.captures[2])
    # gate size stays author-frozen across the count-changing push
    @test mn !== nothing && parse(Int, mn.captures[3]) == parse(Int, mn.captures[1])
    @test contains(output, "COUNTER_MESHSWAP=") &&
          occursin(":positions_transformed_f32c", match(r"COUNTER_MESHSWAP=(.*)", output).captures[1])
    me = match(r"NONBLACK_E=(\d+)", output)
    @test me !== nothing && parse(Int, me.captures[1]) > 300

    # Stage authored exactly once.
    mo = match(r"ROOT_OPENS=(\d+)", output)
    @test mo !== nothing && parse(Int, mo.captures[1]) == 1
end
