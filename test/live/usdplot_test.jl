# usdplot recipe + bind_usd! — pure logic + GPU behaviour (subprocess).
#
# Pure (in-process): the recipe (childless/atomic, abspath convert_arguments, data_limits == bbox),
# target parsing (prim vs attr split, xformOp:* refusal, empty-segment + identifier validation),
# the up-fold matrix, value classification, and the WeakKeyDict registry (add / replace / unbind).
#
# GPU (subprocess, pixel-verified):
#   • compose an on-disk arm.usda into a live stage → red quad renders; a displayColor attribute
#     binding flips it red→green; a prim (transform) binding translates it (centroid moves ≫20 px);
#     a bogus target throws OVRTXError AT BIND TIME (fail-fast); delete! removes it (pixels gone).
#   • accumulate_across_frames: a BOUND update fires ZERO resets (non-structural), a structural
#     insert fires one; default mode resets on the bound update (control).
#   • up = :y folds a +90° X rotation so a quad authored flat in the file's X-Z plane renders
#     face-on (many px) where up = :z leaves it edge-on (few px).

using Test
include(joinpath(@__DIR__, "..", "helpers.jl"))
using OmniverseMakie
const OM = OmniverseMakie
using OmniverseMakie: OV
import Makie
using Makie: Scene, Observable, Rect3f, Rect3d, Point3f, Vec3f, Vec4f, RGBf, translationmatrix

# =============================================================================================
# Pure logic (no GPU)
# =============================================================================================

@testset "usdplot pure: recipe / parsing / registry / dispatch" begin
    @testset "recipe is a childless atomic plot; abspath; data_limits" begin
        sc = Scene()
        p  = usdplot!(sc, "rel/car.usdc")
        @test p isa OM.USDPlot
        @test isempty(p.plots)                       # childless → backend atomic path
        @test isabspath(p[1][])                      # convert_arguments absolutized the path
        @test endswith(p[1][], "/rel/car.usdc")
        @test p.up[] === :z                          # default up
        @test Makie.data_limits(p) == Rect3d(p.bbox[])
        # custom bbox frames the axis
        bb = Rect3f(Point3f(-2, -3, -4), Vec3f(4, 6, 8))
        p2 = usdplot!(Scene(), "x.usd"; bbox = bb)
        @test Makie.data_limits(p2) == Rect3d(bb)
    end

    @testset "up-fold matrix" begin
        pz = usdplot!(Scene(), "x.usd"; up = :z)
        py = usdplot!(Scene(), "x.usd"; up = :y)
        M  = translationmatrix(Vec3f(1, 2, 3))
        @test OM._usdplot_model(pz, M) == M                    # :z → identity passthrough
        @test OM._usdplot_model(py, M) == M * OM.ROT_X_90      # :y → +90° X fold
        # ROT_X_90 maps the asset's +Y onto the scene's +Z
        @test isapprox(OM.ROT_X_90 * Vec4f(0, 1, 0, 1), Vec4f(0, 0, 1, 1); atol = 1e-6)
    end

    @testset "target parsing: prim / attr split + validation" begin
        @test OM._parse_usd_target("/Arm") == ("/Arm", nothing)
        @test OM._parse_usd_target("/Arm/Geo.primvars:displayColor") == ("/Arm/Geo", "primvars:displayColor")
        @test OM._parse_usd_target("/A/B/C.foo:bar") == ("/A/B/C", "foo:bar")
        # rejected targets
        for bad in ("Arm",                       # no leading /
                    "/Arm.xformOp:translate",    # baked transform op
                    "/Arm.xformOpOrder",         # baked op order
                    "/Arm.",                     # empty attribute
                    "/Arm/.x",                   # empty prim segment (stray /.)
                    "/Arm//Geo",                 # double /
                    "/Arm/",                     # trailing /
                    "/",                         # root only
                    "/9bad/x",                   # segment starts with a digit
                    "/a b/c")                    # space in a segment
            @test_throws Exception OM._parse_usd_target(bad)
        end
    end

    @testset "value classification + guards" begin
        @test OM._is_vec3(Vec3f(1, 2, 3))
        @test OM._is_vec3(Point3f(1, 2, 3))
        @test OM._is_vec3((1.0, 2.0, 3.0))
        @test !OM._is_vec3([1.0, 2.0, 3.0])      # a plain Vector is the ARRAY case, not a single vec3
        @test !OM._is_vec3(1.0)
        @test !OM._is_vec3(RGBf(1, 0, 0))
        # an unsupported attribute value throws BEFORE touching the renderer (r = nothing here)
        @test_throws ArgumentError OM._write_usd_attr!(nothing, "/p", "a", :nope; prim_mode = nothing)
        @test_throws ArgumentError OM._write_usd_attr!(nothing, "/p", "a", "str"; prim_mode = nothing)
        # a prim binding needs a 4×4 matrix
        @test OM._to_mat4(translationmatrix(Vec3f(1, 2, 3))) isa AbstractMatrix
        @test_throws ArgumentError OM._to_mat4([1, 2, 3])
        @test_throws ArgumentError OM._to_mat4(Vec3f(1, 2, 3))
    end

    @testset "registry: add / replace / unbind (no live screen → stash only)" begin
        p = usdplot!(Scene(), "x.usd")
        o1 = Observable(translationmatrix(Vec3f(0)))
        @test bind_usd!(p, "/Arm", o1) === o1
        @test length(OM._USD_BINDINGS[p]) == 1
        o2 = Observable(translationmatrix(Vec3f(1, 0, 0)))
        bind_usd!(p, "/Arm", o2)                                     # same target → replace
        @test length(OM._USD_BINDINGS[p]) == 1
        @test OM._USD_BINDINGS[p][1].obs === o2
        bind_usd!(p, "/Arm/Geo.primvars:displayColor", Observable([RGBf(1, 0, 0)]))
        @test length(OM._USD_BINDINGS[p]) == 2
        unbind_usd!(p, "/Arm")
        @test length(OM._USD_BINDINGS[p]) == 1
        @test OM._USD_BINDINGS[p][1].target == "/Arm/Geo.primvars:displayColor"
        unbind_usd!(p, "/not-bound")                                 # no-op
        @test length(OM._USD_BINDINGS[p]) == 1
        # a value wrapped from a plain arg comes back as an Observable
        @test bind_usd!(p, "/Bolt", translationmatrix(Vec3f(0))) isa Observable
    end
end

# =============================================================================================
# GPU subprocess: shared asset authoring + pixel helpers
# =============================================================================================

# The arm.usda-style asset (defaultPrim Model; a red `Mesh` quad from `points_str`, wrapped in an
# `Arm` Xform so a prim binding has a subprim to drive).  Built PARENT-side and embedded into each
# prog via `repr(...)` (a single-line escaped literal) so no nested `"""` collides with the prog's
# own triple-quote.
_arm_usda(points_str) = """#usda 1.0
(
    defaultPrim = "Model"
)
def Xform "Model"
{
    def Xform "Arm"
    {
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Mesh "Geo"
        {
            uniform bool doubleSided = true
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [$(points_str)]
            color3f[] primvars:displayColor = [(1, 0, 0)] (
                interpolation = "constant"
            )
        }
    }
}
"""
# X-Y quad (faces +Z): rendered face-on by a camera on +Z.
const _ARM_XY = _arm_usda("(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)")
# X-Z quad (lies flat, y=0): edge-on to a +Z camera, but a +90° X fold stands it up face-on.
const _ARM_XZ = _arm_usda("(-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1)")

# Prelude spliced into each prog: `using` lines + colour/centroid pixel oracles (no `$`, no `"""`).
const _USDPLOT_PROG_PRELUDE = """
using OmniverseMakie, ColorTypes, FixedPointNumbers
const OM = OmniverseMakie
using OmniverseMakie: OV
using Makie: Scene, Observable, Vec3f, RGBf, translationmatrix, cam3d!, update_cam!

lum(c) = Float32(red(c)) + Float32(green(c)) + Float32(blue(c))
nonblack(img) = count(c -> lum(c) > 0.05f0, img)
reddom(img)   = count(c -> Float32(red(c))   > Float32(green(c)) + 0.15f0 && Float32(red(c))   > Float32(blue(c)) + 0.15f0, img)
greendom(img) = count(c -> Float32(green(c)) > Float32(red(c))   + 0.15f0 && Float32(green(c)) > Float32(blue(c)) + 0.15f0, img)
function centroid(img)
    H, W = size(img); sr = 0.0; sc = 0.0; n = 0
    for h in 1:H, w in 1:W
        lum(img[h, w]) > 0.10f0 && (sr += h; sc += w; n += 1)
    end
    return (n = n, crow = n > 0 ? sr / n : -1.0, ccol = n > 0 ? sc / n : -1.0)
end
"""

# ---------------------------------------------------------------------------------------------
# Prog A — compose + displayColor flip + prim move + fail-fast + delete
# ---------------------------------------------------------------------------------------------

const _USDPLOT_CORE_PROG = """
$(_USDPLOT_PROG_PRELUDE)

OM.activate!(warmup = 40, samples = 128)
const ARM = tempname() * ".usda"
write(ARM, $(repr(_ARM_XY)))

scene = Scene(size = (300, 300)); cam3d!(scene)
update_cam!(scene, Vec3f(0, 0, 14), Vec3f(0, 0, 0), Vec3f(0, 1, 0))
p = usdplot!(scene, ARM)
screen = OM.Screen(scene)

img0 = Makie.colorbuffer(screen); nb0 = nonblack(img0); rd0 = reddom(img0); c0 = centroid(img0)
println("BASE_NB=", nb0, " BASE_RED=", rd0, " BASE_COL=", round(c0.ccol, digits = 1))

col = Observable([RGBf(1, 0, 0)])
bind_usd!(p, "/Arm/Geo.primvars:displayColor", col)
col[] = [RGBf(0, 1, 0)]
img1 = Makie.colorbuffer(screen)
println("COLOR_GREEN=", greendom(img1), " COLOR_RED=", reddom(img1))

arm = Observable(translationmatrix(Vec3f(0, 0, 0)))
bind_usd!(p, "/Arm", arm)
arm[] = translationmatrix(Vec3f(5, 0, 0))
img2 = Makie.colorbuffer(screen); c2 = centroid(img2)
println("MOVE_DCOL=", round(c2.ccol - c0.ccol, digits = 1))

function try_bad_bind()
    try
        bind_usd!(p, "/Nope.primvars:displayColor", Observable([RGBf(1, 0, 0)]))
        return false
    catch e
        return e isa OV.LibOVRTX.OVRTXError
    end
end
threw = try_bad_bind()
stale = any(b -> b.target == "/Nope.primvars:displayColor", OM._USD_BINDINGS[p])
println("FAILFAST_THREW=", threw, " FAILFAST_STALE=", stale)

delete!(screen, scene, p)
img3 = Makie.colorbuffer(screen)
println("DELETE_NB=", nonblack(img3))

close(screen)
rm(ARM; force = true)
println("OK_USDPLOT_CORE")
"""

@testset "usdplot GPU: compose + displayColor + prim move + fail-fast + delete (subprocess)" begin
    exitcode, out = run_ovrtx_subprocess(_USDPLOT_CORE_PROG; timeout = 900, retries = 4,
                                         ready_marker = "OK_USDPLOT_CORE")
    contains(out, "OK_USDPLOT_CORE") || @info "usdplot core output" out
    @test exitcode == 0
    @test contains(out, "OK_USDPLOT_CORE")

    getint(tag)   = (m = match(Regex("$(tag)=(-?\\d+)"), out); m === nothing ? nothing : parse(Int, m.captures[1]))
    getfloat(tag) = (m = match(Regex("$(tag)=(-?[\\d.]+)"), out); m === nothing ? nothing : parse(Float64, m.captures[1]))

    @test getint("BASE_NB")   !== nothing && getint("BASE_NB")   > 300   # asset composed + rendered
    @test getint("BASE_RED")  !== nothing && getint("BASE_RED")  > 300   # red-dominant
    @test getint("COLOR_GREEN") !== nothing && getint("COLOR_GREEN") > 300  # displayColor flipped to green
    @test getint("COLOR_RED")   !== nothing && getint("COLOR_RED")   < 50   # ...and no longer red
    @test getfloat("MOVE_DCOL") !== nothing && getfloat("MOVE_DCOL") > 30   # prim binding moved it right
    @test contains(out, "FAILFAST_THREW=true")                             # bogus bind threw at bind time
    @test contains(out, "FAILFAST_STALE=false")                            # ...and left no bogus binding
    @test getint("DELETE_NB") !== nothing && getint("DELETE_NB") < 75      # delete! removed the reference
end

# ---------------------------------------------------------------------------------------------
# Prog B — accumulate suppresses the reset on a bound update; a structural insert still resets
# ---------------------------------------------------------------------------------------------

const _USDPLOT_ACCUM_PROG = """
$(_USDPLOT_PROG_PRELUDE)

OM.activate!(warmup = 4)
const ARM  = tempname() * ".usda"; write(ARM,  $(repr(_ARM_XY)))
const ARM2 = tempname() * ".usda"; write(ARM2, $(repr(_ARM_XY)))

resets = Ref(0)
OV._RESET_OBSERVER[] = () -> (resets[] += 1)

# accumulate mode
scene = Scene(size = (200, 200)); cam3d!(scene)
update_cam!(scene, Vec3f(0, 0, 14), Vec3f(0, 0, 0), Vec3f(0, 1, 0))
p = usdplot!(scene, ARM)
screen = OM.Screen(scene; accumulate_across_frames = true, warmup = 4, accumulation_preroll = 8)
Makie.colorbuffer(screen)                       # author (structural reset — ignored below)
col = Observable([RGBf(1, 0, 0)])
bind_usd!(p, "/Arm/Geo.primvars:displayColor", col)
resets[] = 0
col[] = [RGBf(0, 1, 0)]                          # a BOUND (non-structural) update
Makie.colorbuffer(screen)
println("ACCUM_BOUND_RESETS=", resets[])        # expect 0 (accumulate keeps the history)

resets[] = 0
p2 = usdplot!(scene, ARM2)
insert!(screen, scene, p2)                       # STRUCTURAL change
Makie.colorbuffer(screen)
println("ACCUM_INSERT_RESETS=", resets[])        # expect ≥ 1
close(screen)

# default mode control: the same bound update DOES reset
scene2 = Scene(size = (200, 200)); cam3d!(scene2)
update_cam!(scene2, Vec3f(0, 0, 14), Vec3f(0, 0, 0), Vec3f(0, 1, 0))
q = usdplot!(scene2, ARM)
screen2 = OM.Screen(scene2; warmup = 4)
Makie.colorbuffer(screen2)
colq = Observable([RGBf(1, 0, 0)])
bind_usd!(q, "/Arm/Geo.primvars:displayColor", colq)
resets[] = 0
colq[] = [RGBf(0, 1, 0)]
Makie.colorbuffer(screen2)
println("DEFAULT_BOUND_RESETS=", resets[])       # expect ≥ 1
close(screen2)

OV._RESET_OBSERVER[] = nothing
rm(ARM; force = true); rm(ARM2; force = true)
println("OK_USDPLOT_ACCUM")
"""

@testset "usdplot GPU: accumulate reset suppression on bound update (subprocess)" begin
    _, out = run_ovrtx_subprocess(_USDPLOT_ACCUM_PROG; timeout = 900, retries = 4,
                                  ready_marker = "OK_USDPLOT_ACCUM")
    contains(out, "OK_USDPLOT_ACCUM") || @info "usdplot accumulate output" out
    @test contains(out, "OK_USDPLOT_ACCUM")

    m_bound  = match(r"ACCUM_BOUND_RESETS=(\d+)", out)
    m_insert = match(r"ACCUM_INSERT_RESETS=(\d+)", out)
    m_def    = match(r"DEFAULT_BOUND_RESETS=(\d+)", out)
    @test m_bound  !== nothing && parse(Int, m_bound.captures[1])  == 0   # accumulate: no reset
    @test m_insert !== nothing && parse(Int, m_insert.captures[1]) >= 1   # structural: resets
    @test m_def    !== nothing && parse(Int, m_def.captures[1])    >= 1   # default: bound update resets
end

# ---------------------------------------------------------------------------------------------
# Prog C — up = :y stands a Y-up (X-Z-plane) quad upright (face-on) vs edge-on for up = :z
# ---------------------------------------------------------------------------------------------

const _USDPLOT_UP_PROG = """
$(_USDPLOT_PROG_PRELUDE)

OM.activate!(warmup = 40, samples = 128)
const ARM = tempname() * ".usda"; write(ARM, $(repr(_ARM_XZ)))   # quad lies flat in the file's X-Z plane

function render_up(up)
    scene = Scene(size = (300, 300)); cam3d!(scene)
    update_cam!(scene, Vec3f(0, 0, 14), Vec3f(0, 0, 0), Vec3f(0, 1, 0))
    p = usdplot!(scene, ARM; up = up)
    screen = OM.Screen(scene)
    img = Makie.colorbuffer(screen)
    n = nonblack(img)
    close(screen)
    return n
end

nb_z = render_up(:z)   # camera on +Z sees the X-Z quad EDGE-ON → few lit px
nb_y = render_up(:y)   # +90° X fold rotates it to X-Y → FACE-ON → many lit px
println("UP_Z_NB=", nb_z)
println("UP_Y_NB=", nb_y)
rm(ARM; force = true)
println("OK_USDPLOT_UP")
"""

@testset "usdplot GPU: up=:y folds a Y-up quad upright (subprocess)" begin
    _, out = run_ovrtx_subprocess(_USDPLOT_UP_PROG; timeout = 900, retries = 4,
                                  ready_marker = "OK_USDPLOT_UP")
    contains(out, "OK_USDPLOT_UP") || @info "usdplot up-axis output" out
    @test contains(out, "OK_USDPLOT_UP")

    m_z = match(r"UP_Z_NB=(\d+)", out)
    m_y = match(r"UP_Y_NB=(\d+)", out)
    @test m_z !== nothing && m_y !== nothing
    if m_z !== nothing && m_y !== nothing
        nb_z = parse(Int, m_z.captures[1]); nb_y = parse(Int, m_y.captures[1])
        @test nb_y > 300           # up=:y → face-on quad renders
        @test nb_y > 4 * nb_z      # ...much larger than the up=:z edge-on strip
    end
end
