using Test
include(joinpath(@__DIR__, "..", "helpers.jl"))

# ---------------------------------------------------------------------------
# Leak-free `delete!` / `delete!(scene)` / `empty!` teardown.
#
# Imperative teardown on the OPEN stage (no re-author) must leave ZERO residual
# prims-with-content, GPU bindings, or diff-nodes:
#
#   delete!(screen, scene, plot)  → destroy bindings, remove_usd! the
#       reference, drop plot2robj, delete the :ovrtx_renderobject node;
#       render drops to bg.
#   delete!(screen, scene)        → recurse children + plots; drop
#       scene2scope.  Add/remove a subscene 50× → registries return to
#       baseline (NO accumulation).
#   empty!(screen)                → tear down everything; all three registries
#       empty + every binding destroyed; the screen still renders a
#       freshly-added plot (structural-re-open carry: references re-add on
#       the still-open stage).
# ---------------------------------------------------------------------------

# ===========================================================================
# Subprocess 1 — single-plot delete!, empty!, and the structural-re-open carry.
# ===========================================================================
const _M25_DELETE_PROG = """
using OmniverseMakie, ColorTypes, FixedPointNumbers, GeometryBasics
const OM = OmniverseMakie

OM.activate!(warmup = 12)

scene = Scene(size = (320, 320))
cam3d!(scene)
update_cam!(scene, Vec3d(7, 7, 5), Vec3d(0, 0, 0), Vec3d(0, 0, 1))
m = mesh!(scene, Rect3f(Point3f(-1, -1, -1), Vec3f(2)); color = :red)

screen = OM.Screen(scene)
Makie.push_screen!(scene, screen)

nonblack(img) = count(c -> (Float32(red(c)) + Float32(green(c)) + Float32(blue(c))) > 0.05f0, img)

# ---- BUILD: author the open stage + the mesh reference + its bindings ----
img0 = Makie.colorbuffer(screen)
nb0  = nonblack(img0)
println("NONBLACK_BUILD=\$(nb0)")
@assert nb0 > 300 "build frame (near) black: \$(nb0)"
@assert haskey(screen.plot2robj, objectid(m)) "mesh not registered"
robj = screen.plot2robj[objectid(m)]
@assert haskey(robj.bindings, :model_f32c) "no xform binding to test teardown"
xb = robj.bindings[:model_f32c]
pb = get(robj.bindings, :positions_transformed_f32c, nothing)
@assert xb.alive "xform binding not alive before delete"

# ---- delete!(screen, scene, plot): leak-free single-plot teardown ----
delete!(screen, scene, m)
println("ROBJ_AFTER_DELETE=\$(length(screen.plot2robj))")
@assert isempty(screen.plot2robj) "plot2robj not empty after delete: \$(length(screen.plot2robj))"
@assert !haskey(m.attributes, :ovrtx_renderobject) "diff node still on plot.attributes after delete (leak / would re-fire)"
@assert !xb.alive "xform binding NOT destroyed on delete (GPU leak)"
@assert pb === nothing || !pb.alive "points binding NOT destroyed on delete"
@assert isempty(robj.bindings) "robj.bindings not cleared on delete"

# ---- render drops to background (the prim is gone from the stage) ----
img1 = Makie.colorbuffer(screen)
nb1  = nonblack(img1)
println("NONBLACK_AFTER_DELETE=\$(nb1)")
@assert nb1 < nb0 ÷ 3 "render did not drop to background after delete (nb0=\$(nb0) nb1=\$(nb1)); prim still present"

# ---- re-add + render works on the still-open stage ----
m2 = mesh!(scene, Rect3f(Point3f(-1, -1, -1), Vec3f(2)); color = :green)
s2 = scatter!(scene, [Point3f(2cos(t), 2sin(t), 0) for t in range(0, 2pi, length = 8)];
              markersize = 0.4, color = :cyan)
img2 = Makie.colorbuffer(screen)
nb2  = nonblack(img2)
println("NONBLACK_READD=\$(nb2)")
@assert nb2 > 300 "re-added plots did not render (nb2=\$(nb2))"
@assert haskey(screen.plot2robj, objectid(m2)) "re-added mesh not registered"

# ---- empty!(screen): tear down EVERYTHING; all three registries empty ----
empty!(screen)
println("PLOT2ROBJ_AFTER_EMPTY=\$(length(screen.plot2robj))")
println("SCENE2SCOPE_AFTER_EMPTY=\$(length(screen.scene2scope))")
@assert isempty(screen.plot2robj) "plot2robj not empty after empty!"
@assert isempty(screen.scene2scope) "scene2scope not empty after empty!"
@assert !haskey(m2.attributes, :ovrtx_renderobject) "m2 node leaked after empty!"

# ---- structural-re-open carry: add a plot + colorbuffer renders correctly ----
m3 = mesh!(scene, Rect3f(Point3f(-1, -1, -1), Vec3f(2)); color = :magenta)
img3 = Makie.colorbuffer(screen)
nb3  = nonblack(img3)
println("NONBLACK_AFTER_EMPTY_READD=\$(nb3)")
@assert nb3 > 300 "plot added after empty! did not render (nb3=\$(nb3)); references not re-added (the carry)"
@assert haskey(screen.plot2robj, objectid(m3)) "post-empty! mesh not registered"

# ---- the stage was opened EXACTLY ONCE the whole time (no re-author) ----
opens = OM._ROOT_OPEN_COUNT[]
println("ROOT_OPENS=\$(opens)")
@assert opens == 1 "stage re-opened during teardown (opens=\$(opens)); delete!/empty! must NOT re-author"

close(screen)
println("OK_DELETE")
"""

@testset "M2.5 delete!/empty! leak-free teardown (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M25_DELETE_PROG; timeout = 900, retries = 2, ready_marker = "NONBLACK_BUILD=")
    @info "M2.5 delete subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_DELETE")

    # plot2robj empty + render drops to background after a single-plot delete.
    @test contains(output, "ROBJ_AFTER_DELETE=0")
    mb = match(r"NONBLACK_BUILD=(\d+)", output)
    md = match(r"NONBLACK_AFTER_DELETE=(\d+)", output)
    if mb !== nothing && md !== nothing
        nb0 = parse(Int, mb.captures[1]); nb1 = parse(Int, md.captures[1])
        @test nb0 > 300            # mesh rendered
        @test nb1 < nb0 ÷ 3        # prim gone → background
    else
        @test false
    end

    # Both registries empty after empty!.
    @test contains(output, "PLOT2ROBJ_AFTER_EMPTY=0")
    @test contains(output, "SCENE2SCOPE_AFTER_EMPTY=0")

    # Structural-re-open carry: a plot added after empty! renders.
    mr = match(r"NONBLACK_AFTER_EMPTY_READD=(\d+)", output)
    @test mr !== nothing && parse(Int, mr.captures[1]) > 300

    # Stage authored exactly once (teardown never re-opens).
    mo = match(r"ROOT_OPENS=(\d+)", output)
    @test mo !== nothing && parse(Int, mo.captures[1]) == 1
end

# ===========================================================================
# Subprocess 2 — add/remove a subscene 50× → registries return to baseline.
# Every add must be matched by delete!(screen, subscene) so scene_listeners +
# plot2robj do NOT accumulate, and remove_usd! 50× frees every per-reference
# handle (no GPU handle exhaustion).
# ===========================================================================
const _M25_SUBSCENE_LEAK_PROG = """
using OmniverseMakie, ColorTypes, FixedPointNumbers, GeometryBasics
const OM = OmniverseMakie

OM.activate!(warmup = 8)

scene = Scene(size = (256, 256))
cam3d!(scene)
update_cam!(scene, Vec3d(7, 7, 5), Vec3d(0, 0, 0), Vec3d(0, 0, 1))
base = mesh!(scene, Rect3f(Point3f(-1, -1, -1), Vec3f(2)); color = :red)

screen = OM.Screen(scene)
Makie.push_screen!(scene, screen)

nonblack(img) = count(c -> (Float32(red(c)) + Float32(green(c)) + Float32(blue(c))) > 0.05f0, img)

# Author the open stage once so insert! registers immediately (not deferred).
img0 = Makie.colorbuffer(screen)
@assert nonblack(img0) > 300 "base build frame (near) black"

base_robj      = length(screen.plot2robj)
base_scope     = length(screen.scene2scope)
println("BASELINE_ROBJ=\$(base_robj)")

peak_robj = base_robj
for i in 1:50
    sub = Scene(scene)              # add a subscene
    pl  = mesh!(sub, Rect3f(Point3f(0, 0, 0), Vec3f(1)); color = :blue)
    OM.insert!(screen, sub, pl)     # register its robj + scope
    global peak_robj = max(peak_robj, length(screen.plot2robj))
    @assert haskey(screen.plot2robj, objectid(pl)) "subscene plot not registered (iter \$(i))"
    delete!(screen, sub)            # remove the subscene
    @assert !haskey(screen.plot2robj, objectid(pl)) "subscene plot still registered after delete (iter \$(i))"
end

println("PEAK_ROBJ=\$(peak_robj)")
println("FINAL_ROBJ=\$(length(screen.plot2robj))")
println("FINAL_SCOPE=\$(length(screen.scene2scope))")

# The add actually grew the registries (the test isn't a no-op)…
@assert peak_robj == base_robj + 1 "subscene add did not grow plot2robj"
# …and 50 add/remove cycles returned them EXACTLY to baseline (NO accumulation).
@assert length(screen.plot2robj)  == base_robj  "plot2robj accumulated over 50× (got \$(length(screen.plot2robj)), baseline \$(base_robj))"
@assert length(screen.scene2scope) == base_scope "scene2scope accumulated over 50× (got \$(length(screen.scene2scope)), baseline \$(base_scope))"

# The stage still renders after all the churn.
img1 = Makie.colorbuffer(screen)
@assert nonblack(img1) > 300 "stage stopped rendering after 50× subscene churn"

opens = OM._ROOT_OPEN_COUNT[]
println("ROOT_OPENS=\$(opens)")
@assert opens == 1 "stage re-opened during subscene churn (opens=\$(opens))"

close(screen)
println("OK_SUBSCENE_LEAK")
"""

@testset "M2.5 subscene add/remove 50× — no registry accumulation (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M25_SUBSCENE_LEAK_PROG; timeout = 900, retries = 2, ready_marker = "BASELINE_ROBJ=")
    @info "M2.5 subscene-leak subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_SUBSCENE_LEAK")

    # Registries returned to baseline (no accumulation over 50 cycles).
    bl_r = match(r"BASELINE_ROBJ=(\d+)", output)
    fn_r = match(r"FINAL_ROBJ=(\d+)", output)
    pk_r = match(r"PEAK_ROBJ=(\d+)", output)
    if all(x -> x !== nothing, (bl_r, fn_r, pk_r))
        # robj baseline
        @test parse(Int, fn_r.captures[1]) == parse(Int, bl_r.captures[1])
        # add did grow it
        @test parse(Int, pk_r.captures[1]) == parse(Int, bl_r.captures[1]) + 1
    else
        @test false
    end

    # Stage authored exactly once across all 50 cycles.
    mo = match(r"ROOT_OPENS=(\d+)", output)
    @test mo !== nothing && parse(Int, mo.captures[1]) == 1
end
