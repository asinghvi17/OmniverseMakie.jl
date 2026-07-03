using Test
include(joinpath(@__DIR__, "..", "helpers.jl"))

# ---------------------------------------------------------------------------
# Integration capstone (formerly m1_mesh_render_test.jl): a real Makie Scene with a
# mesh renders through ovrtx via Makie.colorbuffer.
#
# Subprocess (carb signals + renderer live only in a child process):
#   fig = Figure(); ax = LScene(fig[1,1]); mesh!(ax, Rect3f(...); color=:red)
#   img = Makie.colorbuffer(ax.scene)   # lazy setup → author root → insert! → RT2
#     assert backend registered, eltype == RGBA{N0f8}, size ≥ 300², non-black > 1000,
#     red-dominant
# (The PNG-save half moved out — save/record round-trips live in save_record_test.jl.)
#
# Notes:
#   - ax.scene is a NON-root child scene, so Makie's colorbuffer(fig) crops it via
#     get_sub_picture using the LScene viewport (root coords) — our Screen renders
#     at ROOT size so the crop is in-bounds and returns the LScene sub-image.
#   - warmup=48 is plenty for RT2 to show the red cube.
# ---------------------------------------------------------------------------

const _M15_MESH_PROG = """
using OmniverseMakie, ColorTypes, FixedPointNumbers   # FixedPointNumbers: N0f8

OmniverseMakie.activate!()

# activate! installs the backend (formerly its own subprocess test).
@assert Makie.current_backend() === OmniverseMakie "backend not registered after activate!()"
println("BACKEND_REGISTERED=true")

# Real Makie scene: Figure + LScene + a red unit cube.
fig = Figure()
ax  = LScene(fig[1, 1])
mesh!(ax, Rect3f(Point3f(0), Vec3f(1)); color = :red)

# Analyse an image (function scope avoids the top-level soft-scope `for` trap).
lum(c) = Float32(red(c)) + Float32(green(c)) + Float32(blue(c))
function analyze(img)
    H, W = size(img)
    nonblack = 0; sr = 0.0; sg = 0.0; sb = 0.0
    litrow = 0.0; litcol = 0.0
    for h in 1:H, w in 1:W
        c = img[h, w]
        r = Float32(red(c)); g = Float32(green(c)); b = Float32(blue(c))
        if r + g + b > 0.05f0
            nonblack += 1; sr += r; sg += g; sb += b
            litrow += h; litcol += w
        end
    end
    n  = max(nonblack, 1)
    tq = count(h -> any(w -> lum(img[h, w]) > 0.05f0, 1:W), 1:(H ÷ 4))
    bq = count(h -> any(w -> lum(img[h, w]) > 0.05f0, 1:W), (3H ÷ 4):H)
    return (H = H, W = W, nonblack = nonblack,
            mr = sr / n, mg = sg / n, mb = sb / n,
            cen = nonblack == 0 ? (-1.0, -1.0) : (litrow / n, litcol / n),
            topq = tq, botq = bq,
            corners = (lum(img[1, 1]), lum(img[1, W]), lum(img[H, 1]), lum(img[H, W])))
end

# --- Integration render through Makie.colorbuffer (lazy USD setup → ovrtx RT2) ---
img = Makie.colorbuffer(ax.scene; warmup = 48)
st  = analyze(img)
println("ELTYPE=", eltype(img))
println("SIZE=", (st.H, st.W))
println("NONBLACK=", st.nonblack)
println("MEAN_RGB=", (st.mr, st.mg, st.mb))
println("LIT_CENTROID=", st.cen, " of ", (st.H, st.W))
println("TOPQUARTER_LITROWS=", st.topq, " BOTQUARTER_LITROWS=", st.botq)
println("CORNERS=", st.corners)

@assert eltype(img) == RGBA{N0f8} "eltype is \$(eltype(img)) (expected RGBA{N0f8})"
@assert st.H >= 300 && st.W >= 300 "image too small: \$((st.H, st.W))"
@assert st.nonblack > 1000 "mesh rendered (near) black: nonblack=\$(st.nonblack)"
@assert st.mr > st.mg && st.mr > st.mb "not red-dominant: mean_rgb=\$((st.mr, st.mg, st.mb))"

println("OK_MESH")
"""

@testset "mesh → colorbuffer (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M15_MESH_PROG; timeout = 900, retries = 2, ready_marker = "ELTYPE=")
    @info "mesh_render subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_MESH")
    @test contains(output, "BACKEND_REGISTERED=true")

    # eltype
    @test contains(output, "ELTYPE=RGBA{N0f8}")

    # size ≥ 300²
    ms = match(r"SIZE=\((\d+), (\d+)\)", output)
    if ms !== nothing
        @test parse(Int, ms.captures[1]) >= 300
        @test parse(Int, ms.captures[2]) >= 300
    else
        @test false   # SIZE line missing
    end

    # non-black > 1000
    mnb = match(r"NONBLACK=(\d+)", output)
    if mnb !== nothing
        @test parse(Int, mnb.captures[1]) > 1000
    else
        @test false   # NONBLACK line missing
    end

    # red-dominant: mean red > mean green and mean blue
    mrgb = match(r"MEAN_RGB=\(([0-9.eE+\-]+), ([0-9.eE+\-]+), ([0-9.eE+\-]+)\)", output)
    if mrgb !== nothing
        mr = parse(Float64, mrgb.captures[1])
        mg = parse(Float64, mrgb.captures[2])
        mb = parse(Float64, mrgb.captures[3])
        @test mr > mg
        @test mr > mb
    else
        @test false   # MEAN_RGB line missing
    end

end
