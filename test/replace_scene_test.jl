# replace_scene! — hybrid embedded viewport (subprocess, GLMakie).
#
# Proves the RPRMakie-style hybrid end to end: in a displayed GLMakie figure with an LScene
# (3D, left) beside an Axis (2D diagnostic, right), replace_scene!(lscene) replaces ONLY the
# LScene with a live ovrtx render. Decisive occlusion + isolation check via a pure-GL baseline:
#   • the LEFT (embedded) region CHANGES vs the pre-replace GL baseline — ovrtx now draws there;
#   • the RIGHT (2D diagnostic) region is ~UNCHANGED — the other axis stays GLMakie;
#   • the ovrtx present buffer is non-black and RESPONDS to orbiting the LScene camera;
#   • close() frees the ovrtx Screen but leaves the host GLMakie window open, and is idempotent.
#
# Second prog: a usdplot INSIDE the replaced LScene.  The childless USDPlot reaches GLMakie's
# atomic branch (insert! → draw_atomic), so without the ext's no-op draw_atomic the
# `display(fig)` PRECONDITION of replace_scene! MethodErrors before any RTX work starts.
# Asserts: display succeeds, plain GL draws NOTHING for the usdplot, and the replace_scene!
# overlay then shows the USD asset via ovrtx.

using Test
include("helpers.jl")

const _REPLACE_PROG = """
using OmniverseMakie, GLMakie, ColorTypes, FixedPointNumbers
const OM = OmniverseMakie
using OmniverseMakie: OV

lum(c) = Float32(c.r) + Float32(c.g) + Float32(c.b)
# Mean PER-CHANNEL abs difference over a column range [c0,c1] of two [H,W] images (per-channel so
# it is not blind to equal-luminance swaps like red<->blue).
function region_diff(a, b, c0, c1)
    H, W = size(a); s = 0.0; n = 0
    for h in 1:H, w in c0:min(c1, W)
        p = a[h, w]; q = b[h, w]
        s += abs(Float32(p.r) - Float32(q.r)) + abs(Float32(p.g) - Float32(q.g)) +
             abs(Float32(p.b) - Float32(q.b)); n += 1
    end
    return n == 0 ? 0.0 : s / n
end
function region_nonblack(img, c0, c1)
    H, W = size(img); n = 0
    for h in 1:H, w in c0:min(c1, W)
        lum(img[h, w]) > 0.04 && (n += 1)
    end
    return n
end
# Lit-pixel centroid (row, col) of the ovrtx present buffer.
function pbuf_centroid(buf)
    H, W = size(buf); sr = 0.0; sc = 0.0; n = 0
    for h in 1:H, w in 1:W
        if lum(buf[h, w]) > 0.04; sr += h; sc += w; n += 1; end
    end
    return n == 0 ? (-1.0, -1.0) : (sr / n, sc / n)
end

GLMakie.activate!()
fig = Figure(size = (700, 350))
ls  = LScene(fig[1, 1])
mesh!(ls, Rect3f(Point3f(-0.5), Vec3f(1)); color = :orange)
ax  = Axis(fig[1, 2]); lines!(ax, 0:0.1:10, sin.(0:0.1:10); color = :blue, linewidth = 3)

glscr = GLMakie.Screen(; visible = false)
display(glscr, fig.scene)
base = copy(GLMakie.colorbuffer(glscr))          # pure-GL baseline (lays out the viewport too)
W = size(base, 2); half = W ÷ 2

session = OM.replace_scene!(ls)
println("SESSION_TYPE=", nameof(typeof(session)))
println("AUTHORED_PLOTS=", length(session.screen.plot2robj))

# Drive a few host frames — each colorbuffer fires render_tick → the embedded tick renders + blits.
for _ in 1:6; GLMakie.colorbuffer(glscr); end
shot = copy(GLMakie.colorbuffer(glscr))
c1 = pbuf_centroid(session.present_buf)        # centroid of the ovrtx present buffer

println("PRESENT_NONBLACK=", count(c -> lum(c) > 0.04, session.present_buf))
# Embedded (left) region changed a lot vs GL baseline; 2D (right) region barely changed.
println("LEFT_DIFF=",  region_diff(base, shot, 1, half))
println("RIGHT_DIFF=", region_diff(base, shot, half + 1, W))
println("LEFT_NONBLACK=",  region_nonblack(shot, 1, half))
println("RIGHT_NONBLACK=", region_nonblack(shot, half + 1, W))

# Orbit the LScene camera → the live ovrtx render responds (present buffer moves/changes).
cam = Makie.cameracontrols(ls.scene)
Makie.update_cam!(ls.scene, Vec3f(4, 2, 3), Vec3f(0, 0, 0), Vec3f(0, 0, 1))
for _ in 1:6; GLMakie.colorbuffer(glscr); end
c2 = pbuf_centroid(session.present_buf)
println("ORBIT_MOVED=", abs(c1[1] - c2[1]) + abs(c1[2] - c2[2]))

# ★ Regression guard (the notify(::Computed) no-op bug): assert the COMPOSITED output — a second
# GLMakie.colorbuffer AFTER the orbit — actually changed in the embedded (left) region, not just
# present_buf. `shot` was the pre-orbit composite; if the GL texture never re-uploaded (the bug),
# shot2's left region is byte-frozen and COMPOSITE_ORBIT_DIFF ≈ 0.
shot2 = copy(GLMakie.colorbuffer(glscr))
println("COMPOSITE_ORBIT_DIFF=", region_diff(shot, shot2, 1, half))
println("COMPOSITE_ORBIT_RIGHT_DIFF=", region_diff(shot, shot2, half + 1, W))

# Teardown: host window stays open, embedded ovrtx Screen closes; idempotent.
close(session)
println("PARENT_OPEN_AFTER_CLOSE=", isopen(glscr))
println("SCREEN_CLOSED=", !OM.isopen(session.screen))
close(session)                                # second close = safe no-op
println("OK_REPLACE_SCENE")
"""

@testset "replace_scene!: hybrid embedded viewport (subprocess)" begin
    _, out = run_ovrtx_subprocess(_REPLACE_PROG; timeout = 600, retries = 4,
                                  ready_marker = "OK_REPLACE_SCENE")
    contains(out, "OK_REPLACE_SCENE") || @info "replace_scene! output" out
    @test contains(out, "OK_REPLACE_SCENE")
    @test contains(out, "SESSION_TYPE=EmbeddedSession")

    # ovrtx authored the LScene's mesh + rendered non-black content.
    m_auth = match(r"AUTHORED_PLOTS=(\d+)", out)
    @test m_auth !== nothing && parse(Int, m_auth.captures[1]) >= 1
    m_pnb = match(r"PRESENT_NONBLACK=(\d+)", out)
    @test m_pnb !== nothing && parse(Int, m_pnb.captures[1]) > 300

    # Occlusion + isolation: the embedded (left) region CHANGED vs the GL baseline; the 2D
    # diagnostic (right) region barely changed (stayed GLMakie).
    m_ld = match(r"LEFT_DIFF=([-\d.]+)",  out)
    m_rd = match(r"RIGHT_DIFF=([-\d.]+)", out)
    @test m_ld !== nothing && m_rd !== nothing
    if m_ld !== nothing && m_rd !== nothing
        left_diff  = parse(Float64, m_ld.captures[1])
        right_diff = parse(Float64, m_rd.captures[1])
        @test left_diff  > 0.05           # ovrtx replaced the GL 3D there
        @test right_diff < left_diff / 3  # the diagnostic axis is essentially untouched
    end
    # Both halves still show content (the hybrid composited, not one blank).
    m_rnb = match(r"RIGHT_NONBLACK=(\d+)", out)
    @test m_rnb !== nothing && parse(Int, m_rnb.captures[1]) > 300

    # The live embedded loop responds to orbiting the LScene camera (present buffer moved).
    m_mov = match(r"ORBIT_MOVED=([-\d.]+)", out)
    @test m_mov !== nothing && parse(Float64, m_mov.captures[1]) > 2.0

    # ★ ...AND that response reaches the COMPOSITED frame — a second colorbuffer after the orbit
    # differs in the embedded (left) region, well above the ~unchanged 2D (right) region. This is
    # the regression guard for the notify(::Computed)-is-a-no-op freeze (the old code never
    # re-uploaded the GL texture, so the composite stayed frozen while present_buf advanced).
    m_cod  = match(r"COMPOSITE_ORBIT_DIFF=([-\d.]+)", out)
    m_codr = match(r"COMPOSITE_ORBIT_RIGHT_DIFF=([-\d.]+)", out)
    @test m_cod !== nothing && parse(Float64, m_cod.captures[1]) > 0.02
    if m_cod !== nothing && m_codr !== nothing
        @test parse(Float64, m_cod.captures[1]) > parse(Float64, m_codr.captures[1]) * 3
    end

    # Teardown contract: host window open, ovrtx Screen closed, idempotent.
    @test contains(out, "PARENT_OPEN_AFTER_CLOSE=true")
    @test contains(out, "SCREEN_CLOSED=true")
end

# ---------------------------------------------------------------------------------------------
# usdplot inside the replaced scene
# ---------------------------------------------------------------------------------------------

# Red doubleSided X-Y quad, defaultPrim "Model" — same shape as the usdplot_test fixture.  Built
# parent-side and embedded via `repr(...)` so no nested `\"\"\"` collides with the prog literal.
const _REPLACE_QUAD_USDA = """#usda 1.0
(
    defaultPrim = "Model"
)
def Xform "Model"
{
    def Mesh "Geo"
    {
        uniform bool doubleSided = true
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
        color3f[] primvars:displayColor = [(1, 0, 0)] (
            interpolation = "constant"
        )
    }
}
"""

const _REPLACE_USDPLOT_PROG = """
using OmniverseMakie, GLMakie, ColorTypes, FixedPointNumbers
const OM = OmniverseMakie

lum(c) = Float32(c.r) + Float32(c.g) + Float32(c.b)
# Red-dominant pixel count over a column range [c0,c1] of an [H,W] image.
function region_red(img, c0, c1)
    H, W = size(img); n = 0
    for h in 1:H, w in c0:min(c1, W)
        c = img[h, w]
        Float32(c.r) > Float32(c.g) + 0.15f0 && Float32(c.r) > Float32(c.b) + 0.15f0 && (n += 1)
    end
    return n
end
function region_nonblack(img, c0, c1)
    H, W = size(img); n = 0
    for h in 1:H, w in c0:min(c1, W)
        lum(img[h, w]) > 0.04 && (n += 1)
    end
    return n
end

const QUAD = tempname() * ".usda"
write(QUAD, $(repr(_REPLACE_QUAD_USDA)))

GLMakie.activate!()
fig = Figure(size = (700, 350))
ls  = LScene(fig[1, 1]; show_axis = false)
p   = usdplot!(ls, QUAD)                     # the ONLY plot in the LScene (pure regression)
ax  = Axis(fig[1, 2]); lines!(ax, 0:0.1:10, sin.(0:0.1:10); color = :blue, linewidth = 3)

# The replace_scene! PRECONDITION: display the figure in GLMakie.  Without the ext's no-op
# draw_atomic(::USDPlot) this MethodErrors in insertplots! before any RTX work starts.
glscr = GLMakie.Screen(; visible = false)
display(glscr, fig.scene)
println("DISPLAY_OK=true")

# Plain GL draws NOTHING for the usdplot (the documented no-op) — no red in the left half.
base = copy(GLMakie.colorbuffer(glscr))
W = size(base, 2); half = W ÷ 2
println("GL_LEFT_RED=", region_red(base, 1, half))

session = OM.replace_scene!(ls)
println("AUTHORED_PLOTS=", length(session.screen.plot2robj))
for _ in 1:6; GLMakie.colorbuffer(glscr); end
shot = copy(GLMakie.colorbuffer(glscr))
# The ovrtx overlay now shows the referenced USD asset (red quad) in the left half; the 2D
# diagnostic on the right still renders.
println("RTX_LEFT_RED=", region_red(shot, 1, half))
println("RIGHT_NONBLACK=", region_nonblack(shot, half + 1, W))

close(session)
println("PARENT_OPEN_AFTER_CLOSE=", isopen(glscr))
GLMakie.colorbuffer(glscr)                   # the usdplot stays in the GL scene — still no crash
println("POST_CLOSE_RENDER_OK=true")
println("OK_REPLACE_USDPLOT")
"""

@testset "replace_scene!: usdplot in the replaced scene (subprocess)" begin
    _, out = run_ovrtx_subprocess(_REPLACE_USDPLOT_PROG; timeout = 600, retries = 4,
                                  ready_marker = "OK_REPLACE_USDPLOT")
    contains(out, "OK_REPLACE_USDPLOT") || @info "replace_scene!+usdplot output" out
    @test contains(out, "OK_REPLACE_USDPLOT")

    # GLMakie ingested the childless USDPlot (display precondition) and drew nothing for it.
    @test contains(out, "DISPLAY_OK=true")
    m_glr = match(r"GL_LEFT_RED=(\d+)", out)
    @test m_glr !== nothing && parse(Int, m_glr.captures[1]) == 0

    # replace_scene! authored the usdplot and the overlay shows the red USD quad.
    m_auth = match(r"AUTHORED_PLOTS=(\d+)", out)
    @test m_auth !== nothing && parse(Int, m_auth.captures[1]) >= 1
    m_red = match(r"RTX_LEFT_RED=(\d+)", out)
    @test m_red !== nothing && parse(Int, m_red.captures[1]) > 300
    m_rnb = match(r"RIGHT_NONBLACK=(\d+)", out)
    @test m_rnb !== nothing && parse(Int, m_rnb.captures[1]) > 300

    # Teardown: host window survives, and GL keeps tolerating the (no-op) usdplot.
    @test contains(out, "PARENT_OPEN_AFTER_CLOSE=true")
    @test contains(out, "POST_CLOSE_RENDER_OK=true")
end
