# replace_scene! — hybrid embedded viewport (subprocess, GLMakie).
#
# Proves the RPRMakie-style hybrid end to end: in a displayed GLMakie figure with an LScene
# (3D, left) beside an Axis (2D diagnostic, right), replace_scene!(lscene) replaces ONLY the
# LScene with a live ovrtx render. Decisive occlusion + isolation check via a pure-GL baseline:
#   • the LEFT (embedded) region CHANGES vs the pre-replace GL baseline — ovrtx now draws there;
#   • the RIGHT (2D diagnostic) region is ~UNCHANGED — the other axis stays GLMakie;
#   • the ovrtx present buffer is non-black and RESPONDS to orbiting the LScene camera;
#   • close() frees the ovrtx Screen but leaves the host GLMakie window open, and is idempotent.

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

    # The live embedded loop responds to orbiting the LScene camera.
    m_mov = match(r"ORBIT_MOVED=([-\d.]+)", out)
    @test m_mov !== nothing && parse(Float64, m_mov.captures[1]) > 2.0

    # Teardown contract: host window open, ovrtx Screen closed, idempotent.
    @test contains(out, "PARENT_OPEN_AFTER_CLOSE=true")
    @test contains(out, "SCREEN_CLOSED=true")
end
