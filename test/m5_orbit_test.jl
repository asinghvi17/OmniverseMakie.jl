using Test

# M5 Task 5 — orbit-forwarding test.
#
# Proves that forwarding input to the DISPLAY scene (glscene, campixel!) drives
# the ovrtx Camera3D via the input_listeners chain wired in interactive_display.
#
# Mechanism:
#   glscene.events.<f>[] = v
#   → input_listener (on glscene.events.<f>) fires synchronously
#   → cam_scene.events.<f>[] = v
#   → Camera3D's own listener fires synchronously
#   → cam.eyeposition[] updates (for scroll/zoom)
#
# Controller-verified: scroll moved eye [3,3,3]→[3.99,3.99,3.99].
# Left-drag: see note below on mouseposition handling.
#
# Deterministic: tick_listener is detached before event injection so auto-ticks
# cannot reset the camera state between assertions.
#
# NOTE on mouseposition during drag test:
#   GLMakie's render loop updates glscene.events.mouseposition from GLFW callbacks
#   on a background thread (the real cursor position).  These concurrent writes race
#   with our test injections, so by the time Camera3D's drag-start handler reads
#   cam_scene.events.mouseposition it may have been overwritten.
#
#   cam_scene.events is the FIGURE's shared events object (different from
#   glscene.events), which GLMakie's GLFW callbacks do NOT touch.  We therefore
#   set cam_scene.events.mouseposition[] directly for drag setup and movement, while
#   forwarding mousebutton press/release via glscene to exercise the forwarding path.
#   mouseposition forwarding is separately proved by capturing the value synchronously
#   inside an on() listener before the background thread can overwrite it.

const _M5_ORBIT_PROG = """
using OmniverseMakie, GLMakie, ColorTypes
OM = OmniverseMakie
OM.activate!(warmup = 16)

# Small 3-D scene: LScene + a mesh (gives Camera3D something to look at).
fig = Figure()
ax  = LScene(fig[1,1]; show_axis = false)
mesh!(ax, Sphere(Point3f(0), 1f0); color = :steelblue)

# Open the interactive display at small size.
session = OM.interactive_display(fig; size = (300, 300))
cam = Makie.cameracontrols(session.cam_scene)

# Detach the live render_tick listener so auto-ticks cannot interfere with the
# deterministic event sequence below (same pattern as camera_loop_test).
off(session.tick_listener)

# Record the initial eye position.
eye0 = copy(cam.eyeposition[])
println("EYE0=", eye0)

# ── SCROLL test: zoom in (negative y scroll → eye moves toward target) ────────
# Set cursor inside the glscene window, then inject a scroll event via glscene.
# The input_listener forwards it to cam_scene → Camera3D zooms.
session.glscene.events.mouseposition[] = (150.0, 150.0)
session.glscene.events.scroll[]        = (0.0, -3.0)
eye_scroll = copy(cam.eyeposition[])
println("EYE_SCROLL=", eye_scroll)
@assert eye_scroll != eye0 "scroll did not change eyeposition: still \$(eye0)"

# ── MOUSEPOSITION FORWARDING proof ─────────────────────────────────────────────
# Capture the cam_scene mouseposition value INSIDE an on() listener, which fires
# synchronously before the GLMakie background thread can overwrite it.
mp_captured = Ref{Tuple{Float64,Float64}}((0.0, 0.0))
_temp = on(session.cam_scene.events.mouseposition) do mp
    mp_captured[] = (Float64(mp[1]), Float64(mp[2]))
end
session.glscene.events.mouseposition[] = (42.0, 42.0)
off(_temp)
println("MP_FORWARDED=", mp_captured[])
@assert mp_captured[] == (42.0, 42.0) "mouseposition not forwarded to cam_scene"

# ── DRAG test: left-button press + move + release → orbit ─────────────────────
# GLMakie's GLFW callbacks race-update glscene.events.mouseposition from a background
# thread.  We set cam_scene.events.mouseposition directly (GLMakie does not update
# this) to ensure is_mouseinside(cam_scene) returns true.
# mousebutton press/release is forwarded via glscene, exercising the forwarding path.
#
# LIMITATION: because mouseposition is set directly on cam_scene here, this drag
# @assert does NOT by itself prove the glscene→cam_scene mouseposition forwarding path.
# That path is proven by the MOUSEPOSITION FORWARDING proof above (synchronous capture)
# and by interactive LIVE verification.  This subtest proves mousebutton forwarding and
# that Camera3D orbits on a drag — do not read it as end-to-end mouseposition coverage.
eye_before_drag = copy(cam.eyeposition[])
session.cam_scene.events.mouseposition[] = (150.0, 150.0)   # inside viewport
session.glscene.events.mousebutton[] =
    Makie.MouseButtonEvent(Makie.Mouse.left, Makie.Mouse.press)
session.cam_scene.events.mouseposition[] = (180.0, 160.0)   # drag 30 px right
session.glscene.events.mousebutton[] =
    Makie.MouseButtonEvent(Makie.Mouse.left, Makie.Mouse.release)
eye_drag = copy(cam.eyeposition[])
println("EYE_DRAG=", eye_drag)
@assert eye_drag != eye_before_drag "drag did not change eyeposition: still \$(eye_before_drag)"

println("OK_ORBIT")
"""

include("helpers.jl")
@testset "M5 orbit forwarding: glscene events drive Camera3D (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M5_ORBIT_PROG; timeout = 600, retries = 2, ready_marker = "OK_ORBIT")
    @info "M5 orbit output" output
    @test exitcode == 0
    @test contains(output, "OK_ORBIT")
    # Verify scroll moved the eye
    m = match(r"EYE_SCROLL=\[([0-9.]+),", output)
    @test m !== nothing && parse(Float64, m.captures[1]) > 3.0
    # Verify mouseposition forwarding was captured
    @test contains(output, "MP_FORWARDED=(42.0, 42.0)")
end
