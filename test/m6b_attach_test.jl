using Test

# M6.B Task 5 — attach_picking! attachable picking interaction (subprocess, GLMakie).
#
# SCOPE (user decision: "ship core now, defer live viewport outline"): the live viewport
# presents via the HdrColor path, but a selection-outline Screen is LdrColor-only (no HdrColor
# AOV) — neither present path can show it.  So `interactive_display(selection_outline=true)` is
# REFUSED for now and the in-viewport outline is deferred.  The DATA flow is the deliverable and
# works fully: click → pick_hit → `selected[]` Observable + `on_hit` callback.
#
# DRIVEN SYNCHRONOUSLY via `_pick_at!` — the M5 GLFW background-event-thread race makes synthetic
# mouse events flaky, so we assert the wiring directly (as the M5 orbit test does), with the live
# render_tick detached so background ticks cannot race the renderer while we pick on the main task.
const _M6B_ATTACH_PROG = """
using OmniverseMakie, GLMakie
OM = OmniverseMakie; OM.activate!(warmup = 16)
fig = Figure(); ax = LScene(fig[1,1]; show_axis = false)
p = mesh!(ax, Rect3f(Point3f(-1), Vec3f(2)); color = :gray)

# 1. interactive_display(selection_outline=true) must REFUSE with an actionable error
#    (the deferred LdrColor live-present), BEFORE building any viewport.
threw = false; msg = ""
try
    OM.interactive_display(fig; size = (200,200), selection_outline = true)
catch e
    global threw = true
    global msg = sprint(showerror, e)
end
println("REFUSED_OUTLINE=", threw)
println("REFUSE_MENTIONS_LDR=", occursin("LdrColor", msg) && occursin("selection_outline=false", msg))

# 2. Default viewport (outline off) — a normal HdrColor Screen, byte-identical to before the kwarg.
session = OM.interactive_display(fig; size = (200,200))
# Detach the live render_tick so background auto-ticks cannot step the renderer while we pick on
# the main task (same determinism pattern as m5_orbit_test).
off(session.tick_listener)

# 3. Attach with outline=true → it DEGRADES (the viewport Screen has no selection_outline), warns
#    once, and falls back to outline=false; the data flow stays fully live.  on_hit records hits.
on_hits = Any[]
h = OM.attach_picking!(session; outline = true, on_hit = hit -> push!(on_hits, hit))
println("ATTACHED=", h !== nothing)
println("HAS_LISTENER=", h.listener !== nothing)
println("OUTLINE_DEGRADED=", h.outline == false)
println("SELECTED_INIT_NOTHING=", h.selected[] === nothing)

# 4. Drive the pick handler SYNCHRONOUSLY at the centre (bypassing the GLFW event thread).
OM._pick_at!(session, h, Vec2(100.0, 100.0))
sel = h.selected[]
println("SELECTED_NONNOTHING=", sel !== nothing)
println("SELECTED_IS_MESH=", sel !== nothing && sel.plot === p)
println("ON_HIT_FIRED=", length(on_hits) == 1 && on_hits[1] !== nothing && on_hits[1].plot === p)

# 5. A far corner → background → selected[] === nothing (and on_hit still fires, with nothing).
OM._pick_at!(session, h, Vec2(2.0, 2.0))
println("MISS_IS_NOTHING=", h.selected[] === nothing)
println("ON_HIT_FIRED_ON_MISS=", length(on_hits) == 2 && on_hits[2] === nothing)

# 6. detach removes the listener (+ clears any highlight).
OM.detach_picking!(h)
println("DETACHED=", h.listener === nothing)
OM.detach_picking!(h)                  # idempotent
println("DETACH_IDEMPOTENT=", h.listener === nothing)

OM.close(session)
println("OK_ATTACH")
"""

include("helpers.jl")
@testset "M6.B attach_picking! (subprocess, GLMakie)" begin
    exitcode, output = run_ovrtx_subprocess(_M6B_ATTACH_PROG; timeout = 500)
    @info "M6.B attach output" output
    @test exitcode == 0
    # interactive_display(selection_outline=true) refuses with the LdrColor follow-up message.
    @test contains(output, "REFUSED_OUTLINE=true")
    @test contains(output, "REFUSE_MENTIONS_LDR=true")
    # attach + degrade guard.
    @test contains(output, "ATTACHED=true")
    @test contains(output, "HAS_LISTENER=true")
    @test contains(output, "OUTLINE_DEGRADED=true")
    @test contains(output, "SELECTED_INIT_NOTHING=true")
    # The data flow — the deliverable.
    @test contains(output, "SELECTED_IS_MESH=true")
    @test contains(output, "ON_HIT_FIRED=true")
    # Miss + detach.
    @test contains(output, "MISS_IS_NOTHING=true")
    @test contains(output, "ON_HIT_FIRED_ON_MISS=true")
    @test contains(output, "DETACHED=true")
    @test contains(output, "DETACH_IDEMPOTENT=true")
    @test contains(output, "OK_ATTACH")
end
