using Test
include(joinpath(@__DIR__, "..", "helpers.jl"))

# ---------------------------------------------------------------------------
# USD subscene grouping (`def Scope` hierarchy mirroring the Makie tree).
#
# A `def Scope "Scene_<objectid>"` is authored per Makie subscene, nested to
# mirror `scene.children`, so each plot's USD reference lives at
# `/World/Scene_<id>/plot_<id>` instead of flat `/World/plot_<id>`.  Render
# stays pixel-equivalent (organizational grouping; plots keep their composed
# `:model_f32c`).
#
#   Step 1 (pure): scene_scopes_usda on a constructed tree → nested `def Scope`
#       blocks + a scene2scope map (objectid → full scope path).  No renderer.
#   Step 2 (render, subprocess): a Figure with a real LScene subscene renders
#       non-black with ROOT_OPENS==1, the LScene scene maps to a NESTED
#       scope, and the mesh's USD reference path is nested under that scope.
# ---------------------------------------------------------------------------

# ===========================================================================
# Step 1 — scene_scopes_usda is PURE (no renderer): runs in-process.
# ===========================================================================
@testset "M2.3 scene_scopes_usda nested scope hierarchy (pure)" begin
    # root → c1, c2 (siblings); c1 → gc (grandchild)
    root = Scene()
    c1   = Scene(root)
    c2   = Scene(root)
    gc   = Scene(c1)

    usda, s2s = OmniverseMakie.scene_scopes_usda(root)

    # --- scene2scope: root → /World (no Scene_ scope); subscenes nested ---
    @test s2s[objectid(root)] == "/World"
    @test s2s[objectid(c1)]   == "/World/Scene_$(objectid(c1))"
    @test s2s[objectid(c2)]   == "/World/Scene_$(objectid(c2))"
    @test s2s[objectid(gc)]   == "/World/Scene_$(objectid(c1))/Scene_$(objectid(gc))"
    @test length(s2s) == 4

    # --- USDA fragment: a def Scope per subscene (NOT the root) ---
    @test occursin("def Scope \"Scene_$(objectid(c1))\"", usda)
    @test occursin("def Scope \"Scene_$(objectid(c2))\"", usda)
    @test occursin("def Scope \"Scene_$(objectid(gc))\"", usda)
    # root is /World, no Scene_ scope
    @test !occursin("Scene_$(objectid(root))", usda)

    # --- nesting via indentation: top scopes at 4 spaces, grandchild at 8 ---
    # (present at 4-space, absent at 8-space ⇒ exactly depth 1)
    @test occursin("    def Scope \"Scene_$(objectid(c1))\"", usda)
    @test !occursin("        def Scope \"Scene_$(objectid(c1))\"", usda)
    @test occursin("    def Scope \"Scene_$(objectid(c2))\"", usda)
    @test !occursin("        def Scope \"Scene_$(objectid(c2))\"", usda)
    # depth 2 (inside c1)
    @test occursin("        def Scope \"Scene_$(objectid(gc))\"", usda)

    # --- the grandchild scope is INSIDE c1's block (between c1's braces) ---
    i_c1 = findfirst("def Scope \"Scene_$(objectid(c1))\"", usda)
    i_c2 = findfirst("def Scope \"Scene_$(objectid(c2))\"", usda)
    i_gc = findfirst("def Scope \"Scene_$(objectid(gc))\"", usda)
    @test first(i_c1) < first(i_gc) < first(i_c2)  # gc nested before sibling c2

    # --- scope layers must OMIT upAxis (only the root governs) ---
    @test !occursin("upAxis", usda)
    # --- scopes carry NO transform (organizational only) ---
    @test !occursin("xformOp:transform", usda)

    # --- a childless root → /World only, empty fragment ---
    solo = Scene()
    usda0, s2s0 = OmniverseMakie.scene_scopes_usda(solo)
    @test s2s0[objectid(solo)] == "/World"
    @test length(s2s0) == 1
    @test strip(usda0) == ""
end

# ===========================================================================
# Step 2 — render a real LScene subscene; assert nested paths + non-black:
# the figure root → /World, the LScene 3-D child → a Scene_ scope, the mesh
# reference nested under it; render non-black with opens==1.
# ===========================================================================
const _M23_SUBSCENE_PROG = """
using OmniverseMakie, ColorTypes, FixedPointNumbers
const OM = OmniverseMakie

OM.activate!(warmup = 32)

# Figure ROOT is a 2-D PixelCamera scene; the LScene's scene is its 3-D child.
fig = Figure()
ax  = LScene(fig[1, 1])
m   = mesh!(ax, Rect3f(Point3f(0), Vec3f(1)); color = :red)
root_scene = fig.scene
lscene     = ax.scene

screen = OM.Screen(root_scene)
Makie.push_screen!(root_scene, screen)

nonblack(img) = count(c -> (Float32(red(c)) + Float32(green(c)) + Float32(blue(c))) > 0.05f0, img)

img = Makie.colorbuffer(screen)
println("NONBLACK=\$(nonblack(img))")
@assert nonblack(img) > 500 "nested-subscene render (near) black: \$(nonblack(img))"

# --- root scene → /World, LScene 3-D scene → a NESTED def Scope ---
@assert haskey(screen.scene2scope, objectid(root_scene)) "root scene missing from scene2scope"
@assert screen.scene2scope[objectid(root_scene)] == "/World" "root scene not mapped to /World"
@assert haskey(screen.scene2scope, objectid(lscene)) "LScene scene missing from scene2scope"
lscope = screen.scene2scope[objectid(lscene)]
println("LSCENE_SCOPE=\$(lscope)")
@assert startswith(lscope, "/World/Scene_") "LScene scene not under a nested def Scope: \$(lscope)"
@assert occursin("Scene_\$(objectid(lscene))", lscope) "LScene scope path missing its objectid"

# --- the mesh's USD reference is nested under the LScene scope ---
@assert haskey(screen.plot2robj, objectid(m)) "mesh not registered in plot2robj"
pp = screen.plot2robj[objectid(m)].prim_path
println("MESH_PRIM_PATH=\$(pp)")
@assert pp == "\$(lscope)/plot_\$(objectid(m))" "mesh not nested under its scene scope: \$(pp)"

# --- add_scene! returns the scope path (not a bare /World) ---
returned = OM.add_scene!(screen, lscene)
println("ADD_SCENE_RET=\$(returned)")
@assert returned == lscope "add_scene! did not return the nested scope path: \$(returned)"

opens = OM._ROOT_OPEN_COUNT[]
println("ROOT_OPENS=\$(opens)")
@assert opens == 1 "stage opened \$(opens)× (expected 1); subscene authoring must not re-open"

close(screen)
println("OK_SUBSCENE")
"""

@testset "M2.3 nested-subscene render + paths (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M23_SUBSCENE_PROG; timeout = 900, retries = 2, ready_marker = "NONBLACK=")
    @info "M2.3 subscene subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_SUBSCENE")

    # Non-black render.
    mnb = match(r"NONBLACK=(\d+)", output)
    @test mnb !== nothing && parse(Int, mnb.captures[1]) > 500

    # LScene scene mapped to a nested /World/Scene_<id> scope.
    msc = match(r"LSCENE_SCOPE=(\S+)", output)
    @test msc !== nothing && startswith(msc.captures[1], "/World/Scene_")

    # Mesh prim path nested under that scope.
    mpp = match(r"MESH_PRIM_PATH=(\S+)", output)
    @test mpp !== nothing && startswith(mpp.captures[1], "/World/Scene_")
    @test mpp !== nothing && occursin("/plot_", mpp.captures[1])

    # Stage opened exactly once.
    mo = match(r"ROOT_OPENS=(\d+)", output)
    @test mo !== nothing && parse(Int, mo.captures[1]) == 1
end
