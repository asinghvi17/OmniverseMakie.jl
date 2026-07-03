using Test
include(joinpath(@__DIR__, "..", "helpers.jl"))

# OmniPBR materials (formerly m3_material_test.jl).
#
# VALIDATED CONSTRAINT (recorded in the source docstrings): an OmniPBR Material must be
# PRE-AUTHORED into the stage at open-time (composed into /World/Looks).  A Material added
# to the OPEN stage at runtime via `OV.add_usd_reference!` is a SILENT NO-OP for
# `material:binding` in our ovrtx build.  `OV.bind_material!` works at runtime, but only on
# a pre-authored material.  The compose subprocess below exercises exactly that shape
# through the full Screen/colorbuffer pipeline (the original M3.1 hand-authored-stage
# feasibility spike asserted the same metallic-vs-diffuse signature and was retired as
# redundant with it).

# ---------------------------------------------------------------------------
# `material=` escape hatch + `color`→base composition + authoring trigger.
#
# Unit (parent process, NO render): `is_materialized` + `material_inputs_from` compose
# `color` + the `material=` NamedTuple into OmniPBR shader-input names, applying the
# `base_color` precedence; PLUS the plain-color regression guard at the DATA level —
# the robust primary guard: a plain `mesh!(…; color=:red)` authors NO material, keeps
# `primvars:displayColor`, and gets NO `material:binding`.
#
# Integration (subprocess): the materialized mesh renders METALLIC through the full
# Screen/colorbuffer pipeline (pre-author → bind → render); the plain mesh renders the
# M1 displayColor path red-dominant (the displayColor pipeline is unbroken).
# ---------------------------------------------------------------------------

@testset "M3.2 is_materialized + material_inputs_from (unit)" begin
    fig  = Figure()
    ax   = LScene(fig[1, 1])
    geom = Rect3f(Point3f(0), Vec3f(1))
    m    = mesh!(ax, geom; color = :red, material = (; metallic = 0.0, roughness = 0.2))
    mp   = mesh!(ax, geom; color = :red)

    # The authoring trigger fires only for the materialized plot.
    @test OmniverseMakie.is_materialized(m)  == true
    @test OmniverseMakie.is_materialized(mp) == false

    # color + material= compose into ONE OmniPBR-input dict (mapped names, raw scalars).
    @test OmniverseMakie.material_inputs_from(m) == Dict(
        "diffuse_color_constant"        => (1, 0, 0),
        "metallic_constant"             => 0,
        "reflection_roughness_constant" => 0.2)

    # Precedence: material=(; base_color=…) OVERRIDES color.
    mo = mesh!(ax, geom; color = :red, material = (; base_color = :blue))
    @test OmniverseMakie.material_inputs_from(mo)["diffuse_color_constant"] == (0, 0, 1)

    # --- Regression guard (DATA-level, robust primary): the plain plot authors NO
    #     material; the materialized plot's material IS pre-authored into /World/Looks. ---
    looks = OmniverseMakie.materialized_looks_usda(ax.scene)
    @test occursin("Mat_$(objectid(m))", looks)        # materialized → pre-authored material
    @test !occursin("Mat_$(objectid(mp))", looks)      # plain        → NO material authored

    # usda_mesh: the plain path STILL emits primvars:displayColor (byte-unchanged — see
    # the M1 emitters); the materialized path (nothing sentinel) OMITS it.
    pts = [(0f0, 0f0, 0f0)]; fcs = [[0]]; nrm = [(0f0, 0f0, 1f0)]
    @test occursin("primvars:displayColor",
                   OmniverseMakie.usda_mesh(pts, OmniverseMakie._flat_faces(fcs)..., nrm, (1f0, 0f0, 0f0)))
    @test !occursin("primvars:displayColor",
                    OmniverseMakie.usda_mesh(pts, OmniverseMakie._flat_faces(fcs)..., nrm, nothing))
end

const _M32_COMPOSE_PROG = read(joinpath(@__DIR__, "material_compose_prog.jl"), String)

@testset "M3.2 materialized mesh renders metallic via colorbuffer (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M32_COMPOSE_PROG; timeout = 900, retries = 2, ready_marker = "ELTYPE=")
    @info "M3.2 compose subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_COMPOSE")

    # Both the plain (displayColor) and materialized renders are non-black.
    mP = match(r"PLAIN_STATS nonblack=(\d+)", output)
    mM = match(r"METALLIC_STATS nonblack=(\d+)", output)
    @test mP !== nothing && parse(Int, mP.captures[1]) > 1000
    @test mM !== nothing && parse(Int, mM.captures[1]) > 1000

    # The materialized render differs substantially from the plain render (binding took).
    mad = match(r"MEANABSDIFF=([0-9.eE+\-]+)", output)
    @test mad !== nothing && parse(Float64, mad.captures[1]) > 0.02

    # Metallic specular signature: a much higher contrast than the flat diffuse sphere.
    mcr = match(r"CONTRAST_RATIO=([0-9.eE+\-]+)", output)
    @test mcr !== nothing && parse(Float64, mcr.captures[1]) > 1.5
end

# ---------------------------------------------------------------------------
# M3.5 — primitive coverage (MeshScatter / Surface / Lines / Scatter / LineSegments)
# + the carried emissive/opacity BOOL fix.
#
# Unit (parent process, NO render): the `material=` escape hatch is materialized on the
# OTHER primitive types, the drop-displayColor plumbing OMITS `primvars:displayColor`
# for a materialized primitive (and KEEPS it byte-unchanged for a plain one — the
# regression guard), and `emissive`/`opacity` now author OmniPBR's `bool` enable gates
# (NOT int/float) plus an explicit `emissive_intensity`.
#
# Integration (subprocess): a materialized meshscatter renders METALLIC (distinct from
# diffuse), a materialized surface renders + differs from plain, and an emissive lines
# renders RED — the bool enable_emission + emissive_intensity fix in action.
# ---------------------------------------------------------------------------

@testset "M3.5 primitive materialization + drop-color plumbing (unit)" begin
    fig = Figure(); ax = LScene(fig[1, 1])
    pts = [Point3f(0), Point3f(1, 0, 0), Point3f(1, 1, 0)]

    # is_materialized fires on every primitive type with `material=` (and stays false
    # for a plain one).
    ls  = lines!(ax, pts; material = (; emissive = (1, 0, 0)))
    lsp = lines!(ax, pts; color = :magenta)
    sc  = scatter!(ax, pts; markersize = 0.2, material = (; metallic = 1.0))
    scp = scatter!(ax, pts; markersize = 0.2, color = :cyan)
    ms  = meshscatter!(ax, pts; markersize = 0.2, material = (; metallic = 1.0))
    sf  = surface!(ax, -1:0.5:1, -1:0.5:1, (x, y) -> x * y; material = (; metallic = 1.0))
    for p in (ls, sc, ms, sf)
        @test OmniverseMakie.is_materialized(p) == true
    end
    @test OmniverseMakie.is_materialized(lsp) == false
    @test OmniverseMakie.is_materialized(scp) == false

    # --- drop-displayColor plumbing: materialized (nothing) OMITS, plain KEEPS ----
    I4 = [1.0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
    cpts = [(0f0, 0f0, 0f0), (1f0, 0f0, 0f0)]
    # BasisCurves (Lines / LineSegments)
    @test occursin("primvars:displayColor",
                   OmniverseMakie._usda_basiscurves(cpts, [2], 1f0, (1f0, 0f0, 0f0), "constant"; model = I4))
    @test !occursin("primvars:displayColor",
                    OmniverseMakie._usda_basiscurves(cpts, [2], 1f0, nothing, "constant"; model = I4))
    # PointInstancer (Scatter / MeshScatter) — instancer-level colour
    @test occursin("primvars:displayColor",
                   OmniverseMakie._usda_pointinstancer(cpts, [(1f0,1f0,1f0),(1f0,1f0,1f0)], nothing,
                       [(1f0,0f0,0f0),(0f0,1f0,0f0)], OmniverseMakie._sphere_proto_body(nothing); model = I4))
    @test !occursin("primvars:displayColor",
                    OmniverseMakie._usda_pointinstancer(cpts, [(1f0,1f0,1f0),(1f0,1f0,1f0)], nothing,
                        nothing, OmniverseMakie._sphere_proto_body(nothing); model = I4))
    # Prototype bodies — a materialized instancer's prototype OMITS displayColor
    @test occursin("primvars:displayColor", OmniverseMakie._sphere_proto_body((1f0, 0f0, 0f0)))
    @test !occursin("primvars:displayColor", OmniverseMakie._sphere_proto_body(nothing))
    @test occursin("primvars:displayColor",
                   OmniverseMakie._mesh_proto_body(cpts, OmniverseMakie._flat_faces([[0, 1]])..., [(0f0,0f0,1f0),(0f0,0f0,1f0)], (1f0,0f0,0f0)))
    @test !occursin("primvars:displayColor",
                    OmniverseMakie._mesh_proto_body(cpts, OmniverseMakie._flat_faces([[0, 1]])..., [(0f0,0f0,1f0),(0f0,0f0,1f0)], nothing))
end

@testset "M3.5 emissive/opacity BOOL enable gates + emissive_intensity (unit)" begin
    fig = Figure(); ax = LScene(fig[1, 1])
    le  = lines!(ax, [Point3f(0), Point3f(1, 0, 0)]; material = (; emissive = (1, 0, 0)))
    inp = OmniverseMakie.material_inputs_from(le)

    # enable_emission is a Bool `true` (NOT the int 1) → the OmniPBR `bool` MDL gate binds.
    @test inp["enable_emission"] === true
    @test inp["emissive_color"] == (1, 0, 0)
    # emissive_intensity is authored as a Float so emission is VISIBLE.
    @test haskey(inp, "emissive_intensity")
    @test inp["emissive_intensity"] isa AbstractFloat
    @test inp["emissive_intensity"] > 0

    # opacity also gets a Bool enable gate.
    lo  = mesh!(ax, Rect3f(Point3f(0), Vec3f(1)); material = (; opacity = 0.4))
    inpo = OmniverseMakie.material_inputs_from(lo)
    @test inpo["enable_opacity"] === true
    @test inpo["opacity_constant"] == 0.4

    # The emitter renders the Bool gate as USD `bool` (= 1), NOT `float … = 1.0`, and
    # `emissive_intensity` as a float.  (The `Bool` branch precedes the `Real`/float branch.)
    usda = OmniverseMakie.usda_omnipbr_material("Mat_x", inp)
    @test occursin("bool inputs:enable_emission = 1", usda)
    @test !occursin("float inputs:enable_emission", usda)
    @test occursin("float inputs:emissive_intensity", usda)
end

const _M35_PRIM_MAT_PROG = read(joinpath(@__DIR__, "primitives_material_prog.jl"), String)

@testset "M3.5 materials on MeshScatter/Surface/Lines render (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M35_PRIM_MAT_PROG; timeout = 1500, retries = 2, ready_marker = "MESHSCATTER_MEANABSDIFF=")
    @info "M3.5 primitives-material subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_PRIMITIVES_MATERIAL")

    # MeshScatter: materialized instances read metallic (distinct from diffuse).
    md = match(r"MESHSCATTER_DIFFUSE nonblack=(\d+)", output)
    mm = match(r"MESHSCATTER_METALLIC nonblack=(\d+)", output)
    @test md !== nothing && parse(Int, md.captures[1]) > 300
    @test mm !== nothing && parse(Int, mm.captures[1]) > 300
    mad = match(r"MESHSCATTER_MEANABSDIFF=([0-9.eE+\-]+)", output)
    @test mad !== nothing && parse(Float64, mad.captures[1]) > 0.02
    mcr = match(r"MESHSCATTER_CONTRAST_RATIO=([0-9.eE+\-]+)", output)
    @test mcr !== nothing && parse(Float64, mcr.captures[1]) > 1.3

    # Surface: materialized renders + differs substantially from plain (bind took).
    sp = match(r"SURFACE_PLAIN nonblack=(\d+)", output)
    sm = match(r"SURFACE_METALLIC nonblack=(\d+)", output)
    @test sp !== nothing && parse(Int, sp.captures[1]) > 2000
    @test sm !== nothing && parse(Int, sm.captures[1]) > 2000
    smad = match(r"SURFACE_MEANABSDIFF=([0-9.eE+\-]+)", output)
    @test smad !== nothing && parse(Float64, smad.captures[1]) > 0.02

    # Lines: emissive RED — many red-dominant pixels, far more than a plain white curve
    # (the bool enable_emission + emissive_intensity fix renders emission).
    le = match(r"LINES_EMISSIVE redpix=(\d+)", output)
    lw = match(r"LINES_WHITE redpix=(\d+)", output)
    @test le !== nothing && lw !== nothing
    if le !== nothing && lw !== nothing
        rcE = parse(Int, le.captures[1]); rcW = parse(Int, lw.captures[1])
        @test rcE > 500
        @test rcE > 20 * (rcW + 1)
    end
end
