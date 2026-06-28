using Test

# ---------------------------------------------------------------------------
# M3.1 — OmniPBR material authoring + material:binding (★ validate-first).
#
# THE load-bearing feasibility spike for the M3 (Materials) milestone: proves an
# OmniPBR `UsdShade Material` (authored via `usda_omnipbr_material` under a
# `looks_scope_usda` `/World/Looks` scope) and bound to a mesh via `OV.bind_material!`
# renders real PBR shading in our ovrtx build — a metallic sphere reads METALLIC (a
# concentrated bright specular highlight over a dark body), DISTINCT from the same
# sphere shaded as a flat grey diffuse `displayColor`.
#
# VALIDATED CONSTRAINT (M3.1, recorded in the report + source docstrings): an OmniPBR
# Material must be PRE-AUTHORED into the stage at open-time (composed into /World/Looks).
# A Material added to the OPEN stage at runtime via `OV.add_usd_reference!` is a SILENT
# NO-OP for `material:binding` in our ovrtx build (verified: no visual effect regardless
# of timing).  `OV.bind_material!` works at runtime, but only on a pre-authored material.
# So this test pre-authors the material, then binds it at runtime.
#
# Subprocess-isolated (carb signals + renderer live only in a child process); the body
# is `test/m3_material_prog.jl` (a standalone .jl so the multi-line USDA needs no
# escaping).  It renders the sphere UNBOUND (diffuse baseline) then RUNTIME-bound
# (metallic) and prints the comparison metrics asserted below.
#
# RED (pre-impl): `usda_omnipbr_material` / `material_prim_path` / `looks_scope_usda` /
#   `OV.bind_material!` are undefined → the subprocess throws → exitcode != 0.
# ---------------------------------------------------------------------------

const _M31_MATERIAL_PROG = read(joinpath(@__DIR__, "m3_material_prog.jl"), String)

@testset "M3.1 OmniPBR material + material:binding (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M31_MATERIAL_PROG; timeout = 900)
    @info "M3.1 material subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_MATERIAL")

    # Material authored at the expected /World/Looks/Mat_<id> path.
    @test contains(output, "MATPATH=/World/Looks/Mat_")

    # Metallic render non-black.
    mnb = match(r"METALLIC_STATS nonblack=(\d+)", output)
    if mnb !== nothing
        @test parse(Int, mnb.captures[1]) > 1000
    else
        @test false   # METALLIC_STATS line missing
    end

    # Metallic differs substantially from the diffuse render (material:binding took).
    mad = match(r"MEANABSDIFF=([0-9.eE+\-]+)", output)
    if mad !== nothing
        @test parse(Float64, mad.captures[1]) > 0.03
    else
        @test false   # MEANABSDIFF line missing
    end

    # Metallic specular signature: a much higher luminance contrast than diffuse.
    mcr = match(r"CONTRAST_RATIO=([0-9.eE+\-]+)", output)
    if mcr !== nothing
        @test parse(Float64, mcr.captures[1]) > 3.0
    else
        @test false   # CONTRAST_RATIO line missing
    end
end

# ---------------------------------------------------------------------------
# M3.2 — `material=` escape hatch + `color`→base composition + authoring trigger.
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
                   OmniverseMakie.usda_mesh(pts, fcs, nrm, (1f0, 0f0, 0f0)))
    @test !occursin("primvars:displayColor",
                    OmniverseMakie.usda_mesh(pts, fcs, nrm, nothing))
end

const _M32_COMPOSE_PROG = read(joinpath(@__DIR__, "m3_material_compose_prog.jl"), String)

@testset "M3.2 materialized mesh renders metallic via colorbuffer (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M32_COMPOSE_PROG; timeout = 900)
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
