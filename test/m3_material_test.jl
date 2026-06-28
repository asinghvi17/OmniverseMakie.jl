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
