using Test

# ---------------------------------------------------------------------------
# M3.4 — LIVE material edits on the PRE-AUTHORED material via the M2 diff path.
#
# A live `plot.color[]` / `plot.material[]` PARAM edit on a MATERIALIZED mesh re-writes
# the pre-authored OmniPBR shader inputs (`/World/Looks/Mat_<id>/Shader`) on the OPEN
# stage — the M2 diff path — WITHOUT re-authoring the root (`ROOT_OPENS == 1`):
#
#   color[]=:blue                      → exactly one :scaled_color push → exactly one
#                                        `diffuse_color_constant` shader write → BLUE-dominant
#   material[]=Attributes(metallic=0,  → exactly one :material push → metallic_constant +
#              roughness=0.9)            reflection_roughness_constant shader writes →
#                                        glossy specular collapses to MATTE (contrast drops)
#
# This is the feasible, valuable M3.4 deliverable (M3.1 finding): a true material SWAP
# to a brand-new material is NOT runtime-feasible (needs a root re-author), so M3.4 is
# LIVE EDITS of the pre-authored material — NOT a `material:binding` re-bind.
#
# Subprocess-isolated (carb signals + renderer live only in a child process); the body
# is `test/m3_material_live_prog.jl`.
#
# RED (pre-impl): `OvrtxRObj.material_shader` field + `OV.write_shader_input!` +
#   `_SHADER_WRITE_OBSERVER` + the materialized push routing + `:material` in
#   `consumed_inputs` are all absent → the subprocess errors / the blue + matte +
#   shader-write assertions fail.
# GREEN (M3.4): the materialized `:scaled_color`/`:material` routes write the shader
#   inputs live; the render turns blue then matte; ROOT_OPENS stays 1.
# ---------------------------------------------------------------------------

const _M34_MATERIAL_LIVE_PROG = read(joinpath(@__DIR__, "m3_material_live_prog.jl"), String)

@testset "M3.4 live material edits via the M2 diff path (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M34_MATERIAL_LIVE_PROG; timeout = 900)
    @info "M3.4 material-live subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_MATERIAL_LIVE")

    # The shader prim was recorded on the OvrtxRObj.
    @test contains(output, "MATERIAL_SHADER=/World/Looks/Mat_")

    # Exactly one minimal push per edit, routed to the right compute output.
    @test contains(output, "PUSH_COLOR=Dict(:scaled_color => 1)")
    @test contains(output, "PUSH_MATERIAL=Dict(:material => 1)")

    # Exactly the changed shader inputs were written (one write per changed param).
    @test contains(output, "SHADER_COLOR=[\"diffuse_color_constant\"]")
    @test contains(output, "SHADER_MATERIAL=[\"metallic_constant\", \"reflection_roughness_constant\"]")

    # Visible change: red build → blue after the color edit (live base re-write).
    ma = match(r"RGB_A=\((.*?)\)", output)
    mb = match(r"RGB_B=\((.*?)\)", output)
    if ma !== nothing && mb !== nothing
        a = parse.(Float64, split(ma.captures[1], ", "))
        b = parse.(Float64, split(mb.captures[1], ", "))
        @test a[1] > a[3]    # build red-dominant
        @test b[3] > b[1]    # after the color edit blue-dominant
    else
        @test false
    end

    # glossy metal → matte dielectric: a substantial pixel change AND the matte surface
    # shows its pure diffuse base (metallic→0 removes the white specular wash, so the blue
    # base is markedly MORE saturated — a robust, scene-independent signature of the edit).
    mch = match(r"CHANGED_B_vs_C=(\d+)", output)
    @test mch !== nothing && parse(Int, mch.captures[1]) > 1000
    msb = match(r"SAT_B=([0-9.eE+\-]+)", output)
    msc = match(r"SAT_C=([0-9.eE+\-]+)", output)
    if msb !== nothing && msc !== nothing
        @test parse(Float64, msc.captures[1]) > parse(Float64, msb.captures[1])
    else
        @test false
    end

    # The stage was authored EXACTLY ONCE across all three renders (live writes only).
    mo = match(r"ROOT_OPENS=(\d+)", output)
    @test mo !== nothing && parse(Int, mo.captures[1]) == 1
end
