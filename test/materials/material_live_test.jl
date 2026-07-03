using Test
include(joinpath(@__DIR__, "..", "helpers.jl"))

# ---------------------------------------------------------------------------
# LIVE material edits on the PRE-AUTHORED material via the live diff path
# (formerly m3_material_live_test.jl + m3_meshscatter variant, one subprocess).
#
# A live `plot.color[]` / `plot.material[]` PARAM edit on a MATERIALIZED mesh re-writes
# the pre-authored OmniPBR shader inputs (`/World/Looks/Mat_<id>/Shader`) on the OPEN
# stage — the live diff path — WITHOUT re-authoring the root (`ROOT_OPENS == 1`):
#
#   color[]=:blue                      → exactly one :scaled_color push → exactly one
#                                        `diffuse_color_constant` shader write → BLUE-dominant
#   material[]=Attributes(metallic=0,  → exactly one :material push → metallic_constant +
#              roughness=0.9)            reflection_roughness_constant shader writes →
#                                        glossy specular collapses to MATTE (contrast drops)
#
# A true material SWAP to a brand-new material is NOT runtime-feasible (needs a root
# re-author), so "live material" means LIVE EDITS of the pre-authored material — NOT a
# `material:binding` re-bind.
#
# Phase 2 of the same prog repeats the material-param edit on a materialized
# *MeshScatter* (the final-review fix: `:material ∈ consumed_inputs(::MeshScatter)`;
# before it the edit was a silent no-op).  MS_-prefixed markers, per-phase ROOT_OPENS
# delta.  Body: `material_live_prog.jl`.
# ---------------------------------------------------------------------------

const _MATERIAL_LIVE_PROG = read(joinpath(@__DIR__, "material_live_prog.jl"), String)

@testset "live material edits: mesh + meshscatter (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_MATERIAL_LIVE_PROG; timeout = 900, retries = 2, ready_marker = "MATERIAL_SHADER=")
    @info "material-live subprocess output" output
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
    mo = match(r"(?<!MS_)ROOT_OPENS=(\d+)", output)
    @test mo !== nothing && parse(Int, mo.captures[1]) == 1

    # --- Phase 2: the same material-param edit on a materialized MeshScatter (the
    #     `:material ∈ consumed_inputs(::MeshScatter)` fix — was a silent no-op). ---
    @test contains(output, "MS_MATERIAL_SHADER=/World/Looks/Mat_")
    @test contains(output, "MS_PUSH_MATERIAL=Dict(:material => 1)")
    @test contains(output, "MS_SHADER_MATERIAL=[\"metallic_constant\", \"reflection_roughness_constant\"]")

    # The render CHANGED (glossy metal → matte dielectric) and the matte surface shows its
    # purer diffuse base (blue base MORE saturated once the specular wash is gone).
    mch = match(r"MS_CHANGED_A_vs_B=(\d+)", output)
    @test mch !== nothing && parse(Int, mch.captures[1]) > 300
    msa = match(r"MS_SAT_A=([0-9.eE+\-]+)", output)
    msb = match(r"MS_SAT_B=([0-9.eE+\-]+)", output)
    if msa !== nothing && msb !== nothing
        @test parse(Float64, msb.captures[1]) > parse(Float64, msa.captures[1])
    else
        @test false
    end

    # Phase 2's stage authored exactly once too (per-phase delta — 2nd Screen, same process).
    mo2 = match(r"MS_ROOT_OPENS=(\d+)", output)
    @test mo2 !== nothing && parse(Int, mo2.captures[1]) == 1
end
