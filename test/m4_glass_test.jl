using Test
# GeometryBasics is the package's own dep, reached via the qualifier (the minimal test env
# declares only CEnum/LibOVRTX/Test; uv_normal_mesh/Tesselation/Sphere need the module binding).
const GeometryBasics = OmniverseMakie.GeometryBasics

# ---------------------------------------------------------------------------
# M4 follow-up — TRUE refractive glass via OmniGlass.
#
# `material=(; glass=true, …)` authors an OmniGlass `UsdShade Material` (instead of the OmniPBR
# `opacity` alpha cut-out), mapping glass_color/ior/roughness/thin_walled.
#
# Unit (parent, NO render): `usda_glass_material` emits the OmniGlass shader + typed inputs;
# `_material_kind` / `_glass_inputs_from` read a real `material=(; glass=true, …)` plot.
#
# Integration (subprocess, ★): a glass sphere in front of a red wall TRANSMITS the red through
# its centre (refraction), proving the OmniGlass material binds + renders.  Body `m4_glass_prog.jl`.
# ---------------------------------------------------------------------------

@testset "M4 glass material authoring (unit)" begin
    frag = OmniverseMakie.usda_glass_material("Mat_x",
        Dict("glass_ior" => 1.5f0, "glass_color" => (0.8f0, 0.9f0, 1.0f0), "frosting_roughness" => 0.1f0))
    @test occursin("@OmniGlass.mdl@", frag)
    @test occursin("info:mdl:sourceAsset:subIdentifier = \"OmniGlass\"", frag)
    @test occursin("float inputs:glass_ior = 1.5", frag)
    @test occursin("color3f inputs:glass_color = (0.8, 0.9, 1.0)", frag)
    # OmniPBR authoring is byte-unchanged (still routes through the shared helper)
    pbr = OmniverseMakie.usda_omnipbr_material("Mat_y", Dict("metallic_constant" => 1.0f0))
    @test occursin("@OmniPBR.mdl@", pbr)
    @test occursin("info:mdl:sourceAsset:subIdentifier = \"OmniPBR\"", pbr)

    fig = Figure(); ax = LScene(fig[1, 1])
    sph = GeometryBasics.uv_normal_mesh(GeometryBasics.Tesselation(GeometryBasics.Sphere(Point3f(0), 1.0f0), 16))
    p = mesh!(ax, sph; color = RGBf(0.7, 0.85, 1.0), material = (; glass = true, ior = 1.45, roughness = 0.05))
    @test OmniverseMakie._material_kind(p) == :glass
    gi = OmniverseMakie._glass_inputs_from(p)
    @test gi["glass_ior"] == 1.45
    @test gi["frosting_roughness"] == 0.05
    @test gi["glass_color"] == (0.7f0, 0.85f0, 1.0f0)

    # default ior when unspecified; non-glass material → :omnipbr
    p_def = mesh!(ax, sph; material = (; glass = true))
    @test OmniverseMakie._glass_inputs_from(p_def)["glass_ior"] == 1.491f0
    p_pbr = mesh!(ax, sph; material = (; metallic = 1.0f0))
    @test OmniverseMakie._material_kind(p_pbr) == :omnipbr
end

const _M4_GLASS_PROG = read(joinpath(@__DIR__, "m4_glass_prog.jl"), String)

@testset "M4 refractive glass transmits the backdrop via colorbuffer (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M4_GLASS_PROG; timeout = 600)
    @info "M4 glass subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_GLASS")
    mbr = match(r"CENTER_BRIGHT=(\d+)", output)
    @test mbr !== nothing && parse(Int, mbr.captures[1]) > 3
    mdk = match(r"CENTER_DARK=(\d+)", output)
    @test mdk !== nothing && parse(Int, mdk.captures[1]) > 50
end

const _M4_GLASS_LIVE_PROG = read(joinpath(@__DIR__, "m4_glass_live_prog.jl"), String)

@testset "M4 LIVE glass material edit routes to OmniGlass inputs (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M4_GLASS_LIVE_PROG; timeout = 600)
    @info "M4 glass-live subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_GLASS_LIVE")
    mmad = match(r"MEANABSDIFF=([0-9.eE+\-]+)", output)
    @test mmad !== nothing && parse(Float64, mmad.captures[1]) > 0.01
end
