using Test
import OmniverseMakie   # bind the module name so this file runs standalone too
using OmniverseMakie: Figure, LScene, mesh!, Rect2f, RGBf, Vec2f
include(joinpath(@__DIR__, "..", "helpers.jl"))
# GeometryBasics is the package's own dep, reached via the
# `OmniverseMakie.` qualifier (the minimal test env declares only
# CEnum/LibOVRTX/Test; every other test file accesses deps through
# OmniverseMakie, which is `using`-ed globally by runtests.jl).  `Rect2f`
# and the Makie surface come from OmniverseMakie's verbatim Makie re-export;
# only `GeometryBasics.uv_normal_mesh` needs the module binding.
const GeometryBasics = OmniverseMakie.GeometryBasics

# ---------------------------------------------------------------------------
# Image textures (`color = img` → `diffuse_texture` + the `st` UV primvar)
# plus the `*_texture` escape-hatch maps.  Mesh only.
#
# Unit (parent process, NO render): `material_inputs_from` maps an image
# `color` to a `diffuse_texture` asset (a temp PNG written with PNGFiles) +
# `project_uvw=false`; `usda_mesh` emits the `texCoord2f[] primvars:st` block
# ONLY when texcoords are given; `*_texture` keys resolve to the right
# OmniPBR inputs with paths used as-is; explicit `base_color_texture`
# overrides the image.
#
# Integration (subprocess): a quad textured with a 2-colour red/blue checker
# renders BOTH colours sampled across the surface (not a flat average)
# through the full Screen/colorbuffer pipeline.  Body: `texture_prog.jl`.
# ---------------------------------------------------------------------------

@testset "image texture mapping + st primvar (unit)" begin
    fig  = Figure()
    ax   = LScene(fig[1, 1])
    img  = [RGBf(1, 0, 0) RGBf(0, 0, 1); RGBf(0, 0, 1) RGBf(1, 0, 0)]
    quad = GeometryBasics.uv_normal_mesh(Rect2f(0, 0, 1, 1))
    mimg = mesh!(ax, quad; color = img)

    # Image `color` → a `diffuse_texture` asset PATH (an on-disk temp PNG)
    # + project_uvw.
    inp = OmniverseMakie.material_inputs_from(mimg)
    @test haskey(inp, "diffuse_texture")
    @test inp["diffuse_texture"] isa AbstractString
    @test isfile(inp["diffuse_texture"])            # OPEN-time write, persists
    @test isabspath(inp["diffuse_texture"])         # absolute (no root anchor)
    @test inp["project_uvw"] === false              # st UVs, not triplanar
    @test OmniverseMakie._needs_texcoords(mimg) == true

    # The pre-authored material fragment carries the asset + bool inputs
    # (proven form).
    frag = OmniverseMakie.usda_omnipbr_material("Mat_x", inp)
    @test occursin("asset inputs:diffuse_texture = @", frag)
    @test occursin("bool inputs:project_uvw = 0", frag)

    # `_texture_asset_for`: a String path is validated to EXIST and returned
    # absolutized (a dangling/typo'd path renders silently untextured, so a
    # MISSING file throws rather than being emitted verbatim); an image is
    # written to a fresh temp PNG.  The third arg is the per-input key
    # (`:color` = the image-`color` path) that keeps the temp PNG unique per
    # input per plot.  Real files back the string-path cases below.
    texdir  = mktempdir()
    already = joinpath(texdir, "already.png"); touch(already)
    @test OmniverseMakie._texture_asset_for(already, mimg, :color) == abspath(already)
    @test_throws ArgumentError OmniverseMakie._texture_asset_for(joinpath(texdir, "missing.png"), mimg, :color)
    written = OmniverseMakie._texture_asset_for(img, mimg, :color)
    @test isfile(written) && endswith(written, ".png")

    # The written file carries the 8-byte PNG signature.
    bytes = read(written)
    @test bytes[1:8] == UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]

    # `*_texture` escape-hatch keys → the right OmniPBR inputs, existing paths
    # resolved (absolutized) as-is.
    tb = joinpath(texdir, "b.png"); tn = joinpath(texdir, "n.png")
    tr = joinpath(texdir, "r.png"); tm = joinpath(texdir, "m.png")
    foreach(touch, (tb, tn, tr, tm))
    mtex = mesh!(ax, quad; material = (; base_color_texture = tb,
                                        normal_texture    = tn,
                                        roughness_texture = tr,
                                        metallic_texture  = tm))
    it = OmniverseMakie.material_inputs_from(mtex)
    @test it["diffuse_texture"]             == abspath(tb)
    @test it["normalmap_texture"]           == abspath(tn)
    @test it["reflectionroughness_texture"] == abspath(tr)
    @test it["metallic_texture"]            == abspath(tm)

    # Precedence: an explicit `base_color_texture` OVERRIDES the image
    # `color` texture.
    twin = joinpath(texdir, "win.png"); touch(twin)
    mover = mesh!(ax, quad; color = img, material = (; base_color_texture = twin))
    @test OmniverseMakie.material_inputs_from(mover)["diffuse_texture"] == abspath(twin)

    # Reap the temp files (the touched string-path files + the written PNG).
    rm(texdir; recursive = true, force = true)
    isfile(written) && rm(written)

    # --- Regression guard: `usda_mesh` OMITS `primvars:st` by default; it
    #     appears ONLY with texcoords. ---
    pts = [(0f0, 0f0, 0f0), (1f0, 0f0, 0f0), (1f0, 1f0, 0f0)]
    fcs = [[0, 1, 2]]
    nrm = [(0f0, 0f0, 1f0) for _ in 1:3]
    tc  = [Vec2f(0, 0), Vec2f(1, 0), Vec2f(1, 1)]
    @test !occursin("primvars:st", OmniverseMakie.usda_mesh(pts, OmniverseMakie._flat_faces(fcs)..., nrm, nothing))
    with_st = OmniverseMakie.usda_mesh(pts, OmniverseMakie._flat_faces(fcs)..., nrm, nothing; texcoords = tc)
    @test occursin("texCoord2f[] primvars:st", with_st)
    @test occursin("interpolation = \"vertex\"", with_st)
end

const _M33_TEXTURE_PROG = read(joinpath(@__DIR__, "texture_prog.jl"), String)

@testset "image-textured mesh renders the checker via colorbuffer (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M33_TEXTURE_PROG; timeout = 900, retries = 2, ready_marker = "ELTYPE=")
    @info "M3.3 texture subprocess output" output
    @test exitcode == 0
    @test contains(output, "OK_TEXTURE")

    # The diffuse_texture asset was pre-authored into /World/Looks and
    # exists on disk.
    @test contains(output, "HAS_DIFFUSE_TEXTURE=true")

    # The render is non-black and BOTH checker colours appear (texture
    # SAMPLED, not averaged).
    mnb = match(r"NONBLACK=(\d+)", output)
    @test mnb !== nothing && parse(Int, mnb.captures[1]) > 1000
    mr = match(r"RED_DOMINANT=(\d+)", output)
    mb = match(r"BLUE_DOMINANT=(\d+)", output)
    @test mr !== nothing && parse(Int, mr.captures[1]) > 200
    @test mb !== nothing && parse(Int, mb.captures[1]) > 200
end
