using Test

# ---------------------------------------------------------------------------
# Review Task B6 — USD string hygiene + texture-name collision.
#
# User-supplied strings enter authored USDA in two shapes that were previously
# unvalidated/unescaped:
#   • IDENTIFIERS — a VDB `field` name (`def OpenVDBAsset "$(field)"`, volume.jl)
#     and camera-path segments (`_validate_camera_path` only checked depth, so
#     "/World/My Camera" passed and authored a broken prim).
#   • `@asset@` PATHS — texture inputs + the MDL source (materials.jl) and the
#     `.vdb`/`.nvdb` `filePath` (volume.jl); a literal `@` in the path broke the
#     reference.
# Two new guards in usd.jl fix this: `_usd_identifier` (assert
# [A-Za-z_][A-Za-z0-9_]*, error naming the offender) and `_usd_asset_path`
# (wrap `@…@`, or `@@@…@@@` when the path contains `@`; error on an embedded `@@@`).
#
# Separately, `_texture_asset_for` named its temp PNG by `objectid(plot)`, but
# `_merge_material_input!` passed `plot = nothing` — so EVERY image `*_texture`
# landed at `tex_<objectid(nothing)>.png` (a process constant): two plots
# overwrote each other. The fix threads the real `plot` AND suffixes the filename
# with the input key, so files are unique per input per plot.
#
# Regression anchor (GOLDEN, captured from the PRE-change code): the volume USDA
# for a CLEAN field name must stay byte-identical.
#
# PURE (no GPU): USDA string emission + path/identifier validation + texture-file
# naming are all directly assertable.
# ---------------------------------------------------------------------------

import OmniverseMakie as OM

# Byte-for-byte volume USDA for a clean field ("density"), captured from the
# PRE-change `_vdb_volume_usda`. Regression anchor: the field-validation +
# asset-path wrapping must be identity-preserving on clean input.
const _GOLDEN_VOLUME_USDA = """#usda 1.0
(
    defaultPrim = "Volume"
)
def Volume "Volume" (
    prepend apiSchemas = ["MaterialBindingAPI"]
)
{
    rel field:density = </Volume/density>
    rel material:binding = </Volume/Material>

    def OpenVDBAsset "density"
    {
        token fieldName = "density"
        token fieldDataType = "float"
        asset filePath = @/data/torus.vdb@
    }

    def Material "Material"
    {
        token outputs:nvindex:volume.connect = </Volume/Material/VolumeShader.outputs:volume>

        def Colormap "Colormap"
        {
            custom token colormapSource = "rgbaPoints"
            custom float2 domain = (0.0, 1.0)
            uniform token domainBoundaryMode = "clampToTransparent"
            custom token outputs:colormap
            custom float4[] rgbaPoints = [(0.267004, 0.004874, 0.329415, 0.0), (0.282656, 0.100196, 0.42216, 0.06666667), (0.277134, 0.185228, 0.489898, 0.13333334), (0.253935, 0.265254, 0.529983, 0.2), (0.221989, 0.339161, 0.548752, 0.26666668), (0.190631, 0.407061, 0.556089, 0.33333334), (0.163625, 0.471133, 0.558148, 0.4), (0.139147, 0.533812, 0.555298, 0.46666667), (0.120565, 0.596422, 0.543611, 0.53333336), (0.134692, 0.658636, 0.517649, 0.6), (0.20803, 0.718701, 0.472873, 0.6666667), (0.327796, 0.77398, 0.40664, 0.73333335), (0.477504, 0.821444, 0.318195, 0.8), (0.647257, 0.8584, 0.209861, 0.8666667), (0.82494, 0.88472, 0.106217, 0.93333334), (0.993248, 0.906157, 0.143936, 1.0)]
            custom float[] xPoints = [0.0, 0.06666667, 0.13333334, 0.2, 0.26666668, 0.33333334, 0.4, 0.46666667, 0.53333336, 0.6, 0.6666667, 0.73333335, 0.8, 0.8666667, 0.93333334, 1.0]
        }

        def Shader "VolumeShader"
        {
            token inputs:colormap.connect = </Volume/Material/Colormap.outputs:colormap>
            token outputs:volume
        }
    }
}
"""

# Capture an `error()` and return its message, or "" if the call did NOT throw.
_errmsg(f) = try; f(); ""; catch e; e isa ErrorException ? e.msg : sprint(showerror, e); end

@testset "B6 _usd_identifier accept / reject (offender named)" begin
    # Accept legal USD identifiers; the returned value is a String equal to input.
    for good in ("density", "_private", "Camera", "torus_fog", "Mat_123", "A")
        @test OM._usd_identifier(good) == good
        @test OM._usd_identifier(good) isa String
    end

    # Reject space, dash, leading-digit, unicode, dot, and empty — each error
    # message must NAME the offending string and carry the `what` context.
    # Trailing/embedded newlines are rejected too: PCRE `$` matches BEFORE a final
    # newline, so "density\n" bypassed the guard until the end anchor became `\z`.
    for bad in ("my name", "my-name", "1name", "café", "naïve", "field.x", "",
                "density\n", "density\r\n", "\ndensity")
        msg = _errmsg(() -> OM._usd_identifier(bad; what = "volume `field` name"))
        @test !isempty(msg)                       # it threw
        @test occursin(bad, msg) || bad == ""     # names the offender (empty string is unprintable)
        @test occursin("volume `field` name", msg)   # names WHERE it came from
        @test occursin("valid USD identifier", msg)
    end
    # Empty string still errors (message names the context even if offender is "").
    @test !isempty(_errmsg(() -> OM._usd_identifier("")))
end

@testset "B6 _usd_asset_path wrap / escape / reject" begin
    # Clean path → plain `@path@` (byte-identical to the old hand-written form).
    @test OM._usd_asset_path("/data/torus.vdb") == "@/data/torus.vdb@"
    @test OM._usd_asset_path("OmniPBR.mdl") == "@OmniPBR.mdl@"

    # A path containing `@` → the USDA triple-delimiter form `@@@…@@@`.
    @test OM._usd_asset_path("/tmp/a@b/tex.png") == "@@@/tmp/a@b/tex.png@@@"
    @test OM._usd_asset_path("s3://bucket@host/x.vdb") == "@@@s3://bucket@host/x.vdb@@@"

    # A path containing the reserved `@@@` delimiter is unrepresentable → error
    # naming the offender + context.
    msg = _errmsg(() -> OM._usd_asset_path("a@@@b"; what = "VDB `filePath`"))
    @test occursin("a@@@b", msg)
    @test occursin("VDB `filePath`", msg)
    @test occursin("@@@", msg)
end

@testset "B6 volume USDA golden byte-identity + hostile field + `@` path" begin
    # CLEAN field: byte-identical to the pre-change emit (regression anchor).
    got = OM._vdb_volume_usda("/data/torus.vdb"; prim_path = "/World/Volume",
                              field = "density", field_dtype = "float",
                              colormap = :viridis, colorrange = (0.0, 1.0))
    @test got == _GOLDEN_VOLUME_USDA

    # A HOSTILE grid name (space) errors clearly, naming the offender — instead of
    # authoring a corrupt `def OpenVDBAsset "torus fog"` prim.
    msg = _errmsg(() -> OM._vdb_volume_usda("/data/x.vdb"; field = "torus fog"))
    @test occursin("torus fog", msg)
    @test occursin("field", msg)
    @test occursin("valid USD identifier", msg)
    # dash / leading-digit grid names likewise rejected.
    @test !isempty(_errmsg(() -> OM._vdb_volume_usda("/d/x.vdb"; field = "grid-1")))
    @test !isempty(_errmsg(() -> OM._vdb_volume_usda("/d/x.vdb"; field = "2density")))

    # A `.vdb` path containing `@` is ESCAPED (not broken) in the filePath ref.
    esc = OM._vdb_volume_usda("/tmp/scene@v2/torus.vdb"; field = "density")
    @test occursin("asset filePath = @@@/tmp/scene@v2/torus.vdb@@@", esc)
end

@testset "B6 material asset-path emit golden + `@` escaping" begin
    # OmniPBR: clean texture + MDL source emit the plain `@…@` form (unchanged).
    m = OM.usda_omnipbr_material("Mat_x", Dict{String,Any}("diffuse_texture" => "/abs/tex.png"))
    @test occursin("uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@", m)
    @test occursin("asset inputs:diffuse_texture = @/abs/tex.png@", m)

    # Glass: MDL source is OmniGlass.mdl (unchanged).
    g = OM.usda_glass_material("Mat_g", Dict{String,Any}("glass_ior" => 1.491f0))
    @test occursin("uniform asset info:mdl:sourceAsset = @OmniGlass.mdl@", g)

    # A texture path with a literal `@` is ESCAPED in the asset ref.
    m2 = OM.usda_omnipbr_material("Mat_y", Dict{String,Any}("diffuse_texture" => "/t/a@b.png"))
    @test occursin("asset inputs:diffuse_texture = @@@/t/a@b.png@@@", m2)
end

@testset "B6 camera_path segment validation" begin
    # Legal path accepted (returned as-is).
    @test OM._validate_camera_path("/World/Camera") == "/World/Camera"
    @test OM._validate_camera_path("/World/MyCam_2") == "/World/MyCam_2"

    # A space in the name has the right DEPTH but is not a legal identifier → clear error.
    msg = _errmsg(() -> OM._validate_camera_path("/World/My Camera"))
    @test occursin("My Camera", msg)
    @test occursin("valid USD identifier", msg)
    @test occursin("camera_path segment", msg)

    # Dash likewise rejected as an identifier.
    @test occursin("valid USD identifier", _errmsg(() -> OM._validate_camera_path("/World/Cam-1")))

    # Pre-existing DEPTH/shape checks still hold (these are the "shape" error, not
    # the identifier error).
    for badshape in ("/Foo/Bar", "/World/A/B", "World/Camera", "/World/")
        m = _errmsg(() -> OM._validate_camera_path(badshape))
        @test occursin("direct /World/<name> child", m)
    end
end

@testset "B6 texture temp-file naming is unique per input per plot (collision fix)" begin
    fig  = Figure()
    ax   = LScene(fig[1, 1])
    quad = OM.GeometryBasics.uv_normal_mesh(Rect2f(0, 0, 1, 1))
    img1 = RGBf[RGBf(1, 0, 0) RGBf(0, 1, 0); RGBf(0, 0, 1) RGBf(1, 1, 0)]
    img2 = RGBf[RGBf(0, 0, 0) RGBf(1, 1, 1); RGBf(1, 1, 1) RGBf(0, 0, 0)]

    # Two DIFFERENT plots, each with an image `base_color_texture` (even the SAME
    # image) → DIFFERENT on-disk paths (previously both → tex_<objectid(nothing)>.png).
    p1 = mesh!(ax, quad; material = (; base_color_texture = img1))
    p2 = mesh!(ax, quad; material = (; base_color_texture = img1))
    t1 = OM.material_inputs_from(p1)["diffuse_texture"]
    t2 = OM.material_inputs_from(p2)["diffuse_texture"]
    @test t1 != t2
    @test isfile(t1) && isfile(t2)
    @test isabspath(t1) && isabspath(t2)
    @test endswith(t1, ".png") && endswith(t2, ".png")

    # ONE plot with TWO image inputs (base_color_texture + normal_texture) →
    # two DISTINCT paths (keyed by the input name, not just the plot).
    p3 = mesh!(ax, quad; material = (; base_color_texture = img1, normal_texture = img2))
    i3 = OM.material_inputs_from(p3)
    @test i3["diffuse_texture"] != i3["normalmap_texture"]
    @test isfile(i3["diffuse_texture"]) && isfile(i3["normalmap_texture"])

    # The image-`color` path (key `:color`) is also unique per plot and distinct
    # from a `*_texture` input on the same plot.
    p4 = mesh!(ax, quad; color = img1, material = (; normal_texture = img2))
    i4 = OM.material_inputs_from(p4)
    @test i4["diffuse_texture"] != i4["normalmap_texture"]

    # Direct `_texture_asset_for`: same plot + same key → same file (idempotent
    # per input); different key OR different plot → different file.
    @test OM._texture_asset_for(img1, p1, :color) == OM._texture_asset_for(img1, p1, :color)
    @test OM._texture_asset_for(img1, p1, :color) != OM._texture_asset_for(img1, p1, :normal_texture)
    @test OM._texture_asset_for(img1, p1, :color) != OM._texture_asset_for(img1, p2, :color)
    # A String path is still returned AS-IS regardless of plot/key (no temp write).
    @test OM._texture_asset_for("/abs/x.png", p1, :color) == "/abs/x.png"
end
