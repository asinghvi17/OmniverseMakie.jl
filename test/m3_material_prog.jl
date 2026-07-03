# Subprocess body for the M3.1 OmniPBR material test (read + run by
# test/m3_material_test.jl via run_ovrtx_subprocess).  Kept as a standalone .jl file
# (NOT a triple-quoted string) so the multi-line USDA stage needs no escaping.
#
# Proves the M3.1 material model end to end:
#   - usda_omnipbr_material  → a renderable OmniPBR Material PRE-AUTHORED in /World/Looks
#   - material_prim_path     → /World/Looks/Mat_<objectid(plot)>
#   - looks_scope_usda(body) → the Looks scope wrapping the material
#   - OV.bind_material!      → binds material:binding at RUNTIME on the open stage
# It renders the sphere UNBOUND (flat-grey diffuse displayColor baseline), then binds
# the metallic OmniPBR material at runtime and re-renders, asserting the metallic
# surface reads metallic: a sharp specular highlight over a dark body (much higher
# luminance contrast) and a substantial pixel-wise difference from the diffuse render.
#
# M3.1 VALIDATED CONSTRAINT (recorded in the report): the material is PRE-AUTHORED into
# the stage; a Material added to the OPEN stage via add_usd_reference! is NOT bindable.

using OmniverseMakie, ColorTypes, FixedPointNumbers, GeometryBasics
const OM = OmniverseMakie
const OV = OmniverseMakie.OV

OM.activate!(warmup = 64)

# A high-tessellation unit sphere, used both as a Makie mesh! plot (so
# material_prim_path is exercised on a real plot) and as the geometry we author inline.
sph = GeometryBasics.normal_mesh(GeometryBasics.Tesselation(Sphere(Point3f(0), 1.0f0), 96))
fig = Figure()
ax  = LScene(fig[1, 1])
m   = mesh!(ax, sph; color = RGBf(0.6, 0.6, 0.62))

matpath = OM.material_prim_path(m)
matname = String(last(split(matpath, "/")))
@assert matpath == "/World/Looks/Mat_$(objectid(m))" "material_prim_path wrong: $(matpath)"
println("MATPATH=$(matpath)")

# --- format the sphere geometry inline (0-based USD face indices) ---
pts = GeometryBasics.coordinates(sph)
nrm = GeometryBasics.normals(sph)
fcs = GeometryBasics.faces(sph)
ptsstr = join(["($(Float32(p[1])), $(Float32(p[2])), $(Float32(p[3])))" for p in pts], ", ")
nrmstr = join(["($(Float32(n[1])), $(Float32(n[2])), $(Float32(n[3])))" for n in nrm], ", ")
fvc = join([string(length(f)) for f in fcs], ", ")
fvi = join([string(Int(GeometryBasics.raw(i))) for f in fcs for i in f], ", ")

W, H = 600, 450
cfg  = OM.ScreenConfig(:rt2, 512, 64, 4, false, false, 40, :default)
rtx  = OM.rtx_settings_usda(cfg)
Mcam = OM.camera_to_world((3.6f0, 3.6f0, 3.6f0), (0.0f0, 0.0f0, 0.0f0), (0.0f0, 0.0f0, 1.0f0))
xf   = OM._usda_row_vector_matrix(Mcam)
intr = OM.camera_intrinsics(45.0, W, H)

inputs = Dict("diffuse_color_constant" => (0.6f0, 0.6f0, 0.62f0),
              "metallic_constant" => 1.0f0,
              "reflection_roughness_constant" => 0.15f0)
# PRE-AUTHOR the OmniPBR material inside the /World/Looks scope (M3.1-validated: a
# runtime add_usd_reference of a Material is NOT bindable; it must be present at open).
looks = OM.looks_scope_usda(OM.usda_omnipbr_material(matname, inputs))

stage = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 0.01
    upAxis = "Z"
    startTimeCode = 0
    endTimeCode = 100
    timeCodesPerSecond = 60
)

def Xform "World"
{
$(OM._DEFAULT_LIGHTS_STR)    def Camera "Camera" (
        prepend apiSchemas = ["OmniRtxCameraAutoExposureAPI_1", "OmniRtxCameraExposureAPI_1"]
    )
    {
        float2 clippingRange = (0.01, 1000000)
        float focalLength = $(intr.focal_length)
        float focusDistance = 6
        float horizontalAperture = $(intr.h_aperture)
        float verticalAperture = $(intr.v_aperture)
        token projection = "perspective"
        matrix4d xformOp:transform = $(xf)
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
$(looks)    def Mesh "sphere"
    {
        int[] faceVertexCounts = [$(fvc)]
        int[] faceVertexIndices = [$(fvi)]
        normal3f[] normals = [$(nrmstr)] (
            interpolation = "vertex"
        )
        point3f[] points = [$(ptsstr)]
        color3f[] primvars:displayColor = [(0.6, 0.6, 0.62)] (
            interpolation = "constant"
        )
        uniform token subdivisionScheme = "none"
        matrix4d xformOp:transform = ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
}

def "Render" (
    hide_in_stage_window = true
    no_delete = true
)
{
    def "OVMakie"
    {
        def RenderProduct "RenderProduct" (
            prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1", "OmniRtxSettingsRtAdvancedAPI_1", "OmniRtxSettingsPtAdvancedAPI_1", "OmniRtxPostColorGradingAPI_1", "OmniRtxPostChromaticAberrationAPI_1", "OmniRtxPostBloomPhysicalAPI_1", "OmniRtxPostMatteObjectAPI_1", "OmniRtxPostCompositingAPI_1", "OmniRtxPostDofAPI_1", "OmniRtxPostMotionBlurAPI_1", "OmniRtxPostTvNoiseAPI_1", "OmniRtxPostTonemapIrayReinhardAPI_1", "OmniRtxPostDebugSettingsAPI_1", "OmniRtxDebugSettingsAPI_1"]
            hide_in_stage_window = true
            no_delete = true
        )
        {
            rel camera = </World/Camera>
$(rtx)
            rel orderedVars = </Render/Vars/LdrColor>
            uniform int2 resolution = ($(W), $(H))
        }
    }
    def RenderSettings "GlobalRenderSettings" (
        prepend apiSchemas = ["OmniRtxSettingsGlobalRtAdvancedAPI_1", "OmniRtxSettingsGlobalPtAdvancedAPI_1"]
        no_delete = true
    )
    {
        rel products = </Render/OVMakie/RenderProduct>
    }
    def "Vars"
    {
        def RenderVar "LdrColor" (
            hide_in_stage_window = true
            no_delete = true
        )
        {
            uniform string sourceName = "LdrColor"
        }
    }
}
"""

# --- metrics -----------------------------------------------------------------
lum(c) = 0.2126f0 * Float32(red(c)) + 0.7152f0 * Float32(green(c)) + 0.0722f0 * Float32(blue(c))
function stats(img)
    L   = vec([lum(c) for c in img])
    lit = sort(L[L .> 0.02f0])
    isempty(lit) && return (nonblack = 0, meanlit = 0.0f0, maxl = 0.0f0,
                            p50 = 0.0f0, p95 = 0.0f0, contrast = 0.0f0)
    q(p) = lit[clamp(round(Int, p * length(lit)), 1, length(lit))]
    p50, p95 = q(0.50), q(0.95)
    return (nonblack = length(lit), meanlit = sum(lit) / length(lit), maxl = lit[end],
            p50 = p50, p95 = p95, contrast = p95 / (p50 + 1.0f-3))
end
function meanabsdiff(a, b)
    s = 0.0
    @inbounds for i in eachindex(a, b)
        s += abs(Float32(red(a[i]))   - Float32(red(b[i]))) +
             abs(Float32(green(a[i])) - Float32(green(b[i]))) +
             abs(Float32(blue(a[i]))  - Float32(blue(b[i])))
    end
    return s / (3 * length(a))
end

# --- render: UNBOUND diffuse baseline, then RUNTIME-bound metallic ------------
# (a function so the try/finally + locals scope cleanly at subprocess top level.)
function render_diffuse_then_metallic(stage, matpath)
    r = OV.Renderer()
    try
        OV.open_usd_string!(r, stage)
        imgA = OV.render_to_matrix(r, "/Render/OVMakie/RenderProduct"; warmup = 80)

        OV.bind_material!(r, "/World/sphere", matpath)
        OV.reset!(r)
        imgB = OV.render_to_matrix(r, "/Render/OVMakie/RenderProduct"; warmup = 80)
        return (imgA, imgB)
    finally
        close(r)
    end
end

imgA, imgB = render_diffuse_then_metallic(stage, matpath)

sA  = stats(imgA)
sB  = stats(imgB)
mad = meanabsdiff(imgA, imgB)
cr  = sB.contrast / (sA.contrast + 1.0f-3)

println("ELTYPE=", eltype(imgB))
println("SIZE=", size(imgB))
println("DIFFUSE_STATS nonblack=$(sA.nonblack) meanlit=$(sA.meanlit) maxl=$(sA.maxl) p50=$(sA.p50) p95=$(sA.p95) contrast=$(sA.contrast)")
println("METALLIC_STATS nonblack=$(sB.nonblack) meanlit=$(sB.meanlit) maxl=$(sB.maxl) p50=$(sB.p50) p95=$(sB.p95) contrast=$(sB.contrast)")
println("MEANABSDIFF=$(mad)")
println("CONTRAST_RATIO=$(cr)")

@assert eltype(imgB) == RGBA{N0f8} "eltype is $(eltype(imgB)) (expected RGBA{N0f8})"
# (a) the metallic render is not black.
@assert sB.nonblack > 1000 "metallic render is (near) black: nonblack=$(sB.nonblack)"
# (b) metallic differs SUBSTANTIALLY from the diffuse render (material:binding took).
@assert mad > 0.03 "metallic render too similar to diffuse (mad=$(mad)): material:binding did not take"
# (c) metallic SIGNATURE: a concentrated bright specular highlight over a dark body →
#     a much higher luminance contrast than the flat-shaded diffuse sphere.
@assert sB.contrast > 3.0 "no metallic specular signature: metallic contrast=$(sB.contrast) (expected high dynamic range)"
@assert cr > 3.0 "metallic not distinct enough from diffuse: contrast ratio=$(cr)"

println("OK_MATERIAL")
