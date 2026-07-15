# In-process composite-color probe: Julia-hosted ovrtx (OV FFI), torus.vdb
# volume with a STRONG-color Colormap, `nvindex:composite` + `omni:rtx:skip`
# attrs, and a carb config carrying the IndeX token PLUS the two composite
# settings Kit's omni.rtx.index_composite ext contributes:
#   /nvindex/compositeRenderingAvailable = true   (the availability gate)
#   /rtx/index/compositeEnabled = true
# PROBE_COMPOSITE=0 runs the identical scene WITHOUT the extra settings/attrs
# (expected: IndeX Direct grayscale) for the A/B.
using OmniverseMakie
using OmniverseMakie: OV
import LibOVRTX
using ColorTypes

const COMPOSITE = get(ENV, "PROBE_COMPOSITE", "1") == "1"
const OUTPNG    = ENV["PROBE_OUT"]
const VDB       = ENV["PROBE_VDB"]
const MODE      = get(ENV, "PROBE_MODE", "RealTimePathTracing")

# ---- carb config: token (+ crashreporter-off) via the library's own
# synthesizer, then prepend the composite settings blocks at top level -------
ovrtx_bin = dirname(ENV["OVRTX_LIBRARY_PATH"])
libs = ENV["OMNIVERSEMAKIE_INDEX_LIBS_DIR"]
cfg = OV._synth_index_config(ovrtx_bin, libs)
extra = ""
if COMPOSITE
    avail = get(ENV, "PROBE_AVAILABLE", "1") == "1" ? """
        "nvindex": {
            "compositeRenderingAvailable": true
        },""" : ""
    extra *= """

        "renderer": {
            "enabled": "rtx,index"
        },
$(avail)
        "rtx": {
            "index": {
                "compositeEnabled": true,
                "compositeDepthMode": 3
            }
        },"""
end
body = read(cfg, String)
if !isempty(extra)
    i = findfirst('{', body)
    body = body[1:i] * extra * body[i + 1:end]
end
if get(ENV, "PROBE_LOGINFO", "0") == "1"
    # edit the existing log block in place — a duplicate top-level "log" key
    # loses to the base one
    body = replace(body,
        "\"outputStreamLevel\": \"Warning\"" => "\"outputStreamLevel\": \"Verbose\"",
        "\"level\": \"Info\""                => "\"level\": \"Verbose\"")
end
write(cfg, body)
println("CONFIG_HEAD=", replace(body[1:min(700, end)], "\n" => " "))
ENV["OMNIVERSEMAKIE_OVRTX_CONFIG"] = cfg
println("CONFIG=", cfg)

# ---- stage: torus fog volume + colored TF + camera/light + render root -----
attrs = COMPOSITE && get(ENV, "PROBE_ATTRS", "1") == "1" ? """
        custom bool nvindex:composite = 1
        custom bool omni:rtx:skip = 1
""" : ""
layer_rs = COMPOSITE ? """
    customLayerData = {
        dictionary renderSettings = {
            int "rtx:index:compositeDepthMode" = 3
            bool "rtx:index:compositeEnabled" = 1
        }
    }
""" : ""
usda = """
#usda 1.0
(
$(layer_rs)    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{
    def Volume "Volume" (
        prepend apiSchemas = ["MaterialBindingAPI"]
        customData = {
            dictionary "nvindex.renderSettings" = {
                string filterMode = "trilinear"
                double samplingDistance = 0.20000000298023224
            }
        }
    )
    {
        rel field:fog = </World/Volume/fog>
        rel material:binding = </World/Volume/Material>
$(attrs)
        def OpenVDBAsset "fog"
        {
            token fieldName = "torus_fog"
            asset filePath = @$(VDB)@
        }

        def Material "Material"
        {
            token outputs:nvindex:volume.connect = </World/Volume/Material/VolumeShader.outputs:volume>

            def Colormap "Colormap"
            {
                custom token colormapSource = "rgbaPoints"
                custom float2 domain = (0, 1)
                uniform token domainBoundaryMode = "clampToTransparent"
                custom token outputs:colormap
                custom float4[] rgbaPoints = [(0, 0, 1, 0), (0, 1, 1, 0.2), (0, 1, 0, 0.4), (1, 1, 0, 0.55), (1, 0, 0, 0.7)]
                custom float[] xPoints = [0, 0.25, 0.5, 0.75, 1]
            }

            def Shader "VolumeShader"
            {
                token inputs:colormap.connect = </World/Volume/Material/Colormap.outputs:colormap>
                token outputs:volume
            }
        }
    }

    def DistantLight "Sun" (
        prepend apiSchemas = ["ShapingAPI"]
    )
    {
        float inputs:angle = 1
        float inputs:intensity = 3000
        double3 xformOp:rotateXYZ = (315, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }

    def Camera "Camera"
    {
        float2 clippingRange = (1, 10000000)
        float focalLength = 18.147562
        float focusDistance = 400
        float horizontalAperture = 20.955
        float verticalAperture = 15.2908
        token projection = "perspective"
        float3 xformOp:rotateYXZ = (80.320885, 0, 149.87924)
        double3 xformOp:translate = (23.116637717275065, 40.34599585848169, 9.516162167444032)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateYXZ"]
    }
}

def Scope "Render"
{
    def Scope "OVMakie"
    {
        def RenderProduct "RenderProduct" (
            prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1", "OmniRtxSettingsRtAdvancedAPI_1", "OmniRtxSettingsPtAdvancedAPI_1", "OmniRtxPostColorGradingAPI_1", "OmniRtxPostChromaticAberrationAPI_1", "OmniRtxPostBloomPhysicalAPI_1", "OmniRtxPostMatteObjectAPI_1", "OmniRtxPostCompositingAPI_1", "OmniRtxPostDofAPI_1", "OmniRtxPostMotionBlurAPI_1", "OmniRtxPostTvNoiseAPI_1", "OmniRtxPostTonemapIrayReinhardAPI_1", "OmniRtxPostDebugSettingsAPI_1", "OmniRtxDebugSettingsAPI_1"]
            hide_in_stage_window = true
            no_delete = true
        )
        {
            rel camera = </World/Camera>
            token omni:rtx:rendermode = "$(MODE)"
            rel orderedVars = [</Render/Vars/LdrColor>]
            uniform int2 resolution = (512, 512)
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
stage_path = joinpath(mktempdir(), "composite_probe.usda")
write(stage_path, usda)
println("STAGE=", stage_path)

# ---- render + metrics -------------------------------------------------------
r = OV.Renderer()
println("INDEX_ENABLED=", OV._index_enabled())
OV.open_usd!(r, stage_path)
img = OV.render_to_matrix(r, "/Render/OVMakie/RenderProduct"; warmup = 64)

lum(c)    = Float32(red(c)) + Float32(green(c)) + Float32(blue(c))
chroma(c) = (v = (Float32(red(c)), Float32(green(c)), Float32(blue(c))); maximum(v) - minimum(v))
nb  = count(c -> lum(c) > 0.05f0, img)
ch  = count(c -> chroma(c) > 0.15f0, img)
mch = (lit = [chroma(c) for c in img if lum(c) > 0.05f0];
       isempty(lit) ? 0.0 : sum(lit) / length(lit))
println("NONBLACK=", nb, "  CHROMA_PX=", ch, "  MEAN_CHROMA_LIT=", round(mch; digits = 4))
Makie.FileIO.save(OUTPNG, img)
println("PNG=", OUTPNG)
close(r)
println("OK_INPROCESS_PROBE")
