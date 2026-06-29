module OmniverseMakie

using Makie, GeometryBasics, Colors, ColorTypes, FixedPointNumbers, LinearAlgebra
import LibOVRTX
import ComputePipeline   # :ovrtx_renderobject diff node (register_computation!/mark_resolved!)
import PNGFiles          # M3.3: write an image `color` to a temp PNG for an OmniPBR texture

# OV.jl does `using ..LibOVRTX`; having `import LibOVRTX` above makes
# `..LibOVRTX` resolve to OmniverseMakie.LibOVRTX (the M0 parent-module fix).
include("binding/OV.jl")         # defines module OV with Renderer, StepResult, etc.
include("settings.jl")           # ScreenConfig, rtx_settings_usda
include("translation/usd.jl")    # author_render_root!, author_root_from_scene!, usda_mesh
include("translation/camera.jl") # camera_to_world, author_camera!, sync_camera!, intrinsics
include("translation/lights.jl") # lights_usda, light_prim_path, sync_lights!, author_lights!
include("translation/materials.jl") # displaycolor_for (plot.color → primvars:displayColor)
include("translation/meshes.jl")    # to_ovrtx_object (Makie.Mesh → UsdGeomMesh reference)
include("translation/primitives.jl") # to_ovrtx_object (scatter/meshscatter/lines/surface)
include("compute.jl")            # OvrtxRObj (Screen.plot2robj references it) — before screen.jl
include("screen.jl")             # Screen, open-stage colorbuffer, insert!/insertplots!, activate!

# Re-export every Makie name verbatim (GLMakie/src/GLMakie.jl:36-41).
for name in names(Makie, all = true)
    if Base.isexported(Makie, name)
        @eval using Makie: $(name)
        @eval export $(name)
    end
end

function __init__()
    # Seed the OmniverseMakie sub-theme in Makie's CURRENT_DEFAULT_THEME so that
    # set_screen_config! / merge_screen_config can look up our ScreenConfig defaults.
    # Without this, activate!() throws KeyError(:OmniverseMakie) because Makie only
    # hardcodes GL/WGL/RPR in theming.jl.  See context file "CRITICAL gotcha" section.
    Makie.CURRENT_DEFAULT_THEME[:OmniverseMakie] = Makie.Attributes(
        mode = :rt2, samples = 512, warmup = 64, max_bounces = 4,
    )
    # M3.5: make `material=` a backend-universal attribute so it is accepted on Lines /
    # Scatter / LineSegments too (Makie validates undocumented keywords; only mesh-like
    # recipes document `material` natively).
    _enable_material_attribute!()
    activate!()
    return
end

end # module OmniverseMakie
