module OmniverseMakie

using Makie, GeometryBasics, Colors, ColorTypes, FixedPointNumbers, LinearAlgebra
import LibOVRTX

# OV.jl does `using ..LibOVRTX`; having `import LibOVRTX` above makes
# `..LibOVRTX` resolve to OmniverseMakie.LibOVRTX (the M0 parent-module fix).
include("binding/OV.jl")         # defines module OV with Renderer, StepResult, etc.
include("settings.jl")           # ScreenConfig, rtx_settings_usda
include("translation/usd.jl")    # author_render_root!, usda_mesh, usda_matrix4d
include("translation/camera.jl") # camera_to_world, author_camera!, camera_intrinsics
include("translation/lights.jl") # lights_usda, author_root_from_scene!, author_lights!
include("screen.jl")             # Screen, activate!

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
    activate!()
    return
end

end # module OmniverseMakie
