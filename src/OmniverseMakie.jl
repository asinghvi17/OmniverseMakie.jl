module OmniverseMakie

using Makie, GeometryBasics, Colors, ColorTypes, FixedPointNumbers, LinearAlgebra
import LibOVRTX
import NanoVDBWriter     # dense Array{Float32,3} → .nvdb writer (save_nanovdb)
import ComputePipeline   # :ovrtx_renderobject diff node
import PNGFiles          # image `color` → temp PNG for an OmniPBR texture

# OV.jl does `using ..LibOVRTX`; the `import LibOVRTX` above makes
# `..LibOVRTX` resolve to OmniverseMakie.LibOVRTX.
include("binding/OV.jl")         # module OV: Renderer, StepResult, etc.
include("settings.jl")           # ScreenConfig, rtx_settings_usda
include("translation/usd.jl")    # author_render_root!, usda_mesh
include("translation/camera.jl") # author_camera!, sync_camera!, intrinsics
include("translation/lights.jl") # lights_usda, sync_lights!
include("translation/materials.jl") # displaycolor_for
include("translation/meshes.jl")    # to_ovrtx_object (Makie.Mesh)
include("translation/primitives.jl") # to_ovrtx_object (scatter/lines/surface)
include("translation/volume.jl")     # _vdb_volume_usda / author_vdb_volume!
include("translation/envlight.jl")   # EnvLightState; before screen.jl
include("compute.jl")            # OvrtxRObj; before screen.jl (field type)
include("screen.jl")             # Screen, colorbuffer, insert!, activate!
include("translation/usdplot.jl")  # USDPlot recipe + bind_usd!
include("sensors.jl")            # Lidar/Radar recipes + step_sensors!
include("tonemap.jl")            # shared HDR tonemap math

# usdplot bindings API (the recipe auto-exports USDPlot/usdplot/usdplot!).
export bind_usd!, unbind_usd!
# Sensor simulation (the recipes auto-export the lidar/radar recipes).
export step_sensors!, sensor_returns
# Environment-light image (IBL): set/live-swap the DomeLight environment map.
export push_environment_image!

# The interactive viewport lives in package extensions (GLMakie / CUDA);
# the main module only declares the generics, the exts add the methods.
function interactive_display end
function present! end
function on_render_tick! end
# The CUDA ext defines `_cuda_functional() = CUDA.functional()`; the GLMakie
# ext's `_pick_blitter` calls it (invokelatest) to decide :gpu vs :cpu.
function _cuda_functional end
# The CUDA ext defines `_gpu_teardown!(::GPUBlitState)` (unregister the GL
# texture resource); `Base.close(::ViewportSession)` calls it to tear down
# the duck-typed GPU state without naming the CUDA-ext type.
function _gpu_teardown! end
# The CUDA ext defines `gpu_unregister!(session)`: drop the CUDA-GL texture
# registration (kept GPUBlitState re-registers on the next GPU present!).
# resize_viewport! calls it before the GL texture is recreated (id recycling).
function gpu_unregister! end
# The GLMakie ext defines attach_picking!/detach_picking! (+ `_pick_at!`) for
# a live `interactive_display` viewport.  Declared here so all three resolve
# as `OmniverseMakie.*` without a GLMakie dep.  Not exported; callers qualify.
function attach_picking! end
function detach_picking! end
function _pick_at! end
# Replace ONE scene (LScene / Axis3 / Scene) in a displayed GLMakie figure
# with a live ovrtx raytraced render, leaving the other axes as GLMakie 2D
# (the RPRMakie `replace_scene_rpr!` pattern).  GLMakie ext adds the method.
function replace_scene! end
# Scripted-recording companion (GLMakie ext adds the method): drive `ticks`
# synchronous host frames on a STOPPED render loop and return the composited
# figure image — see the `replace_scene!` docstring's recording recipe.
function record_frame! end
# The CUDA-only ext (OmniverseMakieCUDADirectExt) defines
# `gpu_update_mesh!(screen, plot; points)`: push mesh points straight from a
# CUDA device array through the persistent binding (no host copy).
function gpu_update_mesh! end
export interactive_display, replace_scene!, record_frame!, gpu_update_mesh!

# Error helpfully when no GLMakie extension is loaded (no method otherwise).
interactive_display(::Any; kwargs...) =
    error("interactive_display requires GLMakie — run `using GLMakie` (and `using CUDA` for GPU-direct).")
replace_scene!(::Any; kwargs...) =
    error("replace_scene! requires GLMakie — run `using GLMakie` and display the figure first.")
record_frame!(::Any; kwargs...) =
    error("record_frame! requires GLMakie — run `using GLMakie`; it records a replace_scene! session.")
gpu_update_mesh!(::Any, ::Any; kwargs...) =
    error("gpu_update_mesh! requires CUDA — run `using CUDA` to load the GPU-direct write extension.")

# Re-export every exported Makie name (the standard backend pattern).
for name in names(Makie, all = true)
    if Base.isexported(Makie, name)
        @eval using Makie: $(name)
        @eval export $(name)
    end
end

function __init__()
    # Seed the OmniverseMakie sub-theme in Makie's CURRENT_DEFAULT_THEME so
    # set_screen_config!/merge_screen_config find our ScreenConfig defaults;
    # without it activate!() throws KeyError (Makie hardcodes GL/WGL/RPR).
    Makie.CURRENT_DEFAULT_THEME[:OmniverseMakie] = Makie.Attributes(
        mode = :rt2, samples = 512, warmup = 64, max_bounces = 4,
        selection_outline = false,   # outline off by default
        accumulate_across_frames = false,  # off = per-frame reconverge
        accumulation_preroll = 40,   # first-frame warmup (accumulate mode)
        background = :default,       # :default | :sky | :domelight
        sensors = false,             # motion BVH for post-display sensors
    )
    # Make `material=` backend-universal so Lines/Scatter/LineSegments accept
    # it (Makie validates keywords; only mesh-like recipes document it).
    # `@eval Makie` is forbidden during precompile output → guard skips there.
    ccall(:jl_generating_output, Cint, ()) == 0 && _enable_material_attribute!()
    activate!()
    return
end

end # module OmniverseMakie
