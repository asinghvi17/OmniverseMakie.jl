module OmniverseMakie

using Makie, GeometryBasics, Colors, ColorTypes, FixedPointNumbers, LinearAlgebra
import LibOVRTX
import NanoVDBWriter     # Volumes M2: dense Array{Float32,3} → .nvdb writer (save_nanovdb)
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
include("translation/volume.jl")     # Volumes M1: _vdb_volume_usda / author_vdb_volume! (UsdVol)
include("compute.jl")            # OvrtxRObj (Screen.plot2robj references it) — before screen.jl
include("screen.jl")             # Screen, open-stage colorbuffer, insert!/insertplots!, activate!
include("tonemap.jl")            # shared HDR tonemap math (Task 2)

# M5/M6 interactive viewport lives in package extensions (GLMakie / CUDA). The main
# module only DECLARES the generics; the GLMakie ext adds the methods.
function interactive_display end
function present! end
function on_render_tick! end
# Declared here (GLMakie ext adds the method) so `OmniverseMakie.cpu_blit!` resolves for
# callers/tests; the M6.A ext refactor moved its body to the GLMakie ext.
function cpu_blit! end
# M6.A: the CUDA ext defines `_cuda_functional() = CUDA.functional()`; the GLMakie ext's
# `_pick_blitter` calls it (invokelatest) to decide :gpu vs :cpu.  Declared here so the GLMakie
# ext can reference it without a CUDA dep.
function _cuda_functional end
# M6.A: the CUDA ext defines `_gpu_teardown!(::GPUBlitState)` (unregister the GL texture resource);
# `Base.close(::ViewportSession)` calls it when `gpu_state` was set, tearing down the duck-typed
# GPU state without naming the CUDA-ext type.
function _gpu_teardown! end
# M6.A Task 5: the CUDA ext defines `gpu_unregister!(session)` — cuGraphicsUnregisterResource the
# GL-texture resource + clear `registered` (keeping the GPUBlitState so the next GPU present!
# re-registers).  `resize_viewport!` drops the OLD CUDA-GL registration through this generic BEFORE
# the image! texture is recreated (GL can recycle a freed texture id → resize id-recycle-proof).
function gpu_unregister! end
# M6.B Task 5: the GLMakie ext defines the attachable picking interaction on a live
# `interactive_display` viewport — `attach_picking!`/`detach_picking!` + the `_pick_at!` helper the
# click listener (and test) invoke.  Declared here so all three resolve as `OmniverseMakie.*`
# without a GLMakie dep.  NOT exported — opt-in advanced API; callers qualify the name.
function attach_picking! end
function detach_picking! end
function _pick_at! end
export interactive_display

# Errors helpfully when no GLMakie extension is loaded (no method otherwise).
interactive_display(::Any; kwargs...) =
    error("interactive_display requires GLMakie — run `using GLMakie` (and `using CUDA` for GPU-direct).")

# Re-export every Makie name verbatim (GLMakie/src/GLMakie.jl:36-41).
for name in names(Makie, all = true)
    if Base.isexported(Makie, name)
        @eval using Makie: $(name)
        @eval export $(name)
    end
end

function __init__()
    # Seed the OmniverseMakie sub-theme in Makie's CURRENT_DEFAULT_THEME so set_screen_config! /
    # merge_screen_config find our ScreenConfig defaults.  Without it activate!() throws
    # KeyError(:OmniverseMakie) (Makie hardcodes only GL/WGL/RPR in theming.jl).
    Makie.CURRENT_DEFAULT_THEME[:OmniverseMakie] = Makie.Attributes(
        mode = :rt2, samples = 512, warmup = 64, max_bounces = 4,
        selection_outline = false,   # M6.B: outline feature off by default at every level
    )
    # M3.5: make `material=` a backend-universal attribute so Lines/Scatter/LineSegments accept it
    # too (Makie validates undocumented keywords; only mesh-like recipes document it natively).
    # Guard: _enable_material_attribute! uses `@eval Makie`, forbidden while Julia is generating
    # precompile output (Makie already closed) — skip during ext precompile; it runs at load time.
    ccall(:jl_generating_output, Cint, ()) == 0 && _enable_material_attribute!()
    activate!()
    return
end

end # module OmniverseMakie
