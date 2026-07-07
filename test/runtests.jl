using Test

# Set OVRTX_LIBRARY_PATH before any `using LibOVRTX` so __init__ can dlopen it.
# The package itself does NOT hardcode this path; only the test harness sets
# the default.
get!(ENV, "OVRTX_LIBRARY_PATH",
    "/home/juliahub/temp/omniverse-makie/references/ovrtx/examples/python/minimal/.venv/lib/python3.13/site-packages/ovrtx/bin/libovrtx-dynamic.so")

# ==============================================================================
# Suite layout: feature-grouped subdirectories; each file self-includes
# ../helpers.jl so it also runs standalone.  Conventions every subprocess test
# follows:
#   • run_ovrtx_subprocess(prog; timeout, retries, ready_marker) — retries
#     absorb ovrtx's intermittent pre-render startup crash (see
#     src/binding/signals.jl);
#   • $(PROG_PIXEL_HELPERS) for shared pixel oracles where its thresholds fit;
#   • pixel evidence is the ONLY oracle for ovrtx writes (it silently ignores
#     bad ones).
# GPU-vs-CPU perf benchmarks live in ../bench (present_bench.jl, hot_path.jl);
# timing gates are machine-dependent and are NOT part of this suite.
# ==============================================================================

@testset "workspace loads" begin
    @test (using LibOVRTX; true)
    @test (using OmniverseMakie; true)
end

# Shared subprocess runner + pixel-oracle prelude (used by every group below).
include("helpers.jl")

# --- harness: the test harness itself (exit codes; ready_marker retries) -----
include("harness/watchdog_test.jl")
include("harness/retry_test.jl")

# --- ffi: LibOVRTX ABI + the raw OV binding layer (below Makie/Screen) --------
include("ffi/abi_test.jl")            # generated-binding layout; version pin
include("ffi/raw_render_test.jl")     # raw full-frame render + write_xform!
include("ffi/op_error_test.jl")       # op-errors THROW; closed-Renderer guards
include("ffi/readback_test.jl")       # cwh_to_matrix; with_mapped_hdr unmap
include("ffi/binding_string_test.jl") # ovx_string SubString; finalizer flags

# --- offscreen: the offscreen pipeline (Screen → colorbuffer/save/record) -----
include("offscreen/mesh_render_test.jl")       # capstone: mesh → colorbuffer
include("offscreen/orientation_test.jl")       # orientation: red above blue
include("offscreen/save_record_test.jl")       # save, showable, record→mp4
include("offscreen/usd_reference_test.jl")     # add_usd_reference! + REMOVE
include("offscreen/camera_test.jl")            # intrinsics + author_camera!
include("offscreen/primitives_test.jl")        # primitives + combined gate
include("offscreen/lights_test.jl")            # usda_light dispatch + lum ratio
include("offscreen/lights_structural_test.jl") # fails loud; RectLight golden
include("offscreen/lights_alloc_test.jl")      # sync_lights! alloc + path reuse
include("offscreen/envlight_test.jl")          # IBL push, background, UV tiling
include("offscreen/pathtracing_test.jl")       # :pathtracing + samples wiring

# --- live: the open-stage diff pipeline (insert/edit/delete, no re-author) ----
include("live/insert_test.jl")             # insert! + children + handle no-op
include("live/rendercfg_test.jl")          # live camera orbit + light writes
include("live/diffnode_test.jl")           # one-minimal-write-per-edit contract
include("live/subscene_test.jl")           # Scope hierarchy mirrors scene tree
include("live/binding_test.jl")            # hot-path bindings identity-stable
include("live/delete_test.jl")             # leak-free delete!/empty!; 50× churn
include("live/scatter_positions_test.jl")  # positions route; materialized skip
include("live/nan_lines_test.jl")          # NaN-split gap; frozen-size gate
include("live/empty_fill_test.jl")         # empty→fill rebuild + pick regs
include("live/accumulate_config_test.jl")  # ScreenConfig field/default contract
include("live/accumulate_render_test.jl")  # accumulate reset suppression
include("live/usdplot_test.jl")            # usdplot recipe + bind_usd!

# --- materials: OmniPBR/OmniGlass authoring, textures, colormaps, live edits --
include("materials/material_test.jl")         # material= compose + bool gates
include("materials/texture_test.jl")          # image color → diffuse_texture
include("materials/material_live_test.jl")    # live color/material edits
include("materials/colormap_test.jl")         # numeric→colormap static + live
include("materials/surface_texture_test.jl")  # surface UV convention (rotation)
include("materials/glass_test.jl")            # OmniGlass refraction; live edits
include("materials/colormap_helpers_test.jl") # NaN-safe colorrange/map helpers
include("materials/texture_freshpath_test.jl")# fresh temp-PNG per write

# --- viewport: the GLMakie/CUDA extension (interactive/hybrid/present) --------
include("viewport/ext_load_test.jl")      # offscreen-pure w/o GLMakie; guard
include("viewport/tonemap_test.jl")       # shared ACES+sRGB scalar (pure)
include("viewport/viewport_test.jl")      # interactive_display resize/teardown
include("viewport/camera_loop_test.jl")   # tick accumulation/reset invariants
include("viewport/orbit_test.jl")         # window-input forwarding → Camera3D
include("viewport/present_cpu_test.jl")   # CPU present: pixel + alloc gates
include("viewport/present_gpu_test.jl")   # CUDA kernel: byte-exact + 0 alloc
include("viewport/map_cuda_test.jl")      # OV.map_cuda linear FFI + wait-event
include("viewport/gpu_blit_test.jl")      # GPU present on the REAL render task
include("viewport/replace_scene_test.jl") # hybrid embed + usdplot + recording

# --- picking: native AOV pick → Makie pick protocol + selection outline -------
include("picking/pick_test.jl")     # path2plot lockstep; pick trio + y-flip pin
include("picking/resolver_test.jl") # cache invalidation on composition change
include("picking/outline_test.jl")  # select!/clear_selection! outline pixels
include("picking/attach_test.jl")   # attach_picking! click→hit flow; refusal

# --- sensors: lidar!/radar! through ovrtx's native RTX sensor pipeline --------
include("sensors/sensor_test.jl")   # recipe/emission + e2e GPU point clouds

# --- volumes: UsdVol/NanoVDB via NVIDIA IndeX (env-gated, skip-if-libs-absent)
include("volumes/index_config_synth_test.jl") # carb-config synthesis + escaping
include("volumes/index_config_test.jl")       # _ensure_index enable/idempotent
include("volumes/render_test.jl")             # real .vdb renders via IndeX
include("volumes/plot_test.jl")               # volume! recipe + gray tripwire
include("volumes/live_test.jl")               # reload; empty→fill; recovery

# --- authoring: USD string hygiene + emitter goldens --------------------------
include("authoring/usd_hygiene_test.jl") # identifier/asset-path guards; goldens
include("authoring/alloc_test.jl")       # flat-emitter byte-identity + allocs
