using Test

# Set OVRTX_LIBRARY_PATH before any `using LibOVRTX` so __init__ can dlopen it.
# The package itself does NOT hardcode this path; only the test harness sets the default.
get!(ENV, "OVRTX_LIBRARY_PATH",
    "/home/juliahub/temp/omniverse-makie/references/ovrtx/examples/python/minimal/.venv/lib/python3.13/site-packages/ovrtx/bin/libovrtx-dynamic.so")

# =============================================================================================
# Suite layout (post-rationalization, 2026-07-03).  Feature-grouped subdirectories replace the
# old milestone-codename flat files; each file self-includes ../helpers.jl so it also runs
# standalone.  Conventions every subprocess test follows:
#   • run_ovrtx_subprocess(prog; timeout, retries, ready_marker) — the retry loop absorbs
#     ovrtx's intermittent GeometryGroup::attachToContext startup crash;
#   • $(PROG_PIXEL_HELPERS) for shared pixel oracles where its thresholds fit;
#   • pixel evidence is the ONLY oracle for ovrtx writes (it silently ignores bad ones).
# GPU-vs-CPU perf benchmarks live in ../bench (present_bench.jl, hot_path.jl) — timing gates
# are machine-dependent and are NOT part of this suite.
# =============================================================================================

@testset "workspace loads" begin
    @test (using LibOVRTX; true)
    @test (using OmniverseMakie; true)
end

# Shared subprocess runner + pixel-oracle prelude (used by every group below).
include("helpers.jl")

# --- harness: the test harness itself (truthful exit codes; ready_marker retry loop) --------
include("harness/watchdog_test.jl")
include("harness/retry_test.jl")

# --- ffi: LibOVRTX ABI + the raw OV binding layer (below Makie/Screen) ----------------------
include("ffi/abi_test.jl")            # generated-binding layout + real .so link/version pin
include("ffi/raw_render_test.jl")     # raw Renderer render (full-frame coverage) + write_xform!
include("ffi/op_error_test.jl")       # enqueue_wait op-errors THROW; closed-Renderer guards
include("ffi/readback_test.jl")       # cwh_to_matrix orientation; with_mapped_hdr unmap-on-throw
include("ffi/binding_string_test.jl") # ovx_string SubString; Binding finalizer flag paths

# --- offscreen: the offscreen pipeline (Screen → colorbuffer/save/record) -------------------
include("offscreen/mesh_render_test.jl")       # capstone: mesh → colorbuffer (+ backend registered)
include("offscreen/orientation_test.jl")       # vertical orientation pin (red above blue)
include("offscreen/save_record_test.jl")       # PNG/JPEG save, showable, record→mp4, re-author
include("offscreen/usd_reference_test.jl")     # add_usd_reference! + REMOVE round-trip
include("offscreen/camera_test.jl")            # intrinsics units + author_camera! reframes
include("offscreen/primitives_test.jl")        # scatter/meshscatter/lines/surface/combined gate
include("offscreen/lights_test.jl")            # usda_light dispatch + luminance-ratio render
include("offscreen/lights_structural_test.jl") # live add/remove fails loud; RectLight xform golden
include("offscreen/lights_alloc_test.jl")      # sync_lights! allocation budget + path reuse
include("offscreen/envlight_test.jl")          # env-image push (IBL), background source, UV tiling
include("offscreen/pathtracing_test.jl")       # mode=:pathtracing rendermode + samples SPP wiring

# --- live: the open-stage diff pipeline (insert/edit/delete without re-authoring) -----------
include("live/insert_test.jl")             # live insert! + recipe children + stable-handle no-op
include("live/rendercfg_test.jl")          # live camera orbit + light intensity/color writes
include("live/diffnode_test.jl")           # exactly-one-minimal-write per edit contract
include("live/subscene_test.jl")           # USD Scope hierarchy mirrors the scene tree
include("live/binding_test.jl")            # persistent hot-path bindings, identity-stable 100 frames
include("live/delete_test.jl")             # leak-free delete!/empty!; 50× add/remove no accumulation
include("live/scatter_positions_test.jl")  # instancer `positions` routing; materialized skip
include("live/nan_lines_test.jl")          # NaN-split BasisCurves render with a gap; frozen-size gate
include("live/empty_fill_test.jl")         # empty→fill late rebuild + pick registration
include("live/accumulate_config_test.jl")  # ScreenConfig field-order/default contract
include("live/accumulate_render_test.jl")  # accumulate-across-frames reset suppression
include("live/usdplot_test.jl")            # usdplot recipe + bind_usd! (compose, bind, fail-fast)

# --- materials: OmniPBR/OmniGlass authoring, textures, colormaps, live edits ----------------
include("materials/material_test.jl")         # material= composition + primitive coverage + bool gates
include("materials/texture_test.jl")          # image color → diffuse_texture + st UVs (mesh)
include("materials/material_live_test.jl")    # live color/material edits on mesh + meshscatter
include("materials/colormap_test.jl")         # numeric color → colormap, static + live
include("materials/surface_texture_test.jl")  # surface parametric-UV convention (rotation guard)
include("materials/glass_test.jl")            # OmniGlass refraction + live glass edits
include("materials/colormap_helpers_test.jl") # NaN-safe _resolve_colorrange/_map_through_colormap
include("materials/texture_freshpath_test.jl")# fresh temp-PNG per write (two-Screen re-author)

# --- viewport: the GLMakie/CUDA extension surface (interactive + hybrid + present paths) ----
include("viewport/ext_load_test.jl")      # offscreen-pure without GLMakie; helpful guard error
include("viewport/tonemap_test.jl")       # shared ACES+sRGB scalar (pure)
include("viewport/viewport_test.jl")      # interactive_display: frame + resize + teardown
include("viewport/camera_loop_test.jl")   # tick accumulation/reset-on-move invariants
include("viewport/orbit_test.jl")         # window-input forwarding → Camera3D
include("viewport/present_cpu_test.jl")   # fused CPU present: pixel-exact + allocation gates
include("viewport/present_gpu_test.jl")   # fused oriented CUDA kernel: byte-exact + 0 dev alloc
include("viewport/map_cuda_test.jl")      # OV.map_cuda linear-mode FFI + wait-event
include("viewport/gpu_blit_test.jl")      # GPU present driven on the REAL render task
include("viewport/replace_scene_test.jl") # hybrid embed + usdplot + rotated target + recording

# --- picking: native AOV pick → Makie pick protocol + selection outline ---------------------
include("picking/pick_test.jl")     # path2plot lockstep; pick/pick_closest/pick_sorted + y-flip pin
include("picking/resolver_test.jl") # path-resolver cache invalidation on composition change
include("picking/outline_test.jl")  # select!/clear_selection! outline pixels; flag-gated warn
include("picking/attach_test.jl")   # attach_picking! click→hit flow; refusal + detach idempotency

# --- volumes: UsdVol / NanoVDB through NVIDIA IndeX (env-gated, skip-if-libs-absent) --------
include("volumes/index_config_synth_test.jl") # carb-config synthesis (JSON escape, app anchoring)
include("volumes/index_config_test.jl")       # _ensure_index enable/idempotent; disabled-path guards
include("volumes/render_test.jl")             # real .vdb (OpenVDB) renders via IndeX Direct
include("volumes/plot_test.jl")               # volume! recipe: orientation, temp lifecycle, gray tripwire
include("volumes/live_test.jl")               # live data reload; empty→fill; reload-failure recovery

# --- authoring: USD string hygiene + emitter goldens ----------------------------------------
include("authoring/usd_hygiene_test.jl") # identifier/asset-path guards; volume/material goldens
include("authoring/alloc_test.jl")       # flat-emitter byte-identity + allocation bound
