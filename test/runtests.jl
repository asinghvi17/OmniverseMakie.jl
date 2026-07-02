using Test

# Set OVRTX_LIBRARY_PATH before any `using LibOVRTX` so __init__ can dlopen it.
# The package itself does NOT hardcode this path; only the test harness sets the default.
get!(ENV, "OVRTX_LIBRARY_PATH",
    "/home/juliahub/temp/omniverse-makie/references/ovrtx/examples/python/minimal/.venv/lib/python3.13/site-packages/ovrtx/bin/libovrtx-dynamic.so")

@testset "M0.1 workspace loads" begin
    # Both packages must import without error (stubs at this point).
    @test (using LibOVRTX; true)
    @test (using OmniverseMakie; true)
end

include("libovrtx_struct_test.jl")
include("libovrtx_load_test.jl")
include("m0_signals_test.jl")
include("m0_renderer_test.jl")
include("m0_render_test.jl")
include("m0_update_test.jl")

# Shared subprocess runner (M1+)
include("helpers.jl")
include("m1_screen_test.jl")
include("m1_usd_test.jl")
include("m1_camera_test.jl")
include("m1_lights_test.jl")
include("m1_mesh_render_test.jl")
include("m1_orientation_test.jl")
include("m1_save_record_test.jl")
include("m1_primitives_test.jl")

# M2.1 — open-stage Screen + live camera/lights + imperative insert!
include("m2_openstage_test.jl")
include("m2_insert_test.jl")
include("m2_rendercfg_test.jl")

# M2.2 — :ovrtx_renderobject diff node + push_to_ovrtx! minimal edits
include("m2_diffnode_test.jl")

# M2.3 — USD subscene grouping (def Scope hierarchy mirroring the Makie scene tree)
include("m2_subscene_test.jl")

# M2.4 — persistent hot-path bindings (map_attribute / bind_array_attribute)
include("m2_binding_test.jl")

# M2.5 — leak-free delete! / delete!(scene) / empty! teardown
include("m2_delete_test.jl")

# M2.6 — hot-path throughput benchmark + gate (≥30 Hz for 10^4 xforms OR 10^5 points)
include("m2_bench_test.jl")

# M3.1 — OmniPBR material authoring (usda_omnipbr_material/looks_scope_usda) +
# runtime material:binding (OV.bind_material!), validated metallic-vs-diffuse render.
include("m3_material_test.jl")

# M3.3 — image textures (color=img → diffuse_texture + st UV primvar) + *_texture maps.
include("m3_texture_test.jl")

# M3.4 — live material edits (plot.color[]/plot.material[]) on the pre-authored OmniPBR
# material via the M2 diff path (write_shader_input! on Mat_<id>/Shader, ROOT_OPENS==1).
include("m3_material_live_test.jl")

# M4 follow-up — numeric color + colormap on scatter/lines/per-vertex-mesh
# (_scaled_to_display maps the numeric :scaled_color through the plot colormap).
include("m4_colormap_test.jl")

# M4 follow-up — image textures on surface! (_surface_texcoords emits the grid st UVs).
include("m4_surface_texture_test.jl")

# M4 follow-up — true refractive glass via OmniGlass (material=(; glass=true, …)).
include("m4_glass_test.jl")

# M5.1 — bounded ovrtx step! (timeout_ns kwarg on enqueue_wait/step!).
include("m5_bounded_step_test.jl")

# M5.2 — cpu_blit! orientation (subprocess, offscreen GL).
include("m5_blit_test.jl")

# M5.3 — interactive_display shows non-black RTX frame in a real GLMakie window (subprocess).
include("m5_viewport_test.jl")

# M5.3 — live camera loop: on_render_tick! syncs camera/lights/plots, resets RT2 on change,
# accumulates when idle, and blits the bounded-step frame (subprocess).
include("m5_camera_loop_test.jl")

# M5 Task 4 — orbit-forwarding: glscene input events drive Camera3D via input_listeners.
include("m5_orbit_test.jl")

# M6.A Task 1 — GLMakie package extension: using OmniverseMakie alone must NOT load
# GLMakie/CUDA; interactive_display errors helpfully until GLMakie is loaded.
include("m6_ext_load_test.jl")

# M6.A Task 2 — host tonemap unit tests (ACES + sRGB + exposure, pure math, no GPU).
include("m6_tonemap_test.jl")

# Track-C C2 — CPU present!(::Val{:cpu}) is ONE fused tonemap+orient pass writing the cached
# session buffer in place (zero steady-state display garbage): pixel-equality vs the old
# reverse(permutedims(tonemap_frame)) chain + a per-tick @allocated gate (subprocess, GL).
include("c2_present_test.jl")

# Track-C C3 — GPU present!(::Val{:gpu}) fuses tonemap+orient into ONE device kernel writing the
# cached [W,H] oriented buffer (was kernel + permutedims + reverse = 3 kernels / 3 device allocs)
# and caches the resolved GL Texture: the oriented kernel byte-equals the host orient chain / the
# C2 twin, and a warmed GPU present! allocates ZERO device memory per tick (subprocess, CUDA+GL).
include("c3_present_test.jl")

# M6.A Task 3 — OV.map_cuda: map HdrColor as LINEAR CUDA device memory (mode 2),
# returning RAW handles (CUdeviceptr + dims + map_handle + wait_event) with no
# CUDA.jl dep in the main module (subprocess, CUDA).
include("m6_map_cuda_test.jl")

# M6.A Task 4 — GPU-direct present!: device tonemap kernel (host-vs-kernel agreement is in
# m6_tonemap_test.jl) + the CUDA→GL on-device blit (interactive_display gpu_direct=true
# shows a non-black RTX frame via the :gpu path, no CPU roundtrip) (subprocess, CUDA+GL).
include("m6_gpu_blit_test.jl")

# M6.A Task 5 — GPU-direct vs CPU blit benchmark + gate: times present! at 800×600 and 4K,
# asserts GPU-direct strictly < CPU at 4K (the on-device tonemap+copy beats the CPU host
# roundtrip + host tonemap).  Manual present! is the sole driver on one task/context
# (render loop stopped) to avoid the render-task/main-task interop race (subprocess, CUDA+GL).
include("m6_bench_test.jl")

# M6.B Task 1 — ovrtx pick-query FFI: enqueue_pick_query → step! → read_pick_hit →
# path_resolver/resolve_prim_path resolve the center-pixel hit to the mesh plot's
# authored prim path (CPU-only map, no GLMakie/CUDA) (subprocess).
include("m6b_pick_ffi_test.jl")

# M6.B Task 2 — Screen.path2plot reverse map (prim_path => objectid(plot)) kept in
# strict lockstep with plot2robj: populated at every insert, cleared at delete (subprocess).
include("m6b_pick_test.jl")

# M6.B Task 4 — select! / clear_selection! selection-outline API: select! a plot on a
# selection_outline=true Screen draws an orange outline (NEW orange pixels gained vs baseline),
# clear_selection! removes most of it; on a non-outline Screen select! warns once + no-ops.
include("m6b_outline_test.jl")

# M6.B Task 5 — attach_picking! attachable interaction on the live viewport: a click runs a
# native pick → selected[] Observable + on_hit callback (data flow driven synchronously via
# _pick_at!); interactive_display(selection_outline=true) refuses (LdrColor live-present deferred);
# attach_picking!(outline=true) degrades on the HdrColor viewport (subprocess, GLMakie).
include("m6b_attach_test.jl")

# Volumes M1 Task 1 — NVIDIA IndeX enablement (the cracked carb-token init): _ensure_index
# OncePerProcess carb-config injection.  Enabled path (env-gated, skip-if-libs-absent) +
# disabled path (always-run zero-overhead / no-regression guard) (subprocess).
include("volumes_index_config_test.jl")

# Volumes M1 Task 2 — author_vdb_volume! (UsdVol + OpenVDBAsset + IndeX Colormap material):
# _vdb_volume_usda snippet shape (pure) + clear-error when IndeX disabled (subprocess).
include("volumes_usda_test.jl")

# Volumes M1 Task 3 — end-to-end VDB render: author torus.vdb into a Screen, render via RT2 →
# IndeX Direct, assert a non-black volume + IndeX enabled (subprocess, env-gated, skip-if-absent).
include("volumes_render_test.jl")

# Volumes M2 Task 1 — a WRITTEN .nvdb renders: build a dense ball → .nvdb via
# NanoVDBWriter.save_nanovdb (Codec::NONE, major-32), author it with author_vdb_volume! and
# render via RT2 → IndeX Direct; assert non-black (the writer go/no-go) (subprocess, env-gated).
include("volumes_writer_test.jl")

# Volumes M2 Task 2 — Makie volume!(x,y,z,array) recipe: a graded-blob Volume plot renders through
# the diff-node path (author_usd_prim!(::Volume) → save_nanovdb + _vdb_volume_usda → IndeX Direct);
# asserts non-black, low-octant orientation (centroid below centre), and temp-.nvdb cleanup on close
# (subprocess, env-gated, skip-if-absent).
include("volumes_plot_test.jl")

# Volumes M2 Task 3 — composite colormap COLORS: the verify-or-degrade spike found the IndeX composite
# path (which honors the transfer-function colors) is ABSENT from standalone ovrtx — the composite
# flags/settings are consumed only by the Kit `omni.rtx.index_composite` ext (no .so), and IndeX
# Direct (the sole bundled volume path) ignores the Colormap → GRAYSCALE.  This test is INVERTED:
# it asserts the volume! path renders non-black grayscale (the degrade M2 ships) and documents
# COLORED=false as a tripwire for a future composite-capable build (subprocess, env-gated).
include("volumes_color_test.jl")

# Volumes M2 Task 5 — LIVE volume DATA edits: `plot[4][] = new_array` tracks the `:volume` compute
# output (spike-verified to resolve + fire) and re-renders the new density via a fresh-.nvdb re-write
# + reload (remove_usd! + add_usd_reference! — a filePath write does NOT update; IndeX evicts the
# file).  Asserts the lit-pixel centroid MOVES between two graded diagonally-opposite octants, temp
# files stay bounded (prior deleted each edit, last cleaned on close), and an all-zero volume! no-ops
# cleanly (subprocess, env-gated, skip-if-absent).
include("volumes_live_test.jl")
