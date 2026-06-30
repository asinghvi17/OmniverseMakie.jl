# M6.B — AOV Picking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Makie-style picking — `Makie.pick(scene, screen, xy) → (plot, index)` plus 3D world position and an opt-in ovrtx selection-outline highlight — driven by ovrtx's native ray-query pick API, with the core in the main module (no GLMakie/CUDA) and an opt-in attachable interaction in the GLMakie extension.

**Architecture:** A pick is a **renderer query**: `enqueue_pick_query` before a `step!` yields the synthetic `ovrtx_pick_hit` render var (CPU-only), whose `primPath` resolves (via the path-dictionary) to the exact prim we author (`/World/plot_<objectid(plot)>`); a new `Screen.path2plot` reverse map closes it back to the Makie `Plot`. Surfaced through Makie's standard `pick`/`pick_closest`/`pick_sorted`. The selection outline is a creation-time `ScreenConfig` flag (default off) + per-prim `omni:selectionOutlineGroup` writes. The opt-in `attach_picking!` (GLMakie ext) wires a click → pick → optional highlight.

**Tech Stack:** Julia · the `LibOVRTX` subpackage (`lib/LibOVRTX`) · ovrtx native picking (`ovrtx_enqueue_pick_query` / `OVRTX_RENDER_VAR_PICK_HIT` / `ovrtx_get_path_dictionary` / `ovrtx_set_selection_group_styles`) · Makie's pick interface · GLMakie 0.13.12 (extension only).

## Global Constraints

- **Picking core lives in the MAIN module** (`src/binding/OV.jl`, `src/screen.jl`, `src/settings.jl`) — NO GLMakie/CUDA dependency. Only the attachable interaction (Task 5) is in `ext/OmniverseMakieGLMakieExt.jl`.
- **`ovrtx_pick_hit` maps CPU-only** — always `OVRTX_MAP_DEVICE_TYPE_CPU`; never `map_cuda`. (It is an ovrtx restriction; the payload is tiny, so there is no perf reason to want a device path.)
- **Highlight is OFF by default at every level:** `ScreenConfig.selection_outline` default `false`; `interactive_display(...; selection_outline=false)`; `attach_picking!(...; outline=false)`. Pick **data** works regardless of the flag.
- **The selection-outline flag is creation-time-only** (it is on `ovrtx_create_renderer`'s config). It lives in `ScreenConfig` so `resize_viewport!` (which rebuilds the Screen) preserves it. `attach_picking!(...; outline=true)` on a Screen built without it `@warn`s once and falls back to no-highlight — it does NOT rebuild the Screen.
- **Element-index fidelity:** plot is always exact. Index exact for non-materialized Scatter/MeshScatter (`geometryInstanceId`); Surface exact **iff** a mesh hit exposes a per-face index (VERIFY — else degrade to plot-level); Mesh/Lines/LineSegments/materialized-merged scatter = index `0` (plot-level).
- **Prim naming (already in the codebase):** `plot_prim_path(scene2scope, scene, plot) = "/World[/Scene_<objectid(scene)>]/plot_<objectid(plot)>"` (`src/compute.jl:69`). Forward map `Screen.plot2robj::Dict{UInt64,OvrtxRObj}` keyed by `objectid(plot)`; `OvrtxRObj.prim_path::String`.
- **The C example is the source of truth for the novel FFI:** `references/ovrtx/tests/docs/c/test_picking_selection.cpp` (enqueue→step→read pick-hit→resolve→set-outline) and `references/ovrtx/tests/docs/c/helpers.h` (`docs_resolve_primpath`). Where this plan marks "VERIFY against …", confirm the exact ABI in a REPL before relying on it (a genuine new-dep unknown, as M6.A did for CUDA).
- **Display + lib env for any test/REPL run (export in the same command):** `DISPLAY=:0 XAUTHORITY=/run/user/1000/.mutter-Xwaylandauth.QRQ4Q3 XDG_RUNTIME_DIR=/run/user/1000 OVRTX_LIBRARY_PATH=/home/juliahub/temp/omniverse-makie/references/ovrtx/examples/python/minimal/.venv/lib/python3.13/site-packages/ovrtx/bin/libovrtx-dynamic.so`. (Picking core tests need no GLMakie; Task 5's interaction test does — `test/helpers.jl` already forwards the env + stacks the test project for `using GLMakie`.)
- **Commit trailer:** `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. **Do not push** (commit on a `feat/m6-picking` branch; the user controls merges).

## File Structure

```
src/binding/OV.jl       # MODIFY: pick FFI — enqueue_pick_query, read_pick_hit, resolve_prim_path,
                        #   set_selection_outline_group, set_selection_group_styles; Renderer(; selection_outline,…)
src/settings.jl         # MODIFY: ScreenConfig gains `selection_outline::Bool` (default false)
src/screen.jl           # MODIFY: Screen.path2plot reverse map (insert/clear); Makie.pick/pick_closest/pick_sorted;
                        #   pick_hit; select!/clear_selection!
src/compute.jl          # MODIFY: populate/clear path2plot alongside plot2robj (insert + delete sites)
ext/OmniverseMakieGLMakieExt.jl  # MODIFY: interactive_display(...; selection_outline) ; attach_picking!/detach_picking!
test/m6b_pick_ffi_test.jl        # CREATE: FFI chain — enqueue→step→read→resolve → "/World/plot_…" (subprocess)
test/m6b_pick_test.jl            # CREATE: Makie.pick (plot, index, miss, scatter index) + path2plot (subprocess)
test/m6b_outline_test.jl         # CREATE: select! → render → outline pixels appear (subprocess)
test/m6b_attach_test.jl          # CREATE: attach_picking! click → on_hit + outline (subprocess, GLMakie)
```

---

### Task 1: Pick FFI in `OV.jl`

**Files:**
- Modify: `src/binding/OV.jl`
- Test: `test/m6b_pick_ffi_test.jl`

**Interfaces:**
- Consumes: existing `OV.Renderer`, `OV.StepResult`, `OV.step!(r, product; timeout_ns)`, `OV._find_var(outs, name)`, `OV.Screen` machinery (via the test authoring a scene). LibOVRTX symbols (all already bound): `ovrtx_enqueue_pick_query(instance, desc)` (libovrtx_api.jl:1727), `ovrtx_get_path_dictionary(instance, out)` (:1701), `ovrtx_set_selection_group_styles(instance, group_ids, styles, count)` (:1752), `ovrtx_write_attribute(instance, binding_handle_or_desc, data_array, data_access)` (:1494). Structs/consts: `ovrtx_pick_query_desc_t{render_product_path::ovx_string_t,left,top,right,bottom::Int32,flags::UInt32}` (:896), `ovrtx_render_var_output_t{…,map_handle,num_tensors,tensors::Ptr{ovrtx_render_var_tensor_t},num_params,params::Ptr{ovrtx_render_var_param_t}}` (:977), `ovrtx_render_var_tensor_t{dl::Ptr{DLTensor},name::Ptr{ovx_string_t},…}` (:958), `ovrtx_render_var_param_t` (:699), `ovrtx_selection_group_style_t{outline_color,fill_color::NTuple{4,Cfloat}}` (:922), `path_dictionary_instance_t{vtable::Ptr{path_dictionary_vtable_t},context}` (:405), `path_dictionary_vtable_t{…get_tokens_from_paths,get_strings_from_tokens::Ptr{Cvoid}…}` (:391), `ovx_primpath_t=UInt64` (:365), `ovx_token_t=UInt64` (:363), `ovx_string_t` (:21), and consts `OVRTX_RENDER_VAR_PICK_HIT="ovrtx_pick_hit"` (:2031), `OVRTX_PICK_HIT_MAGIC=0x56505448` (:2037), `OVRTX_PICK_HIT_VERSION=1` (:2039), `OVRTX_ATTR_NAME_SELECTION_OUTLINE_GROUP="omni:selectionOutlineGroup"` (:2027), `OVRTX_PICK_FLAG_*` (:2033-2035), `OVRTX_CONFIG_SELECTION_OUTLINE_ENABLED=5` (:1074), `OVRTX_CONFIG_SELECTION_OUTLINE_WIDTH=0`/`OVRTX_CONFIG_SELECTION_FILL_MODE=1` (:1101-1102).
- Produces:
  - `OV.enqueue_pick_query(r::Renderer, product::AbstractString, (left,top,right,bottom)::NTuple{4,Int}; flags::UInt32=UInt32(0)) -> Nothing`
  - `OV.PickHit` = `NamedTuple{(:primpath_id,:object_type,:instance_id,:world_position,:normal),Tuple{UInt64,UInt32,UInt32,NTuple{3,Float64},NTuple{3,Float32}}}`
  - `OV.read_pick_hit(sr::StepResult) -> Vector{PickHit}` (empty on no hit / magic-version mismatch)
  - `OV.PathResolver` (wraps `path_dictionary_instance_t`) + `OV.path_resolver(r) -> PathResolver` + `OV.resolve_prim_path(pr::PathResolver, id::UInt64) -> String`
  - `OV.set_selection_outline_group!(r, prim_paths::Vector{String}, group_ids::Vector{UInt8}) -> Nothing`
  - `OV.set_selection_group_styles!(r, group_ids::Vector{UInt8}, styles::Vector{LibOVRTX.ovrtx_selection_group_style_t}) -> Nothing`

- [ ] **Step 1: Create the branch.** `git checkout main && git checkout -b feat/m6-picking` (M6.A is merged on main @ `1f622a1`). Confirm `git branch --show-current` prints `feat/m6-picking`.

- [ ] **Step 2: VERIFY the pick FFI chain in a REPL** (before writing code) against `references/ovrtx/tests/docs/c/test_picking_selection.cpp` + `helpers.h`. With the env exported, in a `julia --project=.` REPL: build a `Screen` on a lit scene with a single mesh at a known location (`scene = Scene(size=(128,128)); cam3d!(scene); mesh!(scene, Rect3f(Point3f(-1), Vec3f(2)); color=:red)`; `screen = OmniverseMakie.Screen(scene); OmniverseMakie.author_root_from_scene!(screen, scene; resolution=screen.fb_size)`). Then by hand: fill an `ovrtx_pick_query_desc_t` for the CENTER pixel (`left=64,top=64,right=65,bottom=65`, `render_product_path = LibOVRTX.ovx_string(screen.product)`, `flags=0`), call `LibOVRTX.ovrtx_enqueue_pick_query(screen.renderer.ptr, Ref(desc))`, `sr = OV.step!(...)`, then `LibOVRTX.ovrtx_fetch_results` + `_find_var(outs[], OVRTX_RENDER_VAR_PICK_HIT)` + `ovrtx_map_render_var_output(..., OVRTX_MAP_DEVICE_TYPE_CPU, ...)`. Inspect `ro[].num_params`/`params` and `ro[].num_tensors`/`tensors`. **Record exactly:** how a `ovrtx_render_var_param_t` exposes its name + scalar value (read `magic`/`version`/`hitCount`), and how a tensor's `name` (`Ptr{ovx_string_t}`) decodes to a Julia `String` so you can match `"primPath"`/`"worldPositionM"`/`"worldNormal"`/`"geometryInstanceId"`. If `magic != OVRTX_PICK_HIT_MAGIC` or the var is absent, STOP and report BLOCKED with the actual output.

- [ ] **Step 3: VERIFY path-dictionary resolution in the REPL** against `helpers.h:238-258` (`docs_resolve_primpath`). The resolvers are C `static inline` **vtable dispatches** (`path_dictionary_vtable_t.get_tokens_from_paths` / `.get_strings_from_tokens`, both `Ptr{Cvoid}`), NOT exported symbols — call them through the function pointers. Get the dictionary: `pd = Ref{LibOVRTX.path_dictionary_instance_t}(); LibOVRTX.check(LibOVRTX.ovrtx_get_path_dictionary(r.ptr, pd), "get_path_dictionary")`. Load the vtable: `vt = unsafe_load(pd[].vtable)`. Then, for one `primPath` id from Step 2: call `vt.get_tokens_from_paths` via `@ccall $(vt.get_tokens_from_paths)(pd[].context::Ptr{Cvoid}, Ref(id)::Ptr{UInt64}, 1::Csize_t, token_buf::Ptr{UInt64}, 64::Csize_t, out_tokens::Ptr{Ptr{UInt64}}, out_ntok::Ptr{Csize_t}, out_nproc::Ptr{Csize_t})::LibOVRTX.ovx_api_result_t`, then per token `vt.get_strings_from_tokens(pd.context, &tok, 1, &str::ovx_string_t)`, joining the `ovx_string_t` pieces with `/`. **Record the exact ccall signatures** (arg order/types) that resolve `id` to `"/World/plot_<id>"` matching the prim you authored. If the vtable call segfaults or returns `OVX_API_ERROR`, STOP and report BLOCKED with what you tried (this is THE genuinely-novel FFI of M6.B).

- [ ] **Step 4: Write the failing FFI test** (`test/m6b_pick_ffi_test.jl`). It authors a known scene, picks the center pixel (which is on the mesh), and asserts the resolved prim path is the mesh plot's prim path.

```julia
using Test
const _M6B_FFI_PROG = """
using OmniverseMakie
const OV = OmniverseMakie.OV
OM = OmniverseMakie
OM.activate!(warmup = 16)
scene = Scene(size=(128,128)); cam3d!(scene)
p = mesh!(scene, Rect3f(Point3f(-1), Vec3f(2)); color = :red)
screen = OM.Screen(scene)
OM.author_root_from_scene!(screen, scene; resolution = screen.fb_size)
# Warm a few frames so geometry is resident, then pick the center pixel.
for _ in 1:8; sr0 = OV.step!(screen.renderer, screen.product; timeout_ns = UInt64(60_000_000_000)); close(sr0); end
OV.enqueue_pick_query(screen.renderer, screen.product, (64, 64, 65, 65))
sr = OV.step!(screen.renderer, screen.product; timeout_ns = UInt64(60_000_000_000))
hits = OV.read_pick_hit(sr); close(sr)
println("HITCOUNT=", length(hits))
if !isempty(hits)
    pr = OV.path_resolver(screen.renderer)
    path = OV.resolve_prim_path(pr, hits[1].primpath_id)
    expected = screen.plot2robj[objectid(p)].prim_path
    println("PICK_PATH=", path)
    println("EXPECTED=", expected)
    println("PATH_MATCH=", path == expected)
    wp = hits[1].world_position
    println("WORLDPOS_FINITE=", all(isfinite, wp))
end
close(screen)
println("OK_PICK_FFI")
"""
include("helpers.jl")
@testset "M6.B pick FFI chain (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M6B_FFI_PROG; timeout = 400)
    @info "M6.B pick FFI output" output
    @test exitcode == 0
    @test contains(output, "OK_PICK_FFI")
    m = match(r"HITCOUNT=(\d+)", output)
    @test m !== nothing && parse(Int, m.captures[1]) >= 1
    @test contains(output, "PATH_MATCH=true")
    @test contains(output, "WORLDPOS_FINITE=true")
end
```

- [ ] **Step 5: Run it — RED.** Register `m6b_pick_ffi_test.jl` in `test/runtests.jl`. Run `julia --project=. -e 'using Test; include("test/helpers.jl"); include("test/m6b_pick_ffi_test.jl")'` (env exported) → fails (`enqueue_pick_query` undefined).

- [ ] **Step 6: Implement `enqueue_pick_query`** (`src/binding/OV.jl`), mirroring how `step!`/`map_cpu` build descriptors:

```julia
"""Enqueue a pick query for the NEXT step on `product`. Pixel rect: left/top inclusive, right/bottom exclusive."""
function enqueue_pick_query(r::Renderer, product::AbstractString, rect::NTuple{4,Int}; flags::UInt32 = UInt32(0))
    r.alive || error("enqueue_pick_query on a closed Renderer")
    desc = Ref(LibOVRTX.ovrtx_pick_query_desc_t(LibOVRTX.ovx_string(product),
              Int32(rect[1]), Int32(rect[2]), Int32(rect[3]), Int32(rect[4]), flags))
    LibOVRTX.check(LibOVRTX.ovrtx_enqueue_pick_query(r.ptr, desc), "enqueue_pick_query")
    return nothing
end
```

- [ ] **Step 7: Implement `read_pick_hit`** (`src/binding/OV.jl`), using the param/tensor decode confirmed in Step 2. Map CPU, validate magic/version, read `hitCount`, gather the named tensors. Skeleton (fill the `_param_u32`/`_tensor_by_name`/row-read helpers from Step 2's findings):

```julia
const PickHit = NamedTuple{(:primpath_id,:object_type,:instance_id,:world_position,:normal),
                           Tuple{UInt64,UInt32,UInt32,NTuple{3,Float64},NTuple{3,Float32}}}

function read_pick_hit(sr::StepResult)::Vector{PickHit}
    sr.r.alive || error("read_pick_hit on a closed Renderer")
    outs = Ref{LibOVRTX.ovrtx_render_product_set_outputs_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_fetch_results(sr.r.ptr, sr.handle, LibOVRTX.OVRTX_TIMEOUT_INFINITE, outs), "fetch_results")
    h = _find_var(outs[], LibOVRTX.OVRTX_RENDER_VAR_PICK_HIT)
    h === nothing && return PickHit[]                      # no pick was enqueued
    mdesc = Ref(LibOVRTX.ovrtx_map_output_description_t(LibOVRTX.OVRTX_MAP_DEVICE_TYPE_CPU, Csize_t(0)))
    ro = Ref{LibOVRTX.ovrtx_render_var_output_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_map_render_var_output(sr.r.ptr, h, mdesc, LibOVRTX.OVRTX_TIMEOUT_INFINITE, ro), "map_render_var_output(pick_hit)")
    hits = PickHit[]
    try
        magic   = _pick_param_u32(ro[], "magic")          # ← Step 2 helper (read a uint32 param by name)
        version = _pick_param_u32(ro[], "version")
        (magic == LibOVRTX.OVRTX_PICK_HIT_MAGIC && version == LibOVRTX.OVRTX_PICK_HIT_VERSION) || return PickHit[]
        n = Int(_pick_param_u32(ro[], "hitCount"))
        prim = _pick_tensor(ro[], "primPath")             # ← Step 2 helpers (tensor-by-name → typed view)
        inst = _pick_tensor(ro[], "geometryInstanceId")
        otyp = _pick_tensor(ro[], "objectType")
        wpos = _pick_tensor(ro[], "worldPositionM")       # [h,3] Float64
        wnrm = _pick_tensor(ro[], "worldNormal")          # [h,3] Float32
        for i in 1:n
            push!(hits, (primpath_id = _u64(prim, i), object_type = _u32(otyp, i), instance_id = _u32(inst, i),
                         world_position = (_f64(wpos,i,1), _f64(wpos,i,2), _f64(wpos,i,3)),
                         normal = (_f32(wnrm,i,1), _f32(wnrm,i,2), _f32(wnrm,i,3))))
        end
    finally
        LibOVRTX.ovrtx_unmap_render_var_output(sr.r.ptr, ro[].map_handle, LibOVRTX.NOSYNC)
    end
    return hits
end
```
Define `_find_var` to RETURN `nothing` when the name is absent if it currently errors (check `src/binding/OV.jl:139`); add a `_find_var_opt` returning `nothing` rather than changing the throwing `_find_var` if other callers depend on the throw.

- [ ] **Step 8: Implement `path_resolver` + `resolve_prim_path`** (`src/binding/OV.jl`), using the vtable ccalls confirmed in Step 3:

```julia
struct PathResolver
    pd::Base.RefValue{LibOVRTX.path_dictionary_instance_t}
    vt::LibOVRTX.path_dictionary_vtable_t
end
function path_resolver(r::Renderer)
    pd = Ref{LibOVRTX.path_dictionary_instance_t}()
    LibOVRTX.check(LibOVRTX.ovrtx_get_path_dictionary(r.ptr, pd), "get_path_dictionary")
    return PathResolver(pd, unsafe_load(pd[].vtable))
end
# Resolve one ovx_primpath_t id → "/World/plot_<id>" via the vtable (Step 3 signatures).
function resolve_prim_path(pr::PathResolver, id::UInt64)::String
    ctx = pr.pd[].context
    token_buf = Vector{UInt64}(undef, 64)
    out_tokens = Ref{Ptr{UInt64}}(C_NULL); out_ntok = Ref{Csize_t}(0); out_nproc = Ref{Csize_t}(0)
    GC.@preserve token_buf begin
        res = @ccall $(pr.vt.get_tokens_from_paths)(ctx::Ptr{Cvoid}, Ref(id)::Ptr{UInt64}, Csize_t(1)::Csize_t,
              pointer(token_buf)::Ptr{UInt64}, Csize_t(64)::Csize_t, out_tokens::Ptr{Ptr{UInt64}},
              out_ntok::Ptr{Csize_t}, out_nproc::Ptr{Csize_t})::LibOVRTX.ovx_api_result_t
        res.status == LibOVRTX.OVX_API_SUCCESS || error("path get_tokens_from_paths: $(LibOVRTX.ovx_to_string(res.error))")
        ntok = Int(out_ntok[]); toks = out_tokens[]
        io = IOBuffer()
        for i in 1:ntok
            s = Ref{LibOVRTX.ovx_string_t}()
            r2 = @ccall $(pr.vt.get_strings_from_tokens)(ctx::Ptr{Cvoid}, (toks + (i-1)*sizeof(UInt64))::Ptr{UInt64},
                  Csize_t(1)::Csize_t, s::Ptr{LibOVRTX.ovx_string_t})::LibOVRTX.ovx_api_result_t
            r2.status == LibOVRTX.OVX_API_SUCCESS || error("path get_strings_from_tokens failed")
            print(io, "/", LibOVRTX.ovx_to_string(s[]))
        end
        return String(take!(io))
    end
end
```
(Use the repo's existing `ovx_string`→`String` helper for `ovx_string_t`; if none exists, decode `ovx_string_t{ptr,length}` via `unsafe_string(s.ptr, s.length)` — VERIFY the field names of `ovx_string_t` at libovrtx_api.jl:21.)

- [ ] **Step 9: Implement the outline writers** (`src/binding/OV.jl`):

```julia
# Write omni:selectionOutlineGroup (uint8) on each prim. group 1 = selected, 0 = cleared.
# VERIFY the ovrtx_write_attribute signature + how to build the per-prim attribute descriptor
# against test_picking_selection.cpp's `ovrtx_set_selection_outline_group` inline helper (it calls
# ovrtx_write_attribute under the hood).  This is an FFI verify step.
function set_selection_outline_group!(r::Renderer, prim_paths::Vector{String}, group_ids::Vector{UInt8})
    # ... per the verified ovrtx_write_attribute(OVRTX_ATTR_NAME_SELECTION_OUTLINE_GROUP, …) call ...
end
function set_selection_group_styles!(r::Renderer, group_ids::Vector{UInt8}, styles::Vector{LibOVRTX.ovrtx_selection_group_style_t})
    GC.@preserve group_ids styles begin
        LibOVRTX.check(LibOVRTX.ovrtx_set_selection_group_styles(r.ptr, pointer(group_ids), pointer(styles), Csize_t(length(group_ids))), "set_selection_group_styles")
    end
end
```

- [ ] **Step 10: Run it — GREEN.** Re-run Step 5's command → `PATH_MATCH=true`, `HITCOUNT>=1`, `WORLDPOS_FINITE=true`, `OK_PICK_FFI`.

- [ ] **Step 11: Commit.**
```bash
git add src/binding/OV.jl test/m6b_pick_ffi_test.jl test/runtests.jl
git commit -m "feat(M6.B): ovrtx pick-query FFI — enqueue, read_pick_hit, resolve_prim_path, outline writers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

**Acceptance:** picking the center pixel of a known mesh returns ≥1 hit whose `primPath` resolves to that plot's authored prim path, with a finite world position.

---

### Task 2: `ScreenConfig.selection_outline` + renderer flag + `Screen.path2plot` reverse map

**Files:**
- Modify: `src/settings.jl`, `src/binding/OV.jl` (Renderer config), `src/screen.jl` (Screen field), `src/compute.jl` (populate/clear)
- Test: `test/m6b_pick_test.jl` (the `path2plot` portion)

**Interfaces:**
- Consumes: `OV.Renderer()` (Task 1 module), `ScreenConfig`, `Screen.plot2robj`, `OvrtxRObj.prim_path`.
- Produces: `ScreenConfig.selection_outline::Bool`; `OV.Renderer(; selection_outline::Bool=false, outline_width::Int=8)`; `Screen.path2plot::Dict{String,UInt64}` populated at every `plot2robj` insert, cleared at delete.

- [ ] **Step 1: Add the `ScreenConfig` field.** In `src/settings.jl:9` add `selection_outline::Bool` (default `false`) to `struct ScreenConfig` and to the keyword constructor / `merge_screen_config` defaults (mirror the existing `mode/samples/warmup/max_bounces` defaults). Seed it in `OmniverseMakie.jl`'s `__init__` theme `Attributes(... selection_outline = false)` so `set_screen_config!` resolves it.

- [ ] **Step 2: Write the failing `path2plot` test** (in `test/m6b_pick_test.jl`):

```julia
using Test
const _M6B_PATH2PLOT_PROG = """
using OmniverseMakie
OM = OmniverseMakie; OM.activate!(warmup = 8)
scene = Scene(size=(96,96)); cam3d!(scene)
p = mesh!(scene, Rect3f(Point3f(0), Vec3f(1)); color = :red)
screen = OM.Screen(scene)
OM.author_root_from_scene!(screen, scene; resolution = screen.fb_size)
prim = screen.plot2robj[objectid(p)].prim_path
println("FWD_OK=", haskey(screen.plot2robj, objectid(p)))
println("REV_OK=", get(screen.path2plot, prim, UInt64(0)) == objectid(p))
delete!(screen, p)
println("REV_CLEARED=", !haskey(screen.path2plot, prim))
close(screen)
println("OK_PATH2PLOT")
"""
include("helpers.jl")
@testset "M6.B path2plot reverse map (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M6B_PATH2PLOT_PROG; timeout = 300)
    @info "M6.B path2plot output" output
    @test exitcode == 0
    @test contains(output, "FWD_OK=true")
    @test contains(output, "REV_OK=true")
    @test contains(output, "REV_CLEARED=true")
end
```
(If `delete!(screen, plot)` is not the exact public delete signature, use the one in `src/screen.jl` — VERIFY the typed `delete!`/`remove` API the M2 leak-free path exposes.)

- [ ] **Step 3: Run it — RED.** `julia --project=. -e 'using Test; include("test/helpers.jl"); include("test/m6b_pick_test.jl")'` → fails (`screen.path2plot` undefined).

- [ ] **Step 4: Add `path2plot` to `Screen`.** In `src/screen.jl` add field `path2plot::Dict{String,UInt64}` to `mutable struct Screen` and initialize `Dict{String,UInt64}()` in every `Screen(...)` constructor (mirror `plot2robj`/`scene2scope` init).

- [ ] **Step 5: Populate/clear `path2plot`.** In `src/compute.jl`, at each site that sets `screen.plot2robj[objectid(plot)] = built` (≈ `:668`) and `= robj` (≈ `:626`), also set `screen.path2plot[built.prim_path] = objectid(plot)` (resp. `robj.prim_path`). In the typed `delete!`/`empty!` teardown (where `plot2robj` entries are dropped), also `delete!(screen.path2plot, robj.prim_path)` for the removed plot (and `empty!(screen.path2plot)` alongside `empty!(screen.plot2robj)`). Keep the two maps strictly in lockstep.

- [ ] **Step 6: Thread `selection_outline` into renderer creation.** Change `OV.Renderer()` (`src/binding/OV.jl:18`) to `OV.Renderer(; selection_outline::Bool=false, outline_width::Int=8)`. When `selection_outline`, build a non-empty `ovrtx_config_t` with two entries — `OVRTX_CONFIG_SELECTION_OUTLINE_ENABLED=true` (bool) and `OVRTX_CONFIG_SELECTION_OUTLINE_WIDTH=outline_width` (int64). **VERIFY building `ovrtx_config_entry_t` in a REPL** (it is an opaque 24-byte struct with `key_type` @0, `key` @4, `value` @8 via `Base.setproperty!` on a `Ptr`, libovrtx_api.jl:1204-1232): allocate `entries = Vector{ovrtx_config_entry_t}(undef, 2)`, set each via a `Ptr` to its slot (`p = pointer(entries, i)`: `p.key_type = OVRTX_CONFIG_KEY_TYPE_BOOL`; `p.key = …(OVRTX_CONFIG_SELECTION_OUTLINE_ENABLED)`; `p.value = …(true)`), then `cfg = Ref(ovrtx_config_t(pointer(entries), 2))`, `GC.@preserve entries` across `ovrtx_create_renderer`. Confirm a renderer created this way reports no error and renders. Have `Screen(scene, config::ScreenConfig)` pass `selection_outline = config.selection_outline` to `Renderer(...)` (find where `Renderer()` is constructed for the Screen and thread it).

- [ ] **Step 7: Run it — GREEN.** Re-run Step 3 → `FWD_OK`/`REV_OK`/`REV_CLEARED` true. Also confirm a `Screen` built from a `ScreenConfig` with `selection_outline=true` constructs without error: `julia --project=. -e 'using OmniverseMakie; OmniverseMakie.activate!(selection_outline=true); s=OmniverseMakie.Screen(Scene(size=(64,64))); println("OUTLINE_SCREEN_OK=", OmniverseMakie.isopen(s)); close(s)'` (env exported) → `OUTLINE_SCREEN_OK=true`.

- [ ] **Step 8: Commit.**
```bash
git add src/settings.jl src/binding/OV.jl src/screen.jl src/compute.jl src/OmniverseMakie.jl test/m6b_pick_test.jl test/runtests.jl
git commit -m "feat(M6.B): ScreenConfig.selection_outline + renderer flag + Screen.path2plot reverse map

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

**Acceptance:** `path2plot` mirrors `plot2robj` (populated at insert, cleared at delete); a Screen built with `selection_outline=true` constructs and renders.

---

### Task 3: `Makie.pick` / `pick_closest` / `pick_sorted` + `pick_hit` + element index

**Files:**
- Modify: `src/screen.jl`
- Test: `test/m6b_pick_test.jl` (the `Makie.pick` testsets)

**Interfaces:**
- Consumes: `OV.enqueue_pick_query`, `OV.read_pick_hit`, `OV.path_resolver`/`resolve_prim_path` (Task 1); `Screen.path2plot`, `Screen.plot2robj` (Task 2); `Screen.renderer`/`product`/`fb_size`/`scene`.
- Produces:
  - `OmniverseMakie.pick_hit(screen::Screen, xy) -> Union{Nothing, NamedTuple{(:plot,:index,:world_position,:normal),…}}`
  - `Makie.pick(scene::Makie.Scene, screen::Screen, xy::Vec{2,Float64}) -> Tuple{Union{Nothing,Makie.AbstractPlot},Int}`
  - `Makie.pick_closest(scene, screen::Screen, xy, range) -> (plot, index)`
  - `Makie.pick_sorted(scene, screen::Screen, xy, range) -> Vector{Tuple{plot,index}}`
  - `OmniverseMakie._element_index(plot, instance_id::UInt32)::Int`

- [ ] **Step 1: Write the failing pick test** (append to `test/m6b_pick_test.jl`): a scatter at known points + a mesh; pick a pixel over a specific scatter point → correct plot + index; pick empty space → `(nothing, 0)`.

```julia
const _M6B_PICK_PROG = """
using OmniverseMakie
OM = OmniverseMakie; OM.activate!(warmup = 24)
using OmniverseMakie: pick
scene = Scene(size=(200,200)); cam3d!(scene)
# A single big centered marker so the center pixel lands on point index 1.
sp = scatter!(scene, [Point3f(0,0,0)]; markersize = 60, color = :red)
screen = OM.Screen(scene)
OM.author_root_from_scene!(screen, scene; resolution = screen.fb_size)
for _ in 1:8; sr0 = OM.OV.step!(screen.renderer, screen.product; timeout_ns=UInt64(60_000_000_000)); close(sr0); end
plt, idx = Makie.pick(scene, screen, Vec2(100.0, 100.0))
println("HIT_IS_SCATTER=", plt === sp)
println("HIT_INDEX=", idx)
miss = Makie.pick(scene, screen, Vec2(2.0, 2.0))     # corner → background
println("MISS=", miss == (nothing, 0))
close(screen)
println("OK_PICK")
"""
@testset "M6.B Makie.pick (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M6B_PICK_PROG; timeout = 400)
    @info "M6.B pick output" output
    @test exitcode == 0
    @test contains(output, "HIT_IS_SCATTER=true")
    @test contains(output, "HIT_INDEX=1")
    @test contains(output, "MISS=true")
end
```

- [ ] **Step 2: Run it — RED.** Re-run the `m6b_pick_test.jl` command → fails (no `Makie.pick` method for `Screen`).

- [ ] **Step 3: VERIFY the coordinate convention in a REPL.** Makie `xy` is bottom-left-origin Float64 pixels over `screen.fb_size`; the pick desc is top-left-inclusive Int RenderProduct pixels. Pick a pixel known to be on a marker offset from center (e.g. a scatter point you place in the TOP half of the screen) and confirm whether `py = H - round(Int, xy[2])` (flip) or `py = round(Int, xy[2])` (no flip) returns the hit. **Record the exact mapping** `to_ovrtx_pixels(xy, (W,H))`.

- [ ] **Step 4: Implement `pick_hit` + the element-index helper + the Makie overrides** (`src/screen.jl`):

```julia
# Map a Makie bottom-left float pixel to an ovrtx top-left integer RenderProduct pixel (Step 3).
function _to_ovrtx_pixel(xy, fb_size)
    W, H = fb_size
    px = clamp(round(Int, xy[1]), 0, W - 1)
    py = clamp(H - round(Int, xy[2]), 0, H - 1)     # ← use the flip confirmed in Step 3
    return (px, py)
end

# instance_id → Makie element index. Scatter/MeshScatter PointInstancer: 0-based instance → 1-based point.
# Other kinds (Mesh/Lines/merged scatter): plot-level (index 0). Surface: see VERIFY note in the spec.
function _element_index(plot, instance_id::UInt32)::Int
    plot isa Makie.Scatter || plot isa Makie.MeshScatter ? Int(instance_id) + 1 : 0
end

function pick_hit(screen::Screen, xy)
    px, py = _to_ovrtx_pixel(xy, screen.fb_size)
    OV.enqueue_pick_query(screen.renderer, screen.product, (px, py, px + 1, py + 1))
    sr = OV.step!(screen.renderer, screen.product; timeout_ns = _PICK_TIMEOUT_NS)
    hits = try OV.read_pick_hit(sr) finally close(sr) end
    isempty(hits) && return nothing
    pr = path_resolver_for(screen)                  # cache the PathResolver on the Screen (lazy)
    path = OV.resolve_prim_path(pr, hits[1].primpath_id)
    oid = get(screen.path2plot, path, nothing)
    oid === nothing && return nothing               # camera/light/looks prim, not a plot
    plot = _plot_for_objectid(screen, oid)          # objectid → the Plot object (see registry note below)
    plot === nothing && return nothing
    return (plot = plot, index = _element_index(plot, hits[1].instance_id),
            world_position = hits[1].world_position, normal = hits[1].normal)
end

function Makie.pick(::Makie.Scene, screen::Screen, xy::Makie.Vec{2,Float64})
    h = pick_hit(screen, xy)
    h === nothing ? (nothing, 0) : (h.plot, h.index)
end
Makie.pick_closest(scene::Makie.Scene, screen::Screen, xy, range) = Makie.pick(scene, screen, Makie.Vec{2,Float64}(xy))
Makie.pick_sorted(scene::Makie.Scene, screen::Screen, xy, range) =
    (ph = pick_hit(screen, Makie.Vec{2,Float64}(xy)); ph === nothing ? Tuple{Makie.AbstractPlot,Int}[] : [(ph.plot, ph.index)])
```
`path2plot` gives `objectid(plot)` but not the `Plot` object — add a small `_plot_for_objectid`: keep the `Plot` reference alongside `OvrtxRObj` (add a `plot` field, or a `Dict{UInt64,Makie.AbstractPlot}` populated next to `plot2robj`). Wire `path_resolver_for(screen)` to build + cache an `OV.PathResolver` once per Screen (the dictionary is stable for the renderer's life). Define `const _PICK_TIMEOUT_NS = UInt64(10_000_000_000)`.

- [ ] **Step 5: Run it — GREEN.** Re-run Step 1's command → `HIT_IS_SCATTER=true`, `HIT_INDEX=1`, `MISS=true`. After a pick, the viewport's RT2 accumulation resets — that is expected (the pick consumed a step); no assertion needed here.

- [ ] **Step 6: VERIFY surface index (spec follow-up gate).** In a REPL, author a `surface!` and pick a cell away from center; check whether `hits[1].instance_id` (or another hit field) carries a per-face index. If YES, extend `_element_index` to map it to the linear `(i-1)*ny + j` (see `src/translation/primitives.jl:298` `lin = (i,j) -> (i-1)*ny + j`); if NO, leave surface at plot-level and note it in the report (per the spec's verify-or-degrade decision). Do NOT block the task on full surface fidelity.

- [ ] **Step 7: Commit.**
```bash
git add src/screen.jl src/compute.jl test/m6b_pick_test.jl
git commit -m "feat(M6.B): Makie.pick/pick_closest/pick_sorted + pick_hit + element index

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

**Acceptance:** `Makie.pick` returns the correct plot + scatter point index over a marker and `(nothing, 0)` over background; `pick_closest`/`pick_sorted` work (so `DataInspector` composes).

---

### Task 4: `select!` / `clear_selection!` outline API + outline render test

**Files:**
- Modify: `src/screen.jl`
- Test: `test/m6b_outline_test.jl`

**Interfaces:**
- Consumes: `OV.set_selection_outline_group!`, `OV.set_selection_group_styles!` (Task 1); `Screen.plot2robj` (`prim_path`), `Screen.config.selection_outline` (Task 2); `OV.reset!`, `OV.render_to_matrix`.
- Produces: `OmniverseMakie.select!(screen::Screen, plot; group::UInt8=0x01)`, `OmniverseMakie.clear_selection!(screen::Screen[, plot])`, with a default high-contrast outline style installed when `selection_outline` is on.

- [ ] **Step 1: Write the failing outline test** (`test/m6b_outline_test.jl`): a Screen with `selection_outline=true`; render once (baseline), `select!` a plot, render again, assert outline-colored (orange) pixels appeared that were not in the baseline.

```julia
using Test
const _M6B_OUTLINE_PROG = """
using OmniverseMakie
OM = OmniverseMakie; OM.activate!(warmup = 24, selection_outline = true)
scene = Scene(size=(160,160)); cam3d!(scene)
p = mesh!(scene, Rect3f(Point3f(-1), Vec3f(2)); color = :gray)
screen = OM.Screen(scene)
OM.author_root_from_scene!(screen, scene; resolution = screen.fb_size)
base = OM.OV.render_to_matrix(screen.renderer, screen.product; warmup = 24)
OM.select!(screen, p)                       # orange outline, group 1
OM.OV.reset!(screen.renderer)
sel  = OM.OV.render_to_matrix(screen.renderer, screen.product; warmup = 24)
# Count strongly-orange pixels (R high, G mid, B low) gained vs baseline.
isorange(c) = Float32(c.r) > 0.6 && 0.2 < Float32(c.g) < 0.75 && Float32(c.b) < 0.3
gained = count(i -> isorange(sel[i]) && !isorange(base[i]), eachindex(sel))
println("OUTLINE_GAINED=", gained)
OM.clear_selection!(screen, p)
OM.OV.reset!(screen.renderer)
cleared = OM.OV.render_to_matrix(screen.renderer, screen.product; warmup = 24)
println("OUTLINE_AFTER_CLEAR=", count(i -> isorange(cleared[i]) && !isorange(base[i]), eachindex(cleared)))
close(screen)
println("OK_OUTLINE")
"""
include("helpers.jl")
@testset "M6.B selection outline (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M6B_OUTLINE_PROG; timeout = 400)
    @info "M6.B outline output" output
    @test exitcode == 0
    @test contains(output, "OK_OUTLINE")
    g = match(r"OUTLINE_GAINED=(\d+)", output)
    @test g !== nothing && parse(Int, g.captures[1]) > 50         # an outline appeared
    c = match(r"OUTLINE_AFTER_CLEAR=(\d+)", output)
    @test c !== nothing && parse(Int, c.captures[1]) < parse(Int, g.captures[1]) ÷ 2   # clearing removed most of it
end
```

- [ ] **Step 2: Run it — RED.** `julia --project=. -e 'using Test; include("test/helpers.jl"); include("test/m6b_outline_test.jl")'` → fails (`select!` undefined).

- [ ] **Step 3: Implement `select!` / `clear_selection!`** (`src/screen.jl`):

```julia
const _OUTLINE_ORANGE = (1.0f0, 0.6f0, 0.0f0, 1.0f0)

function _ensure_outline_style!(screen::Screen)
    screen.config.selection_outline || return false
    if !screen._outline_styled                      # one-time per Screen
        OV.set_selection_group_styles!(screen.renderer, UInt8[0x01],
            [LibOVRTX.ovrtx_selection_group_style_t(_OUTLINE_ORANGE, (0f0,0f0,0f0,0f0))])
        screen._outline_styled = true
    end
    return true
end

function select!(screen::Screen, plot; group::UInt8 = 0x01)
    if !_ensure_outline_style!(screen)
        @warn "select!: this Screen was built without selection_outline=true; no highlight drawn" maxlog=1
        return nothing
    end
    robj = get(screen.plot2robj, objectid(plot), nothing)
    robj === nothing && return nothing
    OV.set_selection_outline_group!(screen.renderer, [robj.prim_path], UInt8[group])
    return nothing
end

function clear_selection!(screen::Screen, plot)
    robj = get(screen.plot2robj, objectid(plot), nothing)
    robj === nothing && return nothing
    OV.set_selection_outline_group!(screen.renderer, [robj.prim_path], UInt8[0x00])
    return nothing
end
clear_selection!(screen::Screen) =                # clear ALL currently-tracked plots
    for robj in values(screen.plot2robj)
        OV.set_selection_outline_group!(screen.renderer, [robj.prim_path], UInt8[0x00])
    end
```
Add a `_outline_styled::Bool` field to `Screen` (default `false`), initialized in the constructors.

- [ ] **Step 4: Run it — GREEN.** Re-run Step 2's command → `OUTLINE_GAINED>50`, clearing removes most of it, `OK_OUTLINE`. If the orange thresholds are off (the outline AA blends), widen `isorange` slightly — but keep "gained vs baseline" so the test proves the outline is NEW, not pre-existing geometry color.

- [ ] **Step 5: Commit.**
```bash
git add src/screen.jl test/m6b_outline_test.jl test/runtests.jl
git commit -m "feat(M6.B): select!/clear_selection! selection-outline API + render test

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

**Acceptance:** on a `selection_outline=true` Screen, `select!(plot)` makes an orange outline appear in the render; `clear_selection!` removes it; on a non-outline Screen `select!` warns once and is a no-op.

---

### Task 5: GLMakie-ext `attach_picking!` attachable interaction + viewport kwarg

**Files:**
- Modify: `ext/OmniverseMakieGLMakieExt.jl`, `src/OmniverseMakie.jl` (declare `attach_picking!`/`detach_picking!` generics; export nothing new by default)
- Test: `test/m6b_attach_test.jl` (subprocess, GLMakie)

**Interfaces:**
- Consumes: `interactive_display` / `ViewportSession` (M5, in the GLMakie ext); `OmniverseMakie.pick_hit`, `select!`, `clear_selection!` (Tasks 3-4); `Screen.config.selection_outline`.
- Produces: `interactive_display(...; selection_outline::Bool=false)` (threads the flag into the viewport's `Screen`); `OmniverseMakie.attach_picking!(session; on_hit=nothing, outline::Bool=false, button=Makie.Mouse.left) -> PickHandle`; `detach_picking!(handle)`; `PickHandle.selected::Observable`.

- [ ] **Step 1: Declare the generics in the main module** (`src/OmniverseMakie.jl`, next to the M6.A generics): `function attach_picking! end` and `function detach_picking! end` (the GLMakie ext adds the methods). Do NOT export them (advanced/opt-in API; tests qualify `OmniverseMakie.attach_picking!`).

- [ ] **Step 2: Thread `selection_outline` into `interactive_display`** (`ext/OmniverseMakieGLMakieExt.jl`). Add the kwarg `selection_outline::Bool=false` to `interactive_display(...)`; build the viewport `Screen` from a `ScreenConfig` carrying it (the viewport currently does `Screen(cam_scene)` — switch to constructing/merging a config with `selection_outline` so the renderer gets the creation-time flag). `resize_viewport!` already rebuilds the Screen from the same config path, so it is preserved.

- [ ] **Step 3: Write the failing attach test** (`test/m6b_attach_test.jl`, subprocess, GLMakie). Drive picking via the pick path directly (the M5 GLMakie background-thread event race makes synthetic mouse events flaky — assert the wiring synchronously, as the M5 orbit test does): construct the viewport with `selection_outline=true`, `attach_picking!(session; outline=true)`, then directly invoke the handler at a known pixel and assert `selected[]` got the right plot and the outline was applied.

```julia
using Test
const _M6B_ATTACH_PROG = """
using OmniverseMakie, GLMakie
OM = OmniverseMakie; OM.activate!(warmup = 16)
fig = Figure(); ax = LScene(fig[1,1]; show_axis=false)
p = mesh!(ax, Rect3f(Point3f(-1), Vec3f(2)); color = :gray)
session = OM.interactive_display(fig; size=(200,200), selection_outline=true)
h = OM.attach_picking!(session; outline=true)
println("ATTACHED=", h !== nothing)
# Invoke the pick handler synchronously at the centre (bypassing the GLFW event thread).
OM._pick_at!(session, h, Vec2(100.0, 100.0))      # test-visible helper the handler also calls
sel = h.selected[]
println("SELECTED_IS_MESH=", sel !== nothing && sel.plot === p)
OM.detach_picking!(h)
OM.close(session)
println("OK_ATTACH")
"""
include("helpers.jl")
@testset "M6.B attach_picking! (subprocess, GLMakie)" begin
    exitcode, output = run_ovrtx_subprocess(_M6B_ATTACH_PROG; timeout = 500)
    @info "M6.B attach output" output
    @test exitcode == 0
    @test contains(output, "ATTACHED=true")
    @test contains(output, "SELECTED_IS_MESH=true")
    @test contains(output, "OK_ATTACH")
end
```

- [ ] **Step 4: Run it — RED.** `julia --project=. -e 'using Test; include("test/helpers.jl"); include("test/m6b_attach_test.jl")'` (env exported) → fails (`attach_picking!` undefined).

- [ ] **Step 5: Implement `attach_picking!` / `detach_picking!`** (`ext/OmniverseMakieGLMakieExt.jl`):

```julia
mutable struct PickHandle
    session
    listener                # the glscene mousebutton listener (or nothing)
    on_hit
    outline::Bool
    selected::Observable{Any}
    last_plot               # currently-outlined plot (or nothing)
end

# The pick action (also called directly by the test). Runs on the calling task; pick is a
# renderer query → safe to run from the event listener on the render task.
function _pick_at!(session, h::PickHandle, xy)
    hit = OmniverseMakie.pick_hit(session.screen, xy)
    if h.outline && session.screen.config.selection_outline
        h.last_plot === nothing || OmniverseMakie.clear_selection!(session.screen, h.last_plot)
        h.last_plot = hit === nothing ? nothing : hit.plot
        hit === nothing || OmniverseMakie.select!(session.screen, hit.plot)
    end
    h.selected[] = hit
    h.on_hit === nothing || h.on_hit(hit)
    return hit
end

function OmniverseMakie.attach_picking!(session; on_hit = nothing, outline::Bool = false, button = Makie.Mouse.left)
    if outline && !session.screen.config.selection_outline
        @warn "attach_picking!(outline=true) but the viewport was built without selection_outline=true; \
               no highlight will be drawn (pass selection_outline=true to interactive_display)" maxlog=1
        outline = false
    end
    h = PickHandle(session, nothing, on_hit, outline, Observable{Any}(nothing), nothing)
    # Click = press+release without drag (so it does not fight the left-drag orbit). Wire on the
    # display scene's mousebutton; read the current mouseposition; forward to _pick_at!.
    h.listener = on(session.glscene.events.mousebutton) do ev
        if ev.button == button && ev.action == Makie.Mouse.release && !_was_dragging(session)
            _pick_at!(session, h, session.glscene.events.mouseposition[])
        end
        return Makie.Consume(false)
    end
    return h
end

function OmniverseMakie.detach_picking!(h::PickHandle)
    h.listener === nothing || off(h.listener)
    h.listener = nothing
    h.last_plot === nothing || OmniverseMakie.clear_selection!(h.session.screen, h.last_plot)
    return nothing
end
```
Implement `_was_dragging(session)` minimally (track press position vs release position over a small threshold, or reuse any existing drag state). Ensure `Base.close(::ViewportSession)` (M5 teardown) also detaches any live `PickHandle` if one is stored on the session — simplest: `attach_picking!` pushes the handle onto a `session`-held list the close path tears down, OR document that the user calls `detach_picking!` before `close` (the listener is on `glscene.events`, which `close` already tears down via the window close — so an explicit detach is belt-and-suspenders). Keep it minimal; do not over-engineer.

- [ ] **Step 6: Run it — GREEN.** Re-run Step 4's command → `ATTACHED=true`, `SELECTED_IS_MESH=true`, `OK_ATTACH`.

- [ ] **Step 7: Run the FULL suite — no regression.** `julia --project=. -e 'using Pkg; Pkg.test()'` (env exported) → "Testing OmniverseMakie tests passed" (all M0–M6.A + the new M6.B tests). Picking adds no GLMakie/CUDA load to the offscreen path, so the M6.A offscreen-purity test stays green.

- [ ] **Step 8: Commit.**
```bash
git add ext/OmniverseMakieGLMakieExt.jl src/OmniverseMakie.jl test/m6b_attach_test.jl test/runtests.jl
git commit -m "feat(M6.B): attach_picking! attachable interaction + interactive_display selection_outline kwarg

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

**Acceptance:** `attach_picking!` (off by default, outline off by default) picks on click and exposes the hit via `selected`/`on_hit`; with `selection_outline=true` it highlights; the full `Pkg.test` is green.

---

## Self-Review (completed)

- **Spec coverage:** pick FFI + path resolution + outline writers (Task 1); `ScreenConfig.selection_outline` + renderer flag + `path2plot` (Task 2); `Makie.pick`/`pick_closest`/`pick_sorted` + element index + y-flip (Task 3); `select!`/`clear_selection!` + outline render (Task 4); attachable interaction + viewport kwarg + full suite (Task 5). The spec's non-goals (per-pixel `pick(rect)` matrix, full Mesh/Lines index, hover-by-default, semantic-segmentation) are correctly ABSENT.
- **Highlight-off-by-default** is enforced at all three levels: `ScreenConfig.selection_outline=false` (Task 2), `interactive_display(...; selection_outline=false)` (Task 5), `attach_picking!(...; outline=false)` (Task 5); `attach_picking!(outline=true)` on a non-outline Screen warns + degrades (Task 5 Step 5), matching `select!`'s guard (Task 4 Step 3).
- **CPU-only pick map** is explicit in `read_pick_hit` (Task 1 Step 7, `OVRTX_MAP_DEVICE_TYPE_CPU`).
- **Placeholder scan:** the genuinely-novel FFI (pick-hit param/tensor decode, the path-dictionary vtable ccalls, building the opaque `ovrtx_config_entry_t`, the `ovrtx_write_attribute` outline write, the coordinate y-flip, the surface face-index availability) are marked **VERIFY in REPL** with exact `file:line` references — real new-dep unknowns handled by explicit verification (the M6.A pattern), not hand-waving. All structural code is complete.
- **Type consistency:** `OV.read_pick_hit -> Vector{PickHit}` (Task 1) consumed by `pick_hit` (Task 3); `pick_hit -> (plot,index,world_position,normal)` consumed by `Makie.pick` (Task 3) and `_pick_at!` (Task 5); `Screen.path2plot::Dict{String,UInt64}` (Task 2) used by `pick_hit` (Task 3); `select!(screen, plot; group::UInt8)` consistent across Tasks 4-5; `selection_outline` flows `ScreenConfig` (2) → `interactive_display` (5) → renderer (2). `_element_index(plot, instance_id::UInt32)` consistent (Tasks 3).
- **Open verify carried to execution:** surface element-index fidelity (Task 3 Step 6) is verify-or-degrade per the spec; the report must state which.
