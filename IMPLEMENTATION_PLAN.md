# OmniverseMakie Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Makie backend that translates a Makie `Scene`/`Figure` into an OpenUSD stage, renders it with NVIDIA `ovrtx` (Omniverse RTX path tracer) via direct `ccall`, streams minimal per-frame edits through Makie's `ComputePipeline`, and presents an interactive, orbit-able RTX viewport in a window.

**Architecture:** Three layers — (1) `LibOVRTX` subpackage: Clang.jl-generated raw `ccall` bindings + a hand-written loader; (2) `OV` high-level GC-aware wrapper + Makie⇄USD translation; (3) `Screen <: Makie.MakieScreen` with the backend contract, an on-demand render loop, event injection, and GPU-direct display. USD *is* the wire format; the `:ovrtx_renderobject` compute node *is* the diff engine. Full design rationale: `ARCHITECTURE.md`.

**Tech Stack:** Julia 1.12 · ovrtx 0.3.0 (`libovrtx-dynamic.so`) · Makie 0.24.12 / ComputePipeline 0.1.8 / GLMakie 0.13.12 · CEnum · GeometryBasics · Colors · CUDA.jl (M3/M4) · Clang.jl (codegen only).

**Validation backing this plan (all run live on this machine — NVIDIA RTX A5000, Julia 1.12.6):**
- The full Python pipeline renders `torus-plane.usda` → RT2 → DLPack → PNG (`references/validation/torus-A5000.png`).
- **Spike A (keystone, retired):** the *Julia-native* `ccall` path renders the same scene — **2,073,600/2,073,600 px non-black, ~11 s** — using the generated binding **verbatim**, and a `write_attribute(omni:xform)` + `reset` dynamic update changes 597,204 px. Code: `<scratchpad>/julia-render-spike/`.
- **Spike B:** the `:ovrtx_renderobject` `register_computation!` diff contract and the imperative `insert!`/`delete!` dispatch both work on **real Makie internals with no window**. Code: `<scratchpad>/makie-wiring-spike/`.

`<scratchpad>` = `/tmp/claude-1000/-home-juliahub-temp-omniverse-makie/a522deed-613e-4f45-9792-cd3717a88207/scratchpad`. These spike files are the literal source for the M0 code below — copy from them.

---

## Global Constraints

Every task's requirements implicitly include these (exact values copied from the spec/spikes):

- **Julia ≥ 1.12.** Backends pin Makie exactly via `[compat]`: `Makie = "=0.24.12"`, `ComputePipeline = "=0.1.8"`, `GLMakie = "=0.13.12"` — all three are **registered at exactly these versions** (verified), so they are ordinary registry deps added with `Pkg.add`. **Do NOT `[sources]`-link them to `references/Makie.jl/`** — those paths won't exist for a user who clones the package. To hack on the local Makie clone, `Pkg.develop(path="…/references/Makie.jl/Makie")` into your *local manifest only* (never the committed `Project.toml`).
- **No Python, no PythonCall, no wheel discovery, and no JLL for ovrtx itself.** The native lib is found by `Libdl.dlopen`; resolution order is `ENV["OVRTX_LIBRARY_PATH"]` → default soname. (A JLL *is* used for the libOpenGL dependency — next bullet — just not for ovrtx, which the user installs.)
- **`libOpenGL.so.0` must be `dlopen`ed `RTLD_GLOBAL` *before* `libovrtx-dynamic.so`** (ovrtx's `usd_resolver` plugin needs it; ovrtx does **not** ship it). Source it from the **`Libglvnd_jll`** dependency — verified to provide `libOpenGL.so` with SONAME `libOpenGL.so.0`, so ovrtx's later by-soname `dlopen` resolves to the already-loaded image — via `dlopen(Libglvnd_jll.libOpenGL_path, RTLD_GLOBAL)`. Override with `ENV["OVRTX_LIBOPENGL_PATH"]`. Self-contained: no system `libglvnd`, no hardcoded path, no `LD_LIBRARY_PATH`.
- **The generated bindings are used verbatim** — never hand-edit a generated `@ccall`/struct/`@cenum`. All additions (loader, helpers, static-inline reimplementations) live in the hand-written module shell.
- **The carb breakpad crash reporter must be neutralized before any renderer is created in-process** (it hijacks SIGSEGV/SIGABRT and crashes Julia at exit — see Task M0.4). No `_exit` hacks in shipped code.
- **C zero-copy hot path is mandatory for animation:** per-frame updates use `map_attribute` (fixed-size attrs like `omni:xform`) or `bind_array_attribute`+`write` (arrays like `points`); **never** re-author USDA per frame.
- **Dynamic add *and* delete** of plots and subscenes are first-class, fully deregistered (better than GLMakie's leaky subscene path).
- **`ovx_string_t` is passed by value and borrows its bytes** — `GC.@preserve` the backing Julia string across the call *and* the subsequent async `wait_op`. Returned error strings are transient/thread-local — copy immediately.
- **ovrtx config struct is passed empty, not NULL:** `ovrtx_config_t(C_NULL, 0)`.
- Render-product path for the validation scene: `/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0`; render var `LdrColor` (RGBA8, `[H,W,4]`, top-left origin).
- **NVIDIA binaries are never vendored** — the user installs ovrtx; we only `dlopen`.
- **TDD throughout:** write the failing test, watch it fail, minimal implementation, watch it pass, commit. Every renderer-touching test runs in its **own Julia process** (`run(\`julia ...\`)` + assert on exit code/output) so a renderer crash can't poison the test session and so the signal-handler fix is exercised honestly.
- **Manage dependencies and `Project.toml` via Pkg.jl** — `Pkg.generate`/`Pkg.add`/`Pkg.develop`/`Pkg.compat`/`Pkg.test`, run inside the package env (the `julia` MCP's `env_path`). Do **not** hand-edit `[deps]`/`[compat]` UUIDs or versions; the `Project.toml` blocks shown in this plan are the **expected result** of those Pkg operations, not text to type by hand. The only hand-written committed TOML is the declarative `[sources]`/`[workspace]` for the in-repo `lib/LibOVRTX` subpackage (Pkg has no CLI for those two sections yet).

---

## Development workflow (the `julia` MCP)

Iterate with the **`julia` MCP** (`julia_eval`, `env_path="…/OmniverseMakie.jl"`) rather than cold `julia` runs — a warm session in the package env persists state across calls, so Julia startup/compile (TTFX) is paid once. Project-specific caveats:

- **Revise can't reload `struct`/`@cenum` redefinitions** — and M0 defines many (`Renderer`, `StepResult`, the generated ABI). After editing a struct/enum, `julia_restart(env_path=…)`; function-body edits reload live.
- **Renderer-touching *tests* still spawn their own `julia` subprocess** (per the TDD constraint above) — the carb crash reporter + signal handling can crash or poison a long-lived session. Use `julia_eval` to author/explore; spawn fresh processes for the test assertions on exit code.
- **A persistent renderer in an MCP session pins GPU resources** — `julia_restart` to reclaim them if a session accumulates renderers or wedges.

---

## Milestone roadmap

| MS | Deliverable (independently testable) | Gate |
|---|---|---|
| **M0** | `LibOVRTX` + `OV` wrapper: render `torus-plane.usda` → `Matrix{RGBA}` and apply a live `omni:xform` update, **all from Julia, process exits 0** | Julia-native render proven; crash reporter neutralized |
| **M1** | `Screen` + static `Scene→USD` for the 3D core; `colorbuffer`/`save` produce a correct image of a real Makie `Scene` | `mesh!`/`scatter!` scene → PNG via Makie |
| **M2** | `:ovrtx_renderobject` diff node + hot-path bindings + dynamic add/delete; live attribute/transform/color edits | benchmark meets interactive rates; add/delete leak-free |
| **M3** | Interactive GLMakie window: CPU-blit display, event injection, `cam3d!` orbit/zoom, on-demand loop + RT2 progressive refinement | orbit a live RTX viewport |
| **M4** | Depth: GPU-direct CUDA-GL blit, materials (OmniPBR/MaterialX), AOV picking, subscene hardening | no-CPU-roundtrip display; pick + materials |
| **M5** | Examples gallery: adapt RPRMakieNotes + raydemo scenes into `examples/` (originals untouched) | a real scene gallery renders through OmniverseMakie |

M0 is fully detailed below (TDD steps + complete code from the spikes). **M1–M5 are specified at task granularity** — each task names exact files, the interfaces it consumes/produces, the novel code in full, and its test. Per writing-plans guidance for a multi-phase backend, **M1–M5 tasks are expanded into their own bite-sized step lists just before each milestone is executed**, because their concrete signatures depend on what the previous milestone surfaces (e.g. the exact `OV` wrapper API M1 consumes is produced by M0). Do not start a milestone before its predecessor's gate is green.

---

## File structure

> Task file paths below use `OmniverseMakie/…` as shorthand for the **package root, which is the repo root** `OmniverseMakie.jl/` (e.g. `OmniverseMakie/src/screen.jl` = `OmniverseMakie.jl/src/screen.jl`).

```
omniverse-makie/                         # working dir (holds the repo + references/ clones)
└── OmniverseMakie.jl/                    # THE GIT REPO (github.com/asinghvi17/OmniverseMakie.jl); package lives at its root
    ├── ARCHITECTURE.md  IMPLEMENTATION_PLAN.md   # design docs (committed)
    ├── Project.toml                      # registry deps via Pkg; [sources]+[workspace] only for lib/LibOVRTX  (Task M0.1)
    ├── src/
    │   ├── OmniverseMakie.jl             # module, __init__, activate!, re-export Makie   (M1.1)
    │   ├── binding/
    │   │   ├── OV.jl                      # GC-aware Renderer/StepResult/MappedVar/AttrBinding + async lifecycle (M0.5–0.7)
    │   │   ├── signals.jl                 # carb crash-reporter neutralization (M0.4)
    │   │   └── dlpack.jl                  # DLTensor → Array/CuArray wrapping (M0.6, M4.1)
    │   ├── translation/
    │   │   ├── usd.jl                     # inline-USDA builders, references, render-config root (M1.2)
    │   │   ├── camera.jl  lights.jl  materials.jl                                          (M1.3,1.4,M4.2)
    │   │   ├── meshes.jl  scatter.jl  lines.jl  surface.jl  volume.jl  # to_ovrtx_object   (M1.5,1.7)
    │   ├── compute.jl                     # :ovrtx_renderobject node + push_to_ovrtx! + bind_hot_attributes! (M2.2,2.3)
    │   ├── screen.jl                      # Screen <: MakieScreen + contract + insert!/delete!/empty! (M1.1,M2.1,2.4)
    │   ├── renderloop.jl                  # on-demand loop, progressive refinement, requires_update (M3.3)
    │   ├── events.jl                      # scene.events.* injection + render_tick               (M3.2)
    │   ├── display.jl                     # GLMakie image! target; CPU blit + CUDA-GL blit        (M3.1,M4.1)
    │   └── settings.jl                    # RT2/PathTracing/Minimal, samples, bounces             (M1.2,M3.3)
    ├── lib/LibOVRTX/                      # subpackage (raw bindings)
    │   ├── Project.toml                   # name=LibOVRTX; deps CEnum, Libdl, Libglvnd_jll          (M0.1)
    │   ├── src/
    │   │   ├── LibOVRTX.jl                # hand-written shell: loader + helpers + include          (M0.3)
    │   │   └── libovrtx_api.jl            # GENERATED 1:1 ccalls (verbatim)                         (M0.2)
    │   └── gen/  generator.jl  generator.toml  ovrtx_umbrella.h  Project.toml                       (M0.2)
    ├── test/   runtests.jl + per-milestone test files
    ├── examples/   ported scene gallery — RPRMakieNotes + raydemo               (M5)
    └── ext/    (future: CUDA interop extension)
```

---

# Milestone M0 — Binding foundation + Julia-native render

**Outcome:** a Julia process that loads `libovrtx-dynamic.so`, renders `torus-plane.usda` to a `Matrix{RGBA{N0f8}}`, applies a live transform update, and **exits 0**. No Makie yet.

### Task M0.1: Scaffold `OmniverseMakie` + `LibOVRTX` workspace

**Files:**
- Create: `OmniverseMakie/Project.toml`
- Create: `OmniverseMakie/lib/LibOVRTX/Project.toml`
- Create: `OmniverseMakie/src/OmniverseMakie.jl` (stub module)
- Create: `OmniverseMakie/lib/LibOVRTX/src/LibOVRTX.jl` (stub module)
- Test: `OmniverseMakie/test/runtests.jl`

**Interfaces:**
- Produces: a workspace where `Pkg.activate("OmniverseMakie"); Pkg.instantiate()` resolves both packages against one manifest, and `using LibOVRTX` / `using OmniverseMakie` load.

- [ ] **Step 1: Write the failing test** — `OmniverseMakie/test/runtests.jl`:
```julia
using Test
@testset "M0.1 workspace loads" begin
    # Both packages must import without error (stubs at this point).
    @test (using LibOVRTX; true)
    @test (using OmniverseMakie; true)
end
```

- [ ] **Step 2: Generate `LibOVRTX` and add its deps via Pkg** (don't hand-write the TOML):
```julia
using Pkg
Pkg.generate("OmniverseMakie/lib/LibOVRTX")              # creates Project.toml (name+uuid) + src stub
Pkg.activate("OmniverseMakie/lib/LibOVRTX")
Pkg.add(["CEnum", "Libdl", "Libglvnd_jll"])              # Libglvnd_jll ships libOpenGL.so.0 (M0.3 loader)
Pkg.compat("CEnum", "0.5"); Pkg.compat("Libglvnd_jll", "1"); Pkg.compat("julia", "1.12")
```
Resulting `lib/LibOVRTX/Project.toml`:
```toml
name = "LibOVRTX"
uuid = "48e69356-47aa-4e0c-8089-4409475c3dd9"
version = "0.1.0"

[deps]
CEnum = "fa961155-64e5-5f13-b03f-caf6b980ea82"
Libdl = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
Libglvnd_jll = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"

[compat]
CEnum = "0.5"
Libglvnd_jll = "1"
julia = "1.12"
```

- [ ] **Step 3: Generate `OmniverseMakie` and link the subpackage via Pkg** (deps grow per milestone — M0 needs only `LibOVRTX`):
```julia
Pkg.generate("OmniverseMakie")
Pkg.activate("OmniverseMakie")
Pkg.develop(path="OmniverseMakie/lib/LibOVRTX")          # links the in-repo subpackage
```
Then hand-add **only** the declarative `[sources]`/`[workspace]` for the in-repo subpackage (Pkg has no CLI for these two). Resulting `OmniverseMakie/Project.toml`:
```toml
name = "OmniverseMakie"
uuid = "d60c73f4-20f1-48ea-a3b4-a683962494cb"
version = "0.1.0"

[deps]
LibOVRTX = "48e69356-47aa-4e0c-8089-4409475c3dd9"
Libdl = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[sources]
LibOVRTX = { path = "lib/LibOVRTX" }   # in-repo, ships with the package — correct to commit

[workspace]
projects = ["lib/LibOVRTX"]

[compat]
LibOVRTX = "0.1"
julia = "1.12"
```
> **No `[sources]` for Makie/ComputePipeline/GLMakie** — they're registry deps added with `Pkg.add` (at the pinned versions) in Task M1.1. Only the in-repo `lib/LibOVRTX` is sourced/workspaced.

- [ ] **Step 4: Create stub modules.** `lib/LibOVRTX/src/LibOVRTX.jl`: `module LibOVRTX end`. `src/OmniverseMakie.jl`: `module OmniverseMakie end`.

- [ ] **Step 5: Run the test**
Run: `cd OmniverseMakie && julia --project=. -e 'using Pkg; Pkg.instantiate()' && julia --project=. test/runtests.jl`
Expected: PASS (both modules import).

- [ ] **Step 6: Commit** — `feat(M0.1): scaffold OmniverseMakie + LibOVRTX workspace`

---

### Task M0.2: Vendor generated bindings + regeneration tooling

The bindings were already generated and proven loadable; copy them in and preserve the `gen/` tooling so they can be regenerated when ovrtx bumps.

**Files:**
- Create: `OmniverseMakie/lib/LibOVRTX/src/libovrtx_api.jl` (copy of `<scratchpad>/clang-spike/gen_out/LibOVRTX.jl`, **with the `module …`/`end` wrapper and the prologue `const libovrtx = …` line stripped** so it can be `include`d into the hand-written shell)
- Create: `OmniverseMakie/lib/LibOVRTX/gen/{generator.jl, generator.toml, ovrtx_umbrella.h, Project.toml}` (copy from `<scratchpad>/clang-spike/`, set `output_file_path = "../src/libovrtx_api.jl"`, drop `module_name` so no module wrapper is emitted)
- Test: `OmniverseMakie/test/libovrtx_struct_test.jl`

**Interfaces:**
- Produces: `LibOVRTX.libovrtx_api.jl` defining 42 `ovrtx_*` functions, 58 structs (incl. `ovx_string_t`, `ovrtx_config_t`, `ovrtx_render_product_set_t`, `ovrtx_render_var_output_t`, `ovrtx_render_var_tensor_t`, `DLTensor`), 24 `@cenum` (incl. `ovrtx_api_status_t`, `ovrtx_map_device_type_t`, `ovrtx_attribute_semantic_t`, `ovrtx_data_access_t`), and macro consts.

- [ ] **Step 1: Write the failing test** — `test/libovrtx_struct_test.jl` (loads the generated file into a bare module with only CEnum; no `.so` needed):
```julia
using Test, CEnum
module _Probe
    using CEnum
    include(joinpath(@__DIR__, "..", "lib", "LibOVRTX", "src", "libovrtx_api.jl"))
end
@testset "M0.2 generated ABI" begin
    @test sizeof(_Probe.ovx_string_t) == 16
    @test sizeof(_Probe.ovrtx_xform_matrix44d_t) == 128
    @test sizeof(_Probe.DLTensor) == 48
    @test Int(_Probe.OVRTX_API_SUCCESS) == 0
    @test Int(_Probe.kDLCUDA) == 2
end
```

- [ ] **Step 2: Run it** → FAIL (`libovrtx_api.jl` not found).

- [ ] **Step 3:** Copy the generated file and `gen/` tooling into place; strip the module wrapper/prologue line from `libovrtx_api.jl`.

- [ ] **Step 4: Run it** → PASS.

- [ ] **Step 5: Commit** — `feat(M0.2): vendor generated LibOVRTX bindings + gen tooling`

---

### Task M0.3: `LibOVRTX` module shell — loader + idioms + static-inline reimplementations

**Files:**
- Modify: `OmniverseMakie/lib/LibOVRTX/src/LibOVRTX.jl` (replace stub)
- Test: `OmniverseMakie/test/libovrtx_load_test.jl`

**Interfaces:**
- Consumes: `libovrtx_api.jl` (Task M0.2).
- Produces (the public surface the `OV` layer calls):
  - `LibOVRTX.version() -> Tuple{UInt32,UInt32,UInt32}`
  - `LibOVRTX.ovx_string(s::AbstractString) -> ovx_string_t`  (caller `GC.@preserve`s `s`)
  - `Base.String(s::ovx_string_t) -> String`
  - `LibOVRTX.check(result, op::AbstractString) -> result`  (throws `OVRTXError` unless `.status == OVRTX_API_SUCCESS`)
  - `const OVRTX_TIMEOUT_INFINITE::ovrtx_timeout_t`
  - `const NOSYNC::ovrtx_cuda_sync_t` (`ovrtx_cuda_sync_t(0,0)`)
  - all generated `ovrtx_*`, structs, `@cenum` re-exported.

- [ ] **Step 1: Write the failing test** — `test/libovrtx_load_test.jl` (own process; needs the `.so`):
```julia
using Test
@testset "M0.3 LibOVRTX loads + links" begin
    ENV["OVRTX_LIBRARY_PATH"] = get(ENV, "OVRTX_LIBRARY_PATH",
        "/home/juliahub/temp/omniverse-makie/references/ovrtx/examples/python/minimal/.venv/lib/python3.13/site-packages/ovrtx/bin/libovrtx-dynamic.so")
    @eval using LibOVRTX
    @test LibOVRTX.version() == (UInt32(0), UInt32(3), UInt32(0))
    @test sizeof(LibOVRTX.ovx_string_t) == 16
end
```

- [ ] **Step 2: Run it** → FAIL.

- [ ] **Step 3: Write `LibOVRTX.jl`** (verbatim from Spike A — the loader + helpers all tested working against libc):
```julia
module LibOVRTX
using CEnum
import Libdl
import Libglvnd_jll

# Resolved at RUNTIME in __init__ so OVRTX_LIBRARY_PATH is honored, not baked at precompile.
# The generated `@ccall libovrtx.sym(...)` lines reference this binding unchanged.
global libovrtx::String = "libovrtx-dynamic.so"
const _OVRTX_HANDLE  = Ref{Ptr{Cvoid}}(C_NULL)
const _OPENGL_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

function __init__()
    # libOpenGL FIRST + GLOBAL so ovrtx's usd_resolver plugin resolves GL symbols by soname.
    # Libglvnd_jll.libOpenGL_path has SONAME libOpenGL.so.0, so ovrtx's later by-soname dlopen
    # finds this already-loaded image. Env override allows a system/driver libOpenGL if desired.
    libgl = get(ENV, "OVRTX_LIBOPENGL_PATH", Libglvnd_jll.libOpenGL_path)
    _OPENGL_HANDLE[] = Libdl.dlopen(libgl, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
    global libovrtx = get(ENV, "OVRTX_LIBRARY_PATH", "libovrtx-dynamic.so")
    _OVRTX_HANDLE[] = Libdl.dlopen(libovrtx, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
end

include("libovrtx_api.jl")  # generated 1:1 ccalls + structs + @cenum + const macros

# --- static const / macros Clang does not emit -----------------------------------
const OVRTX_TIMEOUT_INFINITE = ovrtx_timeout_t(typemax(UInt64))
const NOSYNC = ovrtx_cuda_sync_t(0, 0)

# --- ovx_string_t surfacing (caller GC.@preserve the backing String) --------------
ovx_string(s::Union{String,SubString{String}}) =
    ovx_string_t(Base.unsafe_convert(Cstring, s), ncodeunits(s))
function Base.String(s::ovx_string_t)
    s.ptr == C_NULL && return ""
    return unsafe_string(Ptr{UInt8}(s.ptr), s.length)
end

# --- error idiom: ovrtx returns a struct whose .status is the enum -----------------
struct OVRTXError <: Exception; op::String; msg::String; end
function Base.showerror(io::IO, e::OVRTXError)
    print(io, "OVRTXError during ", e.op, ": ", e.msg)
end
function check(result, op::AbstractString)
    result.status == OVRTX_API_SUCCESS && return result
    s = ovrtx_get_last_error()                 # transient thread-local ovx_string_t — copy now
    throw(OVRTXError(op, String(s)))
end

# --- version helper (ovrtx_get_version returns Cvoid via 3 out-params) -------------
function version()
    maj, mn, pt = Ref{UInt32}(0), Ref{UInt32}(0), Ref{UInt32}(0)
    ovrtx_get_version(maj, mn, pt)
    return (maj[], mn[], pt[])
end

end # module
```

- [ ] **Step 4: Run it** → PASS (own process). Confirm exit code 0 *is not yet required* here (no renderer created — the crash reporter isn't loaded until `create_renderer`).

- [ ] **Step 5: Commit** — `feat(M0.3): LibOVRTX loader + ovx_string/check idioms`

---

### Task M0.4: Neutralize the carb breakpad crash reporter

**Why:** Spike A proved that `ovrtx_create_renderer` loads `carb.crashreporter-breakpad.plugin`, which installs SIGSEGV/SIGABRT handlers. All rendering works, but when Julia tears down its scheduler at exit, breakpad catches a benign signal → **exit 139** with a Julia-scheduler backtrace. `bin/ovrtx.config.json` has a `crashreporter` block with no off-switch. The library-friendly fix is to **snapshot the POSIX signal handlers before `create_renderer` and restore Julia's afterward**. (Confirmed minimal repro: create+destroy renderer → 139 without the fix; → 0 with the spike's `_exit`, which a library can't use.)

**Files:**
- Create: `OmniverseMakie/src/binding/signals.jl`
- Test: `OmniverseMakie/test/m0_signals_test.jl`

**Interfaces:**
- Produces: `OV.with_restored_signals(f)` — runs `f()` (which creates the renderer) then restores the previously-saved handlers for SIGSEGV(11), SIGABRT(6), SIGBUS(7), SIGILL(4), SIGFPE(8). Used by `OV.Renderer(...)` (Task M0.5).

- [ ] **Step 1: Write the failing test** — a *subprocess* that creates+destroys a renderer and must exit 0:
```julia
using Test
const PROG = raw"""
    push!(LOAD_PATH, ENV["OM_PROJECT"]); using LibOVRTX
    include(joinpath(ENV["OM_SRC"], "binding", "signals.jl"))
    save = SignalGuard.snapshot()
    cfg = Ref(LibOVRTX.ovrtx_config_t(Ptr{LibOVRTX.ovrtx_config_entry_t}(C_NULL), Csize_t(0)))
    rref = Ref{Ptr{LibOVRTX.ovrtx_renderer_t}}(C_NULL)
    LibOVRTX.check(LibOVRTX.ovrtx_create_renderer(cfg, rref), "create")
    SignalGuard.restore(save)        # <-- the fix
    LibOVRTX.ovrtx_destroy_renderer(rref[])
"""
@testset "M0.4 renderer process exits cleanly" begin
    p = run(pipeline(setenv(`julia --project=$(@__DIR__)/.. -e $PROG`,
            "OVRTX_LIBRARY_PATH"=>ENV["OVRTX_LIBRARY_PATH"],
            "OM_SRC"=>joinpath(@__DIR__,"..","src"),
            "OM_PROJECT"=>joinpath(@__DIR__,"..","lib","LibOVRTX","src"))); wait=false)
    wait(p)
    @test p.exitcode == 0          # without restore() this is 139
end
```

- [ ] **Step 2: Run it** → FAIL (no `signals.jl`; or 139 if you stub a no-op).

- [ ] **Step 3: Write `signals.jl`** — snapshot/restore via `sigaction(2)`:
```julia
module SignalGuard
# struct sigaction is platform-specific; on Linux/glibc it is large. We treat it as an
# opaque fixed-size blob and let the kernel copy it in/out — we only ever *restore* exactly
# what was there before ovrtx clobbered it, so we never need to interpret the fields.
const _SIGS = (4, 6, 7, 8, 11)                 # SIGILL, SIGABRT, SIGBUS, SIGFPE, SIGSEGV
const _SA_SIZE = 256                            # >= sizeof(struct sigaction) on linux-x86_64 (152)

snapshot() = Dict(sig => _getaction(sig) for sig in _SIGS)
function restore(saved::AbstractDict)
    for (sig, blob) in saved
        _setaction(sig, blob)
    end
    return nothing
end

function _getaction(sig::Integer)
    old = zeros(UInt8, _SA_SIZE)
    r = @ccall sigaction(sig::Cint, C_NULL::Ptr{Cvoid}, old::Ptr{UInt8})::Cint
    r == 0 || error("sigaction(get) failed for signal $sig (errno via Libc)")
    return old
end
function _setaction(sig::Integer, blob::Vector{UInt8})
    r = @ccall sigaction(sig::Cint, blob::Ptr{UInt8}, C_NULL::Ptr{Cvoid})::Cint
    r == 0 || error("sigaction(set) failed for signal $sig")
    return nothing
end
end # module
```
> Implementer note: confirm `_SA_SIZE ≥ sizeof(struct sigaction)` on the target (Linux x86_64 = 152). If `restore` proves insufficient on some platform (e.g. breakpad uses an alt-stack), fall back to `@ccall jl_install_default_signal_handlers()::Cvoid` after `create_renderer`. Keep the chosen mechanism behind `OV.with_restored_signals` so callers are unaffected. The acceptance test is exit-code 0 — iterate the mechanism until it passes.

- [ ] **Step 4: Run it** → PASS (exitcode 0).

- [ ] **Step 5: Commit** — `fix(M0.4): neutralize carb breakpad crash reporter (signal save/restore)`

---

### Task M0.5: `OV.Renderer` + async lifecycle + config

**Files:**
- Create: `OmniverseMakie/src/binding/OV.jl`
- Test: `OmniverseMakie/test/m0_renderer_test.jl`

**Interfaces:**
- Consumes: `LibOVRTX.*`, `SignalGuard` (M0.4).
- Produces:
  - `mutable struct Renderer; ptr::Ptr{LibOVRTX.ovrtx_renderer_t}; alive::Bool; end`
  - `Renderer() -> Renderer` (empty config; wraps `create_renderer` in `with_restored_signals`; attaches a finalizer that calls `destroy_renderer`)
  - `enqueue_wait(r, enq, op)` — drives `wait_op` on an `ovrtx_enqueue_result_t`, raises on error
  - `Base.close(r::Renderer)` — idempotent `destroy_renderer`
  - `with_restored_signals(f)`

- [ ] **Step 1: Write the failing test** (subprocess, exit 0 + can create twice):
```julia
const PROG = raw"""
    push!(LOAD_PATH, ENV["OM_PROJECT"]); using LibOVRTX
    include(joinpath(ENV["OM_SRC"], "binding", "signals.jl"))
    include(joinpath(ENV["OM_SRC"], "binding", "OV.jl"))
    r1 = OV.Renderer(); close(r1)
    r2 = OV.Renderer(); close(r2)
    println("OK")
"""
# ... run as in M0.4, assert p.exitcode == 0 and output contains "OK"
```

- [ ] **Step 2–4:** Implement `OV.jl` Renderer section, run → PASS. Core:
```julia
module OV
using ..LibOVRTX
const L = LibOVRTX
include("signals.jl")
with_restored_signals(f) = (s = SignalGuard.snapshot(); try f() finally SignalGuard.restore(s) end)

mutable struct Renderer
    ptr::Ptr{L.ovrtx_renderer_t}
    alive::Bool
    function Renderer()
        cfg  = Ref(L.ovrtx_config_t(Ptr{L.ovrtx_config_entry_t}(C_NULL), Csize_t(0)))
        rref = Ref{Ptr{L.ovrtx_renderer_t}}(C_NULL)
        with_restored_signals() do
            L.check(L.ovrtx_create_renderer(cfg, rref), "create_renderer")
        end
        r = new(rref[], true)
        finalizer(close, r)
        return r
    end
end
Base.unsafe_convert(::Type{Ptr{L.ovrtx_renderer_t}}, r::Renderer) = r.ptr
function Base.close(r::Renderer)
    r.alive || return
    L.ovrtx_destroy_renderer(r.ptr); r.alive = false; return
end

# async lifecycle: enqueue (ovrtx_enqueue_result_t) -> wait_op
function enqueue_wait(r::Renderer, enq, op::AbstractString)
    L.check(enq, op)
    wr = Ref{L.ovrtx_op_wait_result_t}()
    L.check(L.ovrtx_wait_op(r.ptr, enq.op_index, L.OVRTX_TIMEOUT_INFINITE, wr), op*":wait")
    return wr[]
end
end # module
```

- [ ] **Step 5: Commit** — `feat(M0.5): OV.Renderer + async enqueue/wait lifecycle`

---

### Task M0.6: `open_usd!`, `step!`, output-tree walk, `map_render_var`, readback → `Matrix`

**Files:**
- Modify: `OmniverseMakie/src/binding/OV.jl`
- Create: `OmniverseMakie/src/binding/dlpack.jl`
- Test: `OmniverseMakie/test/m0_render_test.jl`

**Interfaces (the API M1's `colorbuffer` consumes):**
- `open_usd!(r, path::AbstractString)` / `open_usd_string!(r, usda::AbstractString)` — sync (enqueue+wait)
- `step!(r, product::AbstractString; dt=1/60) -> StepResult`
- `struct StepResult; r::Renderer; handle::L.ovrtx_step_result_handle_t; open::Bool; end`; `Base.close(::StepResult)` → `destroy_results`
- `map_cpu(sr::StepResult, name="LdrColor") -> (pixels::Array{UInt8,3} [C,W,H], W::Int, H::Int)` (fetch + tree-walk + map CPU + **copy before unmap** + unmap). The CUDA-array analog `map_cuda_array` is added in M4.1.
- `render_to_matrix(r, product; warmup=64) -> Matrix{RGBA{N0f8}}` (convenience: warmup loop + `map_cpu` + reshape `[C,W,H]`→`Matrix`)

- [ ] **Step 1: Write the failing test** (subprocess renders the torus, asserts non-black + exit 0):
```julia
const PROG = raw"""
    ... include LibOVRTX, signals.jl, OV.jl ...
    r = OV.Renderer()
    OV.open_usd!(r, ENV["OM_USDA"])
    img = OV.render_to_matrix(r, "/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0"; warmup=64)
    nonblack = count(c -> (red(c)+green(c)+blue(c)) > 0, img)
    println("SIZE=", size(img), " NONBLACK=", nonblack)
    @assert size(img) == (1080, 1920)
    @assert nonblack == 1080*1920
    close(r)
"""
# assert exitcode 0 and output reports full nonblack
```
with `OM_USDA = references/ovrtx/examples/c/minimal/torus-plane.usda`.

- [ ] **Step 2: Run it** → FAIL.

- [ ] **Step 3: Implement** (verbatim shape from Spike A). `open_usd!`:
```julia
function open_usd!(r::Renderer, path::AbstractString)
    GC.@preserve path begin
        enqueue_wait(r, L.ovrtx_open_usd_from_file(r.ptr, L.ovx_string(path)), "open_usd")
    end
end
```
`step!` + the output-tree walk + `map_render_var` (note: `render_products` is a by-value struct holding a pointer into a `Vector{ovx_string_t}` you must `GC.@preserve`; the output tree is `outputs → outputs[] → output_frames[] → output_render_vars[] → .output_handle`; `ovrtx_render_var_tensor_t.dl` is a `Ptr{DLTensor}`; CPU dtype `{kDLUInt,8,1}` → `UInt8`, shape `[H,W,4]`; **`copy()` the pixels before `unmap`**):
```julia
mutable struct StepResult; r::Renderer; handle::L.ovrtx_step_result_handle_t; open::Bool; end
function step!(r::Renderer, product::AbstractString; dt::Float64=1/60)
    rp = L.ovx_string_t[ L.ovx_string(product) ]
    GC.@preserve product rp begin
        set = L.ovrtx_render_product_set_t(pointer(rp), Csize_t(1))
        h = Ref{L.ovrtx_step_result_handle_t}(0)
        enqueue_wait(r, L.ovrtx_step(r.ptr, set, dt, h), "step")
        sr = StepResult(r, h[], true); finalizer(close, sr); return sr
    end
end
Base.close(sr::StepResult) = (sr.open && (L.ovrtx_destroy_results(sr.r.ptr, sr.handle); sr.open=false); nothing)

function _find_var(outs::L.ovrtx_render_product_set_outputs_t, name::AbstractString)
    for i in 1:outs.output_count
        po = unsafe_load(outs.outputs, i)
        for f in 1:po.output_frame_count
            fr = unsafe_load(po.output_frames, f)
            for v in 1:fr.render_var_count
                rv = unsafe_load(fr.output_render_vars, v)
                String(rv.render_var_name) == name && return rv.output_handle
            end
        end
    end
    error("render var $name not found")
end

function map_cpu(sr::StepResult, name::AbstractString="LdrColor")
    outs = Ref{L.ovrtx_render_product_set_outputs_t}()
    L.check(L.ovrtx_fetch_results(sr.r.ptr, sr.handle, L.OVRTX_TIMEOUT_INFINITE, outs), "fetch")
    h = _find_var(outs[], name)
    mdesc = Ref(L.ovrtx_map_output_description_t(L.OVRTX_MAP_DEVICE_TYPE_CPU, Csize_t(0)))
    ro = Ref{L.ovrtx_render_var_output_t}()
    L.check(L.ovrtx_map_render_var_output(sr.r.ptr, h, mdesc, L.OVRTX_TIMEOUT_INFINITE, ro), "map")
    t0  = unsafe_load(ro[].tensors, 1)
    dlt = unsafe_load(t0.dl)                       # DLTensor
    H = unsafe_load(dlt.shape,1); W = unsafe_load(dlt.shape,2); C = unsafe_load(dlt.shape,3)
    raw = unsafe_wrap(Array, Ptr{UInt8}(dlt.data), (Int(C),Int(W),Int(H)); own=false)
    pixels = copy(raw)                              # OWN before unmap
    L.ovrtx_unmap_render_var_output(sr.r.ptr, ro[].map_handle, L.NOSYNC)
    return (pixels, Int(W), Int(H))                 # pixels is [C,W,H]
end
```
`render_to_matrix` warms up `warmup` steps then maps the last and reshapes `[C,W,H]`→`Matrix{RGBA{N0f8}}` (`dlpack.jl` holds the `[C,W,H]`→`RGBA` reinterpret + the M3 y-flip; M0 just asserts non-black, so any orientation passes).

- [ ] **Step 4: Run it** → PASS (1080×1920, all non-black, exit 0).

- [ ] **Step 5: Commit** — `feat(M0.6): open_usd/step/map readback — Julia-native render`

---

### Task M0.7: hot-path writes — `write_xform!`, `write_array_attribute!`, `reset!`

**Files:**
- Modify: `OmniverseMakie/src/binding/OV.jl`
- Test: `OmniverseMakie/test/m0_update_test.jl`

**Interfaces (the API M2's `push_to_ovrtx!` consumes):**
- `write_xform!(r, prim::AbstractString, mat::AbstractMatrix{Float64})` — 4×4 row-vector, translation in last row; semantic `OVRTX_SEMANTIC_XFORM_MAT4x4`; `OVRTX_DATA_ACCESS_SYNC`
- `write_array_attribute!(r, prim, name::AbstractString, arr::AbstractArray)` — e.g. `points`
- `reset!(r; time=0.0)` — restart RT2 accumulation (call after any geometry/camera change)

- [ ] **Step 1: Write the failing test** — render, capture frame1, `write_xform!` on `/World/Torus`, `reset!`, re-render, assert pixels differ (Spike A got 597,204 px). Subprocess, exit 0.

- [ ] **Step 2: Run it** → FAIL.

- [ ] **Step 3: Implement** (`write_xform!` reimplements the static-inline `ovrtx_set_xform_mat`: build a 1-D `DLTensor` over a row-major 4×4 `Float64` with `dtype={kDLFloat,64,lanes=16}`, wrap in `ovrtx_input_buffer_t`, build `ovrtx_binding_desc_or_handle_t` with `attr="omni:xform"`, `is_array=false`, `semantic=OVRTX_SEMANTIC_XFORM_MAT4x4`, `mode=EXISTING_ONLY`, then `ovrtx_write_attribute(..., OVRTX_DATA_ACCESS_SYNC)` + `wait_op`). `reset!` = `enqueue_wait(r, ovrtx_reset(r.ptr, time), "reset")`. Copy the exact struct construction from `<scratchpad>/julia-render-spike/render.jl` (`set_xform_translation!`).

- [ ] **Step 4: Run it** → PASS (frame2 differs; exit 0).

- [ ] **Step 5: Commit** — `feat(M0.7): hot-path write_xform/write_array + reset`

**M0 GATE:** `julia --project=OmniverseMakie OmniverseMakie/test/runtests.jl` runs M0.1–M0.7, every renderer subprocess exits 0, the torus renders non-black, and a transform update changes the frame. ✅ → proceed to M1.

---

# Milestone M1 — Static `Scene → USD → image`

**Outcome:** `colorbuffer(fig)` / `save("out.png", fig)` render a real Makie `Scene` (mesh + scatter + lights + camera) through ovrtx. Build-once authoring (inline USDA + references); no live diffing yet. Add Makie/GLMakie/GeometryBasics/Colors to deps here.

Each task below carries Files / Interfaces / key code / Test / Acceptance. Expand to bite-sized steps at execution.

### Task M1.1: `Screen` struct, `ScreenConfig`, `activate!`, module wiring
- **Files:** `src/OmniverseMakie.jl`, `src/screen.jl`, `src/settings.jl`; add deps via `Pkg.add(["Makie","ComputePipeline","GeometryBasics","Colors","ColorTypes","FixedPointNumbers"])` then `Pkg.compat` the exact pins (`Makie="=0.24.12"`, `ComputePipeline="=0.1.8"`) — **registry deps, no `[sources]`**. Test: `test/m1_screen_test.jl`.
- **Interfaces — produces:**
  ```julia
  mutable struct OvrtxRObj
      prim_path::String
      usd_handle::UInt64                    # from add_usd_reference_from_string
      bindings::Dict{Symbol, OV.AttrBinding}  # populated in M2
      visible::Bool
  end
  mutable struct Screen <: Makie.MakieScreen
      renderer::OV.Renderer
      stage_root::String                    # "/World"
      product::String                       # render-product path
      size::Tuple{Int,Int}
      plot2robj::Dict{UInt64, OvrtxRObj}    # objectid(plot) => robj
      scene2scope::Dict{WeakRef, String}    # subscene => USD scope path
      scene_listeners::Dict{WeakRef, Vector{Observables.ObserverFunction}}  # M2.4 dereg
      config::ScreenConfig
      requires_update::Bool
      scene::Union{Nothing, Makie.Scene}
      render_tick::Observable{Makie.TickState}   # M3
      display_target::Any                   # M3 GLMakie image!/screen; nothing offscreen
  end
  struct ScreenConfig
      samples::Int                          # offline PathTracing SPP (default 512)
      warmup::Int                           # RT2 warmup frames (default 64)
      mode::Symbol                          # :rt2 | :pathtracing | :minimal
      max_bounces::Int
      visible::Bool
  end
  activate!(; screen_config...)             # set_screen_config! + set_active_backend!
  ```
- **Key code:** copy RPRMakie's `activate!`/`__init__`/re-export-all-Makie-names loop (`references/Makie.jl/RPRMakie/src/RPRMakie.jl`); the three `Screen(scene, config[, io, mime | format])` constructors collapse to one real constructor that builds `OV.Renderer()`, authors the render-config root (Task M1.2), and stores size from `size(scene)`.
- **Test:** `using OmniverseMakie; OmniverseMakie.activate!()`; construct a `Screen` for an empty `Scene`; `size(screen) == size(scene)`; subprocess exits 0.
- **Acceptance:** backend activates as `Makie.current_backend()`; `Screen` builds and tears down cleanly.

### Task M1.2: USD authoring — inline USDA builders, references, render-config root, settings
- **Files:** `src/translation/usd.jl`, `src/settings.jl`. Test: `test/m1_usd_test.jl`.
- **Interfaces — produces:**
  ```julia
  author_render_root!(screen; resolution, camera_path) -> Nothing   # open_usd_string! a root with /World, /Render RenderProduct+RenderVar(LdrColor), omni:rtx:* settings
  add_prim!(screen, usda::String, prefix::String) -> UInt64         # add_usd_reference_from_string -> handle
  remove_prim!(screen, handle::UInt64)                              # remove_usd
  usda_mesh(points, faces, normals, color) -> String                # inline USDA for a UsdGeomMesh
  usda_xform_matrix(model::Mat4d) -> String                         # omni:xform literal
  ```
- **Key code:** the render-config root mirrors `ovrtx-api.md §1` (the `def "Render" { def RenderProduct ... def RenderVar "LdrColor" }` block) plus `omni:rtx:rendermode`/`omni:rtx:rtpt:maxBounces` from `settings.jl`. `add_usd_reference_from_string` wrapper drives enqueue+wait and returns the `ovrtx_usd_handle_t`.
- **Test:** author root + add a hand-written cube USDA reference; `step!` + `map_cpu` → non-black; `remove_prim!` then re-render → the cube's pixels are gone. Subprocess, exit 0.
- **Acceptance:** authoring + reference add/remove round-trips through a render.

### Task M1.3: Camera translation
- **Files:** `src/translation/camera.jl`. Test: `test/m1_camera_test.jl`.
- **Interfaces — produces:** `author_camera!(screen, scene)` — `UsdGeomCamera` at `camera_path` from `scene.camera` (`eyeposition[]`, `view_direction[]`, `upvector[]`, `projection[]` → focal length/aperture; perspective + orthographic) as `omni:xform` + intrinsics; `update_camera!(screen, scene)` writes only `omni:xform` (+ focal length) via `OV.write_xform!`.
- **Key code:** adapt RPRMakie `update_rpr_camera!` (`references/Makie.jl/RPRMakie/src/scene.jl:3`) — same inputs (eye/lookat/up/fov/near/far), author USD instead of RPR. `focal_length = res[2] / (2*tand(fov/2))`.
- **Test:** author a scene with a known camera; render two camera positions (re-author xform + `reset!`); assert the framed content differs. Subprocess.
- **Acceptance:** camera pose drives the rendered viewpoint.

### Task M1.4: Lights translation
- **Files:** `src/translation/lights.jl`. Test: `test/m1_lights_test.jl`.
- **Interfaces — produces:** `author_lights!(screen, scene)` reading `scene.compute.lights[]` + `scene.compute.ambient_color[]`. Mapping (per `ARCHITECTURE.md §5.1`): `PointLight→SphereLight`, `DirectionalLight→DistantLight`, `RectLight→RectLight`, `EnvironmentLight→DomeLight`, `AmbientLight→`low `DomeLight`.
- **Key code:** one `usda_light(::Makie.PointLight)` etc. per type (intensity/color/xform), composed into the render root or added as references.
- **Test:** render the torus scene with a single `DistantLight` vs none; assert mean luminance increases with the light. Subprocess.
- **Acceptance:** lights affect the image.

### Task M1.5: `to_ovrtx_object(::Mesh)` + materials + `display`/`colorbuffer`
- **Files:** `src/translation/meshes.jl`, `src/translation/materials.jl`, `src/screen.jl` (`display`, `colorbuffer`, `insert!` build-once, `backend_showable`, `backend_show`/`show(MIME)`). Test: `test/m1_mesh_render_test.jl`.
- **Interfaces — produces:**
  ```julia
  to_ovrtx_object(screen, scene, plot::Makie.Mesh) -> OvrtxRObj   # author UsdGeomMesh from plot, return robj
  Makie.colorbuffer(screen::Screen, format=Makie.JuliaNative) -> Matrix{RGB{N0f8}}
  Base.display(screen::Screen, scene::Makie.Scene)               # author root+camera+lights+plots, set requires_update
  Base.insert!(screen::Screen, scene, plot::Plot)               # build-once: recurse plot.plots; atomic -> to_ovrtx_object
  ```
- **Key code:** `colorbuffer` = `author/refresh if needed → warmup `config.warmup` steps (RT2) or one PathTracing step → `OV.map_cpu` → `[C,W,H]`→`Matrix{RGB{N0f8}}` with the y-flip/permute matching GLMakie's `JuliaNative` convention (`makie-backend-contract.md §6.6`). `insert!` recurses `plot.plots` (Spike B: called once-per-parent) and is idempotent via `haskey(screen.plot2robj, objectid(plot))`. Material: scalar `color`→`primvars:displayColor`; matrix/colormap→per-vertex `displayColor` or texture (adapt RPRMakie's 4-way `mesh_material` branch); MDL `OmniPBR` deferred to M4. Backend `material=` escape hatch like RPRMakie.
- **Test:** `fig = Figure(); ax = LScene(fig[1,1]); mesh!(ax, Rect3f(...)); scatter is M1.7`. `img = colorbuffer(ax.scene)`; assert size + non-black; `save(tmp*".png", fig)` writes a valid PNG. Subprocess, exit 0.
- **Acceptance:** a real Makie `Scene` with a mesh renders to a correct image; `save`/`record` work via `colorbuffer`.

### Task M1.6: Backend `display`/`save`/`record` plumbing + offscreen `Screen` constructors
- **Files:** `src/screen.jl`. Test: `test/m1_save_record_test.jl`.
- **Interfaces — produces:** the `Screen(scene, config, ::IO, ::MIME)` and `Screen(scene, config, ::Makie.ImageStorageFormat)` constructors; `Makie.backend_showable(::Type{Screen}, ::MIME"image/png"/"image/jpeg") = true`; `to_native`. Makie's `record`/`save` route through `colorbuffer` (no extra work).
- **Test:** `Makie.record(fig, tmp*".mp4", 1:3) do i; rotate!(ax.scene, i*0.1); end` produces a non-empty mp4 (each frame re-renders). Subprocess.
- **Acceptance:** offscreen image + video output work end-to-end.

### Task M1.7: Remaining 3D primitives
- **Files:** `src/translation/{scatter.jl,lines.jl,surface.jl,volume.jl,meshes.jl}`. Test: `test/m1_primitives_test.jl`.
- **Interfaces — produces** one `to_ovrtx_object` per type (each returns `OvrtxRObj`), following the Mesh template:
  - `MeshScatter` → `UsdGeomPointInstancer` (prototype = marker mesh): `positions`, `orientations`, `scales`, per-instance `displayColor`.
  - `Scatter` → `PointInstancer` + **`UsdGeomSphere` prototype (sphere fast path)**; `UsdVol.ParticleField` for huge N.
  - `Surface` → `UsdGeomMesh` (grid re-meshed; adapt RPRMakie's `grid`/`Tessellation` in `meshes.jl:151`).
  - `Lines`/`LineSegments` → `UsdGeomBasisCurves` (`points`, `widths`=linewidth, `type=linear`).
  - `Volume` → `UsdVol` (`VoxelGrid` + volume material).
- **Test:** one render per primitive type asserting non-black + plausible coverage; a combined scene (mesh+scatter+lines) renders. Subprocess.
- **Acceptance:** the full v1 3D core renders statically.

**M1 GATE:** a `Figure` with `LScene` containing mesh/scatter/lines/surface renders via `colorbuffer`/`save`; all primitive tests pass; processes exit 0. ✅ → M2.

---

# Milestone M2 — `ComputePipeline` diff path + dynamic add/delete

**Outcome:** plots update live via the `:ovrtx_renderobject` node (transform/color/points edits push minimal C writes); plots and subscenes can be added and deleted at runtime, leak-free. Hot path benchmarked.

### Task M2.1: Imperative `insert!`/`insertplots!` + `add_scene!`
- **Files:** `src/screen.jl`. Test: `test/m2_insert_test.jl`.
- **Interfaces — produces:** `Base.insert!(screen::Screen, scene::Scene, plot::Plot)` (recurses `plot.plots`, idempotent via `plot2robj`, calls `add_scene!` first); `Makie.insertplots!(screen, scene)` (loop plots, recurse `scene.children`); `add_scene!(screen, scene::Scene)` (idempotent `get!` on `scene2scope`, authors a USD scope for the subscene, registers redraw listeners **stored in `scene_listeners`** for later deregistration).
- **Key code:** mirror `dynamic-add-delete.md §6`. Spike B confirmed dispatch fires with just `Makie.push_screen!(scene, screen)` — no window needed for the test.
- **Test:** `Scene` + `Makie.push_screen!(scene, screen)`; `scatter!(scene, ...)` after attach → `plot2robj` grows by one and the plot renders; inserting a recipe (`poly!`) registers its 2 atomic children. Subprocess.
- **Acceptance:** live `plot!` on a displayed scene authors USD + registers the node.

### Task M2.2: The `:ovrtx_renderobject` compute node + `push_to_ovrtx!`
- **Files:** `src/compute.jl`. Test: `test/m2_diffnode_test.jl`.
- **Interfaces — produces:**
  ```julia
  register_ovrtx_robj!(screen, scene, plot) -> OvrtxRObj   # registers the node, force-resolves once
  push_to_ovrtx!(screen, robj, name::Symbol, value)        # routes one changed output to the right write
  ```
- **Key code** (Spike-B-correct contract — `last` is a NamedTuple keyed by **output** name `:ovrtx_renderobject`; `args`/`last` hold **deref'd values**; on first build `last===nothing` and `changed` is all-true so just build):
  ```julia
  function register_ovrtx_robj!(screen, scene, plot)
      attr = plot.attributes
      inputs = consumed_inputs(plot)   # per-type list (Spike B: all exist by default)
      ComputePipeline.register_computation!(attr, inputs, [:ovrtx_renderobject]) do args, changed, last
          if isnothing(last)
              robj = author_usd_prim!(screen, plot, args)   # M1 authoring
              bind_hot_attributes!(screen, robj, args)      # M2.3
          else
              robj = last.ovrtx_renderobject
              for name in keys(args)
                  changed[name] || continue                 # the minimal-delta gate
                  push_to_ovrtx!(screen, robj, name, args[name])
              end
          end
          screen.requires_update = true
          return (robj,)
      end
      robj = attr[:ovrtx_renderobject][]                    # force first resolve
      screen.plot2robj[objectid(plot)] = robj
      return robj
  end
  ```
  `author_usd_prim!(screen, plot, args)` wraps the per-type `to_ovrtx_object` authoring from M1.5/M1.7, but sources geometry/color from the resolved compute outputs in `args` (not directly from the plot) so the build and update branches use one data source; it returns an `OvrtxRObj` and records `screen.plot2robj`/the USD handle.
  `consumed_inputs(::Mesh)` = `[:positions_transformed_f32c, :model_f32c, :faces, :normals, :scaled_color, :visible]`; `::Scatter` = `[:positions_transformed_f32c, :model_f32c, :quad_scale, :quad_offset, :converted_rotation, :scaled_color, :visible]`; `::Lines` = `[:positions_transformed_f32c, :scaled_color, :model_f32c, :linewidth, :visible]` (Spike B verified these exist; the GL-only nodes `:gl_miter_limit`/`:uniform_pattern*`/normal-matrices do **not** — register ovrtx equivalents or use `Makie.add_computation!(attr, Val(:computed_color))` if a packed color is needed). `push_to_ovrtx!` routes per `ARCHITECTURE.md §6`: `:model_f32c`→`write_xform!`/map; `:positions_transformed_f32c`→array binding; `:faces`→`write_array_attribute!`; `:scaled_color`→`displayColor`; `:visible`→`visibility`.
- **Test:** insert a mesh; pull the node (build); `plot.color = :red`; pull again; assert only the color write fired (instrument `push_to_ovrtx!` with a log) and the render changed. Subprocess.
- **Acceptance:** an attribute edit triggers exactly one minimal C write and a visible change.

### Task M2.3: Persistent hot-path bindings (`map_attribute` / `bind_array_attribute`)
- **Files:** `src/compute.jl`, `src/binding/OV.jl` (binding wrappers). Test: `test/m2_binding_test.jl`.
- **Interfaces — produces:** `OV.create_binding(r, prims, name, dtype, shape; array)`, `OV.map_binding(b; device)`, `OV.unmap!`, `OV.destroy!(b)`; `bind_hot_attributes!(screen, robj, args)` creating persistent bindings (xform via `map_attribute`; positions via `bind_array_attribute`+`write`) stored in `robj.bindings`.
- **Key code:** `ARCHITECTURE.md §6` hot-path tiers; `ovrtx-api.md §4` (`bind_attribute`/`map_attribute`, `BindingFlag.OPTIMIZE`). Fixed-size → zero-copy `map`; arrays → `bind`+`write` (GPU DLPack later; CPU now). Create once, reuse every frame.
- **Test:** animate a prim's `omni:xform` 100 frames via a mapped binding; assert each frame's content moves and no per-frame USDA authoring occurs (assert binding object identity stable). Subprocess.
- **Acceptance:** transforms/points update through persistent bindings, no re-authoring.

### Task M2.4: `delete!`/`delete!(scene)`/`empty!` — leak-free teardown
- **Files:** `src/screen.jl`. Test: `test/m2_delete_test.jl`.
- **Interfaces — produces** (note **typed signatures** — Spike B: untyped `delete!(screen, scene, plot)` is ambiguous with Makie's default and errors):
  ```julia
  Base.delete!(screen::Screen, scene::Makie.Scene, plot::Makie.AbstractPlot)
  Base.delete!(screen::Screen, scene::Makie.Scene)
  Base.empty!(screen::Screen)
  ```
- **Key code** (`dynamic-add-delete.md §6`): in `delete!(screen,scene,plot)` — flatten `collect_atomic_plots`; look up `OvrtxRObj` by `objectid(plot)`; `OV.destroy!` its bindings + `remove_prim!(screen, robj.usd_handle)`; drop from `plot2robj`; `delete!(plot.attributes, :ovrtx_renderobject)` (Spike B: `empty!(graph)` does **not** clear nodes, so this is required); `requires_update = true`. In `delete!(screen,scene)` — recurse children + plots, **`Observables.off` the `scene_listeners[WeakRef(scene)]`** (the GLMakie leak we fix), remove the subscene USD scope, drop its ScreenID. `empty!` — delete every cached plot, `delete!(screen, screen.scene)`, assert registries empty.
- **Test:** add then `delete!(scene, plot)` → prim gone from render, `plot2robj` empty, node dropped; add/remove a subscene 50× → `scene_listeners` returns to empty (no listener accumulation). Subprocess.
- **Acceptance:** add/delete of plots and subscenes leaves no residual prims, bindings, nodes, or listeners.

### Task M2.5: Hot-path benchmark (de-risk gate)
- **Files:** `OmniverseMakie/bench/hot_path.jl`. Test: a threshold assertion.
- **Interfaces — produces:** a benchmark animating N transforms (map) and N points (bind+write) per frame, reporting updates/sec and frame time.
- **Acceptance:** map/bind throughput sustains interactive rates (target: ≥30 Hz for ~10⁴ instance transforms or ~10⁵ points) on the A5000. Record the numbers in `bench/RESULTS.md`. If below target, escalate to GPU-resident DLPack writes before M3.

**M2 GATE:** live edits push minimal writes; add/delete is leak-free; benchmark meets target. ✅ → M3.

---

# Milestone M3 — Interactive window

**Outcome:** a GLMakie window shows the live RTX viewport; `cam3d!` orbits/zooms; the on-demand loop renders only when dirty and progressively refines (RT2). CPU blit. Add GLMakie + CUDA to deps.

### Task M3.1: GLMakie display target + CPU blit
- **Files:** `src/display.jl`, `src/screen.jl` (`display` window path). Test: `test/m3_display_test.jl` (headed; skip if no display).
- **Interfaces — produces:** `open_window!(screen, scene)` — open a GLMakie `Screen`, plot a fullscreen `image!` at frame size, store it as `screen.display_target`; `present_cpu!(screen, pixels)` — write the mapped `LdrColor` into the `image!` data Observable (one host roundtrip; `cuda-gl-interop.md §5`).
- **Acceptance:** ovrtx frames appear in a GLMakie window.

### Task M3.2: Event injection + `render_tick`
- **Files:** `src/events.jl`. Test: `test/m3_events_test.jl`.
- **Interfaces — produces:** `connect_screen(scene, screen)` overloads writing GLMakie window input into `scene.events.*` (mouseposition px/upper-left, mousebutton, scroll, keyboard, resize) per `makie-backend-contract.md §5.3`; `screen.render_tick` bumped each loop iteration to drive mouse-position polling + `frame_tick`.
- **Acceptance:** mouse/keyboard events reach `scene.events`; `ispressed`/`mouseposition` work.

### Task M3.3: On-demand render loop + progressive refinement
- **Files:** `src/renderloop.jl`, `src/settings.jl`. Test: `test/m3_loop_test.jl`.
- **Interfaces — produces:** `start_renderloop!(screen)` (`@async`, GL context on the render task per `cuda-gl-interop.md §2`); each iteration `pollevents → poll_updates (pull every `plot2robj` node) → if requires_update or still accumulating: step! + present`; `requires_update` set by the diff node, `add_scene!` listeners, and `on(scene.camera.projectionview)`. RT2: low-sample denoised while moving, keep accumulating when idle, `OV.reset!` on any change (`ARCHITECTURE.md §8.2`).
- **Key code:** `poll_updates` mirrors GLMakie (`computepipeline.md §3c`): `try plot.attributes[:ovrtx_renderobject][] catch; ComputePipeline.mark_resolved!(node) end`. **Run the renderloop with the signal guard applied** (the renderer lives across the whole loop; ensure the crash-reporter fix from M0.4 holds for a long-lived in-process renderer + window teardown).
- **Acceptance:** idle scene converges and stops; any change re-renders; no busy-spin.

### Task M3.4: End-to-end interactive + live add/delete
- **Files:** integration. Test: `test/m3_interactive_test.jl` (scripted camera moves via `update_cam!`, assert frames change + accumulation resets).
- **Interfaces — produces:** `update_camera!` wired to `on(scene.camera.projectionview)` → re-author camera xform + `reset!`. Confirm `cam3d!` orbit/zoom and live `plot!`/`delete!` during the loop.
- **Acceptance:** a user can `display(fig)`, orbit/zoom a path-traced scene, and add/remove plots live.

**M3 GATE:** interactive orbit of a live RTX viewport with progressive refinement and live add/delete. ✅ → M4.

---

# Milestone M4 — Depth

### Task M4.1: GPU-direct CUDA-GL blit (no CPU roundtrip)
- **Files:** `src/display.jl`, `ext/OmniverseMakieCUDAExt.jl`. Test: `test/m4_gpu_blit_test.jl`.
- **Interfaces — produces:** `OV.map_cuda_array(sr, name="LdrColor")` (CUDA-array analog of M0.6 `map_cpu`, mode `OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY`, returns the `CUarray` + `cuda_sync.wait_event`); `present_gpu!(screen, ...)` — `cuGraphicsGLRegisterImage` the `image!`'s `Texture.id` once → per frame map/`SubResourceGetMappedArray`/`cuMemcpy2D`/unmap, event-gated `ovrtx_unmap` (`cuda-gl-interop.md §3–4`). Runs on the GLMakie render task; falls back to `present_cpu!`.
- **Acceptance:** display path uses no host roundtrip; visually identical to CPU blit; measurable latency drop.

### Task M4.2: Materials — OmniPBR / MaterialX
- **Files:** `src/translation/materials.jl`. Test: `test/m4_materials_test.jl`.
- **Interfaces — produces:** Makie shading/PBR (metalness/roughness/transparency) → MDL `OmniPBR`; backend `material=` escape hatch (MDL/MaterialX path); runtime swap = write `material:binding`.
- **Acceptance:** PBR materials render; runtime material swap works.

### Task M4.3: AOVs + picking
- **Files:** `src/screen.jl` (pick), `src/settings.jl` (extra RenderVars). Test: `test/m4_pick_test.jl`.
- **Interfaces — produces:** add `Depth`/`Normal`/`Id` RenderVars; `Makie.pick(scene, screen, xy)` via `ovrtx_enqueue_pick_query` or the ID AOV → `screen.events`.
- **Acceptance:** `pick`/`mouseover` return the plot under the cursor.

### Task M4.4: Subscene hardening
- **Files:** `src/screen.jl`. Test: `test/m4_subscene_test.jl`.
- **Interfaces — produces:** refcounted shared resources; contiguous ScreenID remap; eager empty-subscene registration (optional `on(parent.children)`); fix the GLMakie gaps catalogued in `dynamic-add-delete.md §5`.
- **Acceptance:** stress add/remove of nested subscenes leaves zero leaks.

**M4 GATE:** GPU-direct display + materials + picking + leak-free subscenes. ✅ → v1 complete.

(Deferred beyond v1: 2D/text/axes parity; remote streaming — `references/notes/wire-protocol-and-webrtc.md`.)

---

# Milestone M5 — Examples gallery

**Outcome:** a set of real, recognizable path-traced scenes rendering through OmniverseMakie, adapted from two existing Makie ray-tracing galleries into `OmniverseMakie.jl/examples/`. **The original repos are never edited** — they live read-only under `references/`; we copy/adapt out of them. This is the end-to-end validation that the backend handles real-world scenes, and the project's showcase.

Source galleries (both are Makie backends, so the *plotting* code ports closely; only backend-specific bits — `activate!`, materials, camera/lights — are translated):
- **RPRMakieNotes** (`references/RPRMakieNotes/scripts/`, ~19 scripts): earth/earthquakes, glass & material balls, transparent + uber materials, volumes, sphere+light studies, freetype text, point fonts, submarine cables. RPRMakie API → OmniverseMakie.
- **raydemo** (`references/raydemo/`, ~20 scene folders): Crown, KillerooGold, BlackHole, Materials, Plants, ProtPlot (proteins), Volumes (bunny cloud, clouds, terrain), GLTF (drone, spacecraft), Waterlily smoke sims, Trixi/koeln flooding. RayMakie/Hikari API → OmniverseMakie.

### Task M5.1: Inventory + triage
- **Files:** `examples/README.md` (gallery index + port-status table).
- **Interfaces — produces:** a table classifying every source scene as **port-now** (uses only the M1–M4 3D core: mesh/meshscatter/scatter/surface/lines/volume + materials/lights/camera), **needs-deferred-feature** (2D/text/axes — e.g. `freetype_text`, `pointsfont`), or **drop** (backend-internal/benchmark scaffolding). Note per-scene asset needs (`.obj`, `.mtlx`, `.hdr`, GLTF).
- **Test:** a script asserts every `*.jl` scene in both source repos appears in the table (no silent omissions).
- **Acceptance:** a complete, justified port list.

### Task M5.2: Examples harness + assets
- **Files:** `examples/common/` (shared `activate!`, camera/light helpers, asset loader), `examples/Project.toml` (its own env — OmniverseMakie + GeometryBasics/Colors/FileIO/MeshIO/… via `Pkg`), `examples/assets/`.
- **Interfaces — produces:** `run_example(path)` + a shared scene scaffold so each ported script is small; assets (meshes, HDRIs, MaterialX) copied into `examples/assets/` (no network at render time).
- **Test:** a trivial example (one mesh + light + camera) renders to PNG via the harness.
- **Acceptance:** harness renders a minimal scene; assets resolve locally.

### Task M5.3: Port the RPRMakieNotes gallery
- **Files:** `examples/<scene>.jl` per port-now RPRMakieNotes script.
- **Interfaces — produces:** each script adapted — `RPRMakie.activate!()` → `OmniverseMakie.activate!()`; RPR material objects/NamedTuples (`RPR.Glass`, uber-material params) → OmniverseMakie's `material=` escape hatch (MDL `OmniGlass`/`OmniPBR`/MaterialX) or `displayColor`; `EnvironmentLight`/HDRIs → `DomeLight`; camera unchanged (reads `cameracontrols`). Plot calls (`mesh!`, `meshscatter!`, `surface!`, `volume!`) stay as-is.
- **Test:** each ported scene renders to a non-black PNG at a fixed sample count; a CI script renders all and assembles a contact sheet.
- **Acceptance:** RPRMakieNotes 3D scenes reproduce recognizably under OmniverseMakie.

### Task M5.4: Port selected raydemo scenes
- **Files:** `examples/<scene>.jl` per chosen raydemo scene (e.g. Crown, Killeroo, Materials, BlackHole, a Volumes cloud, a ProtPlot protein, a GLTF model).
- **Interfaces — produces:** RayMakie/Hikari scene scripts adapted to OmniverseMakie (same translation pattern as M5.3; GLTF/mesh import via MeshIO → USD mesh; volumes → `UsdVol`). Prefer scenes exercising distinct features (instancing, volumes, glass, large meshes).
- **Test:** each renders to a non-black PNG; volume + glass + instancing scenes each covered.
- **Acceptance:** a cross-section of raydemo scenes renders, exercising the full primitive/material set.

### Task M5.5: Gallery doc + render CI
- **Files:** `examples/README.md` (final gallery with thumbnails), `examples/render_all.jl`.
- **Interfaces — produces:** a one-command render of the whole gallery → thumbnails; optional docs page.
- **Acceptance:** `julia examples/render_all.jl` renders the gallery; README shows the results.

**M5 GATE:** the adapted RPRMakieNotes + raydemo galleries render through OmniverseMakie from `examples/`, originals untouched. ✅ → showcase-ready.

---

## Risks (carried from `ARCHITECTURE.md §10`, updated post-spike)

| Risk | Status / mitigation |
|---|---|
| Julia-native `ccall` render feasibility | **RETIRED** — Spike A renders + updates from Julia, exit 0 |
| **carb breakpad crash reporter crashes Julia at exit** | **NEW** — Task M0.4 signal save/restore; re-verify for long-lived loop in M3.3 |
| ovrtx is preview 0.3 (API may churn) | pin a version; isolate in `LibOVRTX`; regen via `gen/` |
| Path-tracer interactivity latency | RT2 + denoiser + low-sample-while-moving + idle accumulation; **M2.5 benchmark gate** |
| Hot-path throughput | M2.5 benchmark; escalate to GPU DLPack writes if under target |
| Array attrs not mappable (`points`) | `bind_array_attribute` + write (M2.3) |
| CUDA-GL interop glue (context/threading) | run on GLMakie render task; CPU fallback always available (M3.1 ships first) |
| Makie internals drift | exact version pins (`=0.24.12` etc.); Spike B confirmed all cited file:lines |

## Reproducing the spikes

- Python baseline: `ARCHITECTURE.md §11`.
- Julia render (Spike A): `cd <scratchpad>/julia-render-spike && OVRTX_LIBRARY_PATH=<...>/libovrtx-dynamic.so julia --project=. render.jl` → two PNGs in `_output/`, exit 0.
- Makie wiring (Spike B): `cd <scratchpad>/makie-wiring-spike && julia --project=. cp_spike.jl && julia --project=. dispatch_spike.jl` → all assertions pass.
