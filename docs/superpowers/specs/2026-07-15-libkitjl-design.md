# libkitjl — in-process Kit runtime for KitScreen (design)

> **STATUS (2026-07-16): IMPLEMENTATION REMOVED.** The transport was built to
> spec and its native/pure tiers verified, then the startup hang was
> root-caused as **architectural**: `OMNI_APP_GLOBALS` must live in the
> process main executable (see "Implementation status" at the end), which
> makes in-process Kit under a `julia` host impossible. The code
> (`lib/LibKitJL`, `InProcessTransport`) was deleted; the subprocess
> transport is the design. This document is kept as the evidence record so
> the approach is not re-attempted on the same axis.

2026-07-15. Short/medium-form spec (project practice). Expands "Phase 2" of
`2026-07-15-omniverse-kit-makie-design.md` into a standalone, buildable plan.
No code in this document; the numbered checklist at the end is the build order.

## Why

`KitScreen` (in `lib/OmniverseKitMakie`) today talks to a persistent headless
`kit --empty` **subprocess** over a line-JSON RPC on a FIFO (`server.jl` +
`kit_server.py`). That is robust and it *coexists* with in-process standalone
ovrtx (two separate processes, two separate carb frameworks). But it pays a
~30-60 s cold Kit start, marshals every frame through a pipe + PNG file, and
can never share a GPU buffer with the Julia process.

`libkitjl` is a **second transport** under the *same* `KitScreen` surface: a
native C shim (`extern "C"`) that hosts the Kit runtime **in this Julia
process** — no subprocess, no FIFO. Kit's whole app loop is a Carbonite
interface (`omni::kit::IApp`: `startup / update / isRunning / shutdown`), so
once Julia holds the interface pointer it can pump frames directly. The
subprocess transport stays the **default and the coexistence path**; the
in-process transport is opt-in for sessions that want fast repeated renders
and, later (v2), zero-copy GPU data planes.

**Constraint that shapes everything:** OmniverseMakie core `src/` stays
untouched (so an upstream ovrtx composite fix drops in cleanly). All new code
lives under `lib/`. Kit is a runtime dependency located via `KIT_RELEASE_DIR`,
**never vendored** (NVIDIA-proprietary), exactly like ovrtx.

## The ABI contract (verified against `kit/dev`)

Kit release dir (env `KIT_RELEASE_DIR`, default the DSX kit-cae build):
`.../release/kit/{kit,libcarb.so}`, dev kit at `.../release/kit/dev`.

- `libcarb.so` exports C: `carbGetSdkVersion`, `omniCoreStart`, `omniCoreStop`,
  `omniGetBuiltInWithoutAcquire`, `carbReallocate`; `carb::acquireFramework()`
  is `CARB_DYNAMICLINK` (exported); `carb::getFramework()` is an inline that
  reads the `g_carbFramework` global set during framework init.
- `omni/kit/IApp.h`: `startup(const AppDesc&)`, `update()`, `isRunning()`,
  `shutdown()`, `postQuit(int)`, `getPythonScripting() -> IAppScripting*`.
  `IAppScripting::executeString(const char*, ...)` is the scripting hatch.
- `omni/kit/AppTypes.h` `AppDesc { const char* carbAppName; const char*
  carbAppPath; int argc; char** argv; }` — this is what `startup` takes, and
  **argv is forwarded to the extension/settings parser** (see below).
- `carb/settings/ISettings.h` → `detail/ISettings1.h`: `CARB_ABI` function
  pointers `setBool(path,bool)`, `setInt64(path,int64)`, `setFloat64(path,f64)`,
  `internalSetString(path, str, len)`, plus `getAsBool/getAsInt64/…` for the
  ready check. Acquire via `framework->acquireInterface<ISettings>()` or
  `carb::settings::getSettings()`.
- `omni.kit.app` plugin ships at `kit/kernel/plugins/libomni.kit.app.plugin.so`
  (+ `libcarb.settings.plugin.so`). `kit/exts` and `kit/extscache` are **empty
  in a fresh build** — extensions resolve/download at first app run, driven by
  the app's own extension manager. The in-process app resolves them the same
  way the subprocess does, because it runs the same extension manager.
- Fabric headers (v2 seam) under `kit/dev/fabric/include/omni/fabric/`:
  `stage/interface/IStageReaderWriter.h`, `IFabric.h`, `FabricUSD.h`.
- g++ 15.2 at `/usr/bin/g++`.

### Key insight: the existing launch *is* the argv vector

`server.jl` launches the subprocess as
`kit --empty --no-window --ext-folder exts … --enable omni.hydra.rtx …
--enable omni.rtx.index_composite … --/rtx/index/compositeEnabled=true … --exec
kit_server.py`. The `kit` executable is a thin bootstrap that forwards **all**
of that argv to `IApp::startup`, whose parser handles `--enable`,
`--ext-folder`, and `--/<setting>=<value>`. Therefore the in-process shim
reproduces the subprocess launch *exactly* by building the **same argv vector**
(minus `flock`/`timeout`/`kit_bin`/`--exec`) and handing it to `AppDesc.argv`.
No new launch semantics are invented. Two adjustments: pass **absolute**
`--ext-folder` paths (no `cd $KIT_RELEASE_DIR`), and add
`--/crashreporter/enabled=false` (see hazards).

## Native vs. scripting: what goes through which path

| Op                | v1 path                     | Rationale |
|-------------------|-----------------------------|-----------|
| lifecycle         | **native** `IApp`           | 5 vtable calls, stable ABI, cheap |
| settings          | **native** `ISettings`      | direct scalar setters; used for the composite-enabled ready check |
| open_stage        | **scripting** (`executeString`) | `omni.usd.get_context().open_stage()` — 1 line; the C++ `omni.usd` interface wiring is many-x the code |
| set_attr (camera) | **scripting**               | typed `pxr.Gf.Matrix4d` construction is trivial in Python, painful across a C ABI |
| render + capture  | **scripting** + Julia pump  | `capture_viewport_to_file` is a **Python-only** convenience — no C++ header ships in `kit/dev`; frame convergence is pumped natively from Julia |
| write_vdb         | **scripting**               | reuses `omni.volume` pyopenvdb exactly as today |

**Recommendation:** v1 keeps stage/capture/set_attr on the scripting hatch and
*reuses `kit_server.py`'s proven handler bodies almost verbatim* — the only
change is transport (no FIFO; Julia drives `update()`). This is the honest
minimum that reaches parity. Native capture (`omni.kit.renderer.capture` /
`omni.syntheticdata` AOV readback) is a **v2 seam** (recommended, not
required): it drops the Python capture and enables GPU-side readback, but its
interface is not a header we can bind against today (the ext isn't materialized
in this release tree), so committing to it in v1 would be speculative.

## Shape

### New subpackage: `lib/LibKitJL` (mirror `lib/LibOVRTX`)

**Decision: a new workspace subpackage, not a fold into OmniverseKitMakie.**
`LibOVRTX` / `OVRTX_jll` split the raw C bindings + native-artifact concerns
away from the Julia logic; `LibKitJL` mirrors that. Folding the g++ build step
into OmniverseKitMakie would drag a compiled-artifact `deps/build.jl` into an
otherwise pure-Julia package and couple its precompile to `KIT_RELEASE_DIR`.

`lib/LibKitJL` layout (add to the root `[workspace] projects` list via Pkg):
- `Project.toml` — `deps`: `Libdl`, `Libglvnd_jll` (GLVND preload); no JLL
  (Kit is env-located, unvendorable).
- `deps/build.jl` — g++ compile of `src/kitjl_shim.cpp` → `libkitjl.so`
  (see Build).
- `src/kitjl_shim.cpp` — the C++ shim (the only C++ in the repo besides ovrtx).
- `src/LibKitJL.jl` — `__init__` load-order dlopens + `@ccall` wrappers over the
  flat C ABI + `KitJlError` exception idiom (copy `LibOVRTX.jl`'s `check`/error
  shape).

### The flat `extern "C"` ABI (`kitjl_shim.cpp`)

Opaque handle keeps all carb/omni C++ types out of Julia (Julia never touches a
vtable). Thread-local last-error string, `LibOVRTX`-style.

```
KitJlApp* kitjl_startup(int argc, const char* const* argv);   // NULL on failure
void      kitjl_update(KitJlApp*);                            // one IApp::update()
int       kitjl_is_running(KitJlApp*);
int       kitjl_shutdown(KitJlApp*);                          // IApp::shutdown() + free
void      kitjl_post_quit(KitJlApp*, int code);

void      kitjl_set_setting_bool  (KitJlApp*, const char* path, int   value);
void      kitjl_set_setting_int   (KitJlApp*, const char* path, long long value);
void      kitjl_set_setting_float (KitJlApp*, const char* path, double value);
void      kitjl_set_setting_string(KitJlApp*, const char* path, const char* value);
int       kitjl_get_setting_bool  (KitJlApp*, const char* path);   // ready-check parity

int       kitjl_exec_string(KitJlApp*, const char* code);     // IAppScripting::executeString; 0 = ok
const char* kitjl_last_error(void);                           // thread-local, copy on read
const char* kitjl_sdk_version(void);                          // carbGetSdkVersion → pure-tier smoke
```

Internals of `kitjl_startup` (the C++ port of `kit_app.py`):
1. `carb::acquireFramework("kitjl")` (→ `omniCoreStart`, sets `g_carbFramework`).
2. Acquire `ISettings`; load the `omni.kit.app` plugin via
   `framework->loadPlugins({ searchPaths=[<CARB_APP_PATH>/kernel/plugins],
   wildcards=["omni.kit.app.plugin"] })`.
3. `IApp* app = framework->acquireInterface<omni::kit::IApp>()`.
4. `AppDesc{ "kit", getenv("CARB_APP_PATH"), argc, argv }`; `app->startup(desc)`.
   The `--enable …` / `--ext-folder …` / `--/…` argv entries load the composite
   extension chain and apply the RTX settings — identical to the subprocess.
5. Pump `app->update()` ~3× so the viewport/renderer come up (matches
   `kit_server.py`'s pre-ready updates); return `new KitJlApp{framework, app,
   settings, scripting}`.

(`carb::startupFramework(StartupFrameworkDesc)` from `StartupUtils.h` is an
alternative higher-level bootstrap that also parses argv config; use the
`kit_app.py`-equivalent low-level form above because it is the *proven*
in-process embedding and avoids `startupFramework`'s default config discovery.)

### Julia transport abstraction (`lib/OmniverseKitMakie`)

Introduce an abstract `KitTransport` with two implementations; `KitScreen`
holds `transport::KitTransport` instead of `server::KitServer`. **The
Julia-facing API — `KitScreen(scene)`, `Makie.colorbuffer`, `open_stage!`,
`render!`, `set_attr!`, `close` — does not change.**

- `SubprocessTransport` — thin wrapper over today's `KitServer`; `_open_stage`
  / `_render` / `_set_attr` / `_write_vdb` / `_close` forward to `rpc`.
- `InProcessTransport` — holds a `LibKitJL` handle + a workdir; the same ops
  either call native `kitjl_*` (settings, ready check) or build the identical
  Python one-liners and call `kitjl_exec_string`, then drive convergence with
  `kitjl_update`.

`authoring.jl` and the `stage_usda` emitters are **transport-agnostic and
unchanged**. `screen.jl`'s `_sync_camera!` / `render!` / `colorbuffer` call the
transport ops instead of `rpc` directly (mechanical rename `server` →
`transport`).

**Transport selection** (default = subprocess):
`KitScreen(scene; transport = :subprocess | :inprocess)`, overridable by
`OMK_KIT_TRANSPORT`. Default subprocess because it is proven *and* it is the
only path that coexists with in-process ovrtx (hazard b). `:inprocess`
constructs an `InProcessTransport`, which asserts ovrtx was never created in
this process (see hazard b) and builds the argv vector from the same knobs
`start_kit_server` uses (`width`/`height`/`extra_settings`/ext chain).

### Reusing `kit_server.py` as an embedded helper

On `InProcessTransport` construction, `kitjl_exec_string` a trimmed copy of
`kit_server.py` that **defines the handler bodies but omits the FIFO/asyncio
transport** (Julia is the transport now). Per-op then execs a one-liner that
calls a handler and writes its result to a small sentinel/response file that
Julia reads (mirrors the existing response-file discipline, without the pipe).
`render` splits cleanly across the ABI: `exec_string` kicks
`capture_viewport_to_file(...)`; Julia then pumps `kitjl_update()` and polls the
output file for bytes — the exact "belt-and-braces" loop `kit_server.py`
already runs, just driven from Julia. This keeps *one* source of truth for the
Python semantics (open_stage synchronization, settle frames, capture-await,
matrix4d/double3 set_attr, write_vdb) shared by both transports.

## Build (`lib/LibKitJL/deps/build.jl`)

Invoke `/usr/bin/g++` (env `CXX` override) roughly:
```
g++ -std=c++17 -O2 -fPIC -shared
    -I$KIT/kit/dev/include -I$KIT/kit/dev/include/omni
    -I$KIT/kit/dev/fabric/include            # v2 headers, harmless in v1
    src/kitjl_shim.cpp -o <out>/libkitjl.so
    -L$KIT/kit -lcarb -Wl,-rpath,$KIT/kit
```
where `$KIT = KIT_RELEASE_DIR` (default the DSX build). Link **only** `-lcarb`;
`omni.kit.app` and the extension chain are loaded at runtime by the framework,
not link-time. Emit `libkitjl.so` into a scratch/artifact dir recorded in a
`deps/deps.jl` the module reads.

**Graceful degradation:** if `KIT_RELEASE_DIR`/headers are absent, `build.jl`
does not fail the whole workspace precompile — it writes a marker and
`LibKitJL.__init__` warns that the in-process transport is unavailable
(subprocess still works). The **pure test tier** asserts the build *did*
produce `libkitjl.so` and export the symbols *only when headers are present*.

Rationale for g++-against-runtime-dir over a BinaryBuilder JLL: Kit is
NVIDIA-proprietary and env-located; a portable JLL is impossible. The local
build against the located runtime is the only option — the direct analogue of
how ovrtx is env-located rather than vendored.

## `__init__` load order (load-bearing — from `kit_app.py`)

Before `dlopen("libkitjl.so")`, `LibKitJL.__init__` must, in order:
1. Set `CARB_APP_PATH = $KIT_RELEASE_DIR/kit` if unset (framework config root).
2. Set `OMNI_KIT_ACCEPT_EULA = YES` if unset (hazard d — no `EULA_ACCEPTED`
   file ships, so absent → the app would prompt/exit).
3. `dlopen` **RTLD_GLOBAL**, in this order: `libcarb.so`, `libre2.so`,
   `libcares.so` (all from `$KIT/kit`), then `libpython3.12.so` (prefer system,
   fall back to `$KIT/kit/kernel/plugins/libpython3.12.so` — Python version
   from `kit/dev` `PlatformInfo`/the shipped libpython soname), then
   libGLU/GLVND (reuse `server.jl`'s `_ensure_libglu!` shim + `Libglvnd_jll`).
4. Then `dlopen("libkitjl.so", RTLD_LAZY|RTLD_GLOBAL)`.

This mirrors `LibOVRTX.__init__`'s RTLD_GLOBAL OpenGL-then-lib discipline.
`libGLU`/`GLVND` is required for the same reason as the subprocess (MDL SDK
`libneuray.so` needs it; the failure is the misleading "Invalid sync scope"
error).

## Update loop & `colorbuffer`

**v1 is single-threaded and synchronous — no free-running pump task.**
`kitjl_update()` is not thread-safe and must be called from one thread; a
background pump would also fight the capture's frame accounting. So convergence
is pumped *inside* the transport ops on the calling task:

`Makie.colorbuffer(::KitScreen)` (in-process) =
1. `_sync_camera!` → `kitjl_exec_string` sets `/World/Camera` `xformOp:transform`
   (matrix4d), pump `settle` updates;
2. pump `frames` × `kitjl_update()` (convergence/accumulation);
3. `kitjl_exec_string` kicks `capture_viewport_to_file`;
4. pump `kitjl_update()` polling the PNG until non-empty or timeout;
5. `Makie.FileIO.load` — byte-identical return to the subprocess path.

`kitjl_update()` can block tens of ms on the GPU; document that `colorbuffer`
blocks the calling task. Do **not** hold the GC off across the pump (unlike the
startup window — see hazard a); only `kitjl_startup` gets the guard. A
background pump for interactive/live use is a **documented v2 seam**: it must
pin to a dedicated thread and share a lock with `colorbuffer`'s bounded pump so
`update()` is never called concurrently.

## Hazards, with mitigations

**(a) carb breakpad vs Julia GC-safepoint signals.** Framework init +
first-renderer creation is a multi-second window in which carb's breakpad
installs SIG{ILL,ABRT,BUS,FPE,SEGV} handlers; a Julia GC-safepoint SIGSEGV
during it is fatal (the exact ovrtx bug). **Mitigation — reuse the ovrtx fix
pattern, both layers:**
- Wrap the `kitjl_startup` `ccall` in
  `OmniverseMakie`'s `SignalGuard.with_restored_signals` (from
  `src/binding/signals.jl`) — snapshots Julia's handlers to `SIG_DFL` and
  disables the GC for the window, serialized by the module lock. It is exported
  through OmniverseMakie's internals, which OmniverseKitMakie already imports as
  `OM`.
- Belt: pass `--/crashreporter/enabled=false` in the startup argv (the native
  analogue of `index_config.jl`'s `_disable_crashreporter` config surgery —
  Kit takes the setting directly on the command line, so no config-file copy is
  needed). This covers any handler re-arming after the guard's one-shot restore.

**(b) In-process Kit cannot coexist with in-process standalone ovrtx.** Both
load `libcarb.so` RTLD_GLOBAL and both want to own `g_carbFramework`,
`CARB_APP_PATH`, and the crashreporter singleton — two frameworks fighting.
**Mitigation:** a session picks ONE in-process backend.
`InProcessTransport` construction must **error clearly** if ovrtx has created a
renderer in this process (probe `LibOVRTX`'s handle / an OmniverseMakie
"renderer created" flag), and vice-versa. The **subprocess `KitScreen` remains
the coexistence path** (ovrtx in-process for meshes + Kit in a subprocess for
colored volumes, as today). Document this loudly in the transport docstring.

**(c) RTLD_GLOBAL load order.** See `__init__` above — `libcarb` → `libre2` →
`libcares` → `libpython` → GLVND → `libkitjl`, all RTLD_GLOBAL, `CARB_APP_PATH`
set first. Getting the order wrong reproduces `kit_app.py`'s implicit-RPATH
failures as symbol-resolution errors at startup.

**(d) OMNI_KIT_ACCEPT_EULA.** No `EULA_ACCEPTED` file ships; set the env var to
`YES` in `__init__` before startup (and honor a pre-set value). Without it the
app prompts on stdin and exits headless.

## Testing

**Pure tier** (`lib/OmniverseKitMakie/test`, extend or add `libkitjl` cases):
- If `kit/dev` headers present: assert `deps/build.jl` produced `libkitjl.so`
  and that `kitjl_startup`, `kitjl_update`, `kitjl_shutdown`,
  `kitjl_set_setting_bool`, `kitjl_exec_string`, `kitjl_sdk_version` resolve via
  `Libdl.dlsym`; assert `kitjl_sdk_version()` returns a non-empty string
  (exercises `carbGetSdkVersion` with no GPU / no app start).
- If headers absent: skip the build assert (transport unavailable is expected).
- Transport-abstraction unit tests with a fake transport: `KitScreen` dispatches
  `open_stage!`/`render!`/`set_attr!` to the transport; `stage_usda` output is
  unchanged (reuse the existing pure oracles verbatim).

**GPU tier** (env-gated on `KIT_RELEASE_DIR`, serialized on
`/tmp/omniversemakie-gpu.lock`):
- Start **one** in-process app (carb cannot cleanly restart in a process — one
  app per process), open the **same** colored-volume `stage_usda` the
  subprocess test uses, render one frame.
- Chroma oracle, identical to the subprocess A/B: `colormap=:viridis` → >500
  high-chroma px; achromatic twin (`to_colormap([:black,:white])`) → ~0 chroma.
- Parity assert: the in-process frame matches the subprocess frame under the
  same chroma oracle.
- **Process isolation:** the in-process GPU test must not share a Julia process
  with any ovrtx-in-process GPU test (hazard b) — separate test file / separate
  `julia` invocation, gated the same way the existing GPU sets are.

## Phasing

- **v1 (this iteration):** lifecycle + settings (native) + stage open + camera
  set_attr + render/capture (scripting hatch, reusing `kit_server.py` bodies) →
  a colored volume frame **in-process**, at parity with the subprocess path.
  Subprocess stays default; `:inprocess` opt-in.
- **v2 (documented seam, NOT built):**
  - Fabric `IStageReaderWriter` (`omni/fabric/stage/interface/IStageReaderWriter.h`)
    CUDA-pointer geometry writes — push device buffers into the running stage
    with no USD text round-trip (the in-process analogue of ovrtx's
    `gpu_update_mesh!`).
  - `omni.syntheticdata` CUDA AOV readback → `CuArray` → the existing CUDA-GL
    blit, replacing the PNG capture (drops the `capture_viewport_to_file`
    scripting hatch and the file poll).
  - Native capture via `omni.kit.renderer.capture` if/when its interface
    materializes in the release tree.
  - Background pump task for interactive/live rendering (thread-pinned, locked
    against `colorbuffer`).

## Implementation checklist (fresh-engineer order, zero prior context)

1. **Scaffold `lib/LibKitJL`.** `Pkg.generate` (or copy `lib/LibOVRTX`'s
   layout); add `"lib/LibKitJL"` to the root `Project.toml` `[workspace]
   projects` list and `LibKitJL = {path="lib/LibKitJL"}` as a source **via Pkg
   ops** (do not hand-edit beyond the tables Pkg leaves). Deps: `Libdl`,
   `Libglvnd_jll`.
2. **Write `src/kitjl_shim.cpp`** — the flat `extern "C"` ABI above:
   `kitjl_startup` (acquireFramework → loadPlugins `omni.kit.app.plugin` →
   acquireInterface `IApp` → `startup(AppDesc)` → 3 warmup updates),
   `kitjl_update/is_running/shutdown/post_quit`, the four
   `kitjl_set_setting_*` + `kitjl_get_setting_bool` over `ISettings`,
   `kitjl_exec_string` over `IApp::getPythonScripting()->executeString`,
   thread-local `kitjl_last_error`, `kitjl_sdk_version` over `carbGetSdkVersion`.
   Every entry point try/catches into the thread-local error (no C++ exception
   crosses the ABI).
3. **Write `deps/build.jl`** — locate `KIT_RELEASE_DIR`; if headers present,
   g++-compile (flags above) to `libkitjl.so` and write `deps/deps.jl` with the
   path; else write the "unavailable" marker without failing precompile.
4. **Write `src/LibKitJL.jl`** — `__init__` load order (§ load order:
   `CARB_APP_PATH`, `OMNI_KIT_ACCEPT_EULA`, RTLD_GLOBAL `libcarb`/`libre2`/
   `libcares`/`libpython`/GLVND, then `libkitjl`); `@ccall` wrappers; a
   `KitJlError` exception + `check`/`kitjl_last_error` idiom copied from
   `LibOVRTX.jl`.
5. **Pure smoke test** for LibKitJL: build produced the lib, symbols resolve,
   `kitjl_sdk_version()` non-empty (headers-present gate).
6. **Add the transport abstraction to `lib/OmniverseKitMakie`.** Define abstract
   `KitTransport`; refactor today's `KitServer` path into `SubprocessTransport`;
   rename `KitScreen.server` → `KitScreen.transport`; route `open_stage!` /
   `render!` / `set_attr!` / `_write_vdb` / `close` / `_sync_camera!` through
   transport ops. **No behavior change** — existing subprocess GPU test must
   still pass (verification gate).
7. **Add `InProcessTransport`** — depends on `LibKitJL` (add via Pkg); build the
   argv vector from the `start_kit_server` knobs (absolute `--ext-folder`, the
   `--enable` chain, `--/rtx/index/*` and resolution settings, plus
   `--/crashreporter/enabled=false`); call `kitjl_startup` **inside**
   `OM.SignalGuard.with_restored_signals`; on construction assert ovrtx has no
   live renderer in this process (hazard b).
8. **Embed the Python helper** — exec the trimmed `kit_server.py` (handlers, no
   FIFO/asyncio) once at construction; implement `_open_stage`/`_set_attr`/
   `_render`/`_write_vdb` as `kitjl_exec_string` + `kitjl_update` pump + sentinel
   file read (§ reusing kit_server.py). `_render` = pump settle+frames, kick
   capture, pump-poll the PNG.
9. **Wire transport selection** — `KitScreen(scene; transport=:subprocess)`
   default, `:inprocess` opt-in, `OMK_KIT_TRANSPORT` override. Confirm
   `Makie.colorbuffer` / `close` are byte-for-byte API-identical across
   transports.
10. **GPU parity test** (env-gated + GPU-lock serialized, own process): one
    in-process app, colored-volume `stage_usda`, viridis vs achromatic chroma
    oracle, and equality against the subprocess frame. Keep it isolated from any
    ovrtx-in-process GPU test.
11. **Docs** — update `lib/OmniverseKitMakie/README.md` and the parent design's
    Phase 2 pointer to "shipped: in-process transport (opt-in); subprocess
    default". Record the v2 Fabric/syntheticdata seams here as the next
    milestone.
12. **Verify** — full gate: subprocess tier unchanged (green), LibKitJL pure
    tier green, in-process GPU parity green when `KIT_RELEASE_DIR` is set;
    confirm the subprocess path still coexists with ovrtx-in-process (hazard b
    guard fires only on the in-process transport).

## Implementation status (2026-07-15) — DONE through step 9; step 10 blocked

Steps 1–9 + 11 landed and verified; step 12's in-process GPU parity is
**blocked by an in-process Kit-startup deadlock** discovered during step 10.

**Green:**
- Steps 1–5: `lib/LibKitJL` scaffolded, `kitjl_shim.cpp` written and g++-built
  to `libkitjl.so`, `deps/build.jl` (graceful degradation verified),
  `LibKitJL.jl` (`__init__` RTLD_GLOBAL load order: `CARB_APP_PATH` +
  `OMNI_KIT_ACCEPT_EULA` + `libcarb`/`libre2`/`libcares`/Kit-bundled
  `libpython3.12`/GLVND/`libkitjl`). Pure smoke tier GREEN: build present, all
  13 ABI symbols resolve, `kitjl_sdk_version()` → `209.0.3+release…` (no GPU).
- Steps 6–9: `KitTransport` abstraction; `SubprocessTransport` (subprocess GPU
  A/B **regression-green** — viridis nb=7620/ch=6536, gray ch=0) +
  `InProcessTransport`; `KitScreen.transport`; transport-kind resolution +
  `OMK_KIT_TRANSPORT`; embedded Python helper (trimmed `kit_server.py`).
- Step 11: docs updated.

**Deviations from the spec, and why:**
- `kitjl_startup` uses the FULL `carb::startupFramework(argv)` (the C++ `kit`
  binary's `OMNI_CORE_INIT(argc,argv)` path), not only the lighter kit_app.py
  `acquireFrameworkAndRegisterBuiltins`. The spec preferred the low-level form
  to avoid startupFramework's config discovery; in practice the low-level form
  deadlocked identically, and startupFramework is what the real `kit` binary
  does — kept for fidelity. (Plugin search path forced to
  `<CARB_APP_PATH>/kernel/plugins` because the host executable is `julia`.)
- `kitjl_shutdown` does postQuit + a bounded update pump, **not** the blocking
  `IApp::shutdown()` / framework release — headless Kit teardown stalls (as
  `kit_server.py` notes, resorting to `os._exit`); the process exits after.
- Hazard-b ovrtx probe: only the "one in-process Kit app per process" guard is
  enforced programmatically (module `_INPROCESS_HANDLE`). A hard ovrtx-renderer
  probe is not added because the core exposes no such flag and may not be
  modified; libovrtx-dynamic.so was verified to export NO carb framework
  symbols (no global symbol clash), and the constraint is documented loudly.

**BLOCKER (step 10): in-process startup hang — ROOT-CAUSED 2026-07-16.**
`IApp::startup` spins (main thread 95% CPU, state R) during the
`omni.usd_resolver` Python-extension `dlopen`; the cpython ext's C++
static-initializer calls `carb::getCachedInterface<ISettings/IDictionary>`,
whose by-name interface lookup (an rwlock-guarded `unordered_map<string>`
probe in libcarb) never terminates.

The earlier explanation in this file (Julia co-hosting / signals / tokens
not initialized / libstdc++) was **WRONG**. Root cause, established by
controlled bisection (scripts in the session scratchpad; a Julia-free C
harness reproduces the hang in ~2 s):

- **NOT Julia.** A pure-C program that dlopens libcarb + calls `kitjl_startup`
  hangs identically — no Julia in the process. (Invalidates all four prior
  "fixes", which only varied Julia-side knobs.)
- **NOT gcov / libpython-preload / libstdc++ / framework-init-sequence.**
  Excluding the coverage-instrumented `libomni.kit.app.gcov.plugin.so` (which
  a folder scan pulls in) only removed a ~9× *slowdown*; dropping the
  libpython preload matched the real kit's timing; the C harness already uses
  the system libstdc++; and our `acquireFrameworkAndRegisterBuiltins` +
  `startupFramework` sequence is exactly what `OMNI_CORE_INIT` (the kit
  binary's own macro) expands to. Each ruled out with a run.
- **THE CAUSE: `OMNI_APP_GLOBALS` (the omni-core client context) must live in
  the process *main executable*, not in a dlopened/preloaded library.** Decisive
  experiment: a C++ **main executable** with `OMNI_APP_GLOBALS` + the identical
  init + identical argv **passes `omni.usd_resolver` and reaches the running
  update loop**, while the same logic in a dlopened `libkitjl.so` (called from
  any host exe) hangs — *even with `LD_PRELOAD=libkitjl.so`*. Kit's client
  mechanism resolves those globals via the main-program handle
  (`dlsym(RTLD_DEFAULT)` / `dlopen(NULL)`), which searches the executable's
  symbol table and never a library's. There is no way to add those symbols to
  an already-running `julia` (or any non-Kit) executable.

**Consequence (architectural, not a shim bug):** Kit **cannot** be hosted
in-process with Julia (or any non-Kit binary) as the main executable — the
`libkitjl.so`-in-a-library approach is fundamentally blocked by this
requirement. Viable options: (a) the **subprocess transport** (shipped,
working — the default `KitScreen` path); or (b) **invert the host**: a small
Kit-globals main executable that embeds `libjulia` (Julia runs *inside* a
Kit-host process) — a larger design change that gives up "Julia is the host."
`LD_PRELOAD`, `RTLD_DEEPBIND`, pre-`dlopen` ordering, and pre-init threads
were considered/tested and do **not** address the main-exe requirement. The
in-process GPU test stays opt-in (`OMK_TEST_INPROCESS_GPU=1`, documented skip);
`InProcessTransport` errors clearly rather than hanging where feasible.
