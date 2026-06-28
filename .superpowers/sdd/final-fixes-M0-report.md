# Final-Review Fixes M0 — Report

Branch: `feat/m0-foundation`  
Base HEAD before this work: `6a22340`  
Date: 2026-06-28

---

## Fix 1 — Guard `StepResult.close` and `map_cpu` against a closed Renderer

### File: `src/binding/OV.jl`

**Before (`Base.close(sr::StepResult)`):**
```julia
function Base.close(sr::StepResult)
    sr.open || return
    L.ovrtx_destroy_results(sr.r.ptr, sr.handle)
    sr.open = false
    return nothing
end
```

**After:**
```julia
function Base.close(sr::StepResult)
    sr.open || return
    sr.r.alive && L.ovrtx_destroy_results(sr.r.ptr, sr.handle)  # pool already freed if renderer closed
    sr.open = false
    return nothing
end
```

**Before (`map_cpu`):** No alive check at top of function.

**After:** Added guard as first line of `map_cpu`:
```julia
sr.r.alive || error("map_cpu: the StepResult's Renderer is already closed")
```

**Rationale:** If `Renderer.close()` runs before all `StepResult` finalizers fire (common during GC or explicit `close(r)` before `close(sr)`), `ovrtx_destroy_results` would be called with a dangling/NULL renderer pointer. The `r.alive` check skips the C call when the renderer's result pool is already freed. The `map_cpu` guard surfaces the error loudly instead of dereferencing into freed memory.

---

## Fix 2 — Remove unused `Libdl` from root package

### File: `Project.toml`

**Before:**
```toml
[deps]
ColorTypes = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
FixedPointNumbers = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
LibOVRTX = "48e69356-47aa-4e0c-8089-4409475c3dd9"
Libdl = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
```

**After:**
```toml
[deps]
ColorTypes = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
FixedPointNumbers = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
LibOVRTX = "48e69356-47aa-4e0c-8089-4409475c3dd9"
```

**Method:** `Pkg.rm("Libdl")` — not hand-edited TOML.  
`lib/LibOVRTX/Project.toml` is unchanged (it keeps its own `Libdl` dep for `dlopen`).

---

## Fix 3 — `OM_USDA` override in `test/m0_update_test.jl`

### File: `test/m0_update_test.jl`

**Before (constant definition, line 8):**
```julia
const _UPDATE_USDA       = "/home/juliahub/temp/omniverse-makie/references/ovrtx/examples/c/minimal/torus-plane.usda"
```

**After:**
```julia
const _UPDATE_USDA       = get(ENV, "OM_USDA",
    "/home/juliahub/temp/omniverse-makie/references/ovrtx/examples/c/minimal/torus-plane.usda")
```

**Before (subprocess script constant):**
```julia
const USDA    = $(repr(_UPDATE_USDA))
```

**After:**
```julia
const USDA    = ENV["OM_USDA"]
```

**Before (`setenv(...)` call):**
```julia
cmd = setenv(
    `julia --project=$(_UPDATE_REPO_ROOT) $script`,
    "OVRTX_LIBRARY_PATH" => _UPDATE_OVRTX_LIB,
    "PATH"               => get(ENV, "PATH", ""),
    "HOME"               => get(ENV, "HOME", ""),
)
```

**After:**
```julia
cmd = setenv(
    `julia --project=$(_UPDATE_REPO_ROOT) $script`,
    "OVRTX_LIBRARY_PATH" => _UPDATE_OVRTX_LIB,
    "OM_USDA"            => _UPDATE_USDA,
    "PATH"               => get(ENV, "PATH", ""),
    "HOME"               => get(ENV, "HOME", ""),
)
```

This mirrors exactly what `m0_render_test.jl` already did, allowing CI overrides of the USDA path via `OM_USDA`.

---

## Fix 4 — Exception-safe `snapshot()` in `src/binding/signals.jl`

### File: `src/binding/signals.jl`

**Before:**
```julia
snapshot() = Dict(sig => _swap_default(sig) for sig in _SIGS)
```

**After:**
```julia
function snapshot()
    saved = Dict{Int,Vector{UInt8}}()
    try
        for sig in _SIGS
            saved[sig] = _swap_default(sig)
        end
    catch
        restore(saved)   # roll back what we already swapped
        rethrow()
    end
    return saved
end
```

**Rationale:** The one-liner dict comprehension has no error recovery: if `sigaction` fails on signal N after succeeding on signals 1..N-1, those already-swapped handlers are stranded at SIG_DFL and Julia's crash handlers are never restored. The new form rolls back partial swaps via `restore(saved)` before re-raising the error.

---

## Item 5 — `_exit` / Render-Teardown Investigation

### Setup

Scratch script written to:
`/tmp/.../scratchpad/scratch_render_no_exit.jl`

Identical to `_RENDER_PROG` but with the trailing `_exit(0)` block removed (the `flush`+`ccall` lines). Ran via:

```
OVRTX_LIBRARY_PATH=".../libovrtx-dynamic.so" \
OM_USDA=".../torus-plane.usda" \
julia --project=/home/juliahub/temp/omniverse-makie/OmniverseMakie.jl \
      /tmp/.../scratch_render_no_exit.jl
```

### Observation

```
[Warning] [omni.log] Source: omni.hydra was already registered.
2026-06-28T04:15:31Z [Warning] [omni.platforminfo.plugin] failed to open the default display.  Can't verify X Server version.
2026-06-28T04:15:31Z [Warning] [omni.rtx] CPU performance profile is set to powersave...
SIZE=(1080, 1920) NONBLACK=2073584
OK
EXIT CODE: 0
```

**Exit code without `_exit(0)`: 0** — no crash, no SIGSEGV.

### Resolution

Per stop rule (b): `_exit(0)` is vestigial. Removed from **both** render test subprocess programs:

- `test/m0_render_test.jl` — removed the three-line `_exit` block inside `_RENDER_PROG`; updated header comment from "hard-exits via _exit(0)" to "exits normally (clean teardown verified: no _exit needed)".
- `test/m0_update_test.jl` — removed the three-line `_exit` block inside `_UPDATE_PROG`; updated header comment similarly.

The tests now prove honest clean teardown (no bypass of Julia's shutdown sequence).

---

## Test Run

Command:
```julia
using Pkg
Pkg.activate("/home/juliahub/temp/omniverse-makie/OmniverseMakie.jl")
Pkg.test()
```

Full output:
```
  Activating project at `~/temp/omniverse-makie/OmniverseMakie.jl`
     Testing OmniverseMakie
     Testing Running tests...
Test Summary:        | Pass  Total  Time
M0.1 workspace loads |    2      2  0.0s
Test Summary:      | Pass  Total  Time
M0.2 generated ABI |    5      5  0.0s
Test Summary:               | Pass  Total  Time
M0.3 LibOVRTX loads + links |    2      2  0.0s
Test Summary:                       | Pass  Total  Time
M0.4 renderer process exits cleanly |    2      2  4.0s
Test Summary:                           | Pass  Total  Time
M0.5 OV.Renderer create/close lifecycle |    2      2  5.7s
Test Summary:                                         | Pass  Total  Time
M0.6 Julia-native render (torus, LdrColor, non-black) |    4      4  8.1s
Test Summary:                                           | Pass  Total   Time
M0.7 write_xform! moves torus (changed >= 50000 pixels) |    3      3  17.1s
     Testing OmniverseMakie tests passed
```

**7 testsets / 20 assertions / all PASS / subprocess exits 0.**

- M0.6: SIZE=(1080,1920), NONBLACK=2073584 (≥ 2073500 threshold ✓)
- M0.7: CHANGED_PIXELS well above 50000 threshold ✓
