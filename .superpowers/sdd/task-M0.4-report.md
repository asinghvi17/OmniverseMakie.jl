# Task M0.4 Report — Neutralize carb breakpad crash reporter

## Status: DONE

---

## Files Changed

### `src/binding/signals.jl` (new)

Provides `module SignalGuard` with three public functions:

- `snapshot() -> Dict{Int,Vector{UInt8}}` — atomically replaces the five crash-reporter signals (SIGILL=4, SIGABRT=6, SIGBUS=7, SIGFPE=8, SIGSEGV=11) with SIG_DFL and returns the previous handler blobs. By installing SIG_DFL before `create_renderer`, carb breakpad chains to a harmless handler rather than Julia's complex SA_ONSTACK handler.
- `restore(saved)` — reinstalls the saved handler blobs via `sigaction(2)`.
- `with_restored_signals(f)` — convenience wrapper: snapshot → f() → restore.

Key design note: `snapshot()` atomically SAVES AND REPLACES (via the 3-argument `sigaction`), rather than read-then-write. A zeroed-out struct sigaction is SIG_DFL.

### `test/m0_signals_test.jl` (new)

Subprocess test that:
1. Writes the render program to a `tempname()` script file (not `-e`)
2. Runs `julia --project=<repo> <script>` with `OVRTX_LIBRARY_PATH` set
3. Asserts `p.exitcode == 0` and `contains(output, "OK")`

Critical discovery: using `julia -e '<code>'` causes a crash INSIDE `ovrtx_create_renderer` (termsignal=11 during Vulkan init), while running the same code as a `.jl` script file works correctly with the crash occurring only at process exit (the true M0.4 scenario). The test uses script-file mode accordingly.

### `test/runtests.jl` (updated)

Added `include("m0_signals_test.jl")` at the end.

### `Project.toml` (updated via Pkg)

Added `CEnum` as a direct dependency (via `Pkg.add`). CEnum was previously only a transitive dep via LibOVRTX and was not directly loadable in the test environment (Julia 1.12 requires explicit listing). This also fixes the pre-existing M0.2 failure when running `julia --project=. test/runtests.jl` directly.

---

## Test Run

Command: `julia --project=<repo> test/runtests.jl` with `OVRTX_LIBRARY_PATH` set.

```
Test Summary:        | Pass  Total  Time
M0.1 workspace loads |    2      2  0.0s
Test Summary:      | Pass  Total  Time
M0.2 generated ABI |    5      5  0.0s
Test Summary:               | Pass  Total  Time
M0.3 LibOVRTX loads + links |    2      2  0.0s
Test Summary:                       | Pass  Total  Time
M0.4 renderer process exits cleanly |    2      2  4.0s
```

All 11 tests pass (2+5+2+2). No regressions.

---

## Investigation Notes

### Root cause of exit 139

`ovrtx_create_renderer` loads `carb.crashreporter-breakpad.plugin`, which installs its own SIGSEGV/SIGBUS/SIGILL/SIGABRT/SIGFPE handlers via `rt_sigaction`. These handlers chain to whatever was installed before them. Julia has complex SA_ONSTACK|SA_SIGINFO handlers for these signals. When Julia's scheduler background thread receives a SIGSEGV during teardown at exit, the chain fires: breakpad handler → Julia's handler → SIGSEGV kills the process → exit code 139.

### Why `snapshot()` uses `_swap_default` (not read-only `_getaction`)

The brief suggested a read-only snapshot then restore. However, strace showed Julia RE-INSTALLS its signal handlers after `_getaction` reads them, so a read-only snapshot captures handlers that are immediately overwritten by Julia — breakpad then chains to Julia's REINSTALLED handlers and the exit crash recurs.

The fix: atomically swap to SIG_DFL BEFORE `create_renderer`. Breakpad then chains to SIG_DFL (no-op at exit), and after `create_renderer` returns, `restore(saved)` puts Julia's original handlers back.

### `-e` vs script-file mode

`julia -e '<code>'` causes a SIGSEGV inside `ovrtx_create_renderer` itself (not at exit), apparently due to a difference in the Julia module initialization context that affects the dynamic linker namespace when loading Vulkan. The identical code in a `.jl` script file runs create_renderer successfully. The test writes the subprocess program to a `tempname()` file to avoid this.

### Spike confirmation

The spike's `render.jl` (no SignalGuard, ends with `_exit(0)`) exits 0/termsignal=0 — confirming `_exit` bypasses the crash. With `_exit` removed and no SignalGuard, same spike code exits 139. With SignalGuard.snapshot()/restore() bracketing create_renderer, same code exits 0/0 with "OK" — confirming the mechanism works end-to-end.

---

## Interface Delivered

```julia
include("src/binding/signals.jl")   # or include from the repo root

# Pattern for M0.5 OV.Renderer(...) constructor:
save = SignalGuard.snapshot()
check(ovrtx_create_renderer(cfg, rref), "create")
SignalGuard.restore(save)

# Or the convenience form:
SignalGuard.with_restored_signals() do
    check(ovrtx_create_renderer(cfg, rref), "create")
end
```
