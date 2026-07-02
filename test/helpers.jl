# Shared subprocess test harness for OmniverseMakie.
# Provides run_ovrtx_subprocess used by M1+ test files.
# Pattern extracted from m0_render_test.jl:46-87.

const _HELPER_OVRTX_LIB = get(ENV, "OVRTX_LIBRARY_PATH",
    "/home/juliahub/temp/omniverse-makie/references/ovrtx/examples/python/minimal/.venv/lib/python3.13/site-packages/ovrtx/bin/libovrtx-dynamic.so")
const _HELPER_REPO_ROOT = joinpath(@__DIR__, "..")
const _HELPER_USDA = get(ENV, "OM_USDA",
    "/home/juliahub/temp/omniverse-makie/references/ovrtx/examples/c/minimal/torus-plane.usda")

"""
    run_ovrtx_subprocess(prog::String; timeout=300, kill_grace=10, env=()) -> (exitcode, output)

Write `prog` to a temp `.jl` file, run it as a child `julia --project=<repo>`
process with the standard ovrtx environment, wait up to `timeout` seconds for a
clean exit, collect stdout, and return `(exitcode, stdout_text)`.  The temp file
is always removed in a `finally` block.

On timeout the watchdog escalates SIGTERM → wait `kill_grace` s → SIGKILL (so even
a GPU-wedged child that ignores SIGTERM is reaped), then ALWAYS `wait`s the child
before reading its exit code and output.  The returned `exitcode` is therefore
TRUTHFUL — callers can assert `exitcode == 0`:

  * `0`       — the child exited 0 (real success);
  * `-9`      — the watchdog timed out and killed it;
  * `-N`      — the child died from uncaught signal `N` (e.g. a crash / segfault);
  * otherwise — the child's own nonzero exit code.

`env` forwards extra `name => value` pairs (a single `Pair` or an iterable of Pairs)
into the child process — used for opt-in vars such as `OMNIVERSEMAKIE_INDEX_LIBS`
(Volumes M1).  `setenv` REPLACES the child environment, so vars neither listed below
nor forwarded via `env` do NOT leak into the child (the disabled-path volume test
relies on this to exercise the "no volume env" branch).

M6.A: The subprocess LOAD_PATH stacks the test project (which has GLMakie in
[deps]) after the main project (which has it only in [weakdeps]).  This lets
subprocess programs do `using OmniverseMakie, GLMakie` — GLMakie is found via
the test project — while `using OmniverseMakie` alone still doesn't load it.
"""
function run_ovrtx_subprocess(prog::String; timeout::Int=300, kill_grace::Real=10, env=())
    script = tempname() * ".jl"
    exitcode = -1
    output = ""
    try
        open(script, "w") do io
            print(io, prog)
        end
        # Stack test/ project after the main project so GLMakie (and other test
        # weakdeps) are loadable without being hard deps of OmniverseMakie itself.
        _test_dir = joinpath(_HELPER_REPO_ROOT, "test")
        # "@" = active project (from --project); _test_dir = test env (has GLMakie)
        _load_path = join(["@", _test_dir, "@v#.#", "@stdlib"], ":")
        env_pairs = Pair{String,String}[
            "OVRTX_LIBRARY_PATH"        => _HELPER_OVRTX_LIB,
            "OM_USDA"                   => _HELPER_USDA,
            "PATH"                      => get(ENV, "PATH", ""),
            "HOME"                      => get(ENV, "HOME", ""),
            # GLMakie (M5) needs a real X display for GL context creation.  Forward the
            # Xwayland display vars (ENV first).  DEV-BOX DEFAULTS: the XAUTHORITY fallback is
            # session-specific (mutter regenerates the `.mutter-Xwaylandauth.*` suffix each
            # Wayland login) and XDG_RUNTIME_DIR hardcodes UID 1000 — same dev-box-only tier as
            # _HELPER_OVRTX_LIB / _HELPER_USDA above; on a fresh session or CI, set these in ENV.
            "DISPLAY"                   => get(ENV, "DISPLAY", ":0"),
            "XAUTHORITY"                => get(ENV, "XAUTHORITY", "/run/user/1000/.mutter-Xwaylandauth.QRQ4Q3"),
            "XDG_RUNTIME_DIR"           => get(ENV, "XDG_RUNTIME_DIR", "/run/user/1000"),
            # GLMakie cannot precompile without a display available.  AUTO=0 skips
            # startup auto-precompile so the subprocess loads OmniverseMakie cleanly.
            "JULIA_PKG_PRECOMPILE_AUTO" => "0",
            # M6.A: when CUDA.jl is loaded (GPU-direct path), it otherwise dlopens the
            # forward-compatible libcuda from CUDA_Driver_jll, which COLLIDES with the
            # system driver that ovrtx's carb.cudainterop plugin uses → ovrtx
            # createDevices fails with "driver API result: 3" (CUDA_ERROR_NOT_INITIALIZED).
            # Forcing the LOCAL system driver makes both share one libcuda. No-op for
            # the non-CUDA subprocess tests (CUDA.jl is never loaded there).
            "JULIA_CUDA_USE_COMPAT"     => "false",
            # M6.A: stack the test project so GLMakie (a weakdep of OmniverseMakie)
            # is loadable by subprocess programs that do `using GLMakie`.
            "JULIA_LOAD_PATH"           => _load_path,
        ]
        # Volumes M1: forward caller-supplied env pairs (e.g. OMNIVERSEMAKIE_INDEX_LIBS).
        # `env` is a single `name=>value` Pair or an iterable of Pairs.  These append to (and
        # can override) the base set above; anything not present is absent from the child.
        for (k, v) in (env isa Pair ? (env,) : env)
            push!(env_pairs, String(k) => String(v))
        end
        cmd = setenv(`julia --project=$(_HELPER_REPO_ROOT) $script`, env_pairs)
        out = IOBuffer()
        err = IOBuffer()
        p = run(pipeline(cmd; stdout=out, stderr=err); wait=false)
        # Watchdog: wait up to `timeout` s for a clean exit; timedwait returns
        # :timed_out on expiry.  Then escalate SIGTERM → grace → SIGKILL so a
        # wedged child that swallows SIGTERM is still reaped.
        timed_out = timedwait(() -> !process_running(p), float(timeout)) === :timed_out
        if timed_out && process_running(p)
            kill(p)                                   # SIGTERM: ask it to stop
            if timedwait(() -> !process_running(p), float(kill_grace)) === :timed_out
                process_running(p) && kill(p, Base.SIGKILL)  # still alive: force it
            end
        end
        wait(p)                                       # ALWAYS reap before reading exit state/output
        # Truthful exit code (see docstring): 0 only on real success; a timed-out
        # kill is -9; an uncaught signal (crash) is -signal; else the child's code.
        exitcode = success(p) ? 0 :
                   timed_out  ? -9 :
                   p.termsignal != 0 ? -p.termsignal : p.exitcode
        output = String(take!(out))
        errtext = String(take!(err))
        isempty(errtext) || @info "subprocess stderr" text=errtext
    finally
        isfile(script) && rm(script)
    end
    return (exitcode, output)
end
