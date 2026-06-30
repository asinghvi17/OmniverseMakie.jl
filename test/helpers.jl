# Shared subprocess test harness for OmniverseMakie.
# Provides run_ovrtx_subprocess used by M1+ test files.
# Pattern extracted from m0_render_test.jl:46-87.

const _HELPER_OVRTX_LIB = get(ENV, "OVRTX_LIBRARY_PATH",
    "/home/juliahub/temp/omniverse-makie/references/ovrtx/examples/python/minimal/.venv/lib/python3.13/site-packages/ovrtx/bin/libovrtx-dynamic.so")
const _HELPER_REPO_ROOT = joinpath(@__DIR__, "..")
const _HELPER_USDA = get(ENV, "OM_USDA",
    "/home/juliahub/temp/omniverse-makie/references/ovrtx/examples/c/minimal/torus-plane.usda")

"""
    run_ovrtx_subprocess(prog::String; timeout=300) -> (exitcode, output)

Write `prog` to a temp `.jl` file, run it as a child `julia --project=<repo>`
process with the standard ovrtx environment, wait up to `timeout` seconds
(watchdog kills on expiry), collect stdout, and return `(exitcode, stdout_text)`.
The temp file is always removed in a `finally` block.

M6.A: The subprocess LOAD_PATH stacks the test project (which has GLMakie in
[deps]) after the main project (which has it only in [weakdeps]).  This lets
subprocess programs do `using OmniverseMakie, GLMakie` — GLMakie is found via
the test project — while `using OmniverseMakie` alone still doesn't load it.
"""
function run_ovrtx_subprocess(prog::String; timeout::Int=300)
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
        cmd = setenv(
            `julia --project=$(_HELPER_REPO_ROOT) $script`,
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
        )
        out = IOBuffer()
        err = IOBuffer()
        p = run(pipeline(cmd; stdout=out, stderr=err); wait=false)
        # Watchdog: timedwait returns :ok when the condition becomes true,
        # :timed_out otherwise.  Kill the process if it exceeds timeout.
        timedwait(() -> !process_running(p), float(timeout))
        process_running(p) && kill(p)
        exitcode = p.exitcode
        output = String(take!(out))
        errtext = String(take!(err))
        isempty(errtext) || @info "subprocess stderr" text=errtext
    finally
        isfile(script) && rm(script)
    end
    return (exitcode, output)
end
