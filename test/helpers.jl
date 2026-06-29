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
"""
function run_ovrtx_subprocess(prog::String; timeout::Int=300)
    script = tempname() * ".jl"
    exitcode = -1
    output = ""
    try
        open(script, "w") do io
            print(io, prog)
        end
        cmd = setenv(
            `julia --project=$(_HELPER_REPO_ROOT) $script`,
            "OVRTX_LIBRARY_PATH"        => _HELPER_OVRTX_LIB,
            "OM_USDA"                   => _HELPER_USDA,
            "PATH"                      => get(ENV, "PATH", ""),
            "HOME"                      => get(ENV, "HOME", ""),
            # GLMakie (M5) needs a real X display for GL context creation.
            # Forward the Xwayland display variables proven to work on this host.
            "DISPLAY"                   => get(ENV, "DISPLAY", ":0"),
            "XAUTHORITY"                => get(ENV, "XAUTHORITY", "/run/user/1000/.mutter-Xwaylandauth.QRQ4Q3"),
            "XDG_RUNTIME_DIR"           => get(ENV, "XDG_RUNTIME_DIR", "/run/user/1000"),
            # GLMakie cannot precompile without a display available.  AUTO=0 skips
            # startup auto-precompile so the subprocess loads OmniverseMakie cleanly.
            "JULIA_PKG_PRECOMPILE_AUTO" => "0",
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
