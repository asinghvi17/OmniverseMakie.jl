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
            "OVRTX_LIBRARY_PATH" => _HELPER_OVRTX_LIB,
            "OM_USDA"            => _HELPER_USDA,
            "PATH"               => get(ENV, "PATH", ""),
            "HOME"               => get(ENV, "HOME", ""),
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
