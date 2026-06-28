using Test

const _OVRTX_LIB_PATH = get(ENV, "OVRTX_LIBRARY_PATH",
    "/home/juliahub/temp/omniverse-makie/references/ovrtx/examples/python/minimal/.venv/lib/python3.13/site-packages/ovrtx/bin/libovrtx-dynamic.so")
const _REPO_ROOT = joinpath(@__DIR__, "..")
const _OM_SRC    = joinpath(_REPO_ROOT, "src")
const _SIGNALS_JL = joinpath(_OM_SRC, "binding", "signals.jl")

# The subprocess program: load LibOVRTX, include signals.jl, create+restore+destroy renderer.
# - Uses --project=<repo root> so the workspace manifest resolves LibOVRTX and all its deps.
# - Written to a temp script file (not -e) — running as a file avoids a Julia startup quirk
#   where `julia -e code` crashes inside ovrtx_create_renderer while the same code in a
#   script file runs cleanly. Root cause: -e mode evaluates code in a different module
#   initialization context; the renderer's internal Vulkan init is sensitive to how the
#   dynamic linker namespace is set up at that point.
# - Uses empty config (C_NULL entries pointer, length 0).
# - Prints "OK" on success so a silent early-exit-0 cannot falsely pass.
function _write_render_script(path::String, signals_jl::String)
    open(path, "w") do io
        println(io, """
using LibOVRTX
include($(repr(signals_jl)))
save = SignalGuard.snapshot()
cfg  = Ref(LibOVRTX.ovrtx_config_t(Ptr{LibOVRTX.ovrtx_config_entry_t}(C_NULL), Csize_t(0)))
rref = Ref{Ptr{LibOVRTX.ovrtx_renderer_t}}(C_NULL)
LibOVRTX.check(LibOVRTX.ovrtx_create_renderer(cfg, rref), "create")
SignalGuard.restore(save)
LibOVRTX.ovrtx_destroy_renderer(rref[])
println("OK")
""")
    end
end

@testset "M0.4 renderer process exits cleanly" begin
    script = tempname() * ".jl"
    try
        _write_render_script(script, _SIGNALS_JL)
        cmd = setenv(
            `julia --project=$(_REPO_ROOT) $script`,
            "OVRTX_LIBRARY_PATH" => _OVRTX_LIB_PATH,
            # Inherit PATH so julia can find its own dependencies
            "PATH"               => get(ENV, "PATH", ""),
            "HOME"               => get(ENV, "HOME", ""),
        )
        out = IOBuffer()
        p = run(pipeline(cmd; stdout=out, stderr=stderr); wait=false)
        wait(p)
        output = String(take!(out))
        @test p.exitcode == 0          # without restore() this is 139
        @test contains(output, "OK")   # guard against silent early-exit-0
    finally
        isfile(script) && rm(script)
    end
end
