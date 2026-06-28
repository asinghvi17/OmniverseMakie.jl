using Test

const _RENDERER_OVRTX_LIB_PATH = get(ENV, "OVRTX_LIBRARY_PATH",
    "/home/juliahub/temp/omniverse-makie/references/ovrtx/examples/python/minimal/.venv/lib/python3.13/site-packages/ovrtx/bin/libovrtx-dynamic.so")
const _RENDERER_REPO_ROOT = joinpath(@__DIR__, "..")
const _OV_JL = joinpath(_RENDERER_REPO_ROOT, "src", "binding", "OV.jl")

# Subprocess program: include OV.jl (which brings its own SignalGuard via include("signals.jl")),
# create two Renderers with close between them, then print "OK".
# The parent must `using LibOVRTX` first so OV's `using ..LibOVRTX` resolves to Main.LibOVRTX.
function _write_renderer_script(path::String, ov_jl::String)
    open(path, "w") do io
        println(io, """
using LibOVRTX
include($(repr(ov_jl)))
r1 = OV.Renderer(); close(r1)
r2 = OV.Renderer(); close(r2)
println("OK")
""")
    end
end

@testset "M0.5 OV.Renderer create/close lifecycle" begin
    script = tempname() * ".jl"
    try
        _write_renderer_script(script, _OV_JL)
        cmd = setenv(
            `julia --project=$(_RENDERER_REPO_ROOT) $script`,
            "OVRTX_LIBRARY_PATH" => _RENDERER_OVRTX_LIB_PATH,
            "PATH"               => get(ENV, "PATH", ""),
            "HOME"               => get(ENV, "HOME", ""),
        )
        out = IOBuffer()
        p = run(pipeline(cmd; stdout=out, stderr=stderr); wait=false)
        wait(p)
        output = String(take!(out))
        @test p.exitcode == 0          # without signal guard this is 139
        @test contains(output, "OK")   # guard against silent early-exit-0
    finally
        isfile(script) && rm(script)
    end
end
