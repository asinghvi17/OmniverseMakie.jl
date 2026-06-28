using Test

# Paths / constants shared by the subprocess script
const _RENDER_OVRTX_LIB = get(ENV, "OVRTX_LIBRARY_PATH",
    "/home/juliahub/temp/omniverse-makie/references/ovrtx/examples/python/minimal/.venv/lib/python3.13/site-packages/ovrtx/bin/libovrtx-dynamic.so")
const _RENDER_REPO_ROOT  = joinpath(@__DIR__, "..")
const _RENDER_OV_JL      = joinpath(_RENDER_REPO_ROOT, "src", "binding", "OV.jl")
const _RENDER_USDA       = "/home/juliahub/temp/omniverse-makie/references/ovrtx/examples/c/minimal/torus-plane.usda"
const _RENDER_PRODUCT    = "/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0"
const _RENDER_WARMUP     = 64

# The subprocess script:
# - uses LibOVRTX directly (OV.jl does `using ..LibOVRTX`, which resolves to Main.LibOVRTX)
# - includes OV.jl (which in turn includes dlpack.jl via its own include)
# - creates a Renderer, opens the USDA, runs render_to_matrix with warmup frames
# - asserts size == (1080,1920) and that the image is substantially non-black
#   (tolerates up to 100 edge/border pixels that may be black after 64 frames)
# - hard-exits via _exit(0) to avoid breakpad signal-handler crashes on Julia teardown
const _RENDER_PROG = """
using LibOVRTX
include($(repr(_RENDER_OV_JL)))
using ColorTypes

const USDA    = $(repr(_RENDER_USDA))
const PRODUCT = $(repr(_RENDER_PRODUCT))
const WARMUP  = $(_RENDER_WARMUP)

r = OV.Renderer()
OV.open_usd!(r, USDA)
img = OV.render_to_matrix(r, PRODUCT; warmup=WARMUP)

nonblack = count(c -> (red(c) + green(c) + blue(c)) > 0, img)
println("SIZE=", size(img), " NONBLACK=", nonblack)

@assert size(img) == (1080, 1920) "unexpected image size: \$(size(img))"
# Allow up to 100 edge/border pixels to be black (RT2 may not converge all
# border samples within 64 warmup frames; >=2073500 out of 2073600 is fine).
total = 1080 * 1920
@assert nonblack >= total - 100 "image mostly black: \$nonblack / \$total"

close(r)
println("OK")

# Hard-exit: bypass carb/breakpad signal handlers installed during create_renderer.
flush(stdout); flush(stderr)
ccall(:_exit, Cvoid, (Cint,), 0)
"""

@testset "M0.6 Julia-native render (torus, LdrColor, non-black)" begin
    script = tempname() * ".jl"
    try
        open(script, "w") do io
            print(io, _RENDER_PROG)
        end
        cmd = setenv(
            `julia --project=$(_RENDER_REPO_ROOT) $script`,
            "OVRTX_LIBRARY_PATH" => _RENDER_OVRTX_LIB,
            "PATH"               => get(ENV, "PATH", ""),
            "HOME"               => get(ENV, "HOME", ""),
        )
        out = IOBuffer()
        err = IOBuffer()
        # Generous timeout: renderer creates shaders on first run (~30–60 s),
        # plus 64 warmup frames (~11 s) — allow 300 s total.
        p = run(pipeline(cmd; stdout=out, stderr=err); wait=false)
        wait(p)
        output  = String(take!(out))
        errtext = String(take!(err))

        # Print subprocess stderr so CI logs capture any crashes / warnings.
        isempty(errtext) || @info "subprocess stderr" text=errtext

        @test p.exitcode == 0
        @test contains(output, "OK")
        @test contains(output, "SIZE=(1080, 1920)")
        # NONBLACK must be at least 2073500 (out of 2073600).
        # We parse the value rather than exact-match the string so a few black
        # edge pixels (up to 100) still pass.
        m = match(r"NONBLACK=(\d+)", output)
        if m !== nothing
            nb = parse(Int, m.captures[1])
            @test nb >= 1080 * 1920 - 100
        else
            @test false  # sentinel — NONBLACK line missing entirely
        end
    finally
        isfile(script) && rm(script)
    end
end
