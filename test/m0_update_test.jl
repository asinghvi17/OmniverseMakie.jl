using Test

# Paths / constants (reuse the same conventions as m0_render_test.jl)
const _UPDATE_OVRTX_LIB = get(ENV, "OVRTX_LIBRARY_PATH",
    "/home/juliahub/temp/omniverse-makie/references/ovrtx/examples/python/minimal/.venv/lib/python3.13/site-packages/ovrtx/bin/libovrtx-dynamic.so")
const _UPDATE_REPO_ROOT  = joinpath(@__DIR__, "..")
const _UPDATE_OV_JL      = joinpath(_UPDATE_REPO_ROOT, "src", "binding", "OV.jl")
const _UPDATE_USDA       = get(ENV, "OM_USDA",
    "/home/juliahub/temp/omniverse-makie/references/ovrtx/examples/c/minimal/torus-plane.usda")
const _UPDATE_PRODUCT    = "/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0"
const _UPDATE_WARMUP     = 64
const _UPDATE_PRIM       = "/World/Torus"

# The subprocess script:
#   1. Renders frame1 (warmup frames).
#   2. Moves /World/Torus by a large translation via write_xform!.
#   3. reset! to restart RT2 accumulation.
#   4. Renders frame2.
#   5. Counts pixels whose ANY channel changed by >= 8/255 (magnitude threshold
#      excludes normal RT2 stochastic noise, proves the geometry actually moved).
#   6. Asserts changed_count >= 50000 (spike measured ~597k — 50k is a very
#      conservative floor that distinguishes "moved" from "noise").
#   7. exits normally (clean teardown verified: no _exit needed).
const _UPDATE_PROG = """
using LibOVRTX
include($(repr(_UPDATE_OV_JL)))
using ColorTypes

const USDA    = ENV["OM_USDA"]
const PRODUCT = $(repr(_UPDATE_PRODUCT))
const WARMUP  = $(_UPDATE_WARMUP)
const PRIM    = $(repr(_UPDATE_PRIM))

# 4×4 row-major identity + large translation in last row (row-vector convention).
# Moving (300, 250, 300) units puts the torus well outside its original footprint.
M = Float64[
    1.0  0.0  0.0  0.0
    0.0  1.0  0.0  0.0
    0.0  0.0  1.0  0.0
    300.0 250.0 300.0 1.0
]

r = OV.Renderer()
OV.open_usd!(r, USDA)

# --- frame 1 (before move) ---
img1 = OV.render_to_matrix(r, PRODUCT; warmup=WARMUP)
# Extract raw UInt8 pixels for per-channel comparison.
# render_to_matrix returns Matrix{RGBA{N0f8}} (H×W).
H, W = size(img1)

# --- move torus ---
OV.write_xform!(r, PRIM, M)
OV.reset!(r)

# --- frame 2 (after move) ---
img2 = OV.render_to_matrix(r, PRODUCT; warmup=WARMUP)

# Count pixels where any channel changed by >= 8 (out of 255).
threshold = 8
changed = let c = 0
    for h in 1:H, w in 1:W
        c1 = img1[h, w]
        c2 = img2[h, w]
        dr = abs(Float32(red(c1))   - Float32(red(c2)))
        dg = abs(Float32(green(c1)) - Float32(green(c2)))
        db = abs(Float32(blue(c1))  - Float32(blue(c2)))
        if dr*255 >= threshold || dg*255 >= threshold || db*255 >= threshold
            c += 1
        end
    end
    c
end

println("CHANGED_PIXELS=", changed, " / ", H*W)

@assert changed >= 50000 "xform write did not move geometry: changed=\$changed < 50000"

close(r)
println("OK_UPDATE")
"""

@testset "M0.7 write_xform! moves torus (changed >= 50000 pixels)" begin
    script = tempname() * ".jl"
    try
        open(script, "w") do io
            print(io, _UPDATE_PROG)
        end
        cmd = setenv(
            `julia --project=$(_UPDATE_REPO_ROOT) $script`,
            "OVRTX_LIBRARY_PATH" => _UPDATE_OVRTX_LIB,
            "OM_USDA"            => _UPDATE_USDA,
            "PATH"               => get(ENV, "PATH", ""),
            "HOME"               => get(ENV, "HOME", ""),
        )
        out = IOBuffer()
        err = IOBuffer()
        # Two 64-frame renders + create_renderer shader compile — allow 360 s total.
        p = run(pipeline(cmd; stdout=out, stderr=err); wait=false)
        wait(p)
        output  = String(take!(out))
        errtext = String(take!(err))

        isempty(errtext) || @info "subprocess stderr" text=errtext

        @test p.exitcode == 0
        @test contains(output, "OK_UPDATE")

        m = match(r"CHANGED_PIXELS=(\d+)", output)
        if m !== nothing
            changed = parse(Int, m.captures[1])
            @test changed >= 50000
        else
            @test false  # sentinel — CHANGED_PIXELS line missing entirely
        end
    finally
        isfile(script) && rm(script)
    end
end
