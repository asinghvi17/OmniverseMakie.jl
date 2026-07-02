using Test

# Review E2 — PROG_PIXEL_HELPERS prelude.  PURE test that the shared pixel-helper prelude
# (helpers.jl) parses + runs when spliced into a subprocess prog, and that its helpers and
# thresholds behave.  The child builds a synthetic image — a Matrix of (r, g, b) NamedTuples
# whose field access matches a real RGBA colorbuffer — so it needs NO Makie/ovrtx/GPU.

@testset "review E2: PROG_PIXEL_HELPERS prelude" begin
    prog = """
    $(PROG_PIXEL_HELPERS)
    # 4×4 synthetic image: all black but one lit pixel at row 3, col 2 (r+g+b = 1.5 > LUM_MIN).
    img = fill((r = 0.0f0, g = 0.0f0, b = 0.0f0), 4, 4)
    img[3, 2] = (r = 0.5f0, g = 0.5f0, b = 0.5f0)
    st = lit_centroid(img)
    println("NONBLACK=", nonblack(img))
    println("LIT_PX_MIN=", LIT_PX_MIN)
    println("LUM_MIN=", LUM_MIN)
    println("SHAPE=", (st.H, st.nb, st.crow, st.ccol))
    println("OK_PRELUDE")
    """
    exitcode, out = run_ovrtx_subprocess(prog; timeout = 120)
    contains(out, "OK_PRELUDE") || @info "prelude output" out
    @test exitcode == 0                              # the prelude parsed + ran (a syntax error → nonzero)
    @test contains(out, "OK_PRELUDE")
    @test contains(out, "NONBLACK=1")                # exactly one non-black pixel
    @test contains(out, "LIT_PX_MIN=300")            # threshold const value
    @test contains(out, "LUM_MIN=0.04")              # threshold const value
    @test contains(out, "SHAPE=(4, 1, 3.0, 2.0)")    # (H, nb, crow, ccol) shape + centroid at the lit pixel
end
