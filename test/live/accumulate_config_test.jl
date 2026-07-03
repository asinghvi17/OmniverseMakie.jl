# Accumulate-across-frames — ScreenConfig plumbing (PURE, no GPU).
#
# The behavioural render test (reset suppression + structural reset + preroll) is the
# GPU subprocess in accumulate_render_test.jl; this pins the config surface only.

using Test
import OmniverseMakie as OM
using OmniverseMakie: OV

@testset "accumulate-across-frames: ScreenConfig plumbing (pure)" begin
    # Field order matters: Makie.merge_screen_config constructs ScreenConfig POSITIONALLY, so new
    # fields must be trailing (selection_outline stays 5th; `background` is 8th/last).
    @test fieldnames(OM.ScreenConfig) ==
          (:mode, :samples, :warmup, :max_bounces, :selection_outline,
           :accumulate_across_frames, :accumulation_preroll, :background)

    # Positional constructor over all 8 fields (the m3_material_prog site relies on this).
    c = OM.ScreenConfig(:rt2, 512, 64, 4, false, true, 8, :default)
    @test c.accumulate_across_frames === true
    @test c.accumulation_preroll === 8
    @test c.background === :default

    # Theme defaults resolve WITHOUT an override (off + preroll 40), matching pre-feature behaviour.
    d = OM.Makie.merge_screen_config(OM.ScreenConfig, Dict{Symbol,Any}())
    @test d.accumulate_across_frames === false
    @test d.accumulation_preroll === 40
    @test d.background === :default

    # Caller overrides flow through the merge.
    o = OM.Makie.merge_screen_config(OM.ScreenConfig,
        Dict{Symbol,Any}(:accumulate_across_frames => true, :accumulation_preroll => 8))
    @test o.accumulate_across_frames === true
    @test o.accumulation_preroll === 8

    # Documented surface exists; the reset observer is a zero-overhead no-op by default.
    @test isdefined(OM, :reset_accumulation!)
    @test OV._RESET_OBSERVER[] === nothing
end
