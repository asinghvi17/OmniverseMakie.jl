# ScreenConfig — per-screen rendering configuration.
# Used by activate!, Screen constructors, and M1.2 USDA stage authoring.
#
# IMPORTANT: keep a positional constructor over all fields in declaration order.
# Makie.merge_screen_config calls Config(arguments...) positionally (display.jl:80).
# A plain struct gets this for free; do NOT add a kwargs-only inner constructor.
# Defaults live in OmniverseMakie.__init__ (theme registration), NOT here.

struct ScreenConfig
    mode::Symbol        # :rt2 (default) | :pathtracing | :minimal
    samples::Int        # offline SPP for :pathtracing (default 512)
    warmup::Int         # RT2 warmup frames (default 64)
    max_bounces::Int    # max ray bounces (default 4)
end

"""
    rtx_settings_usda(cfg::ScreenConfig) -> String

Emit a USDA snippet for the `RenderSettings` prim encoding the render mode and
max bounces selected by `cfg`.  Consumed by M1.2 USD stage authoring.
"""
function rtx_settings_usda(cfg::ScreenConfig)
    rendermode = if cfg.mode === :rt2
        "RealTimePathTracing"
    elseif cfg.mode === :pathtracing
        "PathTracing"
    else
        "RealTimePathTracing"
    end
    return """def RenderSettings "RenderSettings"
{
    uniform token omni:rtx:rendermode = "$(rendermode)"
    uniform int omni:rtx:rtpt:maxBounces = $(cfg.max_bounces)
}
"""
end
