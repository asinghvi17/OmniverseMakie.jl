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

Emit the `omni:rtx:*` attribute lines that belong on the **RenderProduct** prim
(rendermode, maxBounces, ambient).  Consumed by `author_render_root!` which
injects them into the RenderProduct body.

RECONCILIATION (M1.2): the spike proved that `omni:rtx:rendermode` and
`maxBounces` must live on the RenderProduct, not in a separate RenderSettings
prim.  Only `"RealTimePathTracing"` (RT2 path-tracing) is spike-verified;
`:pathtracing` is mapped to it as well until the `PathTracing` token is tested.
"""
function rtx_settings_usda(cfg::ScreenConfig)
    # Only "RealTimePathTracing" is spike-verified.  Both :rt2 and :pathtracing
    # map to it until an explicit PathTracing-mode spike confirms the token.
    rendermode = "RealTimePathTracing"
    ind = "            "   # 12-space indent to match RenderProduct body
    # Avoid triple-quoted string dedentation by using explicit concatenation.
    return "$(ind)token omni:rtx:rendermode = \"$(rendermode)\"\n" *
           "$(ind)int omni:rtx:rtpt:maxBounces = $(cfg.max_bounces)\n" *
           "$(ind)color3f omni:rtx:rt:ambientLight:color = (0.2, 0.2, 0.2)"
end
