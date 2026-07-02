# ScreenConfig — per-screen rendering configuration.  Used by activate!, Screen constructors, and
# M1.2 USDA stage authoring.
#
# IMPORTANT: keep a positional constructor over all fields in declaration order —
# Makie.merge_screen_config calls Config(arguments...) positionally (display.jl:80).  A plain
# struct gets this free; do NOT add a kwargs-only inner constructor.  Defaults live in
# OmniverseMakie.__init__ (theme registration), NOT here.

struct ScreenConfig
    mode::Symbol        # :rt2 (default) | :pathtracing | :minimal
    samples::Int        # offline SPP for :pathtracing (default 512)
    warmup::Int         # RT2 warmup frames (default 64)
    max_bounces::Int    # max ray bounces (default 4)
    selection_outline::Bool  # M6.B: enable creation-time selection-outline feature (default false)
    # Accumulate-across-frames (realtime-style recording): keep RT2 accumulation across frames
    # instead of resetting on every camera/light/attribute change — RT2's temporal reprojection +
    # denoiser handle motion like the interactive viewport, so recording runs ~10× faster.  Only a
    # STRUCTURAL change (add/remove a USD reference) still resets.  Default false = byte-identical
    # per-frame-reconverge behaviour.
    accumulate_across_frames::Bool
    # Extra RTX steps folded into the FIRST frame's warmup so frame 1 is converged, not cold
    # (only meaningful when accumulate_across_frames is on).  Default 40.
    accumulation_preroll::Int
end

"""
    rtx_settings_usda(cfg::ScreenConfig) -> String

Emit the `omni:rtx:*` attribute lines for the **RenderProduct** prim (rendermode, maxBounces,
ambient); `author_render_root!` injects them into the RenderProduct body.

RECONCILIATION (M1.2): `omni:rtx:rendermode` + `maxBounces` must live on the RenderProduct, NOT a
separate RenderSettings prim (spike-proven).  Only `"RealTimePathTracing"` is spike-verified;
`:pathtracing` maps to it too until the `PathTracing` token is tested.
"""
function rtx_settings_usda(cfg::ScreenConfig)
    # Only "RealTimePathTracing" is spike-verified; :rt2 + :pathtracing both map to it.
    rendermode = "RealTimePathTracing"
    indent = "            "   # 12-space indent to match RenderProduct body
    # Explicit concatenation avoids triple-quoted-string dedentation.
    return "$(indent)token omni:rtx:rendermode = \"$(rendermode)\"\n" *
           "$(indent)int omni:rtx:rtpt:maxBounces = $(cfg.max_bounces)\n" *
           "$(indent)color3f omni:rtx:rt:ambientLight:color = (0.2, 0.2, 0.2)"
end
