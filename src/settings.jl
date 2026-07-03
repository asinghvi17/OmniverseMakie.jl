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
    # Background rendered behind the scene — `omni:rtx:background:source:type` on the
    # RenderProduct, a CREATION-time render setting (baked at stage open, not live-togglable):
    #   :default   — author nothing (byte-identical to the pre-feature output).  NOTE (pixel-
    #                verified): ovrtx's own default ALREADY shows a dome/environment map as the
    #                background when the scene has one; with no dome the background is black.
    #   :sky       — authors the procedural-sky token.  ★ NOT rendered by standalone ovrtx (black
    #                background; verified against RealTimePathTracing AND PathTracing, with and
    #                without Kit's waitForEvents) — the procedural sky lives in a Kit viewport
    #                extension, like volume-colormap colors.  Authored anyway (+ one warn) so a
    #                composite/Kit runtime would honor it.
    #   :domelight — explicitly pin the dome/environment-light texture as the visible background
    #                (pairs with `EnvironmentLight` / `push_environment_image!`).
    background::Symbol
end

"""
    rtx_settings_usda(cfg::ScreenConfig) -> String

Emit the `omni:rtx:*` attribute lines for the **RenderProduct** prim (rendermode, maxBounces,
ambient, background source); `author_render_root!` injects them into the RenderProduct body.

RECONCILIATION (M1.2): `omni:rtx:rendermode` + `maxBounces` must live on the RenderProduct, NOT a
separate RenderSettings prim (spike-proven).  Only `"RealTimePathTracing"` is spike-verified;
`:pathtracing` maps to it too until the `PathTracing` token is tested.
"""
function rtx_settings_usda(cfg::ScreenConfig)
    # Only "RealTimePathTracing" is spike-verified; :rt2 + :pathtracing both map to it.
    rendermode = "RealTimePathTracing"
    indent = "            "   # 12-space indent to match RenderProduct body
    # Background source token: `:default` authors NOTHING (the pre-feature byte-identical output);
    # note the camelCase "domeLight" USD token vs the all-lowercase Julia-facing Symbol.
    background_line = if cfg.background === :default
        ""
    elseif cfg.background === :sky
        # Authored for a composite/Kit runtime; standalone ovrtx does NOT implement the
        # procedural sky (pixel-verified black — see the ScreenConfig field docs).
        @warn "OmniverseMakie: `background = :sky` authors the procedural-sky token, but this \
               standalone ovrtx does not render it (black background) — it needs a Kit/composite \
               runtime. Use :domelight with an environment image instead." maxlog = 1
        "\n$(indent)token omni:rtx:background:source:type = \"sky\""
    elseif cfg.background === :domelight
        "\n$(indent)token omni:rtx:background:source:type = \"domeLight\""
    else
        throw(ArgumentError("OmniverseMakie: unsupported `background = $(repr(cfg.background))` — " *
                            "use :default, :sky, or :domelight."))
    end
    # Explicit concatenation avoids triple-quoted-string dedentation.
    return "$(indent)token omni:rtx:rendermode = \"$(rendermode)\"\n" *
           "$(indent)int omni:rtx:rtpt:maxBounces = $(cfg.max_bounces)\n" *
           "$(indent)color3f omni:rtx:rt:ambientLight:color = (0.2, 0.2, 0.2)" *
           background_line
end
