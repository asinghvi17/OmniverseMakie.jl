# ScreenConfig — per-screen rendering configuration.  Used by activate!, Screen constructors, and
# M1.2 USDA stage authoring.
#
# IMPORTANT: keep a positional constructor over all fields in declaration order —
# Makie.merge_screen_config calls Config(arguments...) positionally (display.jl:80).  A plain
# struct gets this free; do NOT add a kwargs-only inner constructor.  Defaults live in
# OmniverseMakie.__init__ (theme registration), NOT here.

struct ScreenConfig
    mode::Symbol        # :rt2 (default, realtime path tracer) | :pathtracing (offline path tracer)
    samples::Int        # total SPP cap for a :pathtracing still (default 512); INERT in :rt2
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
    # Sensors (lidar!/radar!): FORCE the renderer's motion BVH on at creation (required for
    # correct MOVING-object sensor returns; static scenes work without).  The Screen ALSO
    # auto-enables it when the displayed scene already contains sensor plots, so this flag is
    # only needed when sensors are added AFTER display.  Creation-frozen (a new Screen is
    # needed to change it); default false = no BVH cost for sensor-free scenes.
    sensors::Bool
end

"""
    rtx_settings_usda(cfg::ScreenConfig) -> String

Emit the `omni:rtx:*` attribute lines for the **RenderProduct** prim (rendermode, maxBounces,
ambient, path-tracing SPP, background source); `author_render_root!` injects them into the
RenderProduct body.

RECONCILIATION (M1.2): `omni:rtx:rendermode` + `maxBounces` must live on the RenderProduct, NOT a
separate RenderSettings prim (spike-proven).

Render-mode facts (PROBE-PROVEN in standalone ovrtx):
  * `:rt2` → `"RealTimePathTracing"` — the realtime accumulating path tracer + OptiX denoiser
    (default; byte-identical output for default configs).  `samples` is inert here.
  * `:pathtracing` → `"PathTracing"` — the OFFLINE path tracer (it renders the *background*
    non-black, a strong discriminator vs RT2).  `samples` becomes the per-still SPP cap via
    `omni:rtx:pt:samplesPerPixel`, with `omni:rtx:pt:samplesPerIteration = cld(samples, warmup)`
    so the STANDARD `warmup` step loop reaches the cap; accumulated SPP = min(warmup×spi, samples).
    PT ignores the RT2 `rtpt:maxBounces` namespace, so its own `omni:rtx:pt:maxBounces` is authored.
  * UNKNOWN/bogus rendermode tokens are SILENTLY absorbed by ovrtx → RT2 fallback (binaries log
    "Uninitialized or unknown render mode ... Switching to RealTimePathTracing instead.").  That
    silent absorption is WHY `mode` must be validated HERE — an unrecognised token would otherwise
    masquerade as :rt2 instead of failing loudly.
  * `"RaytracedLighting"` is also honored but not yet exposed (a faint discriminator on diffuse
    scenes; a possible future `:raytraced` mode).
  * `:minimal` is NOT selectable via USD rendermode: `"Minimal"`/`"MinimalRendering"` are both
    pixel-proven EXACT (byte-identical) RT2 fallbacks — Minimal needs a renderer-config pipeline
    flag, not a USD token — so `:minimal` throws rather than silently rendering RT2.
"""
function rtx_settings_usda(cfg::ScreenConfig)
    indent = "            "   # 12-space indent to match RenderProduct body
    # Render-mode selection + validation (see docstring: ovrtx silently falls unknown tokens back
    # to RT2, so we must reject anything we don't map).  `pt_lines` carries the PathTracing-only
    # SPP/bounce attributes; it stays "" for :rt2 so that path is byte-identical to the pre-feature
    # output (regression goldens must not move for :rt2).
    pt_lines = ""
    rendermode = if cfg.mode === :rt2
        "RealTimePathTracing"
    elseif cfg.mode === :pathtracing
        # `samples` is honored PURELY at author time: samplesPerIteration = samples-per-step so the
        # ordinary `warmup` step loop in render_to_matrix REACHES the samplesPerPixel cap without
        # any render-path change.  cld → ceil so warmup×spi ≥ samples (default 512/64 → spi = 8).
        spi = cld(cfg.samples, cfg.warmup)
        pt_lines =
            "\n$(indent)int omni:rtx:pt:samplesPerPixel = $(cfg.samples)" *
            "\n$(indent)int omni:rtx:pt:samplesPerIteration = $(spi)" *
            "\n$(indent)int omni:rtx:pt:maxBounces = $(cfg.max_bounces)"   # PT ignores rtpt:maxBounces
        "PathTracing"
    else
        throw(ArgumentError("OmniverseMakie: unsupported `mode = $(repr(cfg.mode))` — use :rt2 " *
                            "(realtime path tracer, default) or :pathtracing (offline path tracer). " *
                            ":minimal is NOT selectable via USD rendermode in standalone ovrtx " *
                            "(pixel-proven EXACT RealTimePathTracing fallback — it needs a renderer-" *
                            "config pipeline flag, not a USD token)."))
    end
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
    # Explicit concatenation avoids triple-quoted-string dedentation.  rtpt:maxBounces is kept
    # unconditionally (harmless in PathTracing; keeps the :rt2 path byte-identical).  pt_lines is
    # empty for :rt2, so the :rt2 emission is unchanged from the pre-feature output.
    return "$(indent)token omni:rtx:rendermode = \"$(rendermode)\"\n" *
           "$(indent)int omni:rtx:rtpt:maxBounces = $(cfg.max_bounces)\n" *
           "$(indent)color3f omni:rtx:rt:ambientLight:color = (0.2, 0.2, 0.2)" *
           pt_lines *
           background_line
end
