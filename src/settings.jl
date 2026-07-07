# ScreenConfig — per-screen rendering configuration (activate!, Screen
# constructors, USDA stage authoring).  Makie.merge_screen_config calls
# Config(arguments...) positionally, so keep a positional constructor over
# all fields in declaration order (no kwargs-only inner constructor).
# Defaults live in OmniverseMakie.__init__ (theme registration), NOT here.

struct ScreenConfig
    mode::Symbol        # :rt2 (realtime, default) | :pathtracing (offline)
    samples::Int        # SPP cap for a :pathtracing still; inert in :rt2
    warmup::Int         # RT2 warmup frames (default 64)
    max_bounces::Int    # max ray bounces (default 4)
    selection_outline::Bool  # creation-time selection outline (default false)
    # Keep RT2 accumulation across frames instead of resetting on every
    # camera/light/attribute change (reprojection handles motion; ~10× faster
    # recording).  Only a structural USD-reference add/remove still resets.
    accumulate_across_frames::Bool
    # Extra RTX steps folded into the FIRST frame's warmup so frame 1 is
    # converged (only meaningful when accumulate_across_frames is on).
    accumulation_preroll::Int
    # Background behind the scene: `omni:rtx:background:source:type` on the
    # RenderProduct — a creation-time render setting, baked at stage open.
    #   :default   — author nothing.  ovrtx already shows the scene's dome/
    #                environment map as the background when one exists; with
    #                no dome the background is black.
    #   :sky       — author the procedural-sky token.  Standalone ovrtx does
    #                NOT render it (black background) — the procedural sky
    #                lives in a Kit viewport extension.  Authored anyway
    #                (+ one warn) so a Kit/composite runtime would honor it.
    #   :domelight — pin the dome/environment-light texture as the visible
    #                background (pairs with `push_environment_image!`).
    background::Symbol
    # Force the renderer's motion BVH on at creation (needed for correct
    # MOVING-object lidar!/radar! returns; auto-enabled when the displayed
    # scene already holds sensor plots, so only needed for sensors added
    # after display).  Creation-frozen; default false (no BVH cost).
    sensors::Bool
end

"""
    rtx_settings_usda(cfg::ScreenConfig) -> String

Emit the `omni:rtx:*` attribute lines for the **RenderProduct** prim
(rendermode, maxBounces, ambient, path-tracing SPP, background source);
`author_render_root!` injects them into the RenderProduct body.

`omni:rtx:rendermode` + `maxBounces` must live on the RenderProduct, NOT a
separate RenderSettings prim.

Render-mode facts (standalone ovrtx):
  * `:rt2` → `"RealTimePathTracing"` — the realtime accumulating path tracer
    + OptiX denoiser (default).  `samples` is inert here.
  * `:pathtracing` → `"PathTracing"` — the OFFLINE path tracer (it renders
    the background non-black, unlike RT2).  `samples` becomes the per-still
    SPP cap via `omni:rtx:pt:samplesPerPixel`, with
    `omni:rtx:pt:samplesPerIteration = cld(samples, warmup)` so the standard
    `warmup` step loop reaches the cap; accumulated SPP =
    min(warmup×spi, samples).  PT ignores the RT2 `rtpt:maxBounces`
    namespace, so its own `omni:rtx:pt:maxBounces` is authored.
  * Unknown rendermode tokens are SILENTLY absorbed by ovrtx as an RT2
    fallback, which is why `mode` must be validated HERE — an unrecognised
    token would otherwise masquerade as :rt2 instead of failing loudly.
  * `"RaytracedLighting"` is also honored but not yet exposed.
  * `:minimal` is NOT selectable via USD rendermode: `"Minimal"` /
    `"MinimalRendering"` are exact RT2 fallbacks (Minimal needs a
    renderer-config pipeline flag, not a USD token), so `:minimal` throws
    rather than silently rendering RT2.
"""
function rtx_settings_usda(cfg::ScreenConfig)
    indent = "            "   # 12-space indent to match RenderProduct body
    # Render-mode selection + validation (ovrtx silently falls unknown tokens
    # back to RT2, so anything unmapped must be rejected).  `pt_lines` carries
    # the PathTracing-only SPP/bounce attributes; it stays "" for :rt2.
    pt_lines = ""
    rendermode = if cfg.mode === :rt2
        "RealTimePathTracing"
    elseif cfg.mode === :pathtracing
        # `samples` is honored purely at author time: samplesPerIteration =
        # samples-per-step so the ordinary `warmup` step loop reaches the
        # samplesPerPixel cap.  cld → ceil so warmup×spi ≥ samples.  Guard the
        # divisor + SPP cap loudly (cld by 0 is a DivideError; ≤0 samples is
        # meaningless).
        cfg.samples > 0 || throw(ArgumentError(
            "OmniverseMakie: `samples` must be positive for `mode = :pathtracing`, got $(cfg.samples)."))
        cfg.warmup > 0 || throw(ArgumentError(
            "OmniverseMakie: `warmup` must be positive for `mode = :pathtracing` (it divides `samples` " *
            "into per-iteration SPP), got $(cfg.warmup)."))
        spi = cld(cfg.samples, cfg.warmup)
        pt_lines =
            "\n$(indent)int omni:rtx:pt:samplesPerPixel = $(cfg.samples)" *
            "\n$(indent)int omni:rtx:pt:samplesPerIteration = $(spi)" *
            "\n$(indent)int omni:rtx:pt:maxBounces = $(cfg.max_bounces)"
        "PathTracing"
    else
        throw(ArgumentError("OmniverseMakie: unsupported `mode = $(repr(cfg.mode))` — use :rt2 " *
                            "(realtime path tracer, default) or :pathtracing (offline path tracer). " *
                            ":minimal is NOT selectable via USD rendermode in standalone ovrtx " *
                            "(an EXACT RealTimePathTracing fallback — it needs a renderer-" *
                            "config pipeline flag, not a USD token)."))
    end
    # Background source token: `:default` authors nothing; note the camelCase
    # "domeLight" USD token vs the all-lowercase Julia-facing Symbol.
    background_line = if cfg.background === :default
        ""
    elseif cfg.background === :sky
        # Authored for a Kit/composite runtime; standalone ovrtx does not
        # render the procedural sky (black — see the ScreenConfig field docs).
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
    # rtpt:maxBounces is authored unconditionally (harmless in PathTracing);
    # pt_lines is empty for :rt2.
    return "$(indent)token omni:rtx:rendermode = \"$(rendermode)\"\n" *
           "$(indent)int omni:rtx:rtpt:maxBounces = $(cfg.max_bounces)\n" *
           "$(indent)color3f omni:rtx:rt:ambientLight:color = (0.2, 0.2, 0.2)" *
           pt_lines *
           background_line
end
