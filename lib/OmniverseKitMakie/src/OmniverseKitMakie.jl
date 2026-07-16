# OmniverseKitMakie — full-color NVIDIA IndeX volume rendering for Makie
# scenes through a persistent headless Kit render server.
#
# Standalone ovrtx (what OmniverseMakie's Screen drives) renders volume
# transfer functions grayscale-only; the colored composite path needs a Kit
# runtime with the omni.rtx.index_composite extension chain — proven and
# documented in examples/kit_index_composite/.  This package contains ALL
# Kit-specific machinery so OmniverseMakie itself stays untouched and can
# simply absorb the fix if a future ovrtx build ships the composite marker
# path (see the seam analysis in that README).
#
# v1 surface:
#   server  = start_kit_server()                    # persistent Kit subprocess
#   screen  = KitScreen(scene; server)              # authors + opens the stage
#   img     = Makie.colorbuffer(screen)             # camera sync + render
#   close(screen)
#
# Design doc: docs/superpowers/specs/2026-07-15-omniverse-kit-makie-design.md
module OmniverseKitMakie

import Makie
import OmniverseMakie as OM
import NanoVDBWriter
using LinearAlgebra: I

export KitServer, KitScreen, start_kit_server, rpc, open_stage!, render!,
    render_stage!, set_attr!, stage_usda, gpu_caps, gpu_update_volume!

include("minijson.jl")
include("server.jl")
include("authoring.jl")
include("screen.jl")
include("gpu_plane.jl")   # out-plane device routing + in-plane entry points

end # module OmniverseKitMakie
