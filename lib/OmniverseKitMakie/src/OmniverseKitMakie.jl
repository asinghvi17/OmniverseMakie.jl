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
# Surface (transport-agnostic):
#   screen  = KitScreen(scene)                       # subprocess transport (default)
#   screen  = KitScreen(scene; transport=:inprocess) # in-process Kit via LibKitJL
#   img     = Makie.colorbuffer(screen)              # camera sync + render
#   close(screen)
# The subprocess spike surface is still available:
#   server  = start_kit_server(); screen = KitScreen(scene; server)
#
# Design docs: docs/superpowers/specs/2026-07-15-omniverse-kit-makie-design.md
#              docs/superpowers/specs/2026-07-15-libkitjl-design.md  (in-process)
module OmniverseKitMakie

import Makie
import OmniverseMakie as OM
import NanoVDBWriter
import LibKitJL
using LinearAlgebra: I

export KitServer, KitScreen, start_kit_server, rpc, open_stage!, render!,
    render_stage!, set_attr!, stage_usda

include("minijson.jl")
include("server.jl")
include("transport.jl")   # KitTransport: SubprocessTransport (default) + InProcessTransport (LibKitJL)
include("authoring.jl")
include("screen.jl")

end # module OmniverseKitMakie
