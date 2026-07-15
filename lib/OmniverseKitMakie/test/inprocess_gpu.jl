# Standalone in-process Kit GPU parity test (hazard-b isolation: its own Julia
# process, never sharing with an ovrtx-in-process test).  Run under the GPU
# recipe (serialize on the shared lock; cap with a timeout):
#
#   DISPLAY=:0 XDG_RUNTIME_DIR=/run/user/1000 \
#   XAUTHORITY=$(ls /run/user/1000/.mutter-Xwaylandauth.* | head -1) \
#   KIT_RELEASE_DIR=<release> OMNI_KIT_ACCEPT_EULA=YES \
#   flock -w 3600 /tmp/omniversemakie-gpu.lock timeout 900 \
#     julia --project=lib/OmniverseKitMakie lib/OmniverseKitMakie/test/inprocess_gpu.jl
#
# Payoff test: render ONE colored-volume frame through the IN-PROCESS transport
# and chroma-check it (viridis > 500 high-chroma px; achromatic twin ~0) — the
# same oracle the subprocess A/B uses.  Pixel evidence is the only proof a
# volume rendered (Kit/ovrtx accept-and-ignore silently; a black frame is the
# classic failure).  Exits non-zero on any failure (so the suite can spawn it).

using Test
using OmniverseKitMakie
import OmniverseKitMakie as OMK
import Makie
using Makie: Scene, cam3d!, update_cam!, volume!, Point3f, Vec3f, RGBf,
    AmbientLight, DirectionalLight, (..)
using ColorTypes: red, green, blue

function blob(n)
    vol = zeros(Float32, n, n, n)
    cx = 0.7n; cy = 0.7n; cz = 0.35n; R = 0.3n
    for k in 1:n, j in 1:n, i in 1:n
        d = sqrt((i - cx)^2 + (j - cy)^2 + (k - cz)^2)
        vol[i, j, k] = d < R ? Float32(3 * (1 - d / R)) : 0.0f0
    end
    return vol
end

function volume_scene(; colormap = :viridis, n = 24, size = (512, 512))
    lights = Makie.AbstractLight[
        AmbientLight(RGBf(0.3, 0.3, 0.3)),
        DirectionalLight(RGBf(1.5, 1.5, 1.4), Vec3f(-0.5, -0.5, -1.0))]
    scene = Scene(; size, lights)
    cam3d!(scene)
    volume!(scene, 0 .. 1, 0 .. 1, 0 .. 1, blob(n); colormap)
    update_cam!(scene, Vec3f(2.4, 2.4, 1.6), Vec3f(0.5, 0.5, 0.4), Vec3f(0, 0, 1))
    return scene
end

lum(c) = Float32(red(c)) + Float32(green(c)) + Float32(blue(c))
chroma(c) = (v = (Float32(red(c)), Float32(green(c)), Float32(blue(c)));
             maximum(v) - minimum(v))

@testset "KitScreen in-process: volume! colors end-to-end (GPU)" begin
    OMK.LibKitJL.available() ||
        error("LibKitJL unavailable: $(OMK.LibKitJL.LIBKITJL_UNAVAILABLE_REASON)")

    # ONE in-process Kit app; both A and B reuse it (carb = one app per process).
    transport = OMK.InProcessTransport(; width = 512, height = 512)
    try
        # A: viridis — the colored transfer function must show as chroma, the
        # thing standalone ovrtx cannot render.
        screen = KitScreen(volume_scene(); transport)
        img = Makie.colorbuffer(screen)
        nb = count(c -> lum(c) > 0.05f0, img)
        ch = count(c -> chroma(c) > 0.15f0, img)
        @info "inprocess viridis" nb ch size(img)
        @test nb > 800
        @test ch > 500
        @test ch > 0.5 * nb   # the lit volume is MOSTLY colored

        # camera motion through the matrix4d set_attr path moves pixels
        update_cam!(screen.scene, Vec3f(1.2, 2.8, 2.2), Vec3f(0.5, 0.5, 0.4), Vec3f(0, 0, 1))
        img2 = Makie.colorbuffer(screen)
        moved = count(k -> abs(lum(img[k]) - lum(img2[k])) > 0.08f0, eachindex(img))
        @info "inprocess cam move" moved
        @test moved > 300

        # B: achromatic colormap through the SAME app -> ~zero chroma
        screen2 = KitScreen(volume_scene(colormap = [RGBf(0, 0, 0), RGBf(1, 1, 1)]); transport)
        img3 = Makie.colorbuffer(screen2)
        nb3 = count(c -> lum(c) > 0.05f0, img3)
        ch3 = count(c -> chroma(c) > 0.15f0, img3)
        @info "inprocess gray" nb3 ch3
        @test nb3 > 800
        @test ch3 < 0.02 * nb3
    finally
        OMK._t_close(transport)
    end
end
