using Test
using OmniverseKitMakie
import OmniverseKitMakie as OMK
import Makie
import CUDA
using Makie: Scene, cam3d!, update_cam!, volume!, Point3f, Vec3f, RGBf,
    AmbientLight, DirectionalLight, (..)
using ColorTypes: red, green, blue, alpha, RGBA
using FixedPointNumbers: N0f8

# Graded blob at a parameterizable center (uniform fields render
# IndeX-transparent; a movable center backs the live-update tests).
function blob(n; center = (0.7, 0.7, 0.35))
    vol = zeros(Float32, n, n, n)
    cx = center[1] * n; cy = center[2] * n; cz = center[3] * n; R = 0.3n
    for k in 1:n, j in 1:n, i in 1:n
        d = sqrt((i - cx)^2 + (j - cy)^2 + (k - cz)^2)
        vol[i, j, k] = d < R ? Float32(3 * (1 - d / R)) : 0.0f0
    end
    return vol
end

function volume_scene(; colormap = :viridis, n = 24, size = (512, 512),
                      data = blob(n))
    lights = Makie.AbstractLight[
        AmbientLight(RGBf(0.3, 0.3, 0.3)),
        DirectionalLight(RGBf(1.5, 1.5, 1.4), Vec3f(-0.5, -0.5, -1.0))]
    scene = Scene(; size, lights)
    cam3d!(scene)
    volume!(scene, 0 .. 1, 0 .. 1, 0 .. 1, data; colormap)
    update_cam!(scene, Vec3f(2.4, 2.4, 1.6), Vec3f(0.5, 0.5, 0.4), Vec3f(0, 0, 1))
    return scene
end

lum(c) = Float32(red(c)) + Float32(green(c)) + Float32(blue(c))
chroma(c) = (v = (Float32(red(c)), Float32(green(c)), Float32(blue(c)));
             maximum(v) - minimum(v))

@testset "minijson: encode/decode round-trip" begin
    line = OMK._json_object("id" => 3, "op" => "set_attr", "value" => [[1.5, 0, 0, 0], [0, 1, 0, 0]],
                            "flag" => true, "name" => "a\"b\\c\nd")
    d = OMK._parse_json(line)
    @test d["id"] == 3
    @test d["op"] == "set_attr"
    @test d["flag"] === true
    @test d["name"] == "a\"b\\c\nd"
    @test d["value"][1] == Any[1.5, 0, 0, 0]
    # python-style \uXXXX escapes (ensure_ascii output), incl. surrogate pairs
    @test OMK._parse_json("{\"s\": \"\\u00e9\\ud83d\\ude00\"}")["s"] == "é😀"
    @test OMK._parse_json("[1, 2.5, null, false]") == Any[1, 2.5, nothing, false]
end

@testset "stage_usda: composite stage from a volume scene (pure)" begin
    wd = mktempdir()
    scene = volume_scene()
    usda = stage_usda(scene; workdir = wd)
    # a clean volume scene authors without warnings
    @test_logs min_level = Base.CoreLogging.Warn stage_usda(volume_scene();
                                                            workdir = mktempdir())

    # composite enablement: root-layer renderSettings + per-prim markers
    @test occursin("bool \"rtx:index:compositeEnabled\" = 1", usda)
    @test occursin("int \"rtx:index:compositeDepthMode\" = 3", usda)
    @test occursin("custom bool nvindex:composite = 1", usda)
    @test occursin("custom bool omni:rtx:skip = 1", usda)

    # camera: bound + authored with a parseable transform
    @test occursin("string boundCamera = \"/World/Camera\"", usda)
    m = match(r"def Camera \"Camera\"(?s).*?matrix4d xformOp:transform = \( (.*?) \)\n", usda)
    @test m !== nothing
    @test m === nothing || count(",", m.captures[1]) == 15  # 16 numbers

    # lights authored through OmniverseMakie's emitters
    @test occursin("def DistantLight", usda)

    # volume payload + fragment on disk; fragment carries the colormap TF
    @test isfile(joinpath(wd, "volume_1.nvdb"))
    frag = read(joinpath(wd, "volume_1.usda"), String)
    @test occursin("float4[] rgbaPoints", frag)
    @test occursin("def Colormap \"Colormap\"", frag)
    @test occursin("OpenVDBAsset", frag)

    # a different colormap changes the authored transfer function
    wd2 = mktempdir()
    stage_usda(volume_scene(colormap = [RGBf(0, 0, 0), RGBf(1, 1, 1)]); workdir = wd2)
    frag2 = read(joinpath(wd2, "volume_1.usda"), String)
    @test frag2 != frag
    ms = match(r"rgbaPoints = \[(.*?)\]", frag2)
    @test ms !== nothing
    pts = collect(eachmatch(r"\(([0-9.eE+-]+), ([0-9.eE+-]+), ([0-9.eE+-]+), [0-9.eE+-]+\)",
                            something(ms).captures[1]))
    @test !isempty(pts) && all(pts) do p  # achromatic ramp: r == g == b per point
        r, g, b = parse.(Float64, p.captures)
        abs(r - g) < 1e-3 && abs(g - b) < 1e-3
    end

    # non-volume atomic plots are skipped with ONE warning
    scene2 = volume_scene()
    Makie.lines!(scene2, [Point3f(0, 0, 0), Point3f(1, 1, 1)])
    @test_logs (:warn, r"volume plots only") stage_usda(scene2; workdir = mktempdir())
end

@testset "gpu plane: device resolution + shm frame decode (pure)" begin
    caps = Dict{Symbol, Any}(:shm_out => true)
    @test OMK._resolve_device(caps, :auto) === :cpu
    @test OMK._resolve_device(Dict{Symbol, Any}(), :auto) === :png
    @test OMK._resolve_device(caps, :shm) === :cpu
    @test OMK._resolve_device(caps, :png) === :png
    @test OMK._resolve_device(caps, :cuda) === :cuda

    # shm decode: known 3×2 RGBA8 payload, row-major top-down -> (H, W) image
    w, h = 3, 2
    path = tempname()
    bytes = UInt8[]
    for row in 0:(h - 1), col in 0:(w - 1)
        push!(bytes, UInt8(10row + col), UInt8(100 + col), UInt8(200 - 10row), 0xff)
    end
    write(path, bytes)
    img = OMK._shm_frame((; shm_path = path, nbytes = length(bytes),
                            width = w, height = h, format = "RGBA8"))
    @test img isa Matrix{RGBA{N0f8}}
    @test size(img) == (h, w)
    p = img[2, 3]   # row 2, col 3 -> bytes (12, 102, 190, 255)
    @test reinterpret(UInt8, red(p)) == 0x0c
    @test reinterpret(UInt8, green(p)) == 0x66
    @test reinterpret(UInt8, blue(p)) == 0xbe
    @test reinterpret(UInt8, alpha(p)) == 0xff
    # size mismatch is a loud error, not a garbled image
    @test_throws ErrorException OMK._shm_frame((; shm_path = path,
        nbytes = length(bytes), width = w + 1, height = h, format = "RGBA8"))
end

# ---------------------------------------------------------------------------
# GPU: end-to-end colors through a live Kit server (skipped without a Kit
# runtime).  The server serializes on the shared GPU lock itself.
# ---------------------------------------------------------------------------
kit_ok = isfile(joinpath(OMK._default_kit_release_dir(), "kit", "kit")) &&
         get(ENV, "OMK_SKIP_GPU", "") != "1"

@testset "KitScreen: volume! colors end-to-end (GPU)" begin
    if !kit_ok
        @test_skip "Kit runtime absent (set KIT_RELEASE_DIR) or OMK_SKIP_GPU=1"
    else
        server = start_kit_server(; width = 512, height = 512)
        try
            # A: viridis — the colored transfer function must show as chroma,
            # the thing standalone ovrtx cannot render.
            screen = KitScreen(volume_scene(); server)
            img = Makie.colorbuffer(screen)
            nb = count(c -> lum(c) > 0.05f0, img)
            ch = count(c -> chroma(c) > 0.15f0, img)
            @info "kitscreen viridis" nb ch size(img)
            @test nb > 800
            @test ch > 500
            @test ch > 0.5 * nb   # the lit volume is MOSTLY colored

            # camera motion through the matrix4d set_attr path moves pixels
            update_cam!(screen.scene, Vec3f(1.2, 2.8, 2.2), Vec3f(0.5, 0.5, 0.4),
                        Vec3f(0, 0, 1))
            img2 = Makie.colorbuffer(screen)
            moved = count(k -> abs(lum(img[k]) - lum(img2[k])) > 0.08f0, eachindex(img))
            @info "kitscreen cam move" moved
            @test moved > 300

            # B: achromatic colormap through the SAME server -> ~zero chroma
            screen2 = KitScreen(volume_scene(colormap = [RGBf(0, 0, 0), RGBf(1, 1, 1)]);
                                server)
            img3 = Makie.colorbuffer(screen2)
            nb3 = count(c -> lum(c) > 0.05f0, img3)
            ch3 = count(c -> chroma(c) > 0.15f0, img3)
            @info "kitscreen gray" nb3 ch3
            @test nb3 > 800
            @test ch3 < 0.02 * nb3

            # ---------------------------------------------------------------
            # GPU data plane (fresh screen: default camera, viridis blob)
            # ---------------------------------------------------------------
            @info "kit server caps" gpu_caps(screen)
            screenP = KitScreen(volume_scene(); server)

            # shm out-plane: parity vs PNG (accumulation drifts between the
            # two captures — tolerance parity, not byte equality)
            if get(gpu_caps(screenP), :shm_out, false)
                t_png = @elapsed img_png = render!(screenP; device = :png)
                t_shm = @elapsed img_shm = render!(screenP; device = :cpu, frames = 8)
                @test img_shm isa Matrix{RGBA{N0f8}}
                @test size(img_shm) == size(img_png)
                ndrift = count(k -> abs(lum(img_png[k]) - lum(img_shm[k])) > 24 / 255,
                               eachindex(img_png))
                @info "shm parity" t_png t_shm ndrift
                @test ndrift < 0.01 * length(img_png)
            else
                @test_skip "server lacks shm_out"
            end

            # cuda out-plane: device-resident frame, chroma oracle holds
            if get(gpu_caps(screenP), :cuda_out, false) && CUDA.functional()
                imgc = render!(screenP; device = :cuda, frames = 8)
                @test imgc isa CUDA.CuArray
                imgc_h = Array(imgc)
                @test size(imgc_h) == (screenP.size[2], screenP.size[1])
                chc = count(c -> chroma(c) > 0.15f0, imgc_h)
                nbc = count(c -> lum(c) > 0.05f0, imgc_h)
                @info "cuda frame" nbc chc
                @test nbc > 800
                @test chc > 500
            else
                @test_skip "cuda out-plane unavailable (syntheticdata or CUDA missing)"
            end

            # in-plane: live volume update from a CuArray (no Julia host copy)
            if get(gpu_caps(screenP), :cuda_ipc, false) && CUDA.functional()
                before = render!(screenP; device = :cpu, frames = 8)
                shifted = blob(24; center = (0.3, 0.3, 0.65))
                gpu_update_volume!(screenP, screenP.volumes[1].plot;
                                   data = CUDA.CuArray(shifted))
                after = render!(screenP; device = :cpu, frames = 24)
                moved = count(k -> abs(lum(before[k]) - lum(after[k])) > 0.08f0,
                              eachindex(before))
                @info "gpu volume update" moved
                @test moved > 300

                # twin equivalence: a CPU-authored screen with the same
                # shifted data must render ~the same image
                screenT = KitScreen(volume_scene(data = shifted); server)
                twin = render!(screenT; device = :cpu, frames = 24)
                tdiff = count(k -> abs(lum(after[k]) - lum(twin[k])) > 0.12f0,
                              eachindex(twin))
                @info "gpu volume twin" tdiff
                @test tdiff < 0.02 * length(twin)
            else
                @test_skip "cuda ipc unavailable"
            end
        finally
            close(server)
        end
    end
end
