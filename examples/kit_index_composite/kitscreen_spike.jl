# KitScreen spike acceptance: one persistent Kit render server, MULTIPLE
# renders through it without restarting — that persistence is the whole point
# (cold Kit startup is expensive; renders are seconds).
#
#   1. start the server (libGLU shim + X env + GPU flock + timeout handled by
#      start_kit_server itself — do NOT wrap this script in flock, the two
#      locks would deadlock)
#   2. render the colored-TF torus volume  -> chroma pixels MUST be > 10000
#   3. render the gray-TF variant through the SAME server -> chroma < 1000
#   4. bonus: set_attr smoke test on the open stage
#
# Run:  julia --project=<repo> examples/kit_index_composite/kitscreen_spike.jl
include(joinpath(@__DIR__, "..", "..", "src", "kit", "kitscreen.jl"))

import .OmniverseMakieKit as OMK
using ColorTypes: red, green, blue

# --- assets -----------------------------------------------------------------

function find_torus_vdb()
    base = joinpath(homedir(), ".local/share/ov/data/exts/v2")
    isdir(base) || error("no ext cache at $base")
    for d in sort(readdir(base); rev = true)
        startswith(d, "omni.rtx.index_composite-") || continue
        p = joinpath(base, d, "data/tests/volumes/torus.vdb")
        isfile(p) && return p
    end
    error("torus.vdb not found under $base (omni.rtx.index_composite ext not cached)")
end

"Write the colored-TF stage and its gray-TF variant; return their paths."
function generate_stages(workdir::String, vdb::String)
    colored_txt = replace(read(joinpath(@__DIR__, "torus_colormap.usda.in"), String),
                          "@VDB_PATH@" => "@$vdb@")
    # Gray variant = NVIDIA's default gray transfer function (same edit
    # launch.sh makes with sed).  Regex `.` stops at newline -> line rewrites.
    gray_txt = replace(colored_txt,
        r"custom float4\[\] rgbaPoints = .*" =>
            "custom float4[] rgbaPoints = [(0.27, 0.27, 0.27, 0), (0.63, 0.63, 0.63, 0.32), (0.5, 0.5, 0.5, 0.5)]",
        r"custom float\[\] xPoints = .*" =>
            "custom float[] xPoints = [0, 0.07, 1]")
    gray_txt == colored_txt && error("gray-variant rewrite did not match the template")
    color_stage = joinpath(workdir, "torus_color.usda")
    gray_stage = joinpath(workdir, "torus_gray.usda")
    write(color_stage, colored_txt)
    write(gray_stage, gray_txt)
    return color_stage, gray_stage
end

# --- metrics ----------------------------------------------------------------

function chroma(c)
    r, g, b = Float32(red(c)), Float32(green(c)), Float32(blue(c))
    return max(r, g, b) - min(r, g, b)
end
chroma_px(img) = count(c -> chroma(c) > 0.15f0, img)

# --- run --------------------------------------------------------------------

function main()
    vdb = find_torus_vdb()
    workdir = mktempdir(; prefix = "kitscreen_spike_", cleanup = false)
    color_stage, gray_stage = generate_stages(workdir, vdb)

    println("== KitScreen spike ==")
    println("vdb      = $vdb")
    println("workdir  = $workdir")

    t0 = time()
    srv = OMK.start_kit_server(; workdir = joinpath(workdir, "server"))
    println("server startup: $(round(time() - t0; digits = 1)) s  ($srv)")

    results = Bool[]
    try
        screen = OMK.KitScreen(srv)

        ping = OMK.rpc(srv, "ping")
        println("ping -> ok=$(ping.ok)")
        push!(results, ping.ok == true)

        t0 = time()
        img_color = OMK.render_stage!(screen, color_stage;
                                      out = joinpath(workdir, "out_color.png"))
        c_color = chroma_px(img_color)
        pass_color = c_color > 10_000
        push!(results, pass_color)
        println("$(pass_color ? "PASS" : "FAIL"): colored render CHROMA_PX=$c_color ",
                "(need > 10000)  size=$(size(img_color))  ",
                "$(round(time() - t0; digits = 1)) s")

        # Second stage through the SAME server — the persistence proof.
        t0 = time()
        img_gray = OMK.render_stage!(screen, gray_stage;
                                     out = joinpath(workdir, "out_gray.png"))
        c_gray = chroma_px(img_gray)
        pass_gray = c_gray < 1_000
        push!(results, pass_gray)
        println("$(pass_gray ? "PASS" : "FAIL"): gray render (same server) CHROMA_PX=$c_gray ",
                "(need < 1000)  $(round(time() - t0; digits = 1)) s")

        # set_attr smoke test (stretch goal): typed float write on the open stage.
        attr_ok = try
            OMK.set_attr!(screen, "/World/Camera", "focalLength", 25.0)
            true
        catch err
            println("set_attr error: $err")
            false
        end
        push!(results, attr_ok)
        println("$(attr_ok ? "PASS" : "FAIL"): set_attr /World/Camera.focalLength = 25.0")
    finally
        t0 = time()
        close(srv)
        println("server shutdown: $(round(time() - t0; digits = 1)) s")
    end

    println("frames: $(joinpath(workdir, "out_color.png")) $(joinpath(workdir, "out_gray.png"))")
    println("kit log: $(srv.log_path)")
    return all(results)
end

if main()
    println("SPIKE_PASS")
else
    println("SPIKE_FAIL")
    exit(1)
end
