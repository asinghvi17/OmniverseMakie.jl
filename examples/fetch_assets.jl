# examples/fetch_assets.jl — one-time setup: populate examples/assets/ from references/ + downloads.
# Usage: julia --project=examples examples/fetch_assets.jl            # all
#        julia --project=examples examples/fetch_assets.jl <scene…>   # subset
# NOT run at render time. assets/ is gitignored.
using Downloads

const EXAMPLES_DIR = @__DIR__
const ASSETS_DIR   = joinpath(EXAMPLES_DIR, "assets")
const REFS = get(ENV, "OM_REFERENCES_DIR", normpath(joinpath(EXAMPLES_DIR, "..", "..", "references")))
const RPRN = joinpath(REFS, "RPRMakieNotes")

const EARTH8K = "https://www.solarsystemscope.com/textures/download/8k_earth_daymap.jpg"
const CSV_BASE = "https://github.com/lazarusA/BeautifulMakie/raw/main/_assets/data/"

# (scene, dest_relpath, kind, src)   kind=:copy → src under RPRMakieNotes; :url → download
const MANIFEST = [
    ("reflections_glass_material", "envLightImage.exr", :copy, "lights/envLightImage.exr"),
    ("materials_julia_room", "makie_logo.png", :copy, "imgs/makie_logo_transparent.png"),
    ("materials_julia_room", "lazaro2.png",    :copy, "imgs/lazaro2.png"),
    ("helix",                "makie_logo.png", :copy, "imgs/makie_logo_transparent.png"),
    ("helix",                "lazaro2.png",    :copy, "imgs/lazaro2.png"),
    ("transparentMaterial",  "earth.jpg", :url, EARTH8K),
    ("uberMExample",         "earth.jpg", :url, EARTH8K),
    ("earth_ina_julia_box",  "earth.jpg", :url, EARTH8K),
    ("earthquakesLight",     "earth.jpg", :url, EARTH8K),
    ("submarineCables",      "earth.jpg", :url, EARTH8K),
    ("twoEarths", "earth.jpg", :url, "https://upload.wikimedia.org/wikipedia/commons/c/c3/Solarsystemscope_texture_2k_earth_daymap.jpg"),
    ("earthquakes",      "2021_01_2021_05.csv", :url, CSV_BASE * "2021_01_2021_05.csv"),
    ("earthquakes",      "2021_06_2022_01.csv", :url, CSV_BASE * "2021_06_2022_01.csv"),
    ("earthquakesLight", "2021_01_2021_05.csv", :url, CSV_BASE * "2021_01_2021_05.csv"),
    ("earthquakesLight", "2021_06_2022_01.csv", :url, CSV_BASE * "2021_06_2022_01.csv"),
    ("submarineCables", "landing-point-geo.json", :url, "https://raw.githubusercontent.com/telegeography/www.submarinecablemap.com/master/web/public/api/v3/landing-point/landing-point-geo.json"),
    ("submarineCables", "cable-geo.json", :url, "https://raw.githubusercontent.com/telegeography/www.submarinecablemap.com/master/web/public/api/v3/cable/cable-geo.json"),
]

function fetch_one(scene, dest, kind, src)
    destdir = joinpath(ASSETS_DIR, scene); mkpath(destdir)
    destpath = joinpath(destdir, dest)
    if isfile(destpath); println("  have     $(scene)/$(dest)"); return destpath; end
    if kind === :copy
        s = joinpath(RPRN, src); isfile(s) || error("missing reference asset: $(s)")
        cp(s, destpath; force = true); println("  copied   $(scene)/$(dest)")
    else
        println("  download $(scene)/$(dest) …")
        try
            Downloads.download(src, destpath)
        catch e
            @warn "  FAILED   $(scene)/$(dest): $(e)"
            return nothing
        end
    end
    return destpath
end

if abspath(PROGRAM_FILE) == @__FILE__
    let want = isempty(ARGS) ? MANIFEST : filter(e -> e[1] in ARGS, MANIFEST)
        for (scene, dest, kind, src) in want
            fetch_one(scene, dest, kind, src)
        end
    end
    println("fetch_assets done → $(ASSETS_DIR)")
end
