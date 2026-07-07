# examples/fetch_assets.jl — one-time setup: populate examples/assets/ from
# references/ + downloads.
# Usage: julia --project=examples examples/fetch_assets.jl            # all
#        julia --project=examples examples/fetch_assets.jl <scene…>   # subset
# NOT run at render time. assets/ is gitignored.
using Downloads

const EXAMPLES_DIR = @__DIR__
const ASSETS_DIR   = joinpath(EXAMPLES_DIR, "assets")
const REFS = get(ENV, "OM_REFERENCES_DIR", normpath(joinpath(EXAMPLES_DIR, "..", "..", "references")))
const RPRN = joinpath(REFS, "RPRMakieNotes")

const EARTH8K = "https://www.solarsystemscope.com/textures/download/8k_earth_daymap.jpg"
# Earthquake CSVs from the USGS FDSN event API (BeautifulMakie's
# _assets/data CSVs were removed upstream). Columns:
# time,latitude,longitude,depth,mag,… — the scene reads
# latitude/longitude/depth/mag, so this is a drop-in source.
const USGS_H1 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2021-01-01&endtime=2021-06-01&minmagnitude=4.5"
const USGS_H2 = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2021-06-01&endtime=2022-01-01&minmagnitude=4.5"

# (scene, dest_relpath, kind, src)
# kind=:copy → src under RPRMakieNotes; :url → download
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
    ("earthquakes",      "2021_01_2021_05.csv", :url, USGS_H1),
    ("earthquakes",      "2021_06_2022_01.csv", :url, USGS_H2),
    ("earthquakesLight", "2021_01_2021_05.csv", :url, USGS_H1),
    ("earthquakesLight", "2021_06_2022_01.csv", :url, USGS_H2),
    # telegeography's GitHub raw path is gone; the live site serves the
    # same GeoJSON API.
    ("submarineCables", "landing-point-geo.json", :url, "https://www.submarinecablemap.com/api/v3/landing-point/landing-point-geo.json"),
    ("submarineCables", "cable-geo.json", :url, "https://www.submarinecablemap.com/api/v3/cable/cable-geo.json"),
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
            @warn "FAILED $(scene)/$(dest): $(e)"
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
