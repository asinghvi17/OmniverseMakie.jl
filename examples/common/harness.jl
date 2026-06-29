# examples/common/harness.jl — shared render harness + property-assert helpers.
# Loaded by common/run_one.jl inside each render subprocess (and by harness_test.jl).
using OmniverseMakie
using ColorTypes, FixedPointNumbers, FileIO

const COMMON_DIR  = @__DIR__
const EXAMPLES_DIR = normpath(joinpath(COMMON_DIR, ".."))
const RENDERS_DIR  = joinpath(EXAMPLES_DIR, "renders")
const ASSETS_DIR   = joinpath(EXAMPLES_DIR, "assets")

"""
    asset(scene, relpath) -> String

Absolute path to `examples/assets/<scene>/<relpath>`; errors (telling the user to run
`fetch_assets.jl`) if missing.
"""
function asset(scene::AbstractString, relpath::AbstractString)
    p = joinpath(ASSETS_DIR, scene, relpath)
    isfile(p) || error("OmniverseMakie examples: asset '$(scene)/$(relpath)' not found at \
                        $p — run `julia --project=examples examples/fetch_assets.jl` first.")
    return p
end

"""
    run_example(name, scene_fn; size, out) -> Matrix

Activate OmniverseMakie, build the figure via `scene_fn()`, `Makie.save` it to `out`
(routes through our colorbuffer), and return the saved image read back for asserts.
"""
function run_example(name::AbstractString, scene_fn;
                     size = (900, 900),
                     out  = joinpath(RENDERS_DIR, name * ".png"))
    OmniverseMakie.activate!()
    fig = scene_fn()
    mkpath(dirname(out))
    Makie.save(out, fig)
    return FileIO.load(out)
end

# --- property-assert helpers (M1–M3 idiom) -----------------------------------
lum(c) = 0.2126f0 * Float32(red(c)) + 0.7152f0 * Float32(green(c)) + 0.0722f0 * Float32(blue(c))

nonblack_count(img; thresh = 0.02f0) = count(c -> lum(c) > thresh, img)

function assert_nonblack(img, name; frac = 0.01)
    n = nonblack_count(img); total = length(img)
    @assert n > frac * total "FAIL: $(name) render is (near) black: $(n)/$(total) nonblack \
                              (< $(round(frac*100; digits=2))%)"
    return n
end

"""
    color_fraction(img, which; thresh, amargin) -> Float64

Fraction of pixels where channel `which` (:red/:green/:blue) **dominates** the other two
channels by an absolute margin of at least `amargin`, and the pixel is bright enough
(max channel > `thresh`).  Grey/white pixels (r≈g≈b) are rejected even if bright, so a
colourless render cannot falsely satisfy the check.
"""
function color_fraction(img, which::Symbol; thresh = 0.12f0, amargin = 0.03f0)
    n = count(img) do c
        r, g, b = Float32(red(c)), Float32(green(c)), Float32(blue(c))
        max(r, g, b) <= thresh && return false
        which === :red   ? (r - max(g, b) > amargin) :
        which === :green ? (g - max(r, b) > amargin) :
                           (b - max(r, g) > amargin)
    end
    return n / length(img)
end
