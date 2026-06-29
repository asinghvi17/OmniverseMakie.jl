# examples/common/run_one.jl — render ONE scene in this (subprocess) child + assert.
# Usage: julia --project=examples examples/common/run_one.jl <scene>
include(joinpath(@__DIR__, "harness.jl"))

const SCENE = length(ARGS) >= 1 ? ARGS[1] : error("usage: run_one.jl <scene>")
const SCENE_FILE = joinpath(@__DIR__, "..", "rpr", SCENE * ".jl")
isfile(SCENE_FILE) || error("no scene file: $(SCENE_FILE)")
include(SCENE_FILE)

const scene_fn  = getfield(Main, Symbol("scene_", SCENE))
const assert_fn = getfield(Main, Symbol("assert_", SCENE))

img = run_example(SCENE, scene_fn)
assert_fn(img)                       # scene-specific asserts; throws on failure
println("PASS:", SCENE)
