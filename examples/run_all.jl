# examples/run_all.jl — subprocess-render every scene + property-assert.
# Usage: julia --project=examples examples/run_all.jl            # all rpr/*.jl
#        julia --project=examples examples/run_all.jl <scene…>   # subset
const EXAMPLES_DIR = @__DIR__
const RPR_DIR  = joinpath(EXAMPLES_DIR, "rpr")
const RUN_ONE  = joinpath(EXAMPLES_DIR, "common", "run_one.jl")
const OVRTX_LIB = get(ENV, "OVRTX_LIBRARY_PATH",
    "/home/juliahub/temp/omniverse-makie/references/ovrtx/examples/python/minimal/.venv/lib/python3.13/site-packages/ovrtx/bin/libovrtx-dynamic.so")

scenes = isempty(ARGS) ?
    sort([replace(basename(f), ".jl" => "") for f in readdir(RPR_DIR; join = true)
          if endswith(f, ".jl") && !startswith(basename(f), "_")]) :
    collect(ARGS)

results = Tuple{String,Symbol}[]
for s in scenes
    println("\n=== rendering $(s) ===")
    cmd = setenv(`julia --project=$(EXAMPLES_DIR) $(RUN_ONE) $(s)`,
                 "OVRTX_LIBRARY_PATH" => OVRTX_LIB,
                 "PATH" => get(ENV, "PATH", ""), "HOME" => get(ENV, "HOME", ""))
    ok = success(pipeline(cmd; stdout = stdout, stderr = stderr))
    push!(results, (s, ok ? :pass : :fail))
end

println("\n==== SUMMARY ====")
for (s, r) in results
    println(rpad(s, 32), r === :pass ? "PASS" : "FAIL")
end
nfail = count(((_, r),) -> r === :fail, results)
exit(nfail == 0 ? 0 : 1)
