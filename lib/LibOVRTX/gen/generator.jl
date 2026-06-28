using Clang
using Clang.Generators

const OVRTX_INCLUDE = get(ENV, "OVRTX_INCLUDE",
    "/home/juliahub/temp/omniverse-makie/references/ovrtx/include")

# The umbrella header in this directory selects which ovrtx headers to wrap.
headers = [joinpath(@__DIR__, "ovrtx_umbrella.h")]

options = load_options(joinpath(@__DIR__, "generator.toml"))

args = get_default_args()              # bundled libclang system headers + target
push!(args, "-I$(OVRTX_INCLUDE)")
push!(args, "-x", "c")                 # force C: the public ABI is `extern "C"`
push!(args, "-std=c11")
# Only wrap declarations that live under the ovrtx include tree (skip libc).
push!(args, "-DOVRTX_GENERATING_BINDINGS=1")

ctx = create_context(headers, args, options)

build!(ctx)

@info "Done. Wrote $(options["general"]["output_file_path"])"
