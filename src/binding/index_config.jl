# IndeX enablement — process-global, set ONCE, BEFORE the first ovrtx_create_renderer.
#
# ovrtx renders a UsdVolVolume by routing it to NVIDIA IndeX Direct.  IndeX's loader resolves
# the carb TOKEN ${omni.index.libs} (registered by Kit's ext manager; ABSENT in standalone
# ovrtx) to find <dir>/bin/nvindex-libs and dlopen its libraries.  We register that token by
# feeding ovrtx a carb config: carb registers tokens from its /app/tokens/* settings subtree and
# honours the config named by CARB_FRAMEWORK_CONFIG_NAME (joined onto CARB_APP_PATH = <ovrtx>/bin)
# at framework init — which happens at the FIRST ovrtx_create_renderer.  So the injection must
# run before the first Renderer() (it is called at the top of Renderer()), once per process.
#
# Env contract (both optional; absent → IndeX not enabled = pre-volume behaviour, zero overhead):
#   OMNIVERSEMAKIE_OVRTX_CONFIG  abs path to a ready `*.config.json` that already registers
#                                /app/tokens/omni.index.libs (user-managed).  Highest precedence.
#   OMNIVERSEMAKIE_INDEX_LIBS    path to the `omni.index.libs` ext ROOT (loader appends
#                                /bin/nvindex-libs).  We SYNTHESIZE a config = copy of the install
#                                ovrtx.config.json + the app.tokens."omni.index.libs" key.
#
# VERIFIED (Volumes M1 plan Task 1 Step 2, faithful timing — module loaded, THEN env injected,
# THEN Renderer()): synthesizing the config, pointing CARB_FRAMEWORK_CONFIG_NAME at it via a path
# RELATIVE to <ovrtx>/bin (carb mangles absolute values and appends ".config.json"), then creating
# a Renderer yields `IndeX Direct: initialization successful.` and a bare torus.vdb renders
# (~9.2k non-black px @ 512²).  Ground truth: .superpowers/sdd/m6b/volume-spike-report.md (exp6).

# Set to true by `_ensure_index` iff IndeX was enabled this process.  Read by `_index_enabled`.
const _INDEX_ENABLED = Ref(false)

# Synthesize a carb config that registers /app/tokens/omni.index.libs = `libs`, returning its
# path.  The install ovrtx.config.json is JSON5 (trailing commas) and ALREADY has a top-level
# `"app": {` block, so we MERGE a `tokens` sub-key into it by minimal text surgery — NOT
# parse+reserialize (carb's loader tolerates JSON5, but a strict Julia JSON writer would drop the
# comments/trailing-commas and might reorder keys).  Copying the whole file preserves the
# install's log/graphics/crashreporter settings, since CARB_FRAMEWORK_CONFIG_NAME's config
# REPLACES the default ovrtx.config.json.
function _synth_index_config(ovrtx_bin::AbstractString, libs::AbstractString)
    base = read(joinpath(ovrtx_bin, "ovrtx.config.json"), String)
    occursin("\"app\": {", base) ||
        error("ovrtx.config.json has no top-level \"app\" block to merge the IndeX token into")
    token_block = "\n        \"tokens\": {\n            \"omni.index.libs\": \"$(libs)\"\n        },"
    merged = replace(base, "\"app\": {" => "\"app\": {" * token_block; count = 1)
    path = joinpath(mktempdir(), "idx.config.json")
    write(path, merged)
    return path
end

# Compute the CARB_FRAMEWORK_CONFIG_NAME value.  carb force-joins it onto CARB_APP_PATH
# (= <ovrtx>/bin, setenv'd by libovrtx-dynamic.so), MANGLES absolute values, and appends
# ".config.json".  So we pass `config_file` (which MUST be named `*.config.json`) as a path
# RELATIVE to <ovrtx>/bin with the ".config.json" suffix removed (carb re-adds it).
function _carb_config_name(config_file::AbstractString, ovrtx_bin::AbstractString)
    endswith(config_file, ".config.json") ||
        error("carb config must be named *.config.json (carb appends .config.json); got: $config_file")
    stem = config_file[1:end - length(".config.json")]
    return relpath(stem, ovrtx_bin)
end

"""
    _ensure_index() -> Bool

Enable NVIDIA IndeX volume rendering for this process, ONCE (memoized via
`Base.OncePerProcess`).  Returns `true` iff IndeX was enabled.  Reads
`OMNIVERSEMAKIE_OVRTX_CONFIG` (precedence) or `OMNIVERSEMAKIE_INDEX_LIBS`; with neither set it is
a no-op returning `false` (zero overhead — non-volume rendering is byte-unchanged).  Never
throws: on any misconfiguration it `@warn`s once and returns `false` (volumes stay unavailable;
non-volume rendering is unaffected).

Called at the top of `Renderer()`, BEFORE `ovrtx_create_renderer`, so carb consumes the config
at framework init.  Idempotent: only the first renderer pays setup; later calls are a Ref read.
"""
const _ensure_index = Base.OncePerProcess{Bool}() do
    cfg  = get(ENV, "OMNIVERSEMAKIE_OVRTX_CONFIG", "")
    libs = get(ENV, "OMNIVERSEMAKIE_INDEX_LIBS", "")
    (isempty(cfg) && isempty(libs)) && return false          # disabled → zero overhead
    try
        ovrtx_bin = dirname(ENV["OVRTX_LIBRARY_PATH"])       # <ovrtx>/bin == CARB_APP_PATH
        config_file = if !isempty(cfg)
            isfile(cfg) || error("OMNIVERSEMAKIE_OVRTX_CONFIG points to a missing file: $cfg")
            cfg
        else
            _synth_index_config(ovrtx_bin, libs)
        end
        ENV["CARB_FRAMEWORK_CONFIG_NAME"] = _carb_config_name(config_file, ovrtx_bin)
        _INDEX_ENABLED[] = true
        return true
    catch e
        @warn "OmniverseMakie: could not enable IndeX volume rendering; volumes will be unavailable" exception = (e, catch_backtrace()) maxlog = 1
        return false
    end
end

"""
    _index_enabled() -> Bool

Non-triggering query: `true` iff `_ensure_index()` has run this process AND succeeded.  Used by
`author_vdb_volume!` to error clearly when volumes are requested without IndeX enabled.  Reading
this does NOT trigger the one-time setup (unlike calling `_ensure_index()`).
"""
_index_enabled() = _INDEX_ENABLED[]
