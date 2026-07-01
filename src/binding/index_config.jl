# IndeX enablement — process-global, set ONCE, BEFORE the first ovrtx_create_renderer.
#
# ovrtx routes a UsdVolVolume to NVIDIA IndeX Direct.  IndeX's loader resolves the carb
# TOKEN ${omni.index.libs} (registered by Kit's ext manager; ABSENT in standalone ovrtx)
# to <dir>/bin/nvindex-libs and dlopens its libs.  We register that token via a carb
# config: carb reads tokens from its /app/tokens/* subtree and honours the config named by
# CARB_FRAMEWORK_CONFIG_NAME (joined onto CARB_APP_PATH = <ovrtx>/bin) at framework init =
# the FIRST ovrtx_create_renderer.  Hence inject once/process before the first Renderer()
# (done at the top of Renderer()).
#
# Env contract (both optional; absent → IndeX off = pre-volume behaviour, zero overhead):
#   OMNIVERSEMAKIE_OVRTX_CONFIG  abs path to a ready `*.config.json` already registering
#                                /app/tokens/omni.index.libs (user-managed; highest prec).
#   OMNIVERSEMAKIE_INDEX_LIBS    path to the `omni.index.libs` ext ROOT (loader appends
#                                /bin/nvindex-libs); we SYNTHESIZE a config = install
#                                ovrtx.config.json + the app.tokens."omni.index.libs" key.
#
# VERIFIED: the config path must be RELATIVE to <ovrtx>/bin (carb mangles absolute values
# and appends ".config.json") → `IndeX Direct: initialization successful.` + torus.vdb
# renders (~9.2k non-black px @ 512²).  Ground truth:
# .superpowers/sdd/m6b/volume-spike-report.md (exp6).

# Set to true by `_ensure_index` iff IndeX was enabled this process.  Read by `_index_enabled`.
const _INDEX_ENABLED = Ref(false)

# Synthesize a carb config registering /app/tokens/omni.index.libs = `libs`; return its path.
# The install ovrtx.config.json is JSON5 (trailing commas) with a top-level `"app": {` block,
# so MERGE a `tokens` sub-key by minimal text surgery — NOT parse+reserialize (a strict JSON
# writer would drop comments/trailing-commas and reorder keys).  Copying the whole file keeps
# the install's log/graphics/crashreporter settings, since the CARB_FRAMEWORK_CONFIG_NAME
# config REPLACES the default ovrtx.config.json.
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

# Compute CARB_FRAMEWORK_CONFIG_NAME.  carb force-joins it onto CARB_APP_PATH (= <ovrtx>/bin,
# setenv'd by libovrtx-dynamic.so), MANGLES absolute values, and appends ".config.json".  So
# pass `config_file` (which MUST be `*.config.json`) as a path RELATIVE to <ovrtx>/bin with
# the ".config.json" suffix stripped (carb re-adds it).
function _carb_config_name(config_file::AbstractString, ovrtx_bin::AbstractString)
    endswith(config_file, ".config.json") ||
        error("carb config must be named *.config.json (carb appends .config.json); got: $config_file")
    stem = config_file[1:end - length(".config.json")]
    return relpath(stem, ovrtx_bin)
end

"""
    _ensure_index() -> Bool

Enable NVIDIA IndeX volume rendering ONCE per process (memoized via `Base.OncePerProcess`);
return `true` iff enabled.  Reads `OMNIVERSEMAKIE_OVRTX_CONFIG` (precedence) or
`OMNIVERSEMAKIE_INDEX_LIBS`; neither set → no-op `false` (zero overhead, non-volume rendering
byte-unchanged).  Never throws — on misconfiguration `@warn`s once and returns `false`.
Called at the top of `Renderer()` before `ovrtx_create_renderer`, so carb consumes the config
at framework init; only the first renderer pays setup (later calls are a Ref read).
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

`true` iff `_ensure_index()` has already run AND succeeded this process.  Does NOT trigger the
one-time setup (unlike `_ensure_index()`).  Used by `author_vdb_volume!` to error clearly when
volumes are requested without IndeX enabled.
"""
_index_enabled() = _INDEX_ENABLED[]
