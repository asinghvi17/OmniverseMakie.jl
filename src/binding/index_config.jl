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

# JSON-escape a value for embedding as a JSON string: backslash, double-quote, and the
# control chars U+0000-U+001F (the JSON-mandatory escape set); everything else (incl. non-ASCII
# UTF-8) is emitted verbatim.  The env-supplied `libs` path flows through here so a `"` or `\`
# in it can't produce invalid JSON that carb rejects at init.
function _json_escape(s::AbstractString)
    io = IOBuffer()
    for c in s
        if     c == '\\';  print(io, "\\\\")
        elseif c == '"';   print(io, "\\\"")
        elseif c == '\b';  print(io, "\\b")
        elseif c == '\f';  print(io, "\\f")
        elseif c == '\n';  print(io, "\\n")
        elseif c == '\r';  print(io, "\\r")
        elseif c == '\t';  print(io, "\\t")
        elseif c < '\x20'; print(io, "\\u", string(UInt32(c); base = 16, pad = 4))
        else               print(io, c)
        end
    end
    return String(take!(io))
end

# Locate the TOP-LEVEL `"app": {` object key carb actually reads — a line-start `"app"` key at
# the file's top-level indentation (that of the first object key line) — and return its
# `RegexMatch` (whose `.match` ends at the opening `{`), or `nothing` if there is no top-level
# block.  A NESTED `"app"` (deeper indent) or a `"app": {` inside a string value is NOT matched:
# the old first-substring merge could plant the token in such a spot, where carb silently
# ignores it.  Spacing around the colon/brace is tolerated (the old exact-string match was
# brittle).  `indent` holds only spaces/tabs, so it is regex-safe to interpolate.
function _find_top_level_app(base::AbstractString)
    first_key = match(r"^([ \t]+)\""m, base)   # first indented object key ⇒ top-level indent
    first_key === nothing && return nothing
    indent = first_key.captures[1]
    return match(Regex("^" * indent * "\"app\"[ \\t]*:[ \\t]*\\{", "m"), base)
end

# Neutralize carb's breakpad crash reporter in a config body: its SIGSEGV handler
# intercepts Julia's RECOVERABLE GC-safepoint page faults and kills the process — the root
# cause of the historical intermittent startup crash (full analysis in
# src/binding/signals.jl).  `/crashreporter/enabled = false` stops the interception:
# PROBE-PROVEN on this install — an unguarded create + post-create GC stress dies 3/3
# without the key and survives 3/3 with it.  The install config carries a top-level
# "crashreporter" block (product/dumpDir), so inject `"enabled": false` INTO it; if a
# future install drops the block, prepend a fresh one after the opening brace.  (Should a
# config ever already set "enabled", ours lands FIRST in the block — carb honors the first
# occurrence, per the probe.)  This is defense-in-depth alongside the GC-window guard in
# signals.jl: the guard protects create even when no config can be synthesized; this key
# also covers any handler re-arming AFTER create that the guard's one-shot restore misses.
function _disable_crashreporter(base::AbstractString)
    m = match(r"\"crashreporter\"[ \t]*:[ \t]*\{", base)
    if m === nothing
        i = findfirst('{', base)
        i === nothing && return String(base)   # not a JSON object — leave untouched
        return base[1:i] * "\n    \"crashreporter\": {\n        \"enabled\": false\n    }," *
               base[i + 1:end]
    end
    endidx = m.offset + ncodeunits(m.match) - 1
    return base[1:endidx] * "\n        \"enabled\": false," * base[endidx + 1:end]
end

# Synthesize a carb config registering /app/tokens/omni.index.libs = `libs`; return its path.
# The install ovrtx.config.json is JSON5 (trailing commas) with a top-level `"app": {` block,
# so MERGE a `tokens` sub-key by minimal text surgery — NOT parse+reserialize (a strict JSON
# writer would drop comments/trailing-commas and reorder keys).  Copying the whole file keeps
# the install's log/graphics settings, since the CARB_FRAMEWORK_CONFIG_NAME config REPLACES
# the default ovrtx.config.json.  The crash reporter is disabled in the copy (see
# `_disable_crashreporter`).  The libs value is JSON-escaped and the merge is anchored to
# the top-level `"app"` block (see `_json_escape` / `_find_top_level_app`).
function _synth_index_config(ovrtx_bin::AbstractString, libs::AbstractString)
    base = read(joinpath(ovrtx_bin, "ovrtx.config.json"), String)
    m = _find_top_level_app(base)
    m === nothing &&
        error("ovrtx.config.json has no top-level \"app\" block to merge the IndeX token into")
    token_block = "\n        \"tokens\": {\n            \"omni.index.libs\": \"$(_json_escape(libs))\"\n        },"
    endidx = m.offset + ncodeunits(m.match) - 1      # byte index of the matched `… "app": {` end
    merged = _disable_crashreporter(base[1:endidx] * token_block * base[endidx + 1:end])
    path = joinpath(mktempdir(), "idx.config.json")
    write(path, merged)
    return path
end

# The crash-reporter fix for the NON-volume path (no IndeX env): synthesize a copy of the
# install config with the crash reporter disabled and point CARB_FRAMEWORK_CONFIG_NAME at
# it — unless carb was already routed to a config (user-set env wins; a user-supplied
# OMNIVERSEMAKIE_OVRTX_CONFIG is likewise left untouched on the volume path).  Never
# throws: on any failure the GC-window guard in signals.jl still protects create.
function _ensure_crashreporter_off!()
    haskey(ENV, "CARB_FRAMEWORK_CONFIG_NAME") && return nothing   # existing routing wins
    try
        lib = get(ENV, "OVRTX_LIBRARY_PATH", "")
        isempty(lib) && return nothing
        ovrtx_bin = dirname(lib)
        base_path = joinpath(ovrtx_bin, "ovrtx.config.json")
        isfile(base_path) || return nothing
        path = joinpath(mktempdir(), "nocrash.config.json")
        write(path, _disable_crashreporter(read(base_path, String)))
        ENV["CARB_FRAMEWORK_CONFIG_NAME"] = _carb_config_name(path, ovrtx_bin)
    catch e
        @warn "OmniverseMakie: could not disable the carb crash reporter (the GC-window \
               guard in signals.jl still protects renderer creation)" exception = e maxlog = 1
    end
    return nothing
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
    if isempty(cfg) && isempty(libs)                         # IndeX disabled…
        _ensure_crashreporter_off!()                         # …but still neutralize breakpad
        return false
    end
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
