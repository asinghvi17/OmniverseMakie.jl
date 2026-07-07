# IndeX enablement — process-global, set once, before the first
# ovrtx_create_renderer.
#
# ovrtx routes a UsdVolVolume to NVIDIA IndeX Direct.  IndeX's loader
# resolves the carb token ${omni.index.libs} (registered by Kit's ext
# manager; absent in standalone ovrtx) to <dir>/bin/nvindex-libs and dlopens
# its libs.  We register that token via a carb config: carb reads tokens from
# its /app/tokens/* subtree and honours the config named by
# CARB_FRAMEWORK_CONFIG_NAME (joined onto CARB_APP_PATH = <ovrtx>/bin) at
# framework init = the first ovrtx_create_renderer.  Hence inject
# once/process before the first Renderer() (done at the top of Renderer()).
#
# Env contract (both optional; absent → IndeX off, zero overhead):
#   OMNIVERSEMAKIE_OVRTX_CONFIG  abs path to a ready `*.config.json` already
#                                registering /app/tokens/omni.index.libs
#                                (user-managed; highest precedence).
#   OMNIVERSEMAKIE_INDEX_LIBS    path to the `omni.index.libs` ext root (the
#                                loader appends /bin/nvindex-libs); a config
#                                is synthesized from the install
#                                ovrtx.config.json plus the
#                                app.tokens."omni.index.libs" key.

# Set to true by `_ensure_index` iff IndeX was enabled this process; read by
# `_index_enabled`.
const _INDEX_ENABLED = Ref(false)

# JSON-escape a value for embedding as a JSON string: backslash, double-quote,
# and control chars U+0000-U+001F (the JSON-mandatory escape set); everything
# else (incl. non-ASCII UTF-8) is verbatim.  The env-supplied `libs` path
# flows through here, so a `"` or `\` in it can't produce invalid JSON.
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

# Locate a top-level `"<key>": {` object key (line-start, at the indentation
# of the file's first object-key line); return its `RegexMatch`, whose
# `.match` ends at the opening `{`, or `nothing`.  A nested `"<key>"` or one
# inside a string value must not match — carb ignores tokens merged there.
# `key` is a literal JSON key (no regex metacharacters); `indent` is
# spaces/tabs only, so both are regex-safe to interpolate.
function _find_top_level_key(base::AbstractString, key::AbstractString)
    first_key = match(r"^([ \t]+)\""m, base)   # first object key ⇒ top indent
    first_key === nothing && return nothing
    indent = first_key.captures[1]
    return match(Regex("^" * indent * "\"" * key * "\"[ \\t]*:[ \\t]*\\{", "m"), base)
end

_find_top_level_app(base::AbstractString) = _find_top_level_key(base, "app")

# Neutralize carb's breakpad crash reporter in a config body: its SIGSEGV
# handler intercepts Julia's recoverable GC-safepoint page faults and kills
# the process (see src/binding/signals.jl); `/crashreporter/enabled = false`
# stops the interception.  Inject `"enabled": false` into the install
# config's top-level "crashreporter" block, or prepend a fresh block if
# absent; ours lands first and carb honors the first occurrence of a key.
# Defense in depth with the signals.jl GC-window guard, which protects create
# even with no config; this key also covers handler re-arming after create,
# which the guard's one-shot restore misses.
function _disable_crashreporter(base::AbstractString)
    # Top-level line-anchored (like _find_top_level_app): a nested
    # "crashreporter" or one inside a string value must not match — carb only
    # honors the top-level block.
    m = _find_top_level_key(base, "crashreporter")
    if m === nothing
        i = findfirst('{', base)
        i === nothing && return String(base)   # not JSON; leave untouched
        return base[1:i] * "\n    \"crashreporter\": {\n        \"enabled\": false\n    }," *
               base[i + 1:end]
    end
    endidx = m.offset + ncodeunits(m.match) - 1
    return base[1:endidx] * "\n        \"enabled\": false," * base[endidx + 1:end]
end

# Synthesize a carb config registering /app/tokens/omni.index.libs = `libs`;
# return its path.  The install ovrtx.config.json is JSON5 (trailing commas),
# so the `tokens` sub-key is merged into the top-level `"app"` block by text
# surgery, not parse+reserialize (a strict JSON writer would drop comments and
# trailing commas).  The whole file is copied because the
# CARB_FRAMEWORK_CONFIG_NAME config replaces (not overlays) the default
# ovrtx.config.json; the copy also gets the crash reporter disabled.
function _synth_index_config(ovrtx_bin::AbstractString, libs::AbstractString)
    base = read(joinpath(ovrtx_bin, "ovrtx.config.json"), String)
    m = _find_top_level_app(base)
    m === nothing &&
        error("ovrtx.config.json has no top-level \"app\" block to merge the IndeX token into")
    token_block = "\n        \"tokens\": {\n            \"omni.index.libs\": \"$(_json_escape(libs))\"\n        },"
    endidx = m.offset + ncodeunits(m.match) - 1   # byte index of match end
    merged = _disable_crashreporter(base[1:endidx] * token_block * base[endidx + 1:end])
    path = joinpath(mktempdir(), "idx.config.json")
    write(path, merged)
    return path
end

# Disable the crash reporter on the non-volume path (no IndeX env): point
# CARB_FRAMEWORK_CONFIG_NAME at a copy of the install config with the reporter
# disabled — unless carb is already routed to a config (user-set env wins; a
# user-supplied OMNIVERSEMAKIE_OVRTX_CONFIG is likewise left untouched on the
# volume path).  Never throws: on any failure the GC-window guard in
# signals.jl still protects create.
function _ensure_crashreporter_off!()
    # existing carb routing (user-set env) wins
    haskey(ENV, "CARB_FRAMEWORK_CONFIG_NAME") && return nothing
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

# Compute CARB_FRAMEWORK_CONFIG_NAME.  carb force-joins it onto CARB_APP_PATH
# (= <ovrtx>/bin, setenv'd by libovrtx-dynamic.so), mangles absolute values,
# and appends ".config.json" — so pass `config_file` (which must be a
# `*.config.json`) relative to <ovrtx>/bin with the suffix stripped.
function _carb_config_name(config_file::AbstractString, ovrtx_bin::AbstractString)
    endswith(config_file, ".config.json") ||
        error("carb config must be named *.config.json (carb appends .config.json); got: $config_file")
    stem = config_file[1:end - length(".config.json")]
    return relpath(stem, ovrtx_bin)
end

"""
    _ensure_index() -> Bool

Enable NVIDIA IndeX volume rendering once per process (memoized via
`Base.OncePerProcess`); return `true` iff enabled.  Reads
`OMNIVERSEMAKIE_OVRTX_CONFIG` (precedence) or `OMNIVERSEMAKIE_INDEX_LIBS`;
neither set → no-op `false` (zero overhead).  Never throws — on
misconfiguration `@warn`s once and returns `false`.  Called at the top of
`Renderer()` before `ovrtx_create_renderer`, so carb consumes the config at
framework init; only the first renderer pays setup (later calls are a Ref
read).
"""
const _ensure_index = Base.OncePerProcess{Bool}() do
    cfg  = get(ENV, "OMNIVERSEMAKIE_OVRTX_CONFIG", "")
    libs = get(ENV, "OMNIVERSEMAKIE_INDEX_LIBS", "")
    if isempty(cfg) && isempty(libs)          # IndeX disabled…
        _ensure_crashreporter_off!()          # …but still neutralize breakpad
        return false
    end
    try
        ovrtx_bin = dirname(ENV["OVRTX_LIBRARY_PATH"])   # == CARB_APP_PATH
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

`true` iff `_ensure_index()` has already run AND succeeded this process.
Does not trigger the one-time setup (unlike `_ensure_index()`).  Used by
`author_vdb_volume!` to error clearly when volumes are requested without
IndeX enabled.
"""
_index_enabled() = _INDEX_ENABLED[]
