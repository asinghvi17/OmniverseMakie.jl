using Test
using OmniverseMakie: OV

# ---------------------------------------------------------------------------
# Index-config synthesis robustness (pure, no GPU). `_synth_index_config`
# builds the carb config that registers the IndeX libs token: the libs path
# is JSON-escaped (`OV._json_escape`) and the token merges at the top-level
# line-start `"app": {` (`OV._find_top_level_app`), erroring when no
# top-level block exists. Text surgery is deliberate (NOT parse+reserialize):
# the real config is JSON5 (comments, trailing commas) that a strict JSON
# writer would mangle; tests assert escaped byte sequences directly.
# ---------------------------------------------------------------------------

# Write a fixture ovrtx.config.json into a fresh temp <ovrtx>/bin dir;
# return that dir.
function _write_bin(cfg::AbstractString)
    bin = mktempdir()
    write(joinpath(bin, "ovrtx.config.json"), cfg)
    return bin
end

# Minimal JSON string-body unescaper (inverse of OV._json_escape) — lets the
# round-trip tests prove the escaped value decodes back to the raw path with
# no JSON package.
function _json_unescape(s::AbstractString)
    cs = collect(s); io = IOBuffer(); i = 1
    while i <= length(cs)
        c = cs[i]
        if c == '\\'
            i += 1
            e = cs[i]
            if     e == 'n';  print(io, '\n')
            elseif e == 't';  print(io, '\t')
            elseif e == 'r';  print(io, '\r')
            elseif e == 'b';  print(io, '\b')
            elseif e == 'f';  print(io, '\f')
            elseif e == '"';  print(io, '"')
            elseif e == '\\'; print(io, '\\')
            elseif e == 'u';  (print(io, Char(parse(UInt32, String(cs[i+1:i+4]); base = 16))); i += 4)
            else              print(io, e)
            end
        else
            print(io, c)
        end
        i += 1
    end
    return String(take!(io))
end

const _CLEAN_LIBS = "/home/juliahub/.local/share/ov/data/exts/v2/omni.index.libs-1287db94366cf6fe"

# A JSON5 config mirroring the real install ovrtx.config.json shape
# (top-level "app" with a nested "graphics" block; JSON5 trailing commas).
# Triple-quoted: the column-0 braces make the common-leading-whitespace
# zero, so Julia does NOT dedent — indentation is verbatim.
const _CLEAN_FIXTURE = """
{
    "log": {
        "level": "Info",
        "outputStreamLevel": "Warning"
    },
    "app": {
        "graphics": {
            "api": "vulkan",
            "raytracing": true
        }
    },
    "crashreporter": {
        "product": "Omniverse.ovrtx",
        "dumpDir": "."
    },
}
"""

# Golden: the exact bytes `_synth_index_config` emits for _CLEAN_FIXTURE +
# _CLEAN_LIBS; the token block is spliced immediately after the top-level
# `"app": {` and must match byte-for-byte.
const _GOLDEN_CLEAN = """
{
    "log": {
        "level": "Info",
        "outputStreamLevel": "Warning"
    },
    "app": {
        "tokens": {
            "omni.index.libs": "$(_CLEAN_LIBS)"
        },
        "graphics": {
            "api": "vulkan",
            "raytracing": true
        }
    },
    "crashreporter": {
        "enabled": false,
        "product": "Omniverse.ovrtx",
        "dumpDir": "."
    },
}
"""

# A config whose "log" block holds a decoy nested `"app": {` (8-space indent)
# before the genuine top-level `"app": {` (4-space indent); the merge must
# target the top-level block, not the decoy.
const _NESTED_FIXTURE = """
{
    "log": {
        "app": {
            "note": "decoy nested app BEFORE the real one"
        }
    },
    "app": {
        "graphics": {
            "raytracing": true
        }
    },
}
"""

# A JSON5 config with a `//` line-comment mentioning a decoy `"app": {`
# before the genuine top-level block — the realistic hazard: the real install
# ovrtx.config.json is JSON5 with comments. `_find_top_level_app`'s
# line-anchored regex skips the comment and targets the real block.
# `_DECOY_COMMENT` is interpolated into the fixture so the
# verbatim-preservation assertion and the fixture text cannot drift apart.
const _DECOY_COMMENT = "// \"app\": { \"raytracing\": false }  (commented-out decoy, must stay ignored)"
const _COMMENT_FIXTURE = """
{
    $(_DECOY_COMMENT)
    "log": {
        "level": "Info"
    },
    "app": {
        "graphics": {
            "raytracing": true
        }
    },
}
"""

# Only a nested `"app"` (inside "log"), no top-level one to merge into.
const _NESTED_ONLY_FIXTURE = """
{
    "log": {
        "app": {
            "note": "nested only, no top-level app"
        }
    },
    "crashreporter": {
        "product": "x"
    },
}
"""

# No `"app"` key anywhere → the no-top-level-block error case.
const _NO_APP_FIXTURE = """
{
    "log": {
        "level": "Info"
    },
    "crashreporter": {
        "product": "x"
    },
}
"""

@testset "_json_escape: JSON string-escape set (pure)" begin
    # Untouched: printable ASCII + non-ASCII UTF-8 (>= U+0020) pass through
    # verbatim.
    @test OV._json_escape("plain/path-42") == "plain/path-42"
    # non-ASCII stays raw (valid JSON)
    @test OV._json_escape("café-π")        == "café-π"

    # The two JSON-mandatory single-char escapes.
    @test OV._json_escape("a\"b")  == "a\\\"b"              # double-quote -> \"
    @test OV._json_escape("a\\b")  == "a\\\\b"              # backslash    -> \\

    # Short control-char escapes.
    @test OV._json_escape("a\nb")  == "a\\nb"
    @test OV._json_escape("a\tb")  == "a\\tb"
    @test OV._json_escape("\r")    == "\\r"
    @test OV._json_escape("\b")    == "\\b"
    @test OV._json_escape("\f")    == "\\f"

    # Other control chars U+0000-U+001F -> \uXXXX (4-digit lowercase hex).
    @test OV._json_escape("\x00")  == "\\u0000"
    @test OV._json_escape("\x0b")  == "\\u000b"  # vertical tab (no short form)
    @test OV._json_escape("\x1f")  == "\\u001f"
    @test OV._json_escape(" ")     == " "   # U+0020 boundary: NOT escaped

    # Round-trip: escaping then unescaping is the identity for tricky paths.
    for s in ("/opt/li\"bs", "C:\\\\ov\\\\libs", "a\"\\\nb\t", "/plain/ok")
        @test _json_unescape(OV._json_escape(s)) == s
    end
end

@testset "crash-reporter disable: injected into an existing block / prepended when absent (pure)" begin
    # An existing top-level "crashreporter" block (the real install shape):
    # the key is injected into it, first position (carb honors the first
    # occurrence of a key).
    with_block = "{\n    \"crashreporter\": {\n        \"product\": \"x\"\n    },\n}"
    out = OV._disable_crashreporter(with_block)
    @test occursin("\"crashreporter\": {\n        \"enabled\": false,\n        \"product\"", out)
    @test count("\"crashreporter\"", out) == 1   # injected, not duplicated

    # No crashreporter block: a fresh disabled block is prepended after the
    # opening brace.
    without = "{\n    \"log\": {}\n}"
    out2 = OV._disable_crashreporter(without)
    @test occursin("\"crashreporter\": {\n        \"enabled\": false\n    },", out2)
    @test findfirst("\"crashreporter\"", out2).start < findfirst("\"log\"", out2).start

    # Not a JSON object at all: passthrough, no throw.
    @test OV._disable_crashreporter("plain text") == "plain text"
end

@testset "_synth_index_config: happy path byte-stable vs golden (pure)" begin
    bin = _write_bin(_CLEAN_FIXTURE)
    out = read(OV._synth_index_config(bin, _CLEAN_LIBS), String)
    @test out == _GOLDEN_CLEAN   # byte-for-byte regression anchor
end

@testset "_synth_index_config: libs with quote/backslash -> valid escaped JSON (pure)" begin
    # A path containing a double-quote: unescaped it would terminate the
    # JSON string early (invalid config carb rejects).  The escaped value
    # must be embedded and quote-terminated.
    libs_q = "/opt/li\"bs/omni.index.libs-1287db94366cf6fe"
    out_q  = read(OV._synth_index_config(_write_bin(_CLEAN_FIXTURE), libs_q), String)
    esc_q  = OV._json_escape(libs_q)
    # the `"` became \"
    @test esc_q == "/opt/li\\\"bs/omni.index.libs-1287db94366cf6fe"
    # escaped + closed + structure intact
    @test occursin("\"omni.index.libs\": \"" * esc_q * "\"\n        },", out_q)
    @test _json_unescape(esc_q) == libs_q   # value round-trips to raw path
    # the RAW (unescaped) value is NOT present
    @test !occursin("libs\": \"$(libs_q)\"", out_q)

    # A path containing a backslash: must be doubled (\\) so JSON stays valid.
    libs_b = "/opt/li\\bs/omni.index.libs-1287db94366cf6fe"
    out_b  = read(OV._synth_index_config(_write_bin(_CLEAN_FIXTURE), libs_b), String)
    esc_b  = OV._json_escape(libs_b)
    @test occursin("\"omni.index.libs\": \"" * esc_b * "\"\n        },", out_b)
    @test _json_unescape(esc_b) == libs_b
    @test occursin("li\\\\bs", out_b)   # backslash was doubled in output
end

@testset "_synth_index_config: nested app before top-level -> token in top-level (pure)" begin
    out = read(OV._synth_index_config(_write_bin(_NESTED_FIXTURE), _CLEAN_LIBS), String)

    r_nested = findfirst("\n        \"app\": {", out)   # 8-space decoy in "log"
    r_top    = findfirst("\n    \"app\": {", out)   # 4-space genuine top-level
    # `"tokens": {` is the inserted block's opener — an unambiguous marker
    # (the libs PATH itself contains the substring "omni.index.libs", so
    # that would over-count).
    r_tok    = findfirst("\"tokens\": {", out)

    @test r_nested !== nothing            # fixture sanity: decoy present
    @test r_top    !== nothing            # fixture sanity: top-level present
    @test r_nested.start < r_top.start    # decoy really precedes the top-level
    @test r_tok    !== nothing
    @test r_tok.start > r_top.start   # token lands AFTER the top-level "app": {
    @test count("\"tokens\": {", out) == 1   # exactly once -> not in the decoy
    # the decoy block is left intact (its "note" key untouched, no token
    # spliced into it)
    @test occursin("        \"app\": {\n            \"note\":", out)
end

@testset "_synth_index_config: line-comment app decoy before top-level -> token in top-level (pure)" begin
    # A `//` line-comment mentioning `"app": {` precedes the genuine
    # top-level block: the token must land in the real block and the comment
    # must survive verbatim. The other fixtures have no comments, so only
    # this case guards the comment hazard.
    out = read(OV._synth_index_config(_write_bin(_COMMENT_FIXTURE), _CLEAN_LIBS), String)

    r_top = findfirst("\n    \"app\": {", out)   # 4-space genuine top-level key
    # the inserted block's unambiguous opener
    r_tok = findfirst("\"tokens\": {", out)

    @test r_top !== nothing   # fixture sanity: real top-level present
    @test r_tok !== nothing
    @test count("\"tokens\": {", out) == 1   # inserted once (the marker trick)
    @test r_tok.start > r_top.start  # token lands after the real `"app": {`
    # The decoy comment line survives verbatim.
    @test occursin("\n    " * _DECOY_COMMENT * "\n", out)
end

@testset "_synth_index_config: no top-level app block errors clearly (pure)" begin
    # No "app" key anywhere — the existing no-block error.
    err = try; OV._synth_index_config(_write_bin(_NO_APP_FIXTURE), _CLEAN_LIBS); nothing
          catch e; e end
    @test err isa ErrorException
    @test occursin("no top-level", err.msg) && occursin("app", err.msg)

    # Only a NESTED "app" (no top-level one) — also no top-level block to
    # merge into.
    err2 = try; OV._synth_index_config(_write_bin(_NESTED_ONLY_FIXTURE), _CLEAN_LIBS); nothing
           catch e; e end
    @test err2 isa ErrorException
    @test occursin("no top-level", err2.msg) && occursin("app", err2.msg)
end
