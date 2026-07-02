using Test
using OmniverseMakie: OV

# ---------------------------------------------------------------------------
# Review Track A / Task A5 — index-config synthesis robustness (PURE, no GPU).
#
# `_synth_index_config` synthesizes the carb config that registers the IndeX libs token.
# Two robustness holes are closed here:
#   (1) the env-supplied libs path was interpolated into JSON UNESCAPED — a `"` or `\` in
#       the path yields invalid JSON that carb rejects at init.  `OV._json_escape` fixes it.
#   (2) the token was merged at the FIRST literal `"app": {` substring — a NESTED `"app"`
#       earlier in the file (or spacing variance) mis-plants the token where carb silently
#       ignores it.  The merge now anchors to the TOP-LEVEL (top-indent, line-start)
#       `"app": {` via `OV._find_top_level_app` and errors when there is no top-level block.
# The text-surgery approach is retained (NOT parse+reserialize): the real config is JSON5
# (comments / trailing commas) a strict JSON writer would mangle — so tests use JSON5
# fixtures built in a temp dir and assert escaped byte sequences directly (no JSON dep).
# ---------------------------------------------------------------------------

# Write a fixture ovrtx.config.json into a fresh temp <ovrtx>/bin dir; return that dir.
function _write_bin(cfg::AbstractString)
    bin = mktempdir()
    write(joinpath(bin, "ovrtx.config.json"), cfg)
    return bin
end

# Minimal JSON string-body unescaper (inverse of OV._json_escape) — lets the round-trip
# tests prove the escaped value decodes back to the raw path with no JSON package.
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

# A JSON5 config mirroring the real install ovrtx.config.json shape (top-level "app" with a
# nested "graphics" block; JSON5 trailing commas).  Triple-quoted: the column-0 braces make
# the common-leading-whitespace zero, so Julia does NOT dedent — indentation is verbatim.
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

# GOLDEN: the exact bytes today's `_synth_index_config` emits for _CLEAN_FIXTURE + _CLEAN_LIBS
# (captured from the unmodified function before this task's change).  The token block is
# spliced immediately after the top-level `"app": {`.  The behavior-preserving change must
# reproduce this byte-for-byte for a clean path.
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
        "product": "Omniverse.ovrtx",
        "dumpDir": "."
    },
}
"""

# A config whose "log" block contains a DECOY nested `"app": {` (8-space indent) BEFORE the
# genuine top-level `"app": {` (4-space indent).  The old first-substring merge would plant
# the token in the decoy; the fix must target the top-level block.
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

# Only a NESTED `"app"` (inside "log"), no top-level one → no top-level block to merge into.
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

# No `"app"` key anywhere → the existing no-block error case.
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

@testset "A5 _json_escape: JSON string-escape set (pure)" begin
    # Untouched: printable ASCII + non-ASCII UTF-8 (>= U+0020) pass through verbatim.
    @test OV._json_escape("plain/path-42") == "plain/path-42"
    @test OV._json_escape("café-π")        == "café-π"      # non-ASCII stays raw (valid JSON)

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
    @test OV._json_escape("\x0b")  == "\\u000b"             # vertical tab (no short form)
    @test OV._json_escape("\x1f")  == "\\u001f"
    @test OV._json_escape(" ")     == " "                   # U+0020 is the boundary: NOT escaped

    # Round-trip: escaping then unescaping is the identity for tricky paths.
    for s in ("/opt/li\"bs", "C:\\\\ov\\\\libs", "a\"\\\nb\t", "/plain/ok")
        @test _json_unescape(OV._json_escape(s)) == s
    end
end

@testset "A5 _synth_index_config: happy path byte-stable vs golden (pure)" begin
    bin = _write_bin(_CLEAN_FIXTURE)
    out = read(OV._synth_index_config(bin, _CLEAN_LIBS), String)
    @test out == _GOLDEN_CLEAN                              # byte-for-byte regression anchor
end

@testset "A5 _synth_index_config: libs with quote/backslash -> valid escaped JSON (pure)" begin
    # A path containing a double-quote: unescaped it would terminate the JSON string early
    # (invalid config carb rejects).  The escaped value must be embedded and quote-terminated.
    libs_q = "/opt/li\"bs/omni.index.libs-1287db94366cf6fe"
    out_q  = read(OV._synth_index_config(_write_bin(_CLEAN_FIXTURE), libs_q), String)
    esc_q  = OV._json_escape(libs_q)
    @test esc_q == "/opt/li\\\"bs/omni.index.libs-1287db94366cf6fe"                # the `"` became \"
    @test occursin("\"omni.index.libs\": \"" * esc_q * "\"\n        },", out_q)   # escaped + closed + structure intact
    @test _json_unescape(esc_q) == libs_q                                          # value round-trips to raw path
    @test !occursin("libs\": \"$(libs_q)\"", out_q)                                # the RAW (unescaped) value is NOT present

    # A path containing a backslash: must be doubled (\\) so JSON stays valid.
    libs_b = "/opt/li\\bs/omni.index.libs-1287db94366cf6fe"
    out_b  = read(OV._synth_index_config(_write_bin(_CLEAN_FIXTURE), libs_b), String)
    esc_b  = OV._json_escape(libs_b)
    @test occursin("\"omni.index.libs\": \"" * esc_b * "\"\n        },", out_b)
    @test _json_unescape(esc_b) == libs_b
    @test occursin("li\\\\bs", out_b)                                              # backslash was doubled in output
end

@testset "A5 _synth_index_config: nested app before top-level -> token in top-level (pure)" begin
    out = read(OV._synth_index_config(_write_bin(_NESTED_FIXTURE), _CLEAN_LIBS), String)

    r_nested = findfirst("\n        \"app\": {", out)   # 8-space decoy, inside "log"
    r_top    = findfirst("\n    \"app\": {", out)       # 4-space genuine top-level
    # `"tokens": {` is the inserted block's opener — an unambiguous marker (the libs PATH
    # itself contains the substring "omni.index.libs", so that would over-count).
    r_tok    = findfirst("\"tokens\": {", out)

    @test r_nested !== nothing                            # fixture sanity: decoy present
    @test r_top    !== nothing                            # fixture sanity: top-level present
    @test r_nested.start < r_top.start                    # decoy really precedes the top-level
    @test r_tok    !== nothing
    @test r_tok.start > r_top.start                       # token block lands AFTER the top-level "app": {
    @test count("\"tokens\": {", out) == 1                # exactly once -> not also in the decoy
    # the decoy block is left intact (its "note" key untouched, no token spliced into it)
    @test occursin("        \"app\": {\n            \"note\":", out)
end

@testset "A5 _synth_index_config: no top-level app block errors clearly (pure)" begin
    # No "app" key anywhere — the existing no-block error.
    err = try; OV._synth_index_config(_write_bin(_NO_APP_FIXTURE), _CLEAN_LIBS); nothing
          catch e; e end
    @test err isa ErrorException
    @test occursin("no top-level", err.msg) && occursin("app", err.msg)

    # Only a NESTED "app" (no top-level one) — also no top-level block to merge into.
    err2 = try; OV._synth_index_config(_write_bin(_NESTED_ONLY_FIXTURE), _CLEAN_LIBS); nothing
           catch e; e end
    @test err2 isa ErrorException
    @test occursin("no top-level", err2.msg) && occursin("app", err2.msg)
end
