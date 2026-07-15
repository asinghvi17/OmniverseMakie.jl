# Minimal JSON codec — hand-rolled so the transport adds no registered deps.
# Covers exactly what the protocol needs: objects of strings/numbers/bools/
# null + arrays (nested arrays for matrix payloads).  Responses come from
# python json.dumps with its default ensure_ascii=true, so byte-wise parsing
# is safe.

function _json_write(io::IO, s::AbstractString)
    print(io, '"')
    for c in s
        if c == '"'
            print(io, "\\\"")
        elseif c == '\\'
            print(io, "\\\\")
        elseif c == '\n'
            print(io, "\\n")
        elseif c == '\r'
            print(io, "\\r")
        elseif c == '\t'
            print(io, "\\t")
        elseif c < ' '
            print(io, "\\u", string(UInt16(c); base = 16, pad = 4))
        else
            print(io, c)
        end
    end
    print(io, '"')
end
_json_write(io::IO, b::Bool) = print(io, b ? "true" : "false")
_json_write(io::IO, n::Integer) = print(io, n)
_json_write(io::IO, x::Real) = print(io, Float64(x))
_json_write(io::IO, ::Nothing) = print(io, "null")
function _json_write(io::IO, v::Union{AbstractVector, Tuple})
    print(io, '[')
    for (i, x) in enumerate(v)
        i > 1 && print(io, ',')
        _json_write(io, x)
    end
    print(io, ']')
end

"One-line JSON object from `\"key\" => value` pairs (insertion order kept)."
function _json_object(pairs::Pair...)
    io = IOBuffer()
    print(io, '{')
    for (i, (k, v)) in enumerate(pairs)
        i > 1 && print(io, ',')
        _json_write(io, String(k))
        print(io, ':')
        _json_write(io, v)
    end
    print(io, '}')
    return String(take!(io))
end

mutable struct _JSONCursor
    s::String
    i::Int
end

function _parse_json(s::AbstractString)
    c = _JSONCursor(String(s), 1)
    v = _jvalue!(c)
    _jskipws!(c)
    c.i > ncodeunits(c.s) || error("json: trailing garbage at byte $(c.i) in: $s")
    return v
end

function _jskipws!(c::_JSONCursor)
    n = ncodeunits(c.s)
    while c.i <= n && codeunit(c.s, c.i) in (0x20, 0x09, 0x0a, 0x0d)
        c.i += 1
    end
end

_jbyte(c::_JSONCursor) =
    c.i <= ncodeunits(c.s) ? codeunit(c.s, c.i) : error("json: unexpected end of input")

function _jvalue!(c::_JSONCursor)
    _jskipws!(c)
    b = _jbyte(c)
    b == UInt8('{') && return _jobject!(c)
    b == UInt8('[') && return _jarray!(c)
    b == UInt8('"') && return _jstring!(c)
    b == UInt8('t') && return (_jliteral!(c, "true"); true)
    b == UInt8('f') && return (_jliteral!(c, "false"); false)
    b == UInt8('n') && return (_jliteral!(c, "null"); nothing)
    return _jnumber!(c)
end

function _jliteral!(c::_JSONCursor, lit::String)
    for u in codeunits(lit)
        _jbyte(c) == u || error("json: bad literal at byte $(c.i)")
        c.i += 1
    end
end

function _jhex4!(c::_JSONCursor)
    ncodeunits(c.s) >= c.i + 3 || error("json: truncated \\u escape")
    v = parse(UInt32, SubString(c.s, c.i, c.i + 3); base = 16)
    c.i += 4
    return v
end

function _jstring!(c::_JSONCursor)
    io = IOBuffer()
    c.i += 1  # opening quote
    while true
        b = _jbyte(c)
        if b == UInt8('"')
            c.i += 1
            return String(take!(io))
        elseif b == UInt8('\\')
            c.i += 1
            e = _jbyte(c)
            c.i += 1
            if e == UInt8('u')
                cp = _jhex4!(c)
                if 0xd800 <= cp <= 0xdbff  # surrogate pair
                    (_jbyte(c) == UInt8('\\')) || error("json: lone surrogate")
                    c.i += 1
                    (_jbyte(c) == UInt8('u')) || error("json: lone surrogate")
                    c.i += 1
                    lo = _jhex4!(c)
                    cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00)
                end
                print(io, Char(cp))
            elseif e == UInt8('n')
                print(io, '\n')
            elseif e == UInt8('t')
                print(io, '\t')
            elseif e == UInt8('r')
                print(io, '\r')
            elseif e == UInt8('b')
                print(io, '\b')
            elseif e == UInt8('f')
                print(io, '\f')
            else
                write(io, e)  # covers \" \\ \/
            end
        else
            write(io, b)  # raw UTF-8 passthrough
            c.i += 1
        end
    end
end

function _jnumber!(c::_JSONCursor)
    n = ncodeunits(c.s)
    start = c.i
    while c.i <= n &&
            (codeunit(c.s, c.i) in UInt8('0'):UInt8('9') ||
             codeunit(c.s, c.i) in (UInt8('-'), UInt8('+'), UInt8('.'), UInt8('e'), UInt8('E')))
        c.i += 1
    end
    t = SubString(c.s, start, c.i - 1)
    v = tryparse(Int, t)
    v === nothing || return v
    f = tryparse(Float64, t)
    f === nothing || return f
    error("json: bad number $(repr(t)) at byte $start")
end

function _jobject!(c::_JSONCursor)
    d = Dict{String, Any}()
    c.i += 1  # '{'
    _jskipws!(c)
    if _jbyte(c) == UInt8('}')
        c.i += 1
        return d
    end
    while true
        _jskipws!(c)
        k = _jstring!(c)
        _jskipws!(c)
        _jbyte(c) == UInt8(':') || error("json: expected ':' at byte $(c.i)")
        c.i += 1
        d[k] = _jvalue!(c)
        _jskipws!(c)
        b = _jbyte(c)
        c.i += 1
        b == UInt8(',') && continue
        b == UInt8('}') && return d
        error("json: expected ',' or '}' at byte $(c.i - 1)")
    end
end

function _jarray!(c::_JSONCursor)
    a = Any[]
    c.i += 1  # '['
    _jskipws!(c)
    if _jbyte(c) == UInt8(']')
        c.i += 1
        return a
    end
    while true
        push!(a, _jvalue!(c))
        _jskipws!(c)
        b = _jbyte(c)
        c.i += 1
        b == UInt8(',') && continue
        b == UInt8(']') && return a
        error("json: expected ',' or ']' at byte $(c.i - 1)")
    end
end

_namedtuple(d::Dict{String, Any}) = (; (Symbol(k) => v for (k, v) in d)...)
