# OmniverseMakieKit — SPIKE: a Makie screen backed by a persistent headless
# NVIDIA Kit render-server subprocess, distinct from the ovrtx-FFI
# `OmniverseMakie.Screen`.
#
# WHY: standalone ovrtx renders volumes grayscale-only (IndeX Direct).  Full
# transfer-function colors need a Kit runtime with the
# `omni.rtx.index_composite` extension chain — proven end-to-end in
# examples/kit_index_composite/ (README.md there has the recipe and evidence).
# This file wraps that proven launch in a persistent line-JSON RPC transport
# (Kit startup is ~30-60 s, so the server must outlive a single render) and
# scaffolds a `KitScreen` over it.
#
# NOT wired into OmniverseMakie's include chain: this is a standalone,
# `include`-able module so the package's precompilation and test suite stay
# untouched while the transport is a spike.  It only depends on Makie (for
# FileIO), which is in the package's own dependency set, so
# `include("src/kit/kitscreen.jl")` works under the repo project.
#
# Protocol (peer: src/kit/kit_server.py):
#   Julia -> Kit : one JSON object per line on a named pipe (FIFO) Julia
#                  creates; Julia holds the write end open for the server's
#                  lifetime.  stdin via `kit --exec` is unreliable; a FIFO is
#                  the simplest robust channel.
#   Kit -> Julia : one JSON line per command appended (+fsync) to a regular
#                  response file, tagged with the command's "id"; Julia polls
#                  from a byte offset.  Line 1 is {"id":0,"op":"ready",...}.
module OmniverseMakieKit

import Makie

export KitServer, KitScreen, start_kit_server, rpc, open_stage!, render!,
    render_stage!, set_attr!

# ---------------------------------------------------------------------------
# Minimal JSON codec — hand-rolled so the spike adds no registered deps.
# Covers exactly what the protocol needs: flat objects of
# strings/numbers/bools/null (+ small arrays like "resolution").  Responses
# come from python json.dumps with its default ensure_ascii=true, so
# byte-wise parsing is safe.
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# KitServer — the persistent subprocess + RPC transport
# ---------------------------------------------------------------------------

const GPU_LOCK = "/tmp/omniversemakie-gpu.lock"
const _SERVER_PY = joinpath(@__DIR__, "kit_server.py")

_default_kit_release_dir() = get(ENV, "KIT_RELEASE_DIR",
    joinpath(homedir(),
        "temp/omniverse-dsx-blueprint-for-ai-factories/deps/kit-cae/_build/linux-x86_64/release"))

"""
    KitServer

Handle to a persistent headless Kit render-server subprocess (see module
docstring for the transport).  Create with [`start_kit_server`](@ref), talk to
it with [`rpc`](@ref), shut it down with `close`.
"""
mutable struct KitServer
    proc::Base.Process
    workdir::String
    fifo_path::String
    fifo_io::Union{IOStream, Nothing}  # nothing once closed
    rsp_path::String
    rsp_offset::Int                    # bytes of the response file consumed
    log_path::String
    next_id::Int
end

Base.isopen(srv::KitServer) = srv.fifo_io !== nothing && process_running(srv.proc)

function Base.show(io::IO, srv::KitServer)
    state = isopen(srv) ? "running" : "closed"
    print(io, "KitServer($state, pid=$(getpid(srv.proc)), workdir=$(srv.workdir))")
end

# X/Wayland env the RTX renderer needs even headless.  The mutter Xwayland
# cookie is a DOTFILE whose suffix changes per boot — glob it.
function _display_env!(env::Dict{String, String})
    haskey(env, "DISPLAY") || (env["DISPLAY"] = ":0")
    rtdir = get(env, "XDG_RUNTIME_DIR", "/run/user/$(ccall(:getuid, Cuint, ()))")
    env["XDG_RUNTIME_DIR"] = rtdir
    xauth = get(env, "XAUTHORITY", "")
    if isempty(xauth) || !isfile(xauth)
        cookies = isdir(rtdir) ?
            filter(startswith(".mutter-Xwaylandauth."), readdir(rtdir)) : String[]
        isempty(cookies) || (env["XAUTHORITY"] = joinpath(rtdir, first(sort(cookies))))
    end
    return env
end

# Kit's RTX scene renderer dlopens the MDL SDK (libneuray.so), which needs
# libGLU.so.1 + GLVND.  Missing system-wide -> extract the debs locally, no
# root.  Symptom without it is MISLEADING: "UsdManager::addHydraEngine... -
# Invalid sync scope created. Failed to add Hydra engine".
function _ensure_libglu!(env::Dict{String, String};
        cache::AbstractString = joinpath(tempdir(), "omniversemakie-libglu"))
    occursin("libGLU.so.1", read(`ldconfig -p`, String)) && return env
    libdir = joinpath(cache, "usr/lib/x86_64-linux-gnu")
    if !isfile(joinpath(libdir, "libGLU.so.1"))
        @info "libGLU.so.1 missing system-wide; extracting locally" cache
        mkpath(cache)
        run(Cmd(`apt-get download libglu1-mesa libopengl0 libglvnd0`; dir = cache))
        for f in readdir(cache)
            endswith(f, ".deb") && run(Cmd(`dpkg-deb -x $f .`; dir = cache))
        end
        isfile(joinpath(libdir, "libGLU.so.1")) ||
            error("libGLU shim extraction failed in $cache")
    end
    old = get(env, "LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = isempty(old) ? libdir : "$libdir:$old"
    return env
end

function _log_tail(srv::KitServer; nbytes::Int = 3000)
    isfile(srv.log_path) || return "(no log at $(srv.log_path))"
    data = read(srv.log_path, String)
    return length(data) <= nbytes ? data : "..." * data[prevind(data, end, nbytes - 1):end]
end

"""
    start_kit_server(; kit_release_dir, width=1280, height=720, kwargs...) -> KitServer

Launch a persistent headless Kit render server: the bare `kit` kernel with the
IndeX-composite extension chain (the proven minimal launch from
examples/kit_index_composite/launch.sh), running `kit_server.py` via `--exec`.
Blocks until the server writes its ready marker.

The subprocess holds the shared GPU lock ($GPU_LOCK) for its whole lifetime —
it is continuously using the GPU — and is capped by `timeout \$lifetime_s` so
a hang can never squat the lock.

Keywords: `kit_release_dir` (default: `KIT_RELEASE_DIR` env or the DSX
kit-cae build), `width`/`height` (viewport resolution), `extra_settings`
(extra `--/path=value` carb settings), `lifetime_s=900`, `lock_wait_s=3600`,
`startup_timeout_s=600` (lock contention counts against it),
`settle_frames=8` (post-open_stage update frames), `workdir` (FIFO, response
file, log, default frame output; a fresh temp dir by default).
"""
function start_kit_server(;
        kit_release_dir::AbstractString = _default_kit_release_dir(),
        width::Integer = 1280,
        height::Integer = 720,
        extra_settings::Vector{String} = String[],
        lifetime_s::Integer = 900,
        lock_wait_s::Integer = 3600,
        startup_timeout_s::Real = 600,
        settle_frames::Integer = 8,
        workdir::AbstractString = mktempdir(; prefix = "omk_kitserver_", cleanup = false))
    kit_bin = joinpath(kit_release_dir, "kit", "kit")
    isfile(kit_bin) || error("no kit kernel at $kit_bin (set KIT_RELEASE_DIR)")
    isfile(_SERVER_PY) || error("kit_server.py not found at $_SERVER_PY")

    mkpath(workdir)
    fifo_path = joinpath(workdir, "cmd.fifo")
    rsp_path = joinpath(workdir, "responses.jsonl")
    log_path = joinpath(workdir, "kit.log")
    run(`mkfifo $fifo_path`)
    touch(rsp_path)

    env = Dict{String, String}(ENV)
    _display_env!(env)
    _ensure_libglu!(env)
    env["OMK_KIT_CMD_FIFO"] = fifo_path
    env["OMK_KIT_RSP_FILE"] = rsp_path
    env["OMK_KIT_SETTLE_FRAMES"] = string(settle_frames)

    # flock execs `timeout`, timeout forwards signals to kit — SIGTERM on the
    # process handle reaches the whole chain.  `--ext-folder` paths are
    # relative to the Kit release dir, hence dir = kit_release_dir.
    settings = [
        "--/app/asyncRendering=false",
        "--/omni.kit.plugin/syncUsdLoads=true",
        "--/rtx/index/compositeEnabled=true",
        "--/rtx/index/overrideSubdivisionMode=kd_tree",
        "--/rtx/index/overrideSubdivisionPartCount=1",
        "--/app/renderer/resolution/width=$width",
        "--/app/renderer/resolution/height=$height",
        extra_settings...,
    ]
    cmd = `flock -w $lock_wait_s $GPU_LOCK timeout $lifetime_s
           $kit_bin --empty --no-window
           --ext-folder exts --ext-folder extscache
           --enable omni.kit.mainwindow
           --enable omni.kit.viewport.window
           --enable omni.kit.viewport.utility
           --enable omni.hydra.rtx
           --enable omni.rtx.index_composite
           --enable omni.kit.exec.core
           $settings
           --exec $_SERVER_PY`
    proc = run(pipeline(setenv(cmd, env; dir = kit_release_dir);
                        stdout = log_path, stderr = log_path); wait = false)

    srv = KitServer(proc, String(workdir), fifo_path, nothing, rsp_path, 0, log_path, 0)

    # Wait for the ready marker before touching the FIFO (opening the write
    # end would block until the server's reader thread exists anyway).
    deadline = time() + startup_timeout_s
    ready = false
    while time() < deadline
        for rsp in _drain_responses!(srv)
            if get(rsp, "op", "") == "ready"
                get(rsp, "composite_enabled", false) == true ||
                    @warn "kit server ready but /rtx/index/compositeEnabled is not true" rsp
                ready = true
            end
        end
        ready && break
        if !process_running(proc)
            error("kit server exited during startup (code $(proc.exitcode)); " *
                  "log: $log_path\n" * _log_tail(srv))
        end
        sleep(0.25)
    end
    ready || (kill(proc); error("kit server not ready after $(startup_timeout_s)s; " *
                                "log: $log_path\n" * _log_tail(srv)))
    srv.fifo_io = open(fifo_path, "w")
    return srv
end

# Read complete JSON lines newly appended to the response file (partial lines
# are left for the next poll).
function _drain_responses!(srv::KitServer)
    out = Dict{String, Any}[]
    isfile(srv.rsp_path) || return out
    data = open(srv.rsp_path, "r") do io
        seek(io, srv.rsp_offset)
        read(io, String)
    end
    nl = findlast('\n', data)
    nl === nothing && return out
    complete = data[1:nl]
    srv.rsp_offset += ncodeunits(complete)
    for line in split(complete, '\n'; keepempty = false)
        try
            push!(out, _parse_json(line))
        catch err
            @warn "unparseable response line from kit server" line err
        end
    end
    return out
end

"""
    rpc(srv::KitServer, op; timeout_s=300, kwargs...) -> NamedTuple

Send one command (`kwargs` become the JSON payload) and block for its
response, matched by id.  The response always has `ok::Bool`; on failure it
carries `error::String`.  Throws only on transport problems (server gone,
deadline passed) — check `ok` for command-level failures, or use the
`KitScreen` wrappers which do.
"""
function rpc(srv::KitServer, op::AbstractString; timeout_s::Real = 300, kwargs...)
    srv.fifo_io === nothing && error("KitServer is closed")
    id = (srv.next_id += 1)
    line = _json_object("id" => id, "op" => String(op),
                        (String(k) => v for (k, v) in pairs(kwargs))...)
    write(srv.fifo_io, line, '\n')
    flush(srv.fifo_io)
    deadline = time() + timeout_s
    while time() < deadline
        for rsp in _drain_responses!(srv)
            get(rsp, "id", nothing) == id && return _namedtuple(rsp)
            @warn "skipping unmatched kit server response" rsp expected_id = id
        end
        process_running(srv.proc) ||
            error("kit server exited (code $(srv.proc.exitcode)) while waiting for " *
                  "rpc $op (id=$id); log: $(srv.log_path)\n" * _log_tail(srv))
        sleep(0.05)
    end
    error("rpc $op (id=$id) timed out after $(timeout_s)s; log: $(srv.log_path)\n" *
          _log_tail(srv))
end

_check(rsp::NamedTuple, what::AbstractString) =
    rsp.ok == true ? rsp :
    error("kit server $what failed: $(get(rsp, :error, "(no error text)"))")

"""
    close(srv::KitServer; grace_s=30)

Orderly shutdown: `quit` RPC, then escalate (SIGTERM, then SIGKILL) if the
process outlives the grace period.
"""
function Base.close(srv::KitServer; grace_s::Real = 30)
    if isopen(srv)
        try
            rpc(srv, "quit"; timeout_s = 15)
        catch err
            @warn "kit server quit RPC failed; escalating" err
        end
    end
    if srv.fifo_io !== nothing
        close(srv.fifo_io)
        srv.fifo_io = nothing
    end
    for sig in (nothing, Base.SIGTERM, Base.SIGKILL)
        sig === nothing || (process_running(srv.proc) && kill(srv.proc, sig))
        t0 = time()
        while process_running(srv.proc) && time() - t0 < grace_s
            sleep(0.25)
        end
        process_running(srv.proc) || return nothing
        @warn "kit server still running" next_signal = sig
    end
    return nothing
end

# ---------------------------------------------------------------------------
# KitScreen — Makie screen scaffold over a KitServer
# ---------------------------------------------------------------------------

"""
    KitScreen(server::KitServer; size=(1280, 720)) <: Makie.MakieScreen

SPIKE SCAFFOLD: a Makie screen whose backend is a persistent Kit render
server, giving full-color NVIDIA IndeX volume rendering (transfer-function
colormaps) that the ovrtx-FFI `OmniverseMakie.Screen` cannot do.

For now it renders externally-authored USD stages
([`render_stage!`](@ref)) and pokes attributes ([`set_attr!`](@ref)).

# TODO phase 2 — full Makie integration seam
#
# * `Base.display(screen::KitScreen, scene::Scene)` / `Makie.colorbuffer`:
#   AUTHORING REUSES src/translation/*.jl — the existing plot->USD emission
#   the ovrtx Screen uses.  Instead of pushing prims through the ovrtx FFI,
#   emit the whole scene to a .usda file (the usda_* emitters already produce
#   layer text), inject the composite renderSettings/customLayerData that
#   torus_colormap.usda.in demonstrates (rtx:index:compositeEnabled,
#   boundCamera, per-Volume nvindex:composite + omni:rtx:skip + Colormap
#   prims from the Makie colormap), then `open_stage` + `render` over this
#   RPC.  `Makie.colorbuffer(::KitScreen)` = render! + FileIO.load, already
#   the shape of `render_stage!` below.
# * LIVE UPDATES go through `set_attr` (typed pxr writes on the open stage):
#   map Makie observable updates -> (prim path, attr name, value) exactly as
#   src/screen.jl's live sync does for ovrtx, but over RPC.  Bulk/array
#   attrs (points, colors) need a binary sidecar channel or base64 payloads
#   — measure latency before choosing (a FIFO line-JSON roundtrip is ~ms;
#   the render itself dominates).
# * Protocol hardening: request pipelining (ids already allow it), an event
#   stream for async render completion, structured errors, server restart on
#   crash (KitServer already exposes everything needed to respawn).
"""
mutable struct KitScreen <: Makie.MakieScreen
    server::KitServer
    size::Tuple{Int, Int}
    stage_path::Union{Nothing, String}  # currently open stage
    framecount::Int                     # for default output naming
end

KitScreen(server::KitServer; size::Tuple{<:Integer, <:Integer} = (1280, 720)) =
    KitScreen(server, (Int(size[1]), Int(size[2])), nothing, 0)

Base.size(screen::KitScreen) = screen.size
Base.isopen(screen::KitScreen) = isopen(screen.server)
Base.close(screen::KitScreen) = close(screen.server)

function Base.show(io::IO, screen::KitScreen)
    print(io, "KitScreen($(screen.size[1])×$(screen.size[2]), ",
          isopen(screen) ? "open" : "closed",
          screen.stage_path === nothing ? "" : ", stage=$(screen.stage_path)", ")")
end

"""
    open_stage!(screen::KitScreen, usda_path) -> KitScreen

Open a USD stage in the server (synchronous; includes post-open settle
frames so the RTX/IndeX pipeline has rebuilt when this returns).
"""
function open_stage!(screen::KitScreen, usda_path::AbstractString; timeout_s::Real = 600)
    path = abspath(usda_path)
    _check(rpc(screen.server, "open_stage"; path, timeout_s), "open_stage($path)")
    screen.stage_path = path
    return screen
end

"""
    render!(screen::KitScreen; frames=240, out=...) -> image matrix

Converge the open stage for `frames` update frames, capture the viewport to
`out` (PNG), and load it (via `Makie.FileIO`).
"""
function render!(screen::KitScreen; frames::Integer = 240,
        out::AbstractString = joinpath(screen.server.workdir,
                                       "frame_$(screen.framecount + 1).png"),
        timeout_s::Real = 600)
    screen.stage_path === nothing && error("render!: no stage open; call open_stage! first")
    rsp = _check(rpc(screen.server, "render"; frames, out = abspath(out), timeout_s),
                 "render")
    screen.framecount += 1
    rsp.bytes > 0 || error("render: capture reported 0 bytes for $out")
    return Makie.FileIO.load(abspath(out))
end

"""
    render_stage!(screen::KitScreen, usda_path; frames=240, out=...) -> image matrix

`open_stage!` + `render!` in one call — the spike's main entry point.
"""
function render_stage!(screen::KitScreen, usda_path::AbstractString; kwargs...)
    open_stage!(screen, usda_path)
    return render!(screen; kwargs...)
end

"""
    set_attr!(screen::KitScreen, prim, attr, value) -> NamedTuple

Best-effort typed attribute write on the open stage (scalar
strings/floats/bools/ints; the server coerces to the attribute's existing
type).  Throws if the prim/attr does not exist or the write fails.
"""
set_attr!(screen::KitScreen, prim::AbstractString, attr::AbstractString, value) =
    _check(rpc(screen.server, "set_attr"; prim = String(prim), attr = String(attr), value),
           "set_attr($prim.$attr)")

end # module OmniverseMakieKit
