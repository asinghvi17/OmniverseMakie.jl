# KitServer — the persistent headless Kit render-server subprocess + its
# line-JSON RPC transport (peer: kit_server.py, launched via `kit --exec`).
#
# Protocol:
#   Julia -> Kit : one JSON object per line on a named pipe (FIFO) Julia
#                  creates; Julia holds the write end open for the server's
#                  lifetime.  stdin via `kit --exec` is unreliable; a FIFO is
#                  the simplest robust channel.
#   Kit -> Julia : one JSON line per command appended (+fsync) to a regular
#                  response file, tagged with the command's "id"; Julia polls
#                  from a byte offset.  Line 1 is {"id":0,"op":"ready",...}.

const GPU_LOCK = "/tmp/omniversemakie-gpu.lock"
const _SERVER_PY = joinpath(@__DIR__, "kit_server.py")

_default_kit_release_dir() = get(ENV, "KIT_RELEASE_DIR",
    joinpath(homedir(),
        "temp/omniverse-dsx-blueprint-for-ai-factories/deps/kit-cae/_build/linux-x86_64/release"))

"""
    KitServer

Handle to a persistent headless Kit render-server subprocess (see the file
header for the transport).  Create with [`start_kit_server`](@ref), talk to
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
           --enable omni.volume
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
