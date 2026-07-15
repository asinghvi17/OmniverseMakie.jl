# KitTransport — the pluggable channel a KitScreen uses to drive a Kit runtime.
#
# Two implementations behind ONE Julia-facing API (KitScreen / colorbuffer /
# open_stage! / render! / set_attr! / close do not change):
#
#   * SubprocessTransport (DEFAULT, coexistence path) — a thin wrapper over the
#     persistent headless `kit` subprocess + line-JSON FIFO RPC (server.jl +
#     kit_server.py).  Proven, and the ONLY path that coexists with in-process
#     standalone ovrtx (two processes, two carb frameworks).
#
#   * InProcessTransport (OPT-IN) — hosts Kit IN THIS Julia process via the
#     LibKitJL C shim (libkitjl.so over Carbonite's ABI).  No subprocess, no
#     FIFO: lifecycle + settings are native `kitjl_*` calls; stage open /
#     set_attr / capture / write_vdb reuse kit_server.py's proven handler
#     bodies through the Python scripting hatch, driven by Julia pumping
#     `kitjl_update()`.  Faster repeated renders; groundwork for zero-copy GPU
#     planes (v2).  CANNOT coexist with in-process ovrtx (hazard b): a session
#     picks ONE in-process backend.
#
# Transport ops are internal generic functions dispatched on the transport
# type; authoring.jl / stage_usda are transport-agnostic and unchanged.

abstract type KitTransport end

# ---------------------------------------------------------------------------
# SubprocessTransport — forwards to the KitServer RPC (no behavior change).
# ---------------------------------------------------------------------------
struct SubprocessTransport <: KitTransport
    server::KitServer
    owns::Bool                      # close the server on _t_close only if we started it
end

_t_workdir(t::SubprocessTransport) = t.server.workdir
_t_isopen(t::SubprocessTransport) = isopen(t.server)
_t_close(t::SubprocessTransport) = (t.owns && close(t.server); nothing)

_t_open_stage!(t::SubprocessTransport, path::AbstractString; timeout_s::Real) =
    (_check(rpc(t.server, "open_stage"; path = String(path), timeout_s), "open_stage($path)"); nothing)

function _t_render(t::SubprocessTransport; frames::Integer, out::AbstractString, timeout_s::Real)
    rsp = _check(rpc(t.server, "render"; frames, out = String(out), timeout_s), "render")
    return Int(rsp.bytes)
end

function _t_set_attr!(t::SubprocessTransport, prim::AbstractString, attr::AbstractString, value;
                      usd_type::Union{Nothing, AbstractString})
    kwargs = usd_type === nothing ? (;) : (; usd_type = String(usd_type))
    _check(rpc(t.server, "set_attr"; prim = String(prim), attr = String(attr), value, kwargs...),
           "set_attr($prim.$attr)")
    return nothing
end

_t_write_vdb(t::SubprocessTransport; raw, shape, origin, voxel_size, out, name) =
    (_check(rpc(t.server, "write_vdb"; raw, shape, origin, voxel_size, out, name),
            "write_vdb($out)"); nothing)

# ---------------------------------------------------------------------------
# InProcessTransport — LibKitJL handle + embedded Python helper.
# ---------------------------------------------------------------------------

import Libdl

# Process-global: carb cannot cleanly restart within a process, and two carb
# frameworks (Kit + in-process ovrtx) fight (hazard b).  Non-null once an
# in-process Kit app exists this process; a second construction errors.
const _INPROCESS_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

# Trimmed copy of kit_server.py's handlers WITHOUT the FIFO/asyncio transport
# (Julia is the transport now, pumping kitjl_update between synchronous
# executeString calls).  Defines dispatchers in Kit's __main__ namespace; each
# op reads a small JSON request file and writes a JSON response file — the same
# response-file discipline as the subprocess, without the pipe.  This keeps ONE
# source of truth for the Python semantics shared conceptually with
# kit_server.py.
const _INPROC_PY = raw"""
import json, os, traceback

def _omk_write_rsp(rsp_path, obj):
    with open(rsp_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj)); f.flush(); os.fsync(f.fileno())

def _omk_h_open_stage(cmd):
    import omni.usd
    ctx = omni.usd.get_context()
    ok = ctx.open_stage(cmd["path"])   # sync under --/omni.kit.plugin/syncUsdLoads=true
    if not ok:
        raise RuntimeError("open_stage(%r) returned False" % cmd["path"])
    return {"stage": cmd["path"]}

def _omk_h_set_attr(cmd):
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("no stage open")
    prim = stage.GetPrimAtPath(cmd["prim"])
    if not prim or not prim.IsValid():
        raise RuntimeError("no prim at %r" % cmd["prim"])
    attr = prim.GetAttribute(cmd["attr"])
    if not attr or not attr.IsValid():
        raise RuntimeError("prim %r has no attribute %r" % (cmd["prim"], cmd["attr"]))
    value = cmd["value"]
    usd_type = cmd.get("usd_type")
    if usd_type == "matrix4d":
        from pxr import Gf
        value = Gf.Matrix4d(*(float(x) for row in value for x in row))
    elif usd_type == "double3":
        from pxr import Gf
        value = Gf.Vec3d(*(float(x) for x in value))
    elif usd_type is not None:
        raise RuntimeError("unsupported usd_type %r" % usd_type)
    else:
        cur = attr.Get()
        if cur is not None and isinstance(cur, (bool, int, float, str)) \
                and not isinstance(value, type(cur)):
            value = type(cur)(value)
    if not attr.Set(value):
        raise RuntimeError("Set(%r) failed on %s" % (value, attr.GetPath()))
    return {"prim": cmd["prim"], "attr": cmd["attr"]}

def _omk_h_write_vdb(cmd):
    import numpy as np, openvdb
    shape = tuple(int(x) for x in cmd["shape"])
    arr = np.fromfile(cmd["raw"], dtype=np.float32)
    if arr.size != shape[0] * shape[1] * shape[2]:
        raise RuntimeError("raw payload has %d floats, expected %r" % (arr.size, shape))
    arr = arr.reshape(shape, order="F")
    grid = openvdb.FloatGrid()
    grid.copyFromArray(np.ascontiguousarray(arr), tolerance=0.0)
    vs = [float(x) for x in cmd["voxel_size"]]
    o = [float(x) for x in cmd["origin"]]
    grid.transform = openvdb.createLinearTransform(
        [[vs[0], 0, 0, 0], [0, vs[1], 0, 0], [0, 0, vs[2], 0], [o[0], o[1], o[2], 1]])
    grid.name = cmd.get("name", "density")
    grid.gridClass = openvdb.GridClass.FOG_VOLUME
    openvdb.write(cmd["out"], grids=[grid])
    return {"out": cmd["out"], "voxels": int(arr.size)}

_omk_cap = None   # keep the ViewportCapture alive across the pumped update frames
def _omk_h_kick_capture(cmd):
    global _omk_cap
    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
    vp = get_active_viewport()
    _omk_cap = capture_viewport_to_file(vp, cmd["out"])
    res = {}
    try:
        res["resolution"] = [int(x) for x in vp.resolution]
    except Exception:
        pass
    return res

_OMK_HANDLERS = {
    "open_stage": _omk_h_open_stage,
    "set_attr": _omk_h_set_attr,
    "write_vdb": _omk_h_write_vdb,
    "kick_capture": _omk_h_kick_capture,
}

def _omk_dispatch(req_path, rsp_path):
    try:
        with open(req_path, "r", encoding="utf-8") as f:
            cmd = json.load(f)
        handler = _OMK_HANDLERS.get(cmd.get("op"))
        if handler is None:
            raise RuntimeError("unknown op %r" % cmd.get("op"))
        res = handler(cmd)
        obj = {"ok": True}
        obj.update(res or {})
        _omk_write_rsp(rsp_path, obj)
    except Exception as e:
        traceback.print_exc()
        _omk_write_rsp(rsp_path, {"ok": False, "error": "%s: %s" % (type(e).__name__, e)})
"""

mutable struct InProcessTransport <: KitTransport
    handle::Ptr{Cvoid}
    workdir::String
    kit_release_dir::String
    settle_frames::Int
    capture_poll_frames::Int
    open::Bool
    _glu_handle::Ptr{Cvoid}         # keep the dlopen'd libGLU alive
end

# Set DISPLAY / XDG_RUNTIME_DIR / XAUTHORITY on the current process ENV (the
# in-process RTX renderer needs them even headless; mutter's Xwayland cookie is
# a dotfile whose suffix changes per boot — glob it).  Mirrors server.jl's
# `_display_env!`, but targets this process's ENV.
function _ensure_display_env!()
    haskey(ENV, "DISPLAY") || (ENV["DISPLAY"] = ":0")
    rtdir = get(ENV, "XDG_RUNTIME_DIR", "/run/user/$(ccall(:getuid, Cuint, ()))")
    ENV["XDG_RUNTIME_DIR"] = rtdir
    xauth = get(ENV, "XAUTHORITY", "")
    if isempty(xauth) || !isfile(xauth)
        cookies = isdir(rtdir) ?
            filter(startswith(".mutter-Xwaylandauth."), readdir(rtdir)) : String[]
        isempty(cookies) || (ENV["XAUTHORITY"] = joinpath(rtdir, first(sort(cookies))))
    end
    return nothing
end

# libGLU.so.1 + GLVND: Kit's RTX renderer dlopens the MDL SDK (libneuray.so),
# which needs libGLU.  For the subprocess this is an LD_LIBRARY_PATH shim, but
# a mid-process LD_LIBRARY_PATH change is unreliable — so here we EXTRACT (reuse
# server.jl's `_ensure_libglu!`) then dlopen the resolved path RTLD_GLOBAL, so
# Kit's later by-soname dlopen finds the already-loaded image.  The misleading
# symptom without it is "Invalid sync scope created".
function _ensure_libglu_loaded!()
    if occursin("libGLU.so.1", read(`ldconfig -p`, String))
        return Libdl.dlopen("libGLU.so.1", Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
    end
    cache = joinpath(tempdir(), "omniversemakie-libglu")
    _ensure_libglu!(Dict{String, String}(); cache = cache)   # extracts the debs if missing
    libpath = joinpath(cache, "usr/lib/x86_64-linux-gnu/libGLU.so.1")
    isfile(libpath) || error("libGLU.so.1 not found after extraction at $libpath")
    return Libdl.dlopen(libpath, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
end

# Build the argv vector handed to IApp::startup — the SAME launch as
# start_kit_server (minus flock/timeout/kit_bin/--exec), with ABSOLUTE
# --ext-folder paths and --/crashreporter/enabled=false (hazards a/c).
function _inproc_argv(kit_release_dir::AbstractString, width::Integer, height::Integer,
                      extra_settings::Vector{String})
    return String[
        "kit", "--empty", "--no-window",
        "--ext-folder", abspath(joinpath(kit_release_dir, "exts")),
        "--ext-folder", abspath(joinpath(kit_release_dir, "extscache")),
        "--enable", "omni.kit.mainwindow",
        "--enable", "omni.kit.viewport.window",
        "--enable", "omni.kit.viewport.utility",
        "--enable", "omni.hydra.rtx",
        "--enable", "omni.rtx.index_composite",
        "--enable", "omni.kit.exec.core",
        "--enable", "omni.volume",
        "--/app/asyncRendering=false",
        "--/omni.kit.plugin/syncUsdLoads=true",
        "--/rtx/index/compositeEnabled=true",
        "--/rtx/index/overrideSubdivisionMode=kd_tree",
        "--/rtx/index/overrideSubdivisionPartCount=1",
        "--/app/renderer/resolution/width=$width",
        "--/app/renderer/resolution/height=$height",
        "--/crashreporter/enabled=false",
        extra_settings...,
    ]
end

"""
    InProcessTransport(; kit_release_dir, width, height, kwargs...) -> InProcessTransport

Start an in-process Kit app (LibKitJL) and load the composite extension chain
via the SAME argv the subprocess uses.  Startup runs inside OmniverseMakie's
crashreporter/GC signal guard (breakpad vs Julia GC-safepoint signals — the
ovrtx fix pattern).

!!! warning "KNOWN LIMITATION — in-process startup deadlock (v1)"
    Kit's in-process startup currently **deadlocks** during the
    `omni.usd_resolver` Python-extension `dlopen` when Kit is co-hosted in the
    Julia process: the calling thread spins (~100% CPU) inside a carb loader
    lock and `IApp::startup` never returns, so this constructor **hangs**.
    Reproduced identically with and without the signal guard,
    with `--handle-signals=no`, with a full `carb::startupFramework` init, and
    with a system-`libstdc++` `LD_PRELOAD`.  The native shim, lifecycle
    (`kitjl_*`), settings, argv construction, embedded Python helper, and the
    whole transport op surface are implemented and the pure tier is green, but
    **no in-process frame renders yet**.  Use the default **subprocess**
    transport for working colored-volume rendering.  The deadlock is the open
    v1 seam (see `docs/superpowers/specs/2026-07-15-libkitjl-design.md` and the
    package README); a fix likely needs the resolver/USD/TBB static-init to run
    outside Julia's co-hosted loader state (e.g. deferring/loading it in a
    controlled order, or an upstream carb/USD change).

HAZARD (b): the in-process Kit app CANNOT coexist with in-process standalone
ovrtx (two carb frameworks fight over `g_carbFramework`, `CARB_APP_PATH`, and
the crashreporter singleton).  A session must pick ONE in-process backend; use
the SUBPROCESS transport (the default) if you also drive ovrtx in-process.
Only one in-process Kit app may exist per process (carb cannot restart it).
"""
function InProcessTransport(;
        kit_release_dir::AbstractString = _default_kit_release_dir(),
        width::Integer = 1280,
        height::Integer = 720,
        extra_settings::Vector{String} = String[],
        settle_frames::Integer = 8,
        capture_poll_frames::Integer = 600,
        workdir::AbstractString = mktempdir(; prefix = "omk_inproc_", cleanup = false))
    LibKitJL.available() ||
        error("InProcessTransport: LibKitJL unavailable ($(LibKitJL.LIBKITJL_UNAVAILABLE_REASON)). " *
              "Build it with KIT_RELEASE_DIR set, or use the subprocess transport (the default).")
    _INPROCESS_HANDLE[] == C_NULL ||
        error("InProcessTransport: an in-process Kit app already exists in this process " *
              "(carb cannot cleanly restart — one app per process; hazard b).")
    isdir(joinpath(kit_release_dir, "kit")) ||
        error("InProcessTransport: no Kit runtime at $kit_release_dir (set KIT_RELEASE_DIR)")

    mkpath(workdir)
    _ensure_display_env!()
    glu = _ensure_libglu_loaded!()

    argv = _inproc_argv(kit_release_dir, width, height, extra_settings)

    # Start inside the signal guard (breakpad vs GC safepoint — see
    # OmniverseMakie's src/binding/signals.jl).  Belt: --/crashreporter/enabled
    # =false in argv.
    handle = OM.OV.SignalGuard.with_restored_signals() do
        LibKitJL.startup(argv)
    end

    _INPROCESS_HANDLE[] = handle
    t = InProcessTransport(handle, String(workdir), String(kit_release_dir),
                           Int(settle_frames), Int(capture_poll_frames), true, glu)

    # Load the embedded Python handlers once (persists in Kit's interpreter).
    LibKitJL.exec_string(handle, _INPROC_PY)
    return t
end

_t_workdir(t::InProcessTransport) = t.workdir
_t_isopen(t::InProcessTransport) = t.open && LibKitJL.is_running(t.handle)

function _t_close(t::InProcessTransport)
    t.open || return nothing
    try
        LibKitJL.shutdown(t.handle)
    catch e
        @warn "InProcessTransport: shutdown error (ignored)" exception = e
    end
    t.open = false
    _INPROCESS_HANDLE[] = C_NULL   # allow a fresh app in a NEW process only
    return nothing
end

# One synchronous op: write the JSON request, exec the dispatcher (executeString
# runs to completion), read the JSON response.  Throws on a command-level error.
function _inproc_call(t::InProcessTransport, op::AbstractString; kwargs...)
    req = joinpath(t.workdir, "req_$(op).json")
    rsp = joinpath(t.workdir, "rsp_$(op).json")
    isfile(rsp) && rm(rsp; force = true)
    write(req, _json_object("op" => String(op), (String(k) => v for (k, v) in kwargs)...))
    LibKitJL.exec_string(t.handle, "_omk_dispatch(r'''$(req)''', r'''$(rsp)''')")
    isfile(rsp) ||
        error("in-process $op: no response file (Python did not run — check Kit log)")
    r = _namedtuple(_parse_json(read(rsp, String)))
    r.ok == true ||
        error("in-process $op failed: $(get(r, :error, "(no error text)"))")
    return r
end

function _t_open_stage!(t::InProcessTransport, path::AbstractString; timeout_s::Real)
    _inproc_call(t, "open_stage"; path = String(path))
    # Settle: Kit rebuilds the RTX/IndeX pipeline over a few updates (the
    # subprocess handler awaits these; here Julia pumps them).
    for _ in 1:t.settle_frames
        LibKitJL.update(t.handle)
    end
    return nothing
end

function _t_set_attr!(t::InProcessTransport, prim::AbstractString, attr::AbstractString, value;
                      usd_type::Union{Nothing, AbstractString})
    kwargs = usd_type === nothing ? (;) : (; usd_type = String(usd_type))
    _inproc_call(t, "set_attr"; prim = String(prim), attr = String(attr), value, kwargs...)
    for _ in 1:2
        LibKitJL.update(t.handle)
    end
    return nothing
end

_t_write_vdb(t::InProcessTransport; raw, shape, origin, voxel_size, out, name) =
    (_inproc_call(t, "write_vdb"; raw, shape, origin, voxel_size, out, name); nothing)

# render = pump convergence frames, kick capture, then pump+poll the PNG until
# it has bytes (the exact belt-and-braces loop kit_server.py runs, driven from
# Julia).  Returns the captured byte count.
function _t_render(t::InProcessTransport; frames::Integer, out::AbstractString, timeout_s::Real)
    outp = String(out)
    isfile(outp) && rm(outp; force = true)
    for _ in 1:frames                       # convergence / accumulation frames
        LibKitJL.update(t.handle)
    end
    _inproc_call(t, "kick_capture"; out = outp)   # registers the async capture
    deadline = time() + timeout_s
    bytes = 0
    for _ in 1:t.capture_poll_frames
        LibKitJL.update(t.handle)             # advances the capture's completion
        if isfile(outp) && (bytes = filesize(outp)) > 0
            break
        end
        time() > deadline && break
    end
    return bytes
end
