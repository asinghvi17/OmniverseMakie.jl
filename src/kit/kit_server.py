# Persistent Kit-side render server for OmniverseMakie's KitScreen spike.
#
# Launched via `kit --exec kit_server.py` (see src/kit/kitscreen.jl for the
# full launch: extension chain, /rtx/index settings, libGLU shim, GPU lock).
# Runs an asyncio command loop for the lifetime of the Kit process so one
# ~30-60 s Kit startup amortizes over many renders.
#
# Transport (chosen over stdin, which is unreliable under --exec):
#   commands : one JSON object per line on a named pipe (FIFO) created by the
#              Julia side (env OMK_KIT_CMD_FIFO).  A plain Python thread
#              blocks on the FIFO and feeds lines into an asyncio queue.
#   responses: one JSON line per command appended (+flush +fsync) to a regular
#              file (env OMK_KIT_RSP_FILE), tagged with the command's "id".
#              Line 1 is the ready marker {"id": 0, "op": "ready", ...}.
#
# Ops: ping | open_stage | render | set_attr | quit.  Every response carries
# "ok" (and "error" text on failure).  Each handler is wrapped in try/except:
# a bad command must never kill the server.
import asyncio
import json
import os
import threading
import time
import traceback

import carb
import omni.kit.app
import omni.usd

CMD_FIFO = os.environ["OMK_KIT_CMD_FIFO"]
RSP_FILE = os.environ["OMK_KIT_RSP_FILE"]
# Frames pumped after a stage switch before open_stage returns: Kit rebuilds
# the RTX/IndeX pipeline over a few updates, and rendering immediately after
# open can capture the previous stage or black.
SETTLE_FRAMES = int(os.environ.get("OMK_KIT_SETTLE_FRAMES", "8"))


def _respond(obj):
    line = json.dumps(obj)
    with open(RSP_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(f"[kit_server] -> {line}", flush=True)


def _fifo_reader(loop, queue):
    # Blocking-open the FIFO; EOF means the writer (Julia) closed its end —
    # reopen and keep serving so a reconnect works.
    while True:
        try:
            with open(CMD_FIFO, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        loop.call_soon_threadsafe(queue.put_nowait, line)
        except Exception:
            traceback.print_exc()
            time.sleep(0.5)


async def _op_ping(app, cmd):
    return {"pong": True, "time": time.time()}


async def _op_open_stage(app, cmd):
    ctx = omni.usd.get_context()
    path = cmd["path"]
    ok = ctx.open_stage(path)  # synchronous under --/omni.kit.plugin/syncUsdLoads=true
    if not ok:
        raise RuntimeError(f"open_stage({path!r}) returned False")
    for _ in range(int(cmd.get("settle", SETTLE_FRAMES))):
        await app.next_update_async()
    return {"stage": path}


async def _op_render(app, cmd):
    frames = int(cmd.get("frames", 240))
    out = cmd["out"]
    for _ in range(frames):  # convergence (accumulation) frames
        await app.next_update_async()
    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

    vp = get_active_viewport()
    cap = capture_viewport_to_file(vp, out)
    # Fire-and-forget writes NOTHING — the capture must be awaited.
    res = await cap.wait_for_result(completion_frames=60)
    for _ in range(180):  # belt-and-braces: wait for bytes on disk
        await app.next_update_async()
        if os.path.exists(out) and os.path.getsize(out) > 0:
            break
    size = os.path.getsize(out) if os.path.exists(out) else 0
    if size <= 0:
        raise RuntimeError(f"capture wrote no bytes to {out!r} (wait_for_result -> {res!r})")
    rsp = {"out": out, "bytes": size}
    try:
        rsp["resolution"] = [int(x) for x in vp.resolution]
    except Exception:
        pass
    return rsp


async def _op_set_attr(app, cmd):
    # Best-effort typed attribute write on the open stage (spike scope:
    # str/float/bool/int scalars).
    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError("no stage open")
    prim = stage.GetPrimAtPath(cmd["prim"])
    if not prim or not prim.IsValid():
        raise RuntimeError(f"no prim at {cmd['prim']!r}")
    attr = prim.GetAttribute(cmd["attr"])
    if not attr or not attr.IsValid():
        raise RuntimeError(f"prim {cmd['prim']!r} has no attribute {cmd['attr']!r}")
    value = cmd["value"]
    cur = attr.Get()
    if cur is not None and isinstance(cur, (bool, int, float, str)) \
            and not isinstance(value, type(cur)):
        value = type(cur)(value)  # e.g. JSON int -> float attr
    if not attr.Set(value):
        raise RuntimeError(f"Set({value!r}) failed on {attr.GetPath()}")
    for _ in range(int(cmd.get("settle", 2))):
        await app.next_update_async()
    return {"prim": cmd["prim"], "attr": cmd["attr"]}


_HANDLERS = {
    "ping": _op_ping,
    "open_stage": _op_open_stage,
    "render": _op_render,
    "set_attr": _op_set_attr,
}


async def _main():
    app = omni.kit.app.get_app()
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    threading.Thread(target=_fifo_reader, args=(loop, queue), daemon=True).start()

    for _ in range(3):  # let the viewport/renderer come up before advertising
        await app.next_update_async()
    settings = carb.settings.get_settings()
    _respond({
        "id": 0, "op": "ready", "ok": True, "pid": os.getpid(),
        "composite_enabled": settings.get("/rtx/index/compositeEnabled"),
    })

    while True:
        line = await queue.get()
        try:
            cmd = json.loads(line)
        except Exception as exc:
            _respond({"id": None, "ok": False,
                      "error": f"bad json: {exc!r}: {line[:200]}"})
            continue
        cid = cmd.get("id")
        op = cmd.get("op")
        if op == "quit":
            _respond({"id": cid, "op": "quit", "ok": True})
            break
        handler = _HANDLERS.get(op)
        if handler is None:
            _respond({"id": cid, "op": op, "ok": False,
                      "error": f"unknown op {op!r}"})
            continue
        try:
            result = await handler(app, cmd)
            rsp = {"id": cid, "op": op, "ok": True}
            rsp.update(result or {})
            _respond(rsp)
        except Exception as exc:
            traceback.print_exc()
            _respond({"id": cid, "op": op, "ok": False,
                      "error": f"{type(exc).__name__}: {exc}"})

    # Shutdown: post_quit, grace frames, then exit hard — headless teardown
    # can stall (same fallback probe.py uses).
    app.post_quit(0)
    for _ in range(120):
        await app.next_update_async()
    print("[kit_server] post_quit stalled; hard exit", flush=True)
    os._exit(0)


asyncio.ensure_future(_main())
