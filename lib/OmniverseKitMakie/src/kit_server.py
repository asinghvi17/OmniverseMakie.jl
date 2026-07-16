# Persistent Kit-side render server for OmniverseKitMakie's KitScreen.
#
# Launched via `kit --exec kit_server.py` (see server.jl for the full
# launch: extension chain, /rtx/index settings, libGLU shim, GPU lock).
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
# Ops: ping | open_stage | render | set_attr | write_vdb | gpu_frame_setup |
# gpu_volume_setup | gpu_write_vdb | quit.  Every response carries "ok" (and
# "error" text on failure).  Each handler is wrapped in try/except: a bad
# command must never kill the server.
#
# GPU data plane (docs/superpowers/specs/2026-07-16-kit-gpu-data-plane-design.md):
#   frames OUT : render {device:"cuda"} — omni.syntheticdata's LdrColorSDPtr
#                node exposes the LdrColor render var as a CUDA device pointer;
#                a device->device copy lands it in a server-owned cudaMalloc'd
#                buffer exported to Julia once via cudaIpcGetMemHandle.
#                render {device:"shm"} — capture_viewport_to_buffer (CPU bytes,
#                RGBA8_UNORM) memmoved into POSIX shared memory (/dev/shm).
#   volumes IN : gpu_volume_setup allocates an IPC staging buffer Julia fills
#                device-side; gpu_write_vdb copies it device->host and writes a
#                fresh OpenVDB .vdb (pyopenvdb; IndeX's importer is file-based).
# All CUDA calls go through the DRIVER API (libcuda.so.1, always present on a
# GPU box) with the device's primary context — UVA makes the annotator's
# pointer (from RTX's internal context) copyable from ours.
import asyncio
import base64
import ctypes
import json
import mmap
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


# ---------------------------------------------------------------------------
# CUDA driver-API shim (ctypes over libcuda.so.1) + capability probing
# ---------------------------------------------------------------------------
class _Cuda:
    """Minimal driver-API wrapper: primary-context alloc/copy/IPC."""

    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        self._check(self.lib.cuInit(0), "cuInit")
        self.device = ctypes.c_int(0)
        self._check(self.lib.cuDeviceGet(ctypes.byref(self.device), 0), "cuDeviceGet")
        self.ctx = ctypes.c_void_p()
        self._check(self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(self.ctx), self.device),
                    "cuDevicePrimaryCtxRetain")

    def _check(self, res, what):
        if res != 0:
            raise RuntimeError(f"CUDA driver error {res} in {what}")

    def _push(self):
        self._check(self.lib.cuCtxPushCurrent_v2(self.ctx), "cuCtxPushCurrent")

    def _pop(self):
        old = ctypes.c_void_p()
        self._check(self.lib.cuCtxPopCurrent_v2(ctypes.byref(old)), "cuCtxPopCurrent")

    def alloc(self, nbytes):
        self._push()
        try:
            ptr = ctypes.c_uint64()
            self._check(self.lib.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(nbytes)),
                        "cuMemAlloc")
            return ptr.value
        finally:
            self._pop()

    def free(self, ptr):
        self._push()
        try:
            self.lib.cuMemFree_v2(ctypes.c_uint64(ptr))
        finally:
            self._pop()

    def ipc_handle_b64(self, ptr):
        handle = (ctypes.c_char * 64)()
        self._push()
        try:
            self._check(self.lib.cuIpcGetMemHandle(handle, ctypes.c_uint64(ptr)),
                        "cuIpcGetMemHandle")
        finally:
            self._pop()
        return base64.b64encode(bytes(handle)).decode("ascii")

    def memcpy_dtod(self, dst, src, nbytes):
        self._push()
        try:
            self._check(self.lib.cuMemcpy(ctypes.c_uint64(dst), ctypes.c_uint64(src),
                                          ctypes.c_size_t(nbytes)), "cuMemcpy DtoD")
            self._check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")
        finally:
            self._pop()

    def memcpy_2d_dtod(self, dst, dst_pitch, src, src_pitch, width_bytes, height):
        # CUDA_MEMCPY2D for a pitched (strided) source -> tight destination.
        class CUDA_MEMCPY2D(ctypes.Structure):
            _fields_ = [
                ("srcXInBytes", ctypes.c_size_t), ("srcY", ctypes.c_size_t),
                ("srcMemoryType", ctypes.c_int), ("srcHost", ctypes.c_void_p),
                ("srcDevice", ctypes.c_uint64), ("srcArray", ctypes.c_void_p),
                ("srcPitch", ctypes.c_size_t),
                ("dstXInBytes", ctypes.c_size_t), ("dstY", ctypes.c_size_t),
                ("dstMemoryType", ctypes.c_int), ("dstHost", ctypes.c_void_p),
                ("dstDevice", ctypes.c_uint64), ("dstArray", ctypes.c_void_p),
                ("dstPitch", ctypes.c_size_t),
                ("WidthInBytes", ctypes.c_size_t), ("Height", ctypes.c_size_t),
            ]
        CU_MEMORYTYPE_DEVICE = 2
        m = CUDA_MEMCPY2D()
        m.srcMemoryType = CU_MEMORYTYPE_DEVICE
        m.srcDevice = src
        m.srcPitch = src_pitch
        m.dstMemoryType = CU_MEMORYTYPE_DEVICE
        m.dstDevice = dst
        m.dstPitch = dst_pitch
        m.WidthInBytes = width_bytes
        m.Height = height
        self._push()
        try:
            self._check(self.lib.cuMemcpy2D_v2(ctypes.byref(m)), "cuMemcpy2D")
            self._check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")
        finally:
            self._pop()

    def memcpy_dtoh(self, host_buf, src, nbytes):
        self._push()
        try:
            self._check(self.lib.cuMemcpyDtoH_v2(host_buf, ctypes.c_uint64(src),
                                                 ctypes.c_size_t(nbytes)), "cuMemcpyDtoH")
        finally:
            self._pop()


_CUDA = None          # _Cuda instance, or None when unavailable
_SD_OK = False        # omni.syntheticdata importable
_CAPS = {"shm_out": True, "cuda_ipc": False, "cuda_out": False, "cuda_device": -1}


def _probe_caps():
    global _CUDA, _SD_OK
    try:
        _CUDA = _Cuda()
        _CAPS["cuda_ipc"] = True
        _CAPS["cuda_device"] = 0
    except Exception as exc:
        print(f"[kit_server] cuda driver unavailable: {exc!r}", flush=True)
    try:
        import omni.syntheticdata  # noqa: F401
        _SD_OK = True
        _CAPS["cuda_out"] = _CAPS["cuda_ipc"]
    except Exception as exc:
        print(f"[kit_server] omni.syntheticdata unavailable: {exc!r}", flush=True)


# GPU-plane state: server-owned buffers + shm segment + activated annotators.
_FRAME_BUF = {"ptr": 0, "nbytes": 0, "width": 0, "height": 0}
_VOL_BUF = {"ptr": 0, "nbytes": 0, "shape": None}
_SHM = {"path": "", "mm": None, "size": 0}
_SD_ACTIVATED = set()  # render_product paths with LdrColorSDPtr activated


def _gpu_cleanup():
    if _CUDA is not None:
        for buf in (_FRAME_BUF, _VOL_BUF):
            if buf["ptr"]:
                try:
                    _CUDA.free(buf["ptr"])
                except Exception:
                    pass
                buf["ptr"] = 0
    if _SHM["mm"] is not None:
        try:
            _SHM["mm"].close()
            os.unlink(_SHM["path"])
        except Exception:
            pass
        _SHM["mm"] = None


def _shm_ensure(nbytes):
    if _SHM["mm"] is not None and _SHM["size"] >= nbytes:
        return
    if _SHM["mm"] is not None:
        _SHM["mm"].close()
        os.unlink(_SHM["path"])
    path = f"/dev/shm/omk_{os.getpid()}_frame"
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.ftruncate(fd, nbytes)
        _SHM["mm"] = mmap.mmap(fd, nbytes)
    finally:
        os.close(fd)
    _SHM["path"] = path
    _SHM["size"] = nbytes


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


async def _render_png(app, vp, out):
    from omni.kit.viewport.utility import capture_viewport_to_file

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
    return {"out": out, "bytes": size}


async def _render_shm(app, vp):
    # CPU fast path: capture_viewport_to_buffer delivers a PyCapsule pointing
    # at RGBA8_UNORM bytes (probed 2026-07-16) — memmove into /dev/shm, which
    # Julia mmaps.  No PNG encode, no disk.
    from omni.kit.viewport.utility import capture_viewport_to_buffer

    result = {}

    def on_capture(buffer, buffer_size, width, height, fmt):
        try:
            _shm_ensure(buffer_size)
            PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
            PyCapsule_GetPointer.restype = ctypes.c_void_p
            PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
            src = PyCapsule_GetPointer(buffer, None)
            dst = ctypes.addressof(ctypes.c_char.from_buffer(_SHM["mm"]))
            ctypes.memmove(dst, src, buffer_size)
            result.update(dict(shm_path=_SHM["path"], nbytes=buffer_size,
                               width=width, height=height, format=str(fmt)))
        except Exception as exc:  # surfaced via the empty-result check below
            result["error"] = f"{type(exc).__name__}: {exc}"

    cap = capture_viewport_to_buffer(vp, on_capture)
    await cap.wait_for_result(completion_frames=60)
    for _ in range(180):
        await app.next_update_async()
        if result:
            break
    if not result:
        raise RuntimeError("shm capture callback never fired")
    if "error" in result:
        raise RuntimeError(f"shm capture failed: {result['error']}")
    return result


async def _render_cuda(app, vp):
    # GPU path: the LdrColorSDPtr syntheticdata node exposes the LdrColor
    # render var as a CUDA device pointer (+ strides); a pitched device->device
    # copy lands the frame tightly packed in the IPC-exported buffer.
    if not _CAPS["cuda_out"]:
        raise RuntimeError("cuda out-plane unavailable (syntheticdata or libcuda missing)")
    if not _FRAME_BUF["ptr"]:
        raise RuntimeError("call gpu_frame_setup before render device=cuda")
    import omni.syntheticdata as syn

    rp = vp.render_product_path
    sdg = syn.SyntheticData.Get()
    if rp not in _SD_ACTIVATED:
        sdg.activate_node_template("LdrColorSD" + "Ptr", 0, [rp])
        _SD_ACTIVATED.add(rp)
        for _ in range(int(os.environ.get("OMK_KIT_SD_SETTLE", "8"))):
            await app.next_update_async()
    outs = sdg.get_node_attributes(
        "LdrColorSDPtr",
        ["outputs:dataPtr", "outputs:cudaDeviceIndex", "outputs:width",
         "outputs:height", "outputs:strides", "outputs:bufferSize", "outputs:format"],
        rp)
    if not outs or not outs.get("outputs:dataPtr"):
        raise RuntimeError(f"LdrColorSDPtr produced no data (outputs={outs!r})")
    if int(outs.get("outputs:cudaDeviceIndex", -1)) < 0:
        raise RuntimeError("LdrColorSDPtr returned host data, expected device")
    width = int(outs["outputs:width"])
    height = int(outs["outputs:height"])
    strides = outs.get("outputs:strides")
    row_bytes = width * 4  # RGBA8
    src_pitch = int(strides[1]) if strides is not None and len(strides) > 1 and int(strides[1]) > 0 else row_bytes
    need = row_bytes * height
    if need > _FRAME_BUF["nbytes"]:
        raise RuntimeError(f"frame buffer too small: need {need}, have {_FRAME_BUF['nbytes']} "
                           "(re-run gpu_frame_setup)")
    _CUDA.memcpy_2d_dtod(_FRAME_BUF["ptr"], row_bytes,
                         int(outs["outputs:dataPtr"]), src_pitch, row_bytes, height)
    return {"width": width, "height": height, "nbytes": need,
            "format": str(outs.get("outputs:format"))}


async def _op_render(app, cmd):
    frames = int(cmd.get("frames", 240))
    device = cmd.get("device", "png")
    for _ in range(frames):  # convergence (accumulation) frames
        await app.next_update_async()
    from omni.kit.viewport.utility import get_active_viewport

    vp = get_active_viewport()
    if device == "png":
        rsp = await _render_png(app, vp, cmd["out"])
    elif device == "shm":
        rsp = await _render_shm(app, vp)
    elif device == "cuda":
        rsp = await _render_cuda(app, vp)
    else:
        raise RuntimeError(f"unknown render device {device!r}")
    try:
        rsp.setdefault("resolution", [int(x) for x in vp.resolution])
    except Exception:
        pass
    return rsp


async def _op_gpu_frame_setup(app, cmd):
    # Idempotent per resolution: (re)allocate the IPC-exported frame buffer.
    if not _CAPS["cuda_ipc"]:
        raise RuntimeError("cuda unavailable (libcuda.so.1 failed to load)")
    width = int(cmd["width"])
    height = int(cmd["height"])
    nbytes = width * height * 4
    if _FRAME_BUF["ptr"] and _FRAME_BUF["nbytes"] != nbytes:
        _CUDA.free(_FRAME_BUF["ptr"])
        _FRAME_BUF["ptr"] = 0
    if not _FRAME_BUF["ptr"]:
        _FRAME_BUF["ptr"] = _CUDA.alloc(nbytes)
        _FRAME_BUF["nbytes"] = nbytes
    _FRAME_BUF["width"] = width
    _FRAME_BUF["height"] = height
    return {"handle_b64": _CUDA.ipc_handle_b64(_FRAME_BUF["ptr"]),
            "nbytes": nbytes, "width": width, "height": height,
            "format": "rgba8", "device": _CAPS["cuda_device"]}


async def _op_gpu_volume_setup(app, cmd):
    # IPC staging buffer for the in-plane (Julia copies its sim CuArray in).
    if not _CAPS["cuda_ipc"]:
        raise RuntimeError("cuda unavailable (libcuda.so.1 failed to load)")
    shape = tuple(int(x) for x in cmd["shape"])
    nbytes = shape[0] * shape[1] * shape[2] * 4  # Float32
    if _VOL_BUF["ptr"] and _VOL_BUF["nbytes"] != nbytes:
        _CUDA.free(_VOL_BUF["ptr"])
        _VOL_BUF["ptr"] = 0
    if not _VOL_BUF["ptr"]:
        _VOL_BUF["ptr"] = _CUDA.alloc(nbytes)
        _VOL_BUF["nbytes"] = nbytes
    _VOL_BUF["shape"] = shape
    return {"handle_b64": _CUDA.ipc_handle_b64(_VOL_BUF["ptr"]),
            "nbytes": nbytes, "device": _CAPS["cuda_device"]}


async def _op_gpu_write_vdb(app, cmd):
    # Device->host the staged volume, then the same pyopenvdb write as
    # write_vdb (fresh .vdb; IndeX's composite importer is file-based).
    import numpy as np
    import openvdb

    if not _VOL_BUF["ptr"]:
        raise RuntimeError("call gpu_volume_setup before gpu_write_vdb")
    shape = tuple(int(x) for x in cmd["shape"])
    if shape != _VOL_BUF["shape"]:
        raise RuntimeError(f"shape {shape} != staged {_VOL_BUF['shape']} "
                           "(re-run gpu_volume_setup)")
    nbytes = _VOL_BUF["nbytes"]
    host = (ctypes.c_byte * nbytes)()
    _CUDA.memcpy_dtoh(host, _VOL_BUF["ptr"], nbytes)
    arr = np.frombuffer(host, dtype=np.float32).reshape(shape, order="F")
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


async def _op_set_attr(app, cmd):
    # Typed attribute write on the open stage: scalars coerce to the
    # attribute's existing type; "usd_type" selects a pxr Gf construction
    # for the structured values the camera sync needs.
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
    usd_type = cmd.get("usd_type")
    if usd_type == "matrix4d":  # 4 rows of 4 numbers (row-vector layout)
        from pxr import Gf

        value = Gf.Matrix4d(*(float(x) for row in value for x in row))
    elif usd_type == "double3":
        from pxr import Gf

        value = Gf.Vec3d(*(float(x) for x in value))
    elif usd_type == "asset":  # filePath swaps (fresh-.vdb live volume updates)
        from pxr import Sdf

        value = Sdf.AssetPath(str(value))
    elif usd_type is not None:
        raise RuntimeError(f"unsupported usd_type {usd_type!r}")
    else:
        cur = attr.Get()
        if cur is not None and isinstance(cur, (bool, int, float, str)) \
                and not isinstance(value, type(cur)):
            value = type(cur)(value)  # e.g. JSON int -> float attr
    if not attr.Set(value):
        raise RuntimeError(f"Set({value!r}) failed on {attr.GetPath()}")
    for _ in range(int(cmd.get("settle", 2))):
        await app.next_update_async()
    return {"prim": cmd["prim"], "attr": cmd["attr"]}


async def _op_write_vdb(app, cmd):
    # Dense Float32 payload (column-major raw file, Julia layout) -> classic
    # OpenVDB .vdb via the omni.volume ext's pyopenvdb bindings.  Kit's IndeX
    # composite importer reads OpenVDB reliably but FAILS to fetch data from
    # NanoVDB files (proven against both NanoVDBWriter output and Warp's own
    # sample .nvdb) — so volume payloads for this server must be .vdb.
    import numpy as np
    import openvdb

    shape = tuple(int(x) for x in cmd["shape"])
    arr = np.fromfile(cmd["raw"], dtype=np.float32)
    if arr.size != shape[0] * shape[1] * shape[2]:
        raise RuntimeError(f"raw payload has {arr.size} floats, expected {shape}")
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


_HANDLERS = {
    "ping": _op_ping,
    "open_stage": _op_open_stage,
    "render": _op_render,
    "set_attr": _op_set_attr,
    "write_vdb": _op_write_vdb,
    "gpu_frame_setup": _op_gpu_frame_setup,
    "gpu_volume_setup": _op_gpu_volume_setup,
    "gpu_write_vdb": _op_gpu_write_vdb,
}


async def _main():
    app = omni.kit.app.get_app()
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    threading.Thread(target=_fifo_reader, args=(loop, queue), daemon=True).start()

    for _ in range(3):  # let the viewport/renderer come up before advertising
        await app.next_update_async()
    _probe_caps()
    settings = carb.settings.get_settings()
    _respond({
        "id": 0, "op": "ready", "ok": True, "pid": os.getpid(),
        "composite_enabled": settings.get("/rtx/index/compositeEnabled"),
        "caps": dict(_CAPS),
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

    # Shutdown: free GPU-plane resources, then post_quit + grace frames + hard
    # exit — headless teardown can stall (same fallback probe.py uses).
    _gpu_cleanup()
    app.post_quit(0)
    for _ in range(120):
        await app.next_update_async()
    print("[kit_server] post_quit stalled; hard exit", flush=True)
    os._exit(0)


asyncio.ensure_future(_main())
