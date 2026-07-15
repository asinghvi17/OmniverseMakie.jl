# Headless IndeX-composite probe: open each PROBE_JOBS stage, let the RTX +
# IndeX composite render converge, capture the viewport to PNG, then quit.
# PROBE_JOBS = "stage1.usda>out1.png;stage2.usda>out2.png"
import asyncio
import os
import traceback

import carb
import omni.kit.app
import omni.usd

JOBS = [j.split(">") for j in os.environ["PROBE_JOBS"].split(";") if j]
WARM = int(os.environ.get("PROBE_WARM_FRAMES", "240"))


async def capture_one(stage_path, out_path):
    app = omni.kit.app.get_app()
    ctx = omni.usd.get_context()
    print(f"[probe] opening {stage_path}", flush=True)
    ok = ctx.open_stage(stage_path)
    print(f"[probe] open_stage -> {ok}", flush=True)
    for _ in range(WARM):
        await app.next_update_async()
    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
    vp = get_active_viewport()
    print(f"[probe] viewport camera={vp.camera_path} res={vp.resolution} "
          f"frame_info={dict(vp.frame_info)}", flush=True)
    cap = capture_viewport_to_file(vp, out_path)
    try:
        res = await cap.wait_for_result(completion_frames=60)
        print(f"[probe] wait_for_result -> {res}", flush=True)
    except Exception:
        traceback.print_exc()
    for _ in range(180):
        await app.next_update_async()
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            break
    size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    print(f"[probe] captured {out_path} bytes={size}", flush=True)
    return size > 0


async def main():
    app = omni.kit.app.get_app()
    try:
        settings = carb.settings.get_settings()
        print(f"[probe] /rtx/index/compositeEnabled = {settings.get('/rtx/index/compositeEnabled')}", flush=True)
        mgr = app.get_extension_manager()
        for ext in ("omni.index", "omni.index.renderer", "omni.index.usd", "omni.rtx.index_composite"):
            print(f"[probe] ext {ext} enabled = {mgr.is_extension_enabled(ext)}", flush=True)
        ok = True
        for stage_path, out_path in JOBS:
            ok = await capture_one(stage_path, out_path) and ok
        print(f"[probe] {'PROBE_OK' if ok else 'PROBE_FAIL'}", flush=True)
    except Exception:
        traceback.print_exc()
        print("[probe] PROBE_FAIL", flush=True)
    finally:
        # post_quit alone can leave a full editor app lingering headless
        # (UI teardown stalls); give it a grace window, then exit hard.
        app.post_quit(0)
        for _ in range(120):
            await app.next_update_async()
        print("[probe] post_quit stalled; hard exit", flush=True)
        os._exit(0)


asyncio.ensure_future(main())
