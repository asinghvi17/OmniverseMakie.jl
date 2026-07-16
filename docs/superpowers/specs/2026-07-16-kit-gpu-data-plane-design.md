# Kit GPU data plane — frames out, live volumes in (design)

2026-07-16. Short-form spec (project practice). Implements "Phase 3" of
`2026-07-15-omniverse-kit-makie-design.md` for the subprocess Kit transport.
Approved in brainstorm: layered out-plane (CUDA with CPU fallback) AND the
in-plane (live volume updates from GPU sims).

## Why

The Kit render server currently returns frames as PNG files on disk
(encode + disk + decode) and ingests volume payloads through a raw-file hop.
Two consumers want better:

- **Frames out:** low-convergence live loops and CUDA-resident downstream
  processing (the water/turbine sim work) want the frame in a `CuArray`
  with zero host copies — or at minimum without PNG/disk.
- **Volumes in:** GPU sims produce density fields as `CuArray`s; today a
  live update would require a Julia-side device→host copy plus a raw file.
  The in-plane's consumer is a **live-update side-channel**
  (`gpu_update_volume!`), symmetric with the standalone backend's
  `gpu_update_mesh!` — NOT stage authoring (Makie's plot pipeline
  materializes CPU arrays anyway, and IndeX's importer is file-based:
  a fresh `.vdb` per update is unavoidable and stays).

## Decisions (from the brainstorm)

1. **Server owns all CUDA-IPC buffers.** Kit-side Python allocates with raw
   `cudaMalloc` via ctypes (no cupy dependency; warp is available but not
   required for allocation) and exports `cudaIpcGetMemHandle` as base64 over
   the existing line-JSON RPC. Julia opens handles once
   (`cuIpcOpenMemHandle` via CUDA.jl's driver API) and `unsafe_wrap`s a
   `CuArray`. Rationale: CUDA.jl's default stream-ordered pool allocations
   are not legacy-IPC-exportable; server-side raw allocation is unambiguous.
   **Same-GPU is a hard requirement** — both sides assert the device and
   error clearly.
2. **Out-plane is layered:**
   - `:cuda` — `omni.syntheticdata`'s LdrColor output on device, device→device
     copy into the shared IPC buffer. The extension is NOT in the local
     caches and must be fetched once from NVIDIA's extension registry
     (network); this is **probe-gated**: if the fetch fails, the layer ships
     dark and everything below carries.
   - `:cpu` (shm) — `omni.kit.viewport.utility.capture_viewport_to_buffer`
     (cached ext, CPU bytes callback) written into POSIX shared memory
     (`/dev/shm/omk_<pid>_frame`), mmapped zero-copy by Julia. Kills
     PNG encode + disk + decode with no new extensions.
   - `:png` — the existing path, kept as the compatibility floor.
   The server probes its capabilities at startup and advertises them in the
   `ready` message (`caps: {cuda_out, shm_out, cuda_device}`); Julia's
   `render!(screen; device = :auto | :cuda | :cpu | :png)` selects, `:auto`
   preferring the best available **CPU-returning** path for the plain API
   (`Makie.colorbuffer` keeps its CPU-image contract but transparently rides
   shm when available — an immediate win for existing users).
3. **In-plane = live volume side-channel.**
   `gpu_update_volume!(screen, plot; data::CuArray)` (and a prim-path
   variant). Consistent with (1), the *server* allocates the staging buffer
   at first use (`gpu_volume_setup` with shape/dtype → handle); Julia wraps
   it as a `CuArray` and `copyto!`s its sim array into it (device→device,
   no host copy in Julia), then `gpu_write_vdb` has the server copy
   device→host into numpy
   (unavoidable: pyopenvdb builds grids on CPU), write a **fresh** `.vdb`
   (proven fresh-file pattern), and swap the volume prim's `filePath`.
   Voxel-size/origin semantics identical to `_vdb_volume_writer`
   (node-centered, extent/(N−1)).
4. **Packaging:** core OmniverseKitMakie gains the shm path only (no CUDA
   dep). A new package extension **`OmniverseKitMakieCUDAExt`** (weakdep:
   CUDA) carries `CuArray` returns and `gpu_update_volume!` — mirroring the
   main package's extension pattern. `KitScreen` tracks volume prim paths
   from `stage_usda` authoring so plots map to prims.

## Protocol additions (kit_server.py)

- `ready` gains `caps` (probed at startup: ctypes cudart loadable + device
  present → `cuda_ipc`; syntheticdata importable → `cuda_out`; always
  `shm_out`).
- `gpu_frame_setup {width, height}` → `{handle_b64, nbytes, format:"rgba8",
  device}` (idempotent per resolution; frees + reallocs on change).
- `render {frames, device:"cuda"|"shm"|"png", out?}` — after convergence:
  `cuda` → annotator ptr D2D→IPC buffer, respond `{frame, width, height}`;
  `shm` → capture_viewport_to_buffer → write shm, respond
  `{shm_path, nbytes, width, height, format}`; `png` unchanged.
- `gpu_volume_setup {shape, dtype:"f32"}` → `{handle_b64, nbytes}`.
- `gpu_write_vdb {shape, voxel_size, origin, out, name}` — reads the shared
  volume buffer (D2H → numpy, column-major reshape), writes fresh `.vdb`.
- `set_attr` (existing) performs the `filePath` swap; orchestrated from
  Julia inside `gpu_update_volume!`.
- Buffer lifecycle: single-buffered; contents are valid until the next
  `render`/`gpu_write_vdb` call — documented, not enforced. All CUDA
  allocations freed and shm unlinked on `quit`.

## Julia surface

- Core: `render!(screen; device=:auto, ...)` (returns `Matrix` image for
  `:cpu`/`:png`); `Makie.colorbuffer(screen)` unchanged in contract, shm
  under the hood; capability query `gpu_caps(screen)`.
- Ext (CUDA loaded): `render!(screen; device=:cuda) -> CuArray` view of the
  IPC buffer (documented: valid until next render; copy to keep);
  `gpu_update_volume!(screen, plot_or_prim; data::CuArray, colorrange=...)`.
- Orientation/format parity: the shm and cuda paths must return images that
  match the PNG path pixel-for-pixel (modulo alpha) — enforced by test, not
  by convention.

## Testing

- Pure: b64 handle round-trip, caps plumbing, op-payload encode/decode,
  device-selection logic (fake transport).
- GPU tier (env-gated, serialized on the shared lock, hard-kill timeouts):
  1. shm frame == png frame (byte parity after format normalization) +
     timing printed.
  2. `:cuda` (skips loudly if syntheticdata unfetchable): CuArray returned,
     frame ≈ png within format tolerance, chroma oracle still passes.
  3. in-plane: author a volume scene, `gpu_update_volume!` with a shifted
     blob from a `CuArray` → pixels move; result matches a CPU-authored twin
     of the same data (fresh-file path equivalence).
- Full-suite regression: existing subprocess A/B baselines unchanged.

## Risks / probes (in build order)

1. **syntheticdata registry fetch** (`kit --enable omni.syntheticdata` on
   this box; needs network + a 109-compatible version). Fail → `:cuda` out
   ships dark; everything else proceeds.
2. **capture_viewport_to_buffer payload format** (RGBA8 expected; callback
   delivers a capsule/pointer + size — probe defines the shm writer).
3. **Annotator output format/pitch** (RGBA8 vs float4; row pitch) — the D2D
   copy must respect pitch; parity test is the oracle.
4. CUDA-IPC legacy handles require same GPU + non-MIG; probed at setup, not
   assumed.

## Build order

1. Spec (this doc) + probes 1–2.
2. shm out-plane: server op + core Julia path + colorbuffer rewire + parity
   test.
3. CUDA out-plane (if probe 1 green): gpu_frame_setup + render device=cuda +
   `OmniverseKitMakieCUDAExt` CuArray return + tolerance test.
4. In-plane: gpu_volume_setup/gpu_write_vdb + `gpu_update_volume!` + moved-
   pixels/twin test; volume prim-path tracking in KitScreen.
5. Full suite + README/spec status updates + push.
