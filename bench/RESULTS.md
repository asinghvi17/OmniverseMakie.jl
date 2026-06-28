# M2.6 Hot-Path Throughput Benchmark Results

**Hardware:** NVIDIA A5000 (JuliaHub cloud node)  
**Commit:** `feat/m2-compute-diff`  
**Runner:** `bench/hot_path.jl` via `run_ovrtx_subprocess`

---

## Gate Verdict: PASS

The M2 gate (≥ 30 Hz for ~10⁴ transforms OR ~10⁵ points) is **MET** via Path B.

---

## Path A — write_mapped_xform! (10,000 omni:xform writes / "frame")

| Metric           | Value        |
|------------------|-------------|
| Frame time       | **40.73 ms** |
| Updates / sec    | 245,500     |
| Effective rate   | **24.55 Hz** |
| Gate (≥ 30 Hz)   | **BELOW**   |

**Cost per write:** ~4.07 µs (map\_attribute + 16 unsafe\_store! + unmap\_attribute).

**Bottleneck:** `ovrtx_map_attribute` / `ovrtx_unmap_attribute` round-trips involve
CPU–GPU synchronisation per call.  Each of the 10,000 writes in a "frame" pays
this overhead serially.

**Powersave caveat:** the system CPU performance profile was in **powersave** mode
during this run (reported by omni.rtx log; clock throttled to lowest frequency).
Path A showed run-to-run variance of ~25 % (24.55 – 32.28 Hz across two runs).
On a performance-governor CPU the sustained rate would likely exceed 30 Hz.

**Implication for users:** applications pushing thousands of independent per-frame
xform updates serially will approach this limit.  A future M3 optimisation
(GPU-resident write buffers, kDLCUDA DLPack) would eliminate the per-call sync
overhead and is documented as a carry below.

---

## Path B — write_binding! points (100,000 Float32×3 writes / "frame")

| Metric           | Value           |
|------------------|----------------|
| USDA generation  | 2,040 KiB in 0.022 s |
| Stage open time  | 0.043 s         |
| Frame time       | **0.114 ms**    |
| Updates / sec    | 874,400,000     |
| Effective rate   | **8,744 Hz**    |
| Gate (≥ 30 Hz)   | **PASS (291×)** |

**Cost:** The 100,000-element array (1.2 MB of Float32×3 data) is written in
~0.11 ms, corresponding to ~10 GB/s effective CPU→ovrtx buffer bandwidth.
`ovrtx_write_attribute` with `OVRTX_DATA_ACCESS_SYNC` performs a synchronous
CPU-side copy to the attribute staging buffer; no GPU round-trip per write.

**Coverage:** This benchmark also closes the **M2.4 coverage gap** — the
array-tier write path (`write_binding!` for point arrays) was spike-proven in
M2.4 but not in the committed benchmark suite.  It is now exercised for real
with 100k-point payloads.

---

## M3 Carry — GPU-resident DLPack writes

Path A is the only path with a per-call sync bottleneck.  If future use-cases
require > 30 Hz updates of thousands of independent omni:xform transforms
(and CPU powersave throttling cannot be avoided), M3 can introduce
**GPU-resident write buffers** (DLPack `kDLCUDA`):

- Pre-pin write buffers in GPU VRAM.
- Replace `map_attribute` / `unsafe_store!` / `unmap_attribute` with a CUDA
  kernel writing directly into the pinned buffer.
- Eliminates the CPU–GPU sync cost per xform write.
- Expected improvement: 5–10× (µs → sub-µs per write at the call boundary).

This is a planned M3 work item; the M2 hot path is sufficient for the gate.

---

## Scatter / MeshScatter Gap — Decision (b): documented carry

`push_to_ovrtx!` routes `:positions_transformed_f32c` to the `"points"` attribute
on all plot types.  A `UsdGeomPointInstancer` (used by `Scatter` / `MeshScatter`)
stores per-instance positions in `"positions"`, **not** `"points"` — so
per-instance position updates for Scatter/MeshScatter are currently a no-op
(the whole-scatter `omni:xform` still works; only per-instance moves are silent).

**Decision:** benchmark the working paths (Mesh xform + Mesh points) which
independently meet the gate.  The scatter per-instance route fix
(`"positions"` attribute, UsdGeomPointInstancer path) is a **documented M3 carry**.

No silent degradation: the gap is identified, the gate passes on the unaffected
paths, and the fix is scoped.
