# Water-cube examples — design

**Date:** 2026-07-13
**Status:** approved (short-form spec by user request; no separate plan doc)

## Goal

Showcase examples that render a convincing animated **cube of water** with the
RTX backend, in the spirit of the Trixi-in-Omniverse demo. Precursor to a
turbine-in-water-cube demo (turbine visual-only, not coupled to the sim).
Phased:

1. **Phase 1 — `examples/water_cube_gerstner.jl`**: procedural sum-of-sines
   waves drive the cube's top surface. Verifies the whole pipeline (geometry,
   OmniGlass water, lighting, recording). Gate: user approves the look.
2. **Phase 2 — `examples/water_cube_dambreak.jl`**: dam-break shallow-water
   equations (h, hu, hv; local Lax–Friedrichs finite volume; reflective
   walls; CFL-bounded dt; NaN guard aborts loudly) drive the same surface.

Out of scope (separate later designs): the turbine scene; GPU-direct plot
updates from CUDA (`ovrtx_write_attribute` documents kDLCUDA input tensors —
viable, deferred by user decision).

## Approach

No library changes. Shared harness `examples/common/water_cube.jl`:

- **Geometry** — watertight box mesh: N×N animated top grid, N×N flat
  bottom, side walls stitched between boundary rings. Constant vertex count
  (the live mesh push gates on frozen npoints). Per frame the driver rewrites
  top-vertex heights + vertex normals (analytic for sines, finite-difference
  for SWE) and assigns the mesh Observable; positions and normals flow
  through the existing binding path (`compute.jl` `:normals` push). N ≈ 128.
- **Material/scene** — cube: `material = (; glass = true, ior = 1.33)`,
  pale-cyan color. Checkerboard OmniPBR floor + 2–3 colored spheres under
  the water as refraction references. Gradient-sky dome via
  `push_environment_image!`, one DirectionalLight, `background = :domelight`,
  slow camera orbit.
- **Recording** — `accumulate_across_frames = true, warmup = 4` + preroll;
  `Makie.record` → `examples/renders/water_cube_<driver>.mp4`, ~8 s @ 24 fps.
  Zeus-style standalone scripts with output asserts.

## Verification

- Script asserts: mp4 exists; non-black lit-pixel count on a sampled frame.
- Pixel-oracle check: floor pattern visibly distorted through the water vs a
  control render without the cube (refraction real, not a gray box).
- Human look-check of extracted frames before phase 2 starts.
