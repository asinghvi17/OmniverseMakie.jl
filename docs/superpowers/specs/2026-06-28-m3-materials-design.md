# M3 — Materials (OmniPBR + textures) — Design

**Date:** 2026-06-28
**Milestone:** M3 (Materials) — the first post-M2 milestone. Reordered 2026-06-28 (option B) so the **M4 examples gallery** has full materials available.
**Status:** design approved in brainstorming; pending spec review → `writing-plans` (the bite-sized `M3_PLAN.md`).

## Goal

Give OmniverseMakie plots real **PBR materials** — **OmniPBR (MDL)** with metallic / roughness / emissive / opacity **plus image textures** — so users, and especially the M4 examples gallery, render beyond the current flat diffuse albedo. Materials are **live** (authored + edited through the M2 diff path) and **runtime-swappable**.

## Non-goals (M3)

- **MaterialX as a primary authored model.** OmniPBR is primary; MaterialX is reachable only via the `material="…"` asset-path pass-through (and is the documented spike fallback if OmniPBR doesn't render in our ovrtx build).
- **Material dedup / sharing.** One `UsdShade Material` per atomic plot (user-confirmed). Sharing is a later optimization.
- Volume materials, nonvisual/sensor materials, per-instance MeshScatter materials.

## User-facing API

`color` and `material=` are **orthogonal axes that merge into ONE OmniPBR material** per atomic plot.

- **`color`** = the **base albedo**, in whatever form Makie gives it:
  - scalar `Colorant`/`Symbol` → `inputs:diffuse_color_constant`.
  - image (`Matrix{<:Colorant}`) + the plot's texture coords → `inputs:diffuse_texture`.
  - per-vertex `Vector{<:Colorant}` (or numeric + `colormap`) → a `displayColor` primvar read in the material **(STRETCH — see Risks; fallback = constant average + `@warn`)**.
- **`material=`** (a `NamedTuple` or `Dict`; **keys = OmniPBR input names**) = **everything else**:
  - scalars: `metallic`, `roughness`, `emissive` (color3f) + `emissive_intensity`, `opacity`.
  - texture maps: `base_color_texture`, `normal_texture`, `roughness_texture`, `metallic_texture` (asset path strings).
  - **OR** `material="path/to/asset.mdl"` / a MaterialX file → **pass-through**: reference the material asset + bind it, no param mapping.

**Authoring trigger:** author an OmniPBR material **iff** `material=` is set **OR** `color` is an image. Otherwise (scalar / per-vertex color, no material) → the existing `primvars:displayColor` path (path-traced diffuse; zero material overhead).

**Precedence:** `color` provides the base **unless** `material=` explicitly names `base_color`/`base_color_texture`, in which case the explicit material value wins and we `@warn` on the conflict.

## USD authoring

- Materials live under a `/World/Looks` `def Scope`. One `def Material "Mat_<objectid(plot)>"` per materialized plot.
- The Material wraps an **OmniPBR** shader: `info:implementationSource="sourceAsset"`, `info:mdl:sourceAsset=@OmniPBR.mdl@`, `info:mdl:sourceAsset:subIdentifier="OmniPBR"`, with the mapped `inputs:*` (`diffuse_color_constant` / `diffuse_texture` / `metallic_constant` / `reflection_roughness_constant` / emissive / opacity / `*_texture`). `outputs:mdl:surface` connects the Material to the shader.
- Bind to the plot's geometry prim via the standard `material:binding` relationship (a path-array write — ovrtx `binding-materials` skill).
- **Texture assets:** OmniPBR texture inputs are USD `asset` paths. A Makie `color=img` (in-memory image) is written to a temp image file (or in-memory USD asset) referenced as `diffuse_texture`; the plot's `:texturecoordinates` author the `st` UV primvar on the mesh.

## M2 integration (live)

- A materialized plot's OmniPBR Material is authored in the `:ovrtx_renderobject` node's **BUILD** branch (alongside the geometry reference) + bound via `material:binding`.
- The **UPDATE** branch routes material-affecting compute outputs through `push_to_ovrtx!`: a changed base color (`:scaled_color`/`color`) → re-write the shader's diffuse input; changed `material=` params → re-write the changed shader inputs; a runtime material **swap** (a different material) → re-write `material:binding`.
- Non-materialized plots keep the M1/M2 `displayColor` path unchanged (byte-identical — a regression guard).

## Primitives

- **Mesh, Surface** → full OmniPBR (base + params + textures), bound on the mesh prim.
- **MeshScatter** → one OmniPBR material on the instancer (or its prototype); per-instance materials out of scope.
- **Lines, Scatter** → base color (+ emissive) via a simpler OmniPBR or the existing path; full PBR/textures not required.

## Validate-first spike (Task 1 — like the camera/lights spikes)

Before building the layer, spike: author an OmniPBR `Material` + `material:binding` on a mesh + render → confirm (a) **real PBR shading** (a metallic sphere reads metallic; varying `reflection_roughness_constant` changes the specular highlight), (b) an image `diffuse_texture` **shows** on the mesh. If OmniPBR does NOT render in our ovrtx build, fall back — MaterialX `standard_surface` (the ovrtx material-editor path) or `UsdPreviewSurface` — and record it. This de-risks the entire milestone before any layer is built.

## Testing

Subprocess render-diffs: metallic ≠ diffuse; roughness changes the specular highlight; a textured mesh shows the texture; a live `plot.material[]`/`plot.color[]` edit changes the render on the open stage; a runtime material swap rebinds; plain-`color` `displayColor` is **byte-unchanged** (no M1/M2 regression).

## Risks / open feasibility (resolved by the spike)

1. **OmniPBR renders in our ovrtx build** — the `mdl/` plugins are bundled in `bin/`, but confirm. Fallback: MaterialX / `UsdPreviewSurface`.
2. **Image-texture loading** — asset-path resolution from a referenced/temp image file in the open stage. Fallback: defer textures to a follow-up; constant/per-vertex color for those plots.
3. **Per-vertex base color in a material** — needs an MDL primvar reader in the material graph. Fallback: constant-average + `@warn`.
4. **`:texturecoordinates` compute output** exists for textured meshes and maps to the OmniPBR `st` UV input.
