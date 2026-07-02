# Milestone M3 — Materials (OmniPBR + textures) — bite-sized plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes. **Design spec:** `docs/superpowers/specs/2026-06-28-m3-materials-design.md`. **Grounded in:** the proven OmniPBR USDA at `references/ovrtx/examples/c/sensors/radar/radar_example.usda` (`def Material` + OmniPBR `Shader` + `rel material:binding`) and the bundled `OmniPBR.mdl` (`references/ovrtx/examples/python/minimal/.venv/lib/python3.13/site-packages/ovrtx/bin/library/mdl/Base/OmniPBR.mdl`); the live shader-input write is demonstrated in `references/ovrtx/examples/c/material-editor/render-test/main.py` (writes `inputs:diffuse_color_constant` + `material:binding`).

**Goal:** Give plots real **PBR materials** — OmniPBR (MDL) with metallic / roughness / emissive / opacity **+ image textures** — authored and edited **live** through the M2 diff path, runtime-swappable; plain `color` stays on cheap `displayColor`.

**Architecture:** A *materialized* plot (one whose `material=` is set **or** whose `color` is an image) gets one `UsdShade Material "Mat_<objectid(plot)>"` under a `/World/Looks` `Scope`, wrapping an OmniPBR `Shader` (`info:mdl:sourceAsset=@OmniPBR.mdl@`), bound to the plot's geometry prim via the standard `material:binding` relationship. `color`→base albedo, `material=`→the rest; they compose into one material. The M2 `:ovrtx_renderobject` node authors the material in its **build** branch and re-writes changed shader inputs / re-binds in its **update** branch. Non-materialized plots keep the M1/M2 `displayColor` path byte-unchanged.

**Tech Stack:** ovrtx OmniPBR (MDL, bundled), `UsdShade` Material/Shader, `material:binding`, the OmniverseMakie `OV` layer + the M2 open-stage diff. No new Julia deps.

## Global Constraints (M3)

Inherit **all** M0/M1/M2 constraints (Pkg-managed pinned deps, **NO new deps**; generated bindings `lib/LibOVRTX/src/libovrtx_api.jl` verbatim; `GC.@preserve` on every FFI path; carb `SignalGuard` intact; subprocess-isolated renderer tests + `timedwait` watchdog; `colorbuffer` returns `Matrix{RGBA{N0f8}}` right-side-up **NO flip**; open-stage M2 model — author once, live-diff, never re-author the root for an edit; reference/scope layers OMIT `upAxis`). Plus M3-specific:

- **OmniPBR is THE material model.** Author a `UsdShade Material` + OmniPBR `Shader` (`uniform token info:implementationSource = "sourceAsset"`, `uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@`, `uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"`); bind via `material:binding`. The bare `@OmniPBR.mdl@` resolves through ovrtx's MDL search path (the asset is bundled). MaterialX / `UsdPreviewSurface` is the **spike fallback only** (M3.1).
- **`color` ⟂ `material=` compose into ONE material** per atomic plot. **Author-a-material trigger:** `material=` is set OR `color` is an image (`Matrix{<:Colorant}`). Otherwise → the existing `displayColor` (byte-unchanged — a regression guard).
- **Escape-hatch keys = OmniPBR input names** via a thin documented map: `metallic`→`metallic_constant`, `roughness`→`reflection_roughness_constant`, `emissive`→`emissive_color`(+`enable_emission`/`emissive_intensity`), `opacity`→`opacity_constant`(+`enable_opacity`), `base_color`→`diffuse_color_constant`, `*_texture`→the matching texture input (`diffuse_texture`, `normalmap_texture`, `reflectionroughness_texture`, `metallic_texture`). An unknown key `@warn`s and is skipped.
- **One Material per atomic plot** under `/World/Looks` (no dedup — user-confirmed).
- **Live edits + runtime swap via the M2 diff path** — re-write shader inputs / re-write `material:binding`; NEVER re-author the root.

## File structure (M3 adds / modifies)

```
src/
  translation/materials.jl   # MODIFY: usda_omnipbr_material; material_inputs_from(plot); is_materialized(plot); _omnipbr_key_map; keep displaycolor_for
  translation/usd.jl (or composition.jl)  # MODIFY: emit a "/World/Looks" def Scope in the root (like the M2.3 scene scopes)
  translation/meshes.jl, primitives.jl    # MODIFY: a materialized plot authors+binds a material instead of emitting displayColor
  binding/OV.jl              # MODIFY: bind_material!(r, geom_prim, material_prim); write_shader_input!(r, shader_prim, name, value::Union{Float32,NTuple{3,Float32},String})
  compute.jl                 # MODIFY: the :ovrtx_renderobject node build branch authors+binds the material; push_to_ovrtx! routes base-color/material changes → write_shader_input!; a material swap → bind_material!
test/
  m3_material_test.jl   m3_texture_test.jl   m3_material_live_test.jl
```

---

## Task M3.1 — OmniPBR material authoring + binding (★ validate-first)

**Files:** `src/translation/materials.jl`, `src/translation/usd.jl`, `src/binding/OV.jl`. Test: `test/m3_material_test.jl`.

**Interfaces — Produces:**
```julia
# the OmniPBR Material+Shader USDA for one material, named Mat_<id>, under a Looks scope:
usda_omnipbr_material(name::AbstractString, inputs::AbstractDict{String,Any}) -> String
# author the "/World/Looks" Scope into the root template (returns the USDA fragment):
looks_scope_usda() -> String                       # `def Scope "Looks" {}` inside /World
material_prim_path(plot) -> String                 # "/World/Looks/Mat_<objectid(plot)>"
OV.bind_material!(r::Renderer, geom_prim::AbstractString, material_prim::AbstractString)  # writes material:binding
```

- **Key code** (the Material USDA mirrors `radar_example.usda:137-156`; `inputs` is a `Dict` of `omnipbr_input_name => value`, value = `Float32` / `NTuple{3,Float32}` / asset-path `String`):
  ```julia
  function usda_omnipbr_material(name, inputs)
      lines = String[]
      for (k, v) in inputs
          if v isa NTuple{3}                     # color3f
              push!(lines, "                color3f inputs:$k = ($(v[1]), $(v[2]), $(v[3]))")
          elseif v isa AbstractString            # asset (texture)
              push!(lines, "                asset inputs:$k = @$(v)@")
          else                                   # float
              push!(lines, "                float inputs:$k = $(Float32(v))")
          end
      end
      return """
          def Material "$(name)"
          {
              token outputs:mdl:surface.connect = </World/Looks/$(name)/Shader.outputs:out>
              def Shader "Shader"
              {
                  uniform token info:implementationSource = "sourceAsset"
                  uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@
                  uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"
  $(join(lines, "\n"))
                  token outputs:out
              }
          }
  """
  end
  ```
  `OV.bind_material!` writes the `material:binding` **relationship** (a path array) — mirror M2.5's existing path/relationship write if one exists, else add it via `_write_attribute!` with `OVRTX_SEMANTIC_PATH_STRING` (the ovrtx `binding-materials` skill: `material:binding`, one path per prim). `GC.@preserve` the path strings.
- ⚠️ **VALIDATE FIRST (this is the load-bearing feasibility for the whole milestone):** Step 1 authors a metallic OmniPBR material on a mesh, binds it, renders, and asserts it reads **metallic** (a bright specular response distinct from a flat diffuse mesh). The `OmniPBR.mdl` asset IS bundled + the radar example proves the USDA, so this should pass — but if OmniPBR does NOT render (black/unshaded), STOP and report; fall back to MaterialX `standard_surface` or `UsdPreviewSurface` (re-validate, record the choice). Do not build M3.2+ on an unproven material path.
- [ ] **Step 1 (failing test, subprocess):** `test/m3_material_test.jl` — author a stage with a unit sphere (high-tessellation `mesh!`) + a `/World/Looks` scope + `usda_omnipbr_material("Mat_test", Dict("diffuse_color_constant"=>(0.6f0,0.6f0,0.62f0), "metallic_constant"=>1.0f0, "reflection_roughness_constant"=>0.15f0))`; `OV.bind_material!` it to the sphere; render. Assert (a) non-black, (b) the render differs **substantially** from the same sphere with a plain `displayColor=(0.6,0.6,0.62)` diffuse (a metallic surface has a concentrated bright specular highlight → e.g. the brightest-pixel fraction / max luminance is markedly higher, or CHANGED ≥ a threshold vs the diffuse render). RED (`usda_omnipbr_material`/`bind_material!` undefined).
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** implement `usda_omnipbr_material`, `looks_scope_usda` (+ wire the `/World/Looks` scope into `author_root_from_scene!`'s `/World` body, alongside the M2.3 scene scopes), `material_prim_path`, `OV.bind_material!`.
- [ ] **Step 4:** run → PASS (metallic reads metallic). If FAIL with OmniPBR unshaded → execute the fallback + record it in the report.
- [ ] **Step 5:** commit `feat(M3.1): OmniPBR material authoring + material:binding`.

**Acceptance:** an OmniPBR material authored under `/World/Looks` + bound via `material:binding` renders real PBR shading (metallic distinct from diffuse). The material model is PROVEN (or the fallback is recorded).

---

## Task M3.2 — `material=` escape hatch + `color`→base composition + authoring trigger

**Files:** `src/translation/materials.jl`, `src/translation/meshes.jl`. Test: `test/m3_material_test.jl` (extend).

**Interfaces — Produces:**
```julia
is_materialized(plot) -> Bool                       # plot.material[] !== nothing  OR  plot.color[] isa AbstractMatrix
material_inputs_from(plot) -> Dict{String,Any}      # color→base + material= params, mapped to OmniPBR input names, precedence applied
const _OMNIPBR_KEY_MAP = Dict(:metallic=>"metallic_constant", :roughness=>"reflection_roughness_constant",
                              :opacity=>"opacity_constant", :base_color=>"diffuse_color_constant", …)
```

- **Mapping rules** (per the spec): start the dict from `color` — scalar `Colorant`→`"diffuse_color_constant"=>_rgb(color)`; image→deferred to M3.3 (just mark it for the texture path). Then merge `material=` (a `NamedTuple`/`Dict`): each key via `_OMNIPBR_KEY_MAP` (`metallic`→`metallic_constant`, `roughness`→`reflection_roughness_constant`, `emissive`→`emissive_color` + `enable_emission=1`, `opacity`→`opacity_constant` + `enable_opacity=1`); unknown key → `@warn` + skip. **Precedence:** if `material=` names `base_color`/`base_color_texture`, it overrides `color` (`@warn` on conflict). Per-vertex `color` + `material=` → constant-average base + `@warn` (the spec's stretch fallback).
- **Build-path wiring** (`meshes.jl` / the M2 build): if `is_materialized(plot)` → author `usda_omnipbr_material(material_name, material_inputs_from(plot))` + `bind_material!`, and emit the geom prim WITHOUT `displayColor`; else → the existing `displaycolor_for`/`displayColor` path, unchanged.
- [ ] **Step 1 (failing test):** subprocess — (a) `mesh!(…; color=:red, material=(; metallic=0.0, roughness=0.2))` → assert `material_inputs_from` yields `Dict("diffuse_color_constant"=>(1,0,0), "metallic_constant"=>0, "reflection_roughness_constant"=>0.2)` (unit test, no render) AND renders red+glossy; (b) `mesh!(…; color=:red)` (no material) → `is_materialized` is `false` and the render is **byte-identical** to the M2 `displayColor` render (regression guard). RED.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** implement `is_materialized`, `_OMNIPBR_KEY_MAP`, `material_inputs_from` (color+material merge + precedence + the per-vertex fallback), and the build-path branch.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** commit `feat(M3.2): material= escape hatch + color/material composition + authoring trigger`.

**Acceptance:** `material=` params + `color` base compose into one OmniPBR material; the authoring trigger fires only when needed; plain-color plots are byte-unchanged.

---

## Task M3.3 — Image textures (`color=img` + `*_texture`)

**Files:** `src/translation/materials.jl`. Test: `test/m3_texture_test.jl`.

**Interfaces — Produces:**
```julia
_texture_asset_for(img_or_path, plot) -> String     # an on-disk asset path: a passed path as-is, or an in-memory Matrix written to a temp PNG
# material_inputs_from gains: color::AbstractMatrix → "diffuse_texture"=>_texture_asset_for(img); material=(; *_texture=path) → that texture input
```

- **Approach:** `color = img` (a `Matrix{<:Colorant}`) → write `img` to a temp PNG (FileIO/ImageIO is already in the test/runtime image path) under a session temp dir, reference it as `asset inputs:diffuse_texture = @<path>@`, and ensure the mesh authors `st` texcoords from `plot[:texturecoordinates]` (the compute output) as a `texCoord2f[] primvar:st`. `material=(; base_color_texture="a.png", normal_texture="n.png", roughness_texture="r.png")` → the matching texture inputs (paths used as-is). An OmniPBR texture input requires the `st` primvar to exist on the mesh.
- ⚠️ **Validate (texture loads):** Step 1 renders a mesh textured with a 2-colour checker image and asserts BOTH colours appear in the render (the texture is sampled, not a flat average). If the texture does not load (flat/black), record it + fall back (constant base) and scope textures to a follow-up.
- [ ] **Step 1 (failing test, subprocess):** `test/m3_texture_test.jl` — a quad `mesh!` with `color = checker_img` (a 2×2 red/blue checker, scaled) + texcoords → assert the render contains BOTH a red-dominant and a blue-dominant region (texture sampled across the surface), not a single averaged colour. RED.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** implement `_texture_asset_for` (temp-PNG write + path), the `diffuse_texture` + `st`-primvar authoring, and the `*_texture` escape-hatch inputs.
- [ ] **Step 4:** run → PASS (both checker colours present).
- [ ] **Step 5:** commit `feat(M3.3): image textures — color=img diffuse_texture + *_texture maps + st primvar`.

**Acceptance:** an image-textured mesh shows the texture; explicit `*_texture` maps author the right inputs.

---

## Task M3.4 — Live material edits + runtime swap (M2 diff integration)

**Files:** `src/compute.jl`, `src/binding/OV.jl`. Test: `test/m3_material_live_test.jl`.

**Interfaces — Produces:**
```julia
OV.write_shader_input!(r::Renderer, shader_prim::AbstractString, name::AbstractString,
                       value::Union{Float32,NTuple{3,Float32}})    # live re-write of inputs:<name> (float / color3f)
# compute.jl: the :ovrtx_renderobject node, for a materialized plot:
#   build branch  → author material + bind_material! (M3.1/3.2)
#   update branch → a changed base color / material param → write_shader_input! on Mat_<id>/Shader;
#                   a material *swap* (plot.material[] becomes a different material object) → bind_material!
```

- **Key code:** the shader prim is `material_prim_path(plot) * "/Shader"`. `write_shader_input!` mirrors M0 `_write_attribute!` (a scalar float `kDLFloat/32/1`, or a `color3f` written as 3-lane / `OVRTX_SEMANTIC_NONE`) on the shader prim — the material-editor render-test proves `inputs:diffuse_color_constant` is writable live. In `push_to_ovrtx!`: `:scaled_color` (or the base color) on a materialized plot → `write_shader_input!(…, "diffuse_color_constant", rgb)`; a changed `:material` param → `write_shader_input!` per changed input; any material write ⇒ `OV.reset!`. A `material:binding` swap (different material identity) → `bind_material!`.
- [ ] **Step 1 (failing test, subprocess):** `m3_material_live_test.jl` — insert a materialized mesh (`material=(; metallic=1, roughness=0.1)`); render A; set `plot.color[] = :blue` → render B → assert the base colour changed (blue dominant) on the OPEN stage (`ROOT_OPENS==1`, no re-author); set `plot.material[] = (; metallic=0.0, roughness=0.9)` → render C → assert the glossy→matte change. Instrument that exactly the changed inputs were written. RED.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** implement `OV.write_shader_input!` and the `push_to_ovrtx!` material routing (+ the swap → `bind_material!`).
- [ ] **Step 4:** run → PASS (live colour + param edits change the render; `ROOT_OPENS==1`).
- [ ] **Step 5:** commit `feat(M3.4): live material edits + runtime swap via the M2 diff path`.

**Acceptance:** a live `plot.color[]`/`plot.material[]` edit re-writes shader inputs on the open stage (no re-author); a material swap re-binds.

---

## Task M3.5 — Primitive coverage (MeshScatter, Surface, Lines/Scatter)

**Files:** `src/translation/primitives.jl`, `src/translation/materials.jl`. Test: `test/m3_material_test.jl` (extend).

- **Approach:** route the M3.2 materialize-or-displayColor branch through the other primitives' builders. **MeshScatter** → ONE OmniPBR material bound to the instancer prim (`/World/plot_<id>`), not per-instance (out of scope). **Surface** → identical to `Mesh` (it is a `UsdGeomMesh`). **Lines/Scatter** → support `material=(; emissive, opacity, base_color)` mapped onto the curve/instancer (base color + emissive), no textures/normalmaps required. Reuse `material_inputs_from` for all; only the bind-target prim differs per type.
- [ ] **Step 1 (failing test, subprocess):** a `meshscatter!` with `material=(; metallic=1, roughness=0.2)` → instances render metallic; a `surface!` with a `material=` → renders with the material; a `lines!` with `material=(; emissive=(1,0,0))` → emissive red. RED.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** thread the materialize branch + `material_inputs_from` through `primitives.jl` (meshscatter/surface/lines/scatter), binding to the correct prim per type.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** commit `feat(M3.5): materials on MeshScatter/Surface/Lines/Scatter`.

**Acceptance:** materials apply across the primitive types (instancer-level for scatter); each renders correctly.

---

**M3 GATE:** OmniPBR materials (base + metallic/roughness/emissive/opacity + image textures) author under `/World/Looks` + bind via `material:binding`; `color`/`material=` compose; **live edits of the pre-authored material** (color + `material=` params) go through the M2 diff path (no re-author, `ROOT_OPENS==1`); plain-color plots are byte-unchanged; materials apply across Mesh/Surface/MeshScatter/Lines/Scatter. ✅ → **M4 (Examples gallery)** — which consumes these materials.

> **GATE reconciliation (M3.1 finding):** the original "runtime swap" wording is delivered as **live edits of the pre-authored material**, not arbitrary re-binding to a brand-new material. Because a `UsdShade Material` must be PRE-AUTHORED at open-time (a material added to the open stage is not bindable) AND the open-stage model forbids re-authoring the root for an edit, a true swap to a not-pre-authored material is infeasible without a root re-author — carried to a later milestone. Live color + material-param edits (the valuable subset) ARE delivered and tested.
> **Primitive deviations (accepted, see ledger):** (a) materialized MeshScatter/Scatter render as a **merged `UsdGeomMesh`** (ovrtx does not honor materials on a `UsdGeomPointInstancer`; non-materialized scatter stays on the instancer, byte-unchanged) — loses instancing for large-N materialized scatter (carry); (b) `material=` on Lines/Scatter is enabled via a global `Makie.attribute_name_allowlist()` append at `__init__`.

---

## Open assumptions this plan validates early (with fallbacks)
1. **OmniPBR renders in our ovrtx build** (M3.1 Step 1). The `OmniPBR.mdl` asset is bundled and `radar_example.usda` proves the USDA, so this is low-risk — but validate-first. Fallback: MaterialX `standard_surface` / `UsdPreviewSurface`.
2. **Image textures load from a referenced asset path** on the open stage (M3.3). Fallback: defer textures; constant/per-vertex base.
3. **Shader inputs are writable live** (`inputs:diffuse_color_constant` etc.) for the M2 diff path (M3.4) — the material-editor render-test demonstrates it; re-confirm through our `write_shader_input!`.
4. **`:texturecoordinates` resolves** for textured meshes and maps to the `st` UV primvar (M3.3). Fallback: require explicit texcoords / skip the texture.
