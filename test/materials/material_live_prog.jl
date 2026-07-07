# Subprocess body for the LIVE material-edit test (read + run by
# test/materials/material_live_test.jl via run_ovrtx_subprocess).  Standalone
# .jl so the Makie scene setup needs no escaping.
#
# End to end THROUGH the real Screen / colorbuffer pipeline: a live
# `plot.color[]` / `plot.material[]` PARAM edit on a MATERIALIZED mesh
# re-writes the PRE-AUTHORED OmniPBR shader inputs on the OPEN stage (the
# live diff path) — NO re-author (`ROOT_OPENS == 1`), the render changes, and
# EXACTLY the changed shader inputs are written (instrumented via
# `_PUSH_OBSERVER` + `_SHADER_WRITE_OBSERVER`).
#
#   render A  — materialized mesh (metallic=1, roughness=0.1), red base →
#               glossy red
#   color[]=:blue  → render B → exactly one :scaled_color push → one
#                    diffuse_color_constant shader write → BLUE-dominant
#   material[]=Attributes(metallic=0, roughness=0.9) → render C → exactly one
#                    :material push → metallic_constant +
#                    reflection_roughness_constant shader writes → the glossy
#                    specular collapses to a MATTE surface (contrast drops)
#
# NOTE: the graph stores `:material` as `Makie.Attributes`, so a live edit
# MUST pass an `Attributes` (a raw NamedTuple fails `convert(Attributes, …)`
# at resolve; `plot.material = (…)` setproperty is method-ambiguous).

using OmniverseMakie, ColorTypes, FixedPointNumbers, GeometryBasics
const OM = OmniverseMakie

OM.activate!(warmup = 120)

sph = GeometryBasics.normal_mesh(GeometryBasics.Tesselation(Sphere(Point3f(0), 1.0f0), 96))

fig = Figure()
ax  = LScene(fig[1, 1])
m   = mesh!(ax, sph; color = :red, material = (; metallic = 1.0, roughness = 0.1))

screen = OM.Screen(ax.scene)
Makie.push_screen!(ax.scene, screen)

# --- metrics ------------------------------------------------------------------
lum(c) = 0.2126f0 * Float32(red(c)) + 0.7152f0 * Float32(green(c)) + 0.0722f0 * Float32(blue(c))
function mean_rgb(img)
    tr = 0.0; tg = 0.0; tb = 0.0; cnt = 0
    for c in img
        if lum(c) > 0.02f0
            tr += Float32(red(c)); tg += Float32(green(c)); tb += Float32(blue(c)); cnt += 1
        end
    end
    cnt == 0 ? (0.0, 0.0, 0.0) :
        (round(tr / cnt; digits = 3), round(tg / cnt; digits = 3), round(tb / cnt; digits = 3))
end
function contrast(img)
    L = sort(Float32[lum(c) for c in img if lum(c) > 0.02f0])
    isempty(L) && return 0.0f0
    q(p) = L[clamp(round(Int, p * length(L)), 1, length(L))]
    return q(0.95) / (q(0.50) + 1.0f-3)
end
nonblack(img) = count(c -> lum(c) > 0.02f0, img)
function changed(x, y; thr = 0.15)
    n = 0
    @inbounds for i in eachindex(x, y)
        d = abs(Float32(red(x[i]))   - Float32(red(y[i]))) +
            abs(Float32(green(x[i])) - Float32(green(y[i]))) +
            abs(Float32(blue(x[i]))  - Float32(blue(y[i])))
        d > thr && (n += 1)
    end
    n
end

# ---- RENDER A: build open stage + diff node, bind pre-authored material ----
imgA = Makie.colorbuffer(screen)
@assert haskey(screen.plot2robj, objectid(m)) "mesh not registered as an OvrtxRObj"
robj = screen.plot2robj[objectid(m)]
@assert robj.material_shader == OM.material_prim_path(m) * "/Shader" "material_shader not recorded: $(robj.material_shader)"
println("MATERIAL_SHADER=$(robj.material_shader)")
rgbA = mean_rgb(imgA); cA = contrast(imgA)
println("RGB_A=$(rgbA) CONTRAST_A=$(cA) NONBLACK_A=$(nonblack(imgA))")
@assert nonblack(imgA) > 1000 "build frame (near) black: nonblack=$(nonblack(imgA))"
@assert rgbA[1] > rgbA[3] "build frame not red-dominant: $(rgbA)"

# ---- install the per-name push counter + the shader-write recorder ----
pushcount  = Dict{Symbol,Int}()
shaderwrit = String[]
OM._PUSH_OBSERVER[]         = name -> (pushcount[name] = get(pushcount, name, 0) + 1)
OM._SHADER_WRITE_OBSERVER[] = name -> push!(shaderwrit, String(name))

# ---- COLOR EDIT: one :scaled_color push → one shader write → blue ----
empty!(pushcount); empty!(shaderwrit)
m.color[] = :blue
imgB = Makie.colorbuffer(screen)
rgbB = mean_rgb(imgB); cB = contrast(imgB)
println("PUSH_COLOR=$(pushcount)")
println("SHADER_COLOR=$(sort(shaderwrit))")
println("RGB_B=$(rgbB) CONTRAST_B=$(cB)")
@assert pushcount == Dict(:scaled_color => 1) "color edit fired $(pushcount), expected one :scaled_color"
@assert sort(shaderwrit) == ["diffuse_color_constant"] "color edit wrote $(shaderwrit), expected only diffuse_color_constant"
@assert rgbB[3] > rgbB[1] "color edit did not turn the material blue: $(rgbB)"

# ---- MATERIAL EDIT: :material push → metallic + roughness writes → matte ----
empty!(pushcount); empty!(shaderwrit)
m.material[] = Makie.Attributes(; metallic = 0.0, roughness = 0.9)
imgC = Makie.colorbuffer(screen)
rgbC = mean_rgb(imgC); cC = contrast(imgC)
diffBC = changed(imgB, imgC)
println("PUSH_MATERIAL=$(pushcount)")
println("SHADER_MATERIAL=$(sort(shaderwrit))")
satB = rgbB[3] - rgbB[1]; satC = rgbC[3] - rgbC[1]  # blue-base saturation
println("RGB_C=$(rgbC) CONTRAST_C=$(cC) CHANGED_B_vs_C=$(diffBC)")
println("SAT_B=$(round(satB; digits=3)) SAT_C=$(round(satC; digits=3))")
@assert pushcount == Dict(:material => 1) "material edit fired $(pushcount), expected one :material"
@assert sort(shaderwrit) == ["metallic_constant", "reflection_roughness_constant"] "material edit wrote $(shaderwrit), expected metallic_constant + reflection_roughness_constant"
@assert diffBC > 1000 "material edit did not change the render (changed=$(diffBC))"
# metallic(1)→dielectric(0) + glossy→matte: the white specular wash is
# removed, so the surface shows its PURE diffuse base colour — markedly
# MORE saturated blue than the metal.
@assert satC > satB "material edit (metallic→matte) did not show the matte diffuse base (satB=$(satB) satC=$(satC))"

# ---- stage authored EXACTLY ONCE across all three renders (live writes) ----
opens = OM._ROOT_OPEN_COUNT[]
println("ROOT_OPENS=$(opens)")
@assert opens == 1 "stage re-opened during live material edits (opens=$(opens))"

close(screen)

# =============================================================================
# PHASE 2 — the same live material-PARAM edit on a MATERIALIZED *MeshScatter*
# (`:material ∈ consumed_inputs(::Makie.MeshScatter)` makes the edit take).
# A fresh figure + Screen in the SAME process (ovrtx startup amortized), so
# ROOT_OPENS is asserted as a PER-PHASE DELTA of 1.  Markers are MS_-prefixed.
# =============================================================================

basecol = RGBf(0.20, 0.20, 0.80)  # saturated blue — saturation rises when matte
msmarkers = [Point3f(x, 0.0, 0.0) for x in -1.5:1.0:1.5]

fig2 = Figure()
ax2  = LScene(fig2[1, 1])
ms   = meshscatter!(ax2, msmarkers; markersize = 0.45, color = basecol,
                    material = Makie.Attributes(; metallic = 1.0, roughness = 0.1))

opens_before_ms = OM._ROOT_OPEN_COUNT[]
screen2 = OM.Screen(ax2.scene)
Makie.push_screen!(ax2.scene, screen2)

imgMA = Makie.colorbuffer(screen2)
@assert haskey(screen2.plot2robj, objectid(ms)) "meshscatter not registered as an OvrtxRObj"
robj2 = screen2.plot2robj[objectid(ms)]
@assert robj2.material_shader == OM.material_prim_path(ms) * "/Shader" "material_shader not recorded: $(robj2.material_shader)"
println("MS_MATERIAL_SHADER=$(robj2.material_shader)")
rgbMA = mean_rgb(imgMA)
println("MS_RGB_A=$(rgbMA) MS_NONBLACK_A=$(nonblack(imgMA))")
@assert nonblack(imgMA) > 300 "meshscatter build frame (near) black: nonblack=$(nonblack(imgMA))"

empty!(pushcount); empty!(shaderwrit)
ms.material[] = Makie.Attributes(; metallic = 0.0, roughness = 0.9)
imgMB = Makie.colorbuffer(screen2)
rgbMB = mean_rgb(imgMB)
diffMAB = changed(imgMA, imgMB)
println("MS_PUSH_MATERIAL=$(pushcount)")
println("MS_SHADER_MATERIAL=$(sort(shaderwrit))")
satMA = rgbMA[3] - rgbMA[1]; satMB = rgbMB[3] - rgbMB[1]  # blue-base saturation
println("MS_RGB_B=$(rgbMB) MS_CHANGED_A_vs_B=$(diffMAB)")
println("MS_SAT_A=$(round(satMA; digits=3)) MS_SAT_B=$(round(satMB; digits=3))")
@assert pushcount == Dict(:material => 1) "meshscatter material edit fired $(pushcount), expected one :material"
@assert sort(shaderwrit) == ["metallic_constant", "reflection_roughness_constant"] "meshscatter material edit wrote $(shaderwrit), expected metallic_constant + reflection_roughness_constant"
@assert diffMAB > 300 "meshscatter material edit did not change the render (changed=$(diffMAB))"
@assert satMB > satMA "meshscatter material edit (metallic→matte) did not show the matte diffuse base (satA=$(satMA) satB=$(satMB))"

opens_ms = OM._ROOT_OPEN_COUNT[] - opens_before_ms
println("MS_ROOT_OPENS=$(opens_ms)")
@assert opens_ms == 1 "meshscatter stage re-opened during the live material edit (delta=$(opens_ms))"

close(screen2)
println("OK_MATERIAL_LIVE")
