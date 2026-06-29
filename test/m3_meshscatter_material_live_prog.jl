# Subprocess body for the M3 LIVE material-param edit test on a MATERIALIZED MeshScatter
# (read + run by test/m3_material_live_test.jl via run_ovrtx_subprocess).  Standalone .jl
# so the Makie scene setup needs no escaping.
#
# Proves the M3 final-review fix: a live `plot.material[]` PARAM edit on a MATERIALIZED
# *MeshScatter* (not just a Mesh) re-writes the PRE-AUTHORED OmniPBR shader inputs on the
# OPEN stage (the M2 diff path) — NO re-author (`ROOT_OPENS == 1`), the render CHANGES,
# and EXACTLY the changed shader inputs are written (instrumented via `_PUSH_OBSERVER` +
# `_SHADER_WRITE_OBSERVER`).  Before the fix `:material` was NOT in
# `consumed_inputs(::Makie.MeshScatter)`, so this edit was a SILENT no-op.
#
#   render A  — materialized meshscatter (metallic=1, roughness=0.1), blue base → glossy
#   material[]=Attributes(metallic=0, roughness=0.9) → render B → exactly one :material push
#                    → metallic_constant + reflection_roughness_constant shader writes →
#                    the glossy specular collapses to a MATTE surface that shows its pure
#                    diffuse base (blue base is MORE saturated when matte)
#
# NOTE (M3.4 MCP quirk): the graph stores `:material` as `Makie.Attributes`, so a live edit
# MUST pass an `Attributes` (a raw NamedTuple fails `convert(Attributes, …)` at resolve).

using OmniverseMakie, ColorTypes, FixedPointNumbers, GeometryBasics
const OM = OmniverseMakie

OM.activate!(warmup = 120)

basecol = RGBf(0.20, 0.20, 0.80)   # saturated blue base — its saturation rises when matte
msmarkers = [Point3f(x, 0.0, 0.0) for x in -1.5:1.0:1.5]

fig = Figure()
ax  = LScene(fig[1, 1])
m   = meshscatter!(ax, msmarkers; markersize = 0.45, color = basecol,
                   material = Makie.Attributes(; metallic = 1.0, roughness = 0.1))

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

# ---- RENDER A: build the open stage + the diff node, bind the pre-authored material ----
imgA = Makie.colorbuffer(screen)
@assert haskey(screen.plot2robj, objectid(m)) "meshscatter not registered as an OvrtxRObj"
robj = screen.plot2robj[objectid(m)]
@assert robj.material_shader == OM.material_prim_path(m) * "/Shader" "material_shader not recorded: $(robj.material_shader)"
println("MATERIAL_SHADER=$(robj.material_shader)")
rgbA = mean_rgb(imgA)
println("RGB_A=$(rgbA) NONBLACK_A=$(nonblack(imgA))")
@assert nonblack(imgA) > 300 "build frame (near) black: nonblack=$(nonblack(imgA))"

# ---- install the per-name push counter + the shader-write recorder ----
pushcount  = Dict{Symbol,Int}()
shaderwrit = String[]
OM._PUSH_OBSERVER[]         = name -> (pushcount[name] = get(pushcount, name, 0) + 1)
OM._SHADER_WRITE_OBSERVER[] = name -> push!(shaderwrit, String(name))

# ---- MATERIAL EDIT: one :material push → metallic + roughness shader writes → glossy→matte ----
empty!(pushcount); empty!(shaderwrit)
m.material[] = Makie.Attributes(; metallic = 0.0, roughness = 0.9)
imgB = Makie.colorbuffer(screen)
rgbB = mean_rgb(imgB)
diffAB = changed(imgA, imgB)
println("PUSH_MATERIAL=$(pushcount)")
println("SHADER_MATERIAL=$(sort(shaderwrit))")
satA = rgbA[3] - rgbA[1]; satB = rgbB[3] - rgbB[1]   # blue-base saturation (blue − red)
println("RGB_B=$(rgbB) CHANGED_A_vs_B=$(diffAB)")
println("SAT_A=$(round(satA; digits=3)) SAT_B=$(round(satB; digits=3))")
@assert pushcount == Dict(:material => 1) "material edit fired $(pushcount), expected one :material"
@assert sort(shaderwrit) == ["metallic_constant", "reflection_roughness_constant"] "material edit wrote $(shaderwrit), expected metallic_constant + reflection_roughness_constant"
@assert diffAB > 300 "material edit did not change the render (changed=$(diffAB))"
# metallic(1)→dielectric(0) + glossy→matte: the white specular wash is removed, so the
# surface shows its PURE diffuse base colour — markedly MORE saturated blue than the metal.
@assert satB > satA "material edit (metallic→matte) did not show the matte diffuse base (satA=$(satA) satB=$(satB))"

# ---- the stage was authored EXACTLY ONCE across both renders (live writes only) ----
opens = OM._ROOT_OPEN_COUNT[]
println("ROOT_OPENS=$(opens)")
@assert opens == 1 "stage re-opened during the live material edit (opens=$(opens))"

close(screen)
println("OK_MESHSCATTER_MATERIAL_LIVE")
