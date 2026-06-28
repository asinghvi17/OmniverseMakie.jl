# Subprocess body for the M3.4 LIVE material-edit test (read + run by
# test/m3_material_live_test.jl via run_ovrtx_subprocess).  Standalone .jl so the Makie
# scene setup needs no escaping.
#
# Proves M3.4 end to end THROUGH the real Screen / colorbuffer pipeline: a live
# `plot.color[]` / `plot.material[]` PARAM edit on a MATERIALIZED mesh re-writes the
# PRE-AUTHORED OmniPBR shader inputs on the OPEN stage (the M2 diff path) — NO re-author
# (`ROOT_OPENS == 1`), the render changes, and EXACTLY the changed shader inputs are
# written (instrumented via `_PUSH_OBSERVER` + `_SHADER_WRITE_OBSERVER`).
#
#   render A  — materialized mesh (metallic=1, roughness=0.1), red base → glossy red
#   color[]=:blue  → render B → exactly one :scaled_color push → one diffuse_color_constant
#                    shader write → the surface turns BLUE-dominant (no re-author)
#   material[]=Attributes(metallic=0, roughness=0.9) → render C → exactly one :material push
#                    → metallic_constant + reflection_roughness_constant shader writes →
#                    the glossy specular collapses to a MATTE surface (contrast drops)
#
# NOTE (M3.4 MCP finding): the graph stores `:material` as `Makie.Attributes`, so a live
# edit MUST pass an `Attributes` (a raw NamedTuple fails `convert(Attributes, …)` at
# resolve; `plot.material = (…)` setproperty is method-ambiguous).  Hence `Attributes(…)`.

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

# ---- RENDER A: build the open stage + the diff node, bind the pre-authored material ----
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

# ---- COLOR EDIT: one :scaled_color push → one diffuse_color_constant shader write → blue ----
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

# ---- MATERIAL EDIT: one :material push → metallic + roughness shader writes → glossy→matte ----
empty!(pushcount); empty!(shaderwrit)
m.material[] = Makie.Attributes(; metallic = 0.0, roughness = 0.9)
imgC = Makie.colorbuffer(screen)
rgbC = mean_rgb(imgC); cC = contrast(imgC)
diffBC = changed(imgB, imgC)
println("PUSH_MATERIAL=$(pushcount)")
println("SHADER_MATERIAL=$(sort(shaderwrit))")
satB = rgbB[3] - rgbB[1]; satC = rgbC[3] - rgbC[1]   # blue-base saturation (blue − red)
println("RGB_C=$(rgbC) CONTRAST_C=$(cC) CHANGED_B_vs_C=$(diffBC)")
println("SAT_B=$(round(satB; digits=3)) SAT_C=$(round(satC; digits=3))")
@assert pushcount == Dict(:material => 1) "material edit fired $(pushcount), expected one :material"
@assert sort(shaderwrit) == ["metallic_constant", "reflection_roughness_constant"] "material edit wrote $(shaderwrit), expected metallic_constant + reflection_roughness_constant"
@assert diffBC > 1000 "material edit did not change the render (changed=$(diffBC))"
# metallic(1)→dielectric(0) + glossy→matte: the white specular wash is removed, so the
# surface shows its PURE diffuse base colour — markedly MORE saturated blue than the metal.
@assert satC > satB "material edit (metallic→matte) did not show the matte diffuse base (satB=$(satB) satC=$(satC))"

# ---- the stage was authored EXACTLY ONCE across all three renders (live writes only) ----
opens = OM._ROOT_OPEN_COUNT[]
println("ROOT_OPENS=$(opens)")
@assert opens == 1 "stage re-opened during live material edits (opens=$(opens))"

close(screen)
println("OK_MATERIAL_LIVE")
