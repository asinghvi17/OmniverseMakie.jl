# Chroma oracle for the IndeX-composite color proof.
# usage: julia analyze.jl out_gray.png out_color.png
using FileIO, ColorTypes

gray  = load(ARGS[1])
color = load(ARGS[2])
size(gray) == size(color) || error("size mismatch: $(size(gray)) vs $(size(color))")

lum(c)    = Float32(red(c)) + Float32(green(c)) + Float32(blue(c))
chroma(c) = (v = (Float32(red(c)), Float32(green(c)), Float32(blue(c))); maximum(v) - minimum(v))

println("GRAY_NONBLACK=",  count(c -> lum(c) > 0.05f0, gray))
println("COLOR_NONBLACK=", count(c -> lum(c) > 0.05f0, color))

# pixels the colormap swap changed = the volume's footprint
diffmask = map((a, b) -> abs(lum(a) - lum(b)) > 0.08f0 || abs(chroma(a) - chroma(b)) > 0.08f0,
               gray, color)
println("DIFF_PX=", count(diffmask))

println("CHROMA_PX_GRAY=",  count(c -> chroma(c) > 0.15f0, gray))
println("CHROMA_PX_COLOR=", count(c -> chroma(c) > 0.15f0, color))

if any(diffmask)
    region_g = [chroma(gray[k])  for k in eachindex(gray)  if diffmask[k]]
    region_c = [chroma(color[k]) for k in eachindex(color) if diffmask[k]]
    println("REGION_MEAN_CHROMA_GRAY=",  round(sum(region_g) / length(region_g); digits = 4))
    println("REGION_MEAN_CHROMA_COLOR=", round(sum(region_c) / length(region_c); digits = 4))
end
println("OK_ANALYZE")
