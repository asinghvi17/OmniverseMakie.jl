# Vendored from references/RPRMakieNotes/scripts/pointsfont.jl (Lazaro Alonso).
# Builds 3-D letter-glyph meshes from font outlines via Luxor/Cairo path extraction.
# Included by helix.jl; the leading underscore excludes this file from run_all.jl's scene glob.
using OmniverseMakie, GeometryBasics, Luxor

"""
    pointsfont(letter; fs, sx, sy, shiftx, shifty) -> Vector{Point2f}

Extract a Luxor/Cairo path for `letter` at fontsize `fs` and return it as a list of
`Point2f` values (NaN separators between sub-paths, as Makie band expects).
"""
function pointsfont(letter; fs=60, sx=500, sy=500, shiftx=0, shifty=0)
    Drawing(500, 500)
    newpath()
    fontsize(fs)
    fontface("Mono")
    textpath(letter)
    pathspoints = pathtopoly()
    ymin = []
    xmin = []
    for pnts in pathspoints
        for p in pnts
            push!(ymin, -p[2])
            push!(xmin, p[1])
        end
    end
    ymin = minimum(ymin)
    xmin = minimum(xmin)
    path = Point2f[]
    for pnts in pathspoints
        tmp = [Point2f(p[1] - xmin + shiftx, -p[2] - ymin + shifty) for p in pnts]
        if length(pnts) > 1
            path = vcat(path, tmp)
            path = vcat(path, [tmp[1]])
        end
        path = vcat(path, [Point2f(NaN, NaN)])
    end
    return path
end

"""
    poly_3d(points3d) -> GeometryBasics.Mesh

Triangulate a flat polygon defined by `points3d` (each point has z = constant).
Uses `GeometryBasics.Polygon` for 2-D face generation, then lifts to 3-D.
"""
function poly_3d(points3d)
    xy = Point2f.(points3d)
    f = faces(GeometryBasics.Polygon(xy))
    return normal_mesh(Point3f.(points3d), f)
end

"""
    getMesh(top_poly, bottom_poly) -> GeometryBasics.Mesh

Combine top and bottom letter-face meshes with lateral band faces to produce
an extruded 3-D glyph mesh. Uses `Makie.band_connect` for the side faces.
"""
function getMesh(top_poly, bottom_poly)
    top      = poly_3d(top_poly)
    bottom   = poly_3d(bottom_poly)
    combined = merge([top, bottom])
    nvertices = length(top.position)
    connection = Makie.band_connect(nvertices)
    meshletter = GeometryBasics.Mesh(
        GeometryBasics.coordinates(combined),
        vcat(faces(combined), connection),
    )
    return meshletter
end
