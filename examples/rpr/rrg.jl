# Ported from references/RPRMakieNotes/scripts/rrg.jl (Lazaro Alonso).
# Random Relational Graph in 3D: weighted edges via linesegments! + meshscatter! nodes.
# DiffuseMaterial on the plane → dropped (plain color= / USD displayColor matte).
# LinearAlgebra avoided: Diagonal and diag replaced with inline idioms.
using OmniverseMakie, GeometryBasics, Colors, Random

# Build a 3D random geometric adjacency matrix (pure base Julia, no graph package).
# radius = 0.17 connects nodes whose Euclidean distance is within threshold.
function _rrg_adjacency(; radius = 0.17, nodes = 500)
    xy = rand(nodes, 3)
    x  = xy[:, 1]
    y  = xy[:, 2]
    z  = xy[:, 3]

    matrixAdj = zeros(nodes, nodes)
    # Off-diagonal edges: pairwise distance threshold
    for point in 1:nodes-1
        xseps = (x[point+1:end] .- x[point]) .^ 2
        yseps = (y[point+1:end] .- y[point]) .^ 2
        zseps = (z[point+1:end] .- z[point]) .^ 2
        distance = sqrt.(xseps .+ yseps .+ zseps)
        dindx = findall(distance .<= radius) .+ point
        if length(dindx) > 0
            rnd = randn(length(dindx))
            matrixAdj[point, dindx] = rnd
            matrixAdj[dindx, point] = rnd
        end
    end
    # Diagonal weights (replaces Diagonal(√2 * randn(nodes)))
    for i in 1:nodes
        matrixAdj[i, i] = √2 * randn()
    end
    return matrixAdj, x, y, z
end

# Extract edge segments and per-point weights for linesegments!
function _rrg_edges(adjMatrix, x, y, z)
    xyzos   = Point3f[]
    weights = Float32[]
    for i in 1:length(x), j in i+1:length(x)
        if adjMatrix[i, j] != 0.0
            push!(xyzos, Point3f(x[i], y[i], z[i]))
            push!(xyzos, Point3f(x[j], y[j], z[j]))
            push!(weights, adjMatrix[i, j])
            push!(weights, adjMatrix[i, j])
        end
    end
    return xyzos, weights
end

function scene_rrg()
    Random.seed!(42)

    adjacencyM3D, x, y, z = _rrg_adjacency()

    cmap       = (:Hiroshige, 0.75)
    adjmin     = minimum(adjacencyM3D)
    adjmax     = maximum(adjacencyM3D)
    # Diagonal entries (replaces diag(adjacencyM3D))
    diagValues = Float32[adjacencyM3D[i, i] for i in 1:length(x)]
    segm, weights = _rrg_edges(adjacencyM3D, x, y, z)

    grey   = [colorant"grey90" for _ in 1:1, _ in 1:1]
    # ★ PointLight: color first, then position
    lights = [EnvironmentLight(1.0, grey'), PointLight(RGBf(8.0, 6.0, 5.0), Vec3f(2, 0, 2.0))]
    plane  = Rect3f(Vec3f(-5, -2, -1.05), Vec3f(10, 4, 0.05))

    fig = Figure(; size = (1000, 1000))
    ax  = LScene(fig[1, 1]; show_axis = false, scenekw = (; lights = lights))

    linesegments!(ax, segm;
        color      = weights,
        colormap   = cmap,
        colorrange = (adjmin, adjmax),
        linewidth  = abs.(weights))

    meshscatter!(ax, x, y, z;
        color      = diagValues,
        colormap   = cmap,
        colorrange = (adjmin, adjmax),
        markersize = abs.(diagValues) ./ 90)

    # DiffuseMaterial → plain color= (drop material=)
    mesh!(ax, plane; color = :gainsboro)

    # Camera: preserve eyeposition from original
    cam = cameracontrols(ax.scene)
    cam.eyeposition[] = Vec3f(1.8, 1.8, 1.5)
    update_cam!(ax.scene, cam)

    return fig
end

function assert_rrg(img)
    # A sparse graph fills a modest pixel fraction — 2% is a conservative floor.
    assert_nonblack(img, "rrg"; frac = 0.02)
end
