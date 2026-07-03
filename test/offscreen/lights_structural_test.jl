using Test
using Test: TestLogger
using Logging: with_logger, Warn
using OmniverseMakie: sync_lights!, _lights_snapshot, usda_light, _direction_to_xform_matrix,
                      Scene, RectLight, DirectionalLight, PointLight,
                      RGBf, Vec3f, Vec2f, Point3f, AbstractLight

# ---------------------------------------------------------------------------
# Track L / Task L1 — structural light change: fail loud, not corrupt.
#
# Pure tests, no renderer/GPU: sync_lights!'s first-call and count-mismatch
# branches both return BEFORE any `screen.renderer` access, so a minimal mutable
# stand-in exposing only `.last_lights` exercises them. Plus a golden regression
# anchor for the de-duplicated RectLight transform.
# ---------------------------------------------------------------------------

# Minimal Screen stand-in: sync_lights! touches ONLY `.last_lights` on the
# first-call and structural-mismatch paths (it returns before reading `.renderer`).
mutable struct _FakeLightScreen
    last_lights::Any
end

_scene_with(lights::Vector) = Scene(lights = AbstractLight[lights...])

# Parse the `matrix4d xformOp:transform = ( (…),(…),(…),(…) )` literal out of a USDA
# light block into a 4×4 row-major Float64 matrix.
function _parse_xform(usda::AbstractString)
    line = only(filter(l -> occursin("matrix4d xformOp:transform", l), split(usda, '\n')))
    rhs  = split(line, " = ")[2]
    nums = [parse(Float64, m.match) for m in eachmatch(r"-?[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?", rhs)]
    @assert length(nums) == 16 "expected 16 matrix entries, got $(length(nums))"
    return permutedims(reshape(nums, 4, 4))
end

@testset "L1 count mismatch: warn once, preserve snapshot, no live write" begin
    # NB: Makie folds AmbientLight out of scene.compute[:lights][] (separate ambient
    # term), so use two light types that both survive the compute → a genuine count of 2.
    one_light  = [DirectionalLight(RGBf(1f0, 1f0, 1f0), Vec3f(-1f0, -1f0, -1f0), false)]
    two_lights = [DirectionalLight(RGBf(1f0, 1f0, 1f0), Vec3f(-1f0, -1f0, -1f0), false),
                  PointLight(RGBf(1f0, 0f0, 0f0), Vec3f(0f0, 0f0, 100f0), Vec2f(0f0, 0f0))]

    seed  = _lights_snapshot(one_light)   # baked baseline: 1 light
    scene = _scene_with(two_lights)       # scene grew to 2 lights (count now differs)

    # A structural change must fail LOUD, not corrupt: warn (once, maxlog=1) and return
    # false WITHOUT advancing the snapshot — advancing it would diff the next edit against
    # never-authored prims. Because the baseline is never advanced, the mismatch is
    # re-detected every frame. Re-seeding a fresh stand-in to `seed` each iteration models
    # the fix holding `last_lights` at the baked baseline across frames.
    logger = TestLogger()
    rets = Bool[]
    with_logger(logger) do
        for _ in 1:2
            screen = _FakeLightScreen(seed)
            push!(rets, sync_lights!(screen, scene))
            @test screen.last_lights === seed          # snapshot NEVER advanced
        end
    end
    warns = filter(r -> r.level == Warn, logger.logs)

    @test all(r -> r === false, rets)                  # nothing live-written on a mismatch
    @test length(warns) == 1                           # warned exactly once (maxlog=1)
    @test any(w -> occursin("adding/removing lights on a live Screen is not supported",
                            w.message), warns)         # the locked message
end

@testset "L1 first call (no baked snapshot) seeds silently" begin
    lights = [DirectionalLight(RGBf(1f0, 1f0, 1f0), Vec3f(-1f0, -1f0, -1f0), false)]
    scene  = _scene_with(lights)
    screen = _FakeLightScreen(nothing)                 # first sync, nothing baked yet

    ret = @test_logs min_level = Warn sync_lights!(screen, scene)   # seeds; must NOT warn
    @test ret === false
    @test screen.last_lights == _lights_snapshot(lights)           # snapshot seeded
end

@testset "L1 RectLight transform == DistantLight orientation + translation row" begin
    dir = Vec3f(-1f0, -2f0, -3f0)     # non-axis-aligned: exercises the general branch
    pos = Point3f(10f0, 20f0, 30f0)
    col = RGBf(1f0, 1f0, 1f0)

    rect = RectLight(col, pos, Vec3f(4f0, 0f0, 0f0), Vec3f(0f0, 5f0, 0f0), dir)
    dl   = DirectionalLight(col, dir, false)

    # Golden regression anchor: exact matrix string captured from the pre-refactor inline
    # implementation — the de-duplication must reproduce it byte-for-byte.
    golden = "( (-0.8944271909999159, 0.4472135954999579, 0.0, 0.0), " *
             "(-0.3585685828003181, -0.7171371656006362, 0.5976143046671968, 0.0), " *
             "(0.2672612419124244, 0.5345224838248488, 0.8017837257372732, 0.0), " *
             "(10.0, 20.0, 30.0, 1.0) )"
    @test occursin(golden, usda_light(rect, 0))

    # Relationship (the point of sharing _direction_to_xform_matrix): the RectLight
    # orientation (rows 1-3) equals the DistantLight's; only row 4 (the translation)
    # differs.
    rect_mat = _parse_xform(usda_light(rect, 0))
    dl_mat   = _parse_xform(usda_light(dl, 0))
    @test rect_mat[1:3, :] == dl_mat[1:3, :]
    @test rect_mat[1:3, :] == _direction_to_xform_matrix(dir)[1:3, :]
    @test rect_mat[4, :] == [10.0, 20.0, 30.0, 1.0]
    @test dl_mat[4, :]   == [0.0, 0.0, 0.0, 1.0]
end
