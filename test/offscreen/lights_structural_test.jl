using Test
using Test: TestLogger
using Logging: with_logger, Warn
using OmniverseMakie: sync_lights!, _lights_snapshot, usda_light, _direction_to_xform_matrix,
                      Scene, RectLight, DirectionalLight, PointLight, SpotLight,
                      RGBf, Vec3f, Vec2f, Point3f, AbstractLight

# ---------------------------------------------------------------------------
# Structural light change (add/remove) must fail loud, not corrupt.
# Pure tests, no renderer/GPU: sync_lights!'s first-call and count-mismatch
# branches return before any `screen.renderer` access, so a minimal stand-in
# exposing only `.last_lights` exercises them. Plus a golden anchor for the
# shared RectLight transform.
# ---------------------------------------------------------------------------

# Minimal Screen stand-in: sync_lights! touches ONLY `.last_lights` on the
# first-call and structural-mismatch paths (it returns before reading
# `.renderer`).
mutable struct _FakeLightScreen
    last_lights::Any
end

_scene_with(lights::Vector) = Scene(lights = AbstractLight[lights...])

# Parse the `matrix4d xformOp:transform = ( (…),(…),(…),(…) )` literal out
# of a USDA light block into a 4×4 row-major Float64 matrix.
function _parse_xform(usda::AbstractString)
    line = only(filter(l -> occursin("matrix4d xformOp:transform", l), split(usda, '\n')))
    rhs  = split(line, " = ")[2]
    nums = [parse(Float64, m.match) for m in eachmatch(r"-?[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?", rhs)]
    @assert length(nums) == 16 "expected 16 matrix entries, got $(length(nums))"
    return permutedims(reshape(nums, 4, 4))
end

@testset "L1 count mismatch: warn once, preserve snapshot, no live write" begin
    # NB: Makie folds AmbientLight out of scene.compute[:lights][] (separate
    # ambient term), so use two light types that both survive the compute →
    # a genuine count of 2.
    one_light  = [DirectionalLight(RGBf(1f0, 1f0, 1f0), Vec3f(-1f0, -1f0, -1f0), false)]
    two_lights = [DirectionalLight(RGBf(1f0, 1f0, 1f0), Vec3f(-1f0, -1f0, -1f0), false),
                  PointLight(RGBf(1f0, 0f0, 0f0), Vec3f(0f0, 0f0, 100f0), Vec2f(0f0, 0f0))]

    seed  = _lights_snapshot(one_light)   # baked baseline: 1 light
    scene = _scene_with(two_lights)       # grew to 2 lights (count differs)

    # A structural change must fail loud, not corrupt: warn once (maxlog=1)
    # and return false without advancing the snapshot (advancing would diff
    # the next edit against never-authored prims). A fresh stand-in per
    # iteration models `last_lights` held at the baked baseline across frames.
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

    @test all(r -> r === false, rets)    # nothing live-written on a mismatch
    @test length(warns) == 1             # warned exactly once (maxlog=1)
    @test any(w -> occursin("adding/removing lights on a live Screen is not supported",
                            w.message), warns)         # the locked message
end

@testset "L1 first call (no baked snapshot) seeds silently" begin
    lights = [DirectionalLight(RGBf(1f0, 1f0, 1f0), Vec3f(-1f0, -1f0, -1f0), false)]
    scene  = _scene_with(lights)
    screen = _FakeLightScreen(nothing)   # first sync, nothing baked yet

    # seeds; must NOT warn
    ret = @test_logs min_level = Warn sync_lights!(screen, scene)
    @test ret === false
    @test screen.last_lights == _lights_snapshot(lights)   # snapshot seeded
end

@testset "L1 RectLight transform == DistantLight orientation + translation row" begin
    dir = Vec3f(-1f0, -2f0, -3f0)   # non-axis-aligned: hits general branch
    pos = Point3f(10f0, 20f0, 30f0)
    col = RGBf(1f0, 1f0, 1f0)

    rect = RectLight(col, pos, Vec3f(4f0, 0f0, 0f0), Vec3f(0f0, 5f0, 0f0), dir)
    dl   = DirectionalLight(col, dir, false)

    # Golden anchor: usda_light must reproduce this matrix string byte-for-byte.
    golden = "( (-0.8944271909999159, 0.4472135954999579, 0.0, 0.0), " *
             "(-0.3585685828003181, -0.7171371656006362, 0.5976143046671968, 0.0), " *
             "(0.2672612419124244, 0.5345224838248488, 0.8017837257372732, 0.0), " *
             "(10.0, 20.0, 30.0, 1.0) )"
    @test occursin(golden, usda_light(rect, 0))

    # Relationship (the point of sharing _direction_to_xform_matrix): the
    # RectLight orientation (rows 1-3) equals the DistantLight's; only row 4
    # (the translation) differs.
    rect_mat = _parse_xform(usda_light(rect, 0))
    dl_mat   = _parse_xform(usda_light(dl, 0))
    @test rect_mat[1:3, :] == dl_mat[1:3, :]
    @test rect_mat[1:3, :] == _direction_to_xform_matrix(dir)[1:3, :]
    @test rect_mat[4, :] == [10.0, 20.0, 30.0, 1.0]
    @test dl_mat[4, :]   == [0.0, 0.0, 0.0, 1.0]
end

@testset "L1 SpotLight cone orientation follows direction, not pure translation" begin
    # The SphereLight+ShapingAPI cone emits along local −Z, so the xform must
    # ORIENT it toward `direction`. Pre-fix the SpotLight branch emitted a
    # pure-translation xform (identity orientation) → every cone pointed −Z
    # regardless of `direction`. A +X direction forces a non-identity rotation.
    dir = Vec3f(1f0, 0f0, 0f0)
    pos = Point3f(10f0, 20f0, 30f0)
    sl  = SpotLight(RGBf(1f0, 1f0, 1f0), pos, dir, Vec2f(0.1f0, 0.3f0))
    usda = usda_light(sl, 0)

    mat = _parse_xform(usda)
    # Orientation (rows 1-3) == the shared direction→orientation matrix,
    # identical to what DistantLight/RectLight derive; translation in row 4.
    @test mat[1:3, :] == _direction_to_xform_matrix(dir)[1:3, :]
    @test mat[4, :]   == [10.0, 20.0, 30.0, 1.0]
    # The pre-fix bug: a +X cone can NOT be the identity orientation.
    @test mat[1:3, 1:3] != [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    # Cone shaping is still authored.
    @test occursin("inputs:shaping:cone:angle", usda)
end
