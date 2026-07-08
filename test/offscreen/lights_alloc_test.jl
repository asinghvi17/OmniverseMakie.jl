using Test
using OmniverseMakie: _lights_snapshot, _light_render_state, LightState, LightSnapshot,
                      DirectionalLight, PointLight, AmbientLight, RectLight,
                      RGBf, Vec3f, Vec2f, Point3f, AbstractLight

# ---------------------------------------------------------------------------
# sync_lights! rebuilds the light snapshot every frame, so the rebuild must
# stay allocation-lean: immutable LightState in a concrete-eltype vector,
# with authored path strings reused while the light count is unchanged.
# Pure tests, no renderer/GPU: snapshot building + LightState comparison.
# ---------------------------------------------------------------------------

_three_lights() = AbstractLight[
    DirectionalLight(RGBf(1f0, 1f0, 1f0), Vec3f(-1f0, -2f0, -3f0), false),
    PointLight(RGBf(1f0, 0f0, 0f0), Vec3f(0f0, 0f0, 100f0), Vec2f(0f0, 0f0)),
    DirectionalLight(RGBf(0.5f0, 0.7f0, 0.9f0), Vec3f(1f0, -1f0, -1f0), false),
]

@testset "per-frame snapshot rebuild is allocation-lean" begin
    lights = _three_lights()
    # author-time seed (fresh; allocates paths once)
    seed   = _lights_snapshot(lights)

    @test seed isa LightSnapshot               # concrete eltype (no Any[])
    @test eltype(seed) === Tuple{String,Union{LightState,Nothing}}

    _lights_snapshot(lights, seed)   # warmup: compile before measuring
    GC.gc()
    a1 = @allocated _lights_snapshot(lights, seed)
    GC.gc()
    a2 = @allocated _lights_snapshot(lights, seed)

    # The 4096 B ceiling is deliberately generous: Julia minor versions shift
    # Union-field/tuple boxing and small-array layout; the bound only needs
    # to catch a revert to a Dict/fresh-strings/Matrix-temporary rebuild.
    @test a1 ≤ 4096
    @test a2 ≤ 4096

    # Reuse preserves the authored paths AND hands back the exact same String
    # OBJECTS (the point of the diet: zero fresh path allocation on an
    # unchanged frame).
    reused = _lights_snapshot(lights, seed)
    @test [p for (p, _) in reused] == [p for (p, _) in seed]
    @test all(reused[i][1] === seed[i][1] for i in eachindex(seed))
end

@testset "count change falls back to a fresh (correct) snapshot" begin
    three = _three_lights()
    two   = AbstractLight[three[1], three[2]]
    seed3 = _lights_snapshot(three)

    # old-snapshot count ≠ current light count → the reuse form must NOT
    # graft mismatched paths; it rebuilds fresh so sync_lights!'s
    # count-mismatch guard still fires.
    snap2 = _lights_snapshot(two, seed3)
    @test length(snap2) == 2
    @test [p for (p, _) in snap2] == [p for (p, _) in _lights_snapshot(two)]
end

@testset "same-count type swap rebuilds fresh paths (no stale index-wise reuse)" begin
    # Prim paths encode <Type>_<idx>: a same-count type swap must NOT reuse
    # old paths with new states (each state would write to the wrong prim).
    # Reuse is valid only when every position's type matches; else rebuild.
    dp = AbstractLight[
        DirectionalLight(RGBf(1f0, 1f0, 1f0), Vec3f(-1f0, -1f0, -1f0), false),
        PointLight(RGBf(1f0, 0f0, 0f0), Vec3f(0f0, 0f0, 100f0), Vec2f(0f0, 0f0)),
    ]
    seed_dp = _lights_snapshot(dp)
    @test [p for (p, _) in seed_dp] == ["/World/DirectionalLight_0", "/World/PointLight_0"]

    pd      = AbstractLight[dp[2], dp[1]]      # [Point, Directional] — swapped
    swapped = _lights_snapshot(pd, seed_dp)    # two-arg reuse form under test
    fresh   = _lights_snapshot(pd)             # ground truth: recomputed paths

    # The swap must produce the FRESH paths (PointLight_0 now at position 1),
    # NOT the stale index-wise reuse (which would keep [DirectionalLight_0,
    # PointLight_0]).
    @test [p for (p, _) in swapped] == [p for (p, _) in fresh]
    @test [p for (p, _) in swapped] == ["/World/PointLight_0", "/World/DirectionalLight_0"]
    @test [p for (p, _) in swapped] != [p for (p, _) in seed_dp]

    # Each state is paired with the CORRECTLY-named prim: the Point light's
    # state at position 1 is written to PointLight_0 (not the stale
    # DirectionalLight_0), the Directional's to _0.
    @test swapped[1] == ("/World/PointLight_0",      _light_render_state(pd[1]))
    @test swapped[2] == ("/World/DirectionalLight_0", _light_render_state(pd[2]))

    # Happy path (types unchanged): still reuses the exact String OBJECTS
    # (===) — zero fresh path allocation, the steady-state win the type-swap
    # guard must not regress.
    reused = _lights_snapshot(dp, seed_dp)
    @test [p for (p, _) in reused] == [p for (p, _) in seed_dp]
    @test all(reused[i][1] === seed_dp[i][1] for i in eachindex(seed_dp))
end

@testset "LightState comparison detects intensity/color/direction changes" begin
    base = DirectionalLight(RGBf(1f0, 1f0, 1f0), Vec3f(-1f0, -1f0, -1f0), false)
    # dim / hue / turn differ from base in intensity / color / direction
    dim  = DirectionalLight(RGBf(0.5f0, 0.5f0, 0.5f0), Vec3f(-1f0, -1f0, -1f0), false)
    hue  = DirectionalLight(RGBf(1f0, 0f0, 0f0), Vec3f(-1f0, -1f0, -1f0), false)
    turn = DirectionalLight(RGBf(1f0, 1f0, 1f0), Vec3f(1f0, 0f0, 0f0), false)

    sb = _light_render_state(base)
    @test sb isa LightState
    # intensity/color/direction (xform) changes detected; unchanged →
    # identical (no write)
    @test sb.intensity != _light_render_state(dim).intensity
    @test sb.color     != _light_render_state(hue).color
    @test sb.xform     != _light_render_state(turn).xform
    @test _light_render_state(base) == sb

    # DomeLight (AmbientLight) carries no xform; exotic types stay baked
    # (no live state).
    @test _light_render_state(AmbientLight(RGBf(0.5f0, 0.5f0, 0.5f0))).xform === nothing
    rect = RectLight(RGBf(1f0, 1f0, 1f0), Point3f(0f0, 0f0, 0f0),
                     Vec3f(4f0, 0f0, 0f0), Vec3f(0f0, 5f0, 0f0), Vec3f(-1f0, -1f0, -1f0))
    @test _light_render_state(rect) === nothing
end
