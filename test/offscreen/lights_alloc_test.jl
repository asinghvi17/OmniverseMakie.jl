using Test
using OmniverseMakie: _lights_snapshot, _light_render_state, LightState, LightSnapshot,
                      DirectionalLight, PointLight, AmbientLight, RectLight,
                      RGBf, Vec3f, Vec2f, Point3f, AbstractLight

# ---------------------------------------------------------------------------
# Track L / Task L2 — sync_lights! allocation diet + typed snapshots.
#
# sync_lights! runs on EVERY frame (colorbuffer + interactive tick), rebuilding the
# light snapshot each time. Pre-L2 that rebuild allocated a `Dict{DataType,Int}`
# (_enumerate_lights), fresh interpolated path strings, and a 4×4 `Matrix{Float64}`
# through ~6 temporaries per directional light, all boxed in an `Any[]`. L2 replaces
# that with an immutable `LightState` (stack-tuple xform) in a concrete-eltype vector
# and REUSES the authored path strings when the light count is unchanged (paths are
# invariant unless a light is added/removed, which L1 makes terminal).
#
# Pure tests, no renderer/GPU: snapshot building + LightState comparison only.
# ---------------------------------------------------------------------------

_three_lights() = AbstractLight[
    DirectionalLight(RGBf(1f0, 1f0, 1f0), Vec3f(-1f0, -2f0, -3f0), false),
    PointLight(RGBf(1f0, 0f0, 0f0), Vec3f(0f0, 0f0, 100f0), Vec2f(0f0, 0f0)),
    DirectionalLight(RGBf(0.5f0, 0.7f0, 0.9f0), Vec3f(1f0, -1f0, -1f0), false),
]

@testset "L2 per-frame snapshot rebuild is allocation-lean" begin
    lights = _three_lights()
    seed   = _lights_snapshot(lights)          # author-time seed (fresh; allocates paths once)

    @test seed isa LightSnapshot               # concrete eltype (no Any[])
    @test eltype(seed) === Tuple{String,Union{LightState,Nothing}}

    _lights_snapshot(lights, seed)             # warmup: force compilation before measuring
    GC.gc()
    a1 = @allocated _lights_snapshot(lights, seed)
    GC.gc()
    a2 = @allocated _lights_snapshot(lights, seed)

    # Measured ≈1088 B for this 3-light scene (three boxed LightState, Julia 1.12.6) vs
    # the pre-L2 6048 B (Dict + fresh path strings + Matrix temporaries in Any[]).  The
    # 4096 B ceiling is DELIBERATELY generous (~3.8× the measurement): Julia minor
    # versions shift Union-field/tuple boxing and small-array layout, and the bound's job
    # is only to trip a revert to the Dict/strings/Matrix path — which still lands well
    # above it at ~6048 B.
    @test a1 ≤ 4096
    @test a2 ≤ 4096

    # Reuse preserves the authored paths AND hands back the exact same String OBJECTS
    # (the point of the diet: zero fresh path allocation on an unchanged frame).
    reused = _lights_snapshot(lights, seed)
    @test [p for (p, _) in reused] == [p for (p, _) in seed]
    @test all(reused[i][1] === seed[i][1] for i in eachindex(seed))
end

@testset "L2 count change falls back to a fresh (correct) snapshot" begin
    three = _three_lights()
    two   = AbstractLight[three[1], three[2]]
    seed3 = _lights_snapshot(three)

    # old-snapshot count ≠ current light count → the reuse form must NOT graft mismatched
    # paths; it rebuilds fresh so sync_lights!'s count-mismatch guard still fires.
    snap2 = _lights_snapshot(two, seed3)
    @test length(snap2) == 2
    @test [p for (p, _) in snap2] == [p for (p, _) in _lights_snapshot(two)]
end

@testset "L2 same-count type swap rebuilds fresh paths (no stale index-wise reuse)" begin
    # Prim paths encode <Type>_<idx>. A constant-count TYPE SWAP must NOT keep the old paths
    # and pair them with the new states — that writes each state to the WRONG prim, silently
    # corrupting both. Path reuse is valid ONLY when every position's type still matches its
    # old path; a swap must fall back to the fresh (recomputed-path) build. (Regression guard
    # for the L2 review finding: pre-L2 rebuilt paths fresh every frame, so this rendered
    # correctly; blind index-wise reuse introduced the corruption.)
    dp = AbstractLight[
        DirectionalLight(RGBf(1f0, 1f0, 1f0), Vec3f(-1f0, -1f0, -1f0), false),
        PointLight(RGBf(1f0, 0f0, 0f0), Vec3f(0f0, 0f0, 100f0), Vec2f(0f0, 0f0)),
    ]
    seed_dp = _lights_snapshot(dp)
    @test [p for (p, _) in seed_dp] == ["/World/DirectionalLight_0", "/World/PointLight_0"]

    pd      = AbstractLight[dp[2], dp[1]]            # [Point, Directional] — same count, swapped
    swapped = _lights_snapshot(pd, seed_dp)          # two-arg reuse form (the code under test)
    fresh   = _lights_snapshot(pd)                   # ground truth: recomputed paths

    # The swap must produce the FRESH paths (PointLight_0 now at position 1), NOT the stale
    # index-wise reuse (which would keep [DirectionalLight_0, PointLight_0]).
    @test [p for (p, _) in swapped] == [p for (p, _) in fresh]
    @test [p for (p, _) in swapped] == ["/World/PointLight_0", "/World/DirectionalLight_0"]
    @test [p for (p, _) in swapped] != [p for (p, _) in seed_dp]

    # Each state is paired with the CORRECTLY-named prim: the Point light's state at position 1
    # is written to PointLight_0 (not the stale DirectionalLight_0), the Directional's to _0.
    @test swapped[1] == ("/World/PointLight_0",      _light_render_state(pd[1]))
    @test swapped[2] == ("/World/DirectionalLight_0", _light_render_state(pd[2]))

    # Happy path (types unchanged): still reuses the exact String OBJECTS (===) — zero fresh
    # path allocation, the steady-state win the type-swap guard must not regress.
    reused = _lights_snapshot(dp, seed_dp)
    @test [p for (p, _) in reused] == [p for (p, _) in seed_dp]
    @test all(reused[i][1] === seed_dp[i][1] for i in eachindex(seed_dp))
end

@testset "L2 LightState comparison detects intensity/color/direction changes" begin
    base = DirectionalLight(RGBf(1f0, 1f0, 1f0), Vec3f(-1f0, -1f0, -1f0), false)
    dim  = DirectionalLight(RGBf(0.5f0, 0.5f0, 0.5f0), Vec3f(-1f0, -1f0, -1f0), false) # intensity
    hue  = DirectionalLight(RGBf(1f0, 0f0, 0f0), Vec3f(-1f0, -1f0, -1f0), false)        # color
    turn = DirectionalLight(RGBf(1f0, 1f0, 1f0), Vec3f(1f0, 0f0, 0f0), false)           # direction

    sb = _light_render_state(base)
    @test sb isa LightState
    @test sb.intensity != _light_render_state(dim).intensity   # intensity change detected
    @test sb.color     != _light_render_state(hue).color       # color change detected
    @test sb.xform     != _light_render_state(turn).xform      # direction (xform) change detected
    @test _light_render_state(base) == sb                      # unchanged → identical (no write)

    # DomeLight (AmbientLight) carries no xform; exotic types stay baked (no live state).
    @test _light_render_state(AmbientLight(RGBf(0.5f0, 0.5f0, 0.5f0))).xform === nothing
    rect = RectLight(RGBf(1f0, 1f0, 1f0), Point3f(0f0, 0f0, 0f0),
                     Vec3f(4f0, 0f0, 0f0), Vec3f(0f0, 5f0, 0f0), Vec3f(-1f0, -1f0, -1f0))
    @test _light_render_state(rect) === nothing
end
