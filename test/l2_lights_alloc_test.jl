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

    # Measured ≈1088 B for this 3-light scene (three boxed LightState) vs the pre-L2
    # 6048 B (Dict + fresh path strings + Matrix temporaries in Any[]). 2048 B leaves
    # struct-layout headroom while staying ~3× under the pre-L2 baseline, so a revert to
    # the Dict/strings/Matrix path trips it.
    @test a1 ≤ 2048
    @test a2 ≤ 2048

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
