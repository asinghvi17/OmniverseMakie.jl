using Test
# Track-C Task C3 — GPU-direct present!: FUSED oriented tonemap kernel + cached device/texture
# buffers.  Proves (subprocess, CUDA [+ offscreen GL + ovrtx]):
#   1. ORIENTED AGREEMENT — the fused device kernel (out[j, H+1-i]) matches host
#      reverse(permutedims(tonemap_frame(hdr, ev)), dims=2) == the C2 `_tonemap_orient!` twin,
#      on the same HDR input (a deterministic case + a random HDR-range sweep), within
#      ≤1 LSB PER BYTE.  Exact byte-equality is NOT cross-GPU-portable: the tonemap's sole
#      transcendental is the sRGB `c^(1/2.4)` (host libm `powf` vs device `__nv_powf` are
#      independent implementations, contractually ≤ a few ULP apart), and a 1-ULP pow
#      difference flips the output byte whenever a pixel lands within ~1/510 of a rounding
#      boundary.  On the A5000 the diff happens to be exactly 0; the gate allows ≤1 so a
#      different GPU/driver/CUDA toolkit stays green while any REAL kernel/orientation bug
#      (large, structured diffs) still fails.
#   2. ZERO STEADY-STATE DEVICE ALLOCATION — a warmed GPU present! allocates 0 bytes of CUDA
#      device memory per tick (cached oriented buffer + cached GL Texture; no permutedims/reverse
#      temporaries, no per-frame plot2robjs/uniforms Dict), measured with CUDA.@allocated (which
#      tracks the CUDA.jl memory pool — the ONLY pool allocation the old chain made was the 3
#      display buffers this fix caches away; ovrtx's own device memory is invisible to it).  A
#      GENUINE GPU path is asserted (blitter==:gpu, gpu_state set) so a silent CPU fallback —
#      which also allocates 0 device bytes — cannot pass the gate vacuously.
const _C3_PRESENT_PROG = """
using OmniverseMakie, GLMakie, CUDA, ColorTypes, FixedPointNumbers
OM = OmniverseMakie
using OmniverseMakie: tonemap_frame
Ext   = Base.get_extension(OmniverseMakie, :OmniverseMakieCUDAExt)
ExtGL = Base.get_extension(OmniverseMakie, :OmniverseMakieGLMakieExt)
println("EXT_LOADED=", Ext !== nothing && ExtGL !== nothing)
println("CUDA_FUNCTIONAL=", CUDA.functional())

# 1. ORIENTED-KERNEL AGREEMENT (no window): fused device kernel vs host orient chain vs twin,
#    max PER-BYTE deviation (device powf may differ ≤1 ULP from host libm — see header).
bytedev(a, b) = maximum(max(abs(Int(reinterpret(UInt8, x.r)) - Int(reinterpret(UInt8, y.r))),
                            abs(Int(reinterpret(UInt8, x.g)) - Int(reinterpret(UInt8, y.g))),
                            abs(Int(reinterpret(UInt8, x.b)) - Int(reinterpret(UInt8, y.b))),
                            abs(Int(reinterpret(UInt8, x.alpha)) - Int(reinterpret(UInt8, y.alpha))))
                        for (x, y) in zip(a, b))
ev  = 0.5f0
hdr = Float32[c==1 ? 0.5f0 : (c==2 ? 0.1f0 : 2.0f0) for c in 1:4, x in 1:8, y in 1:6]  # [C,W,H]
dev_or  = Ext.tonemap_oriented_kernel_to_matrix(CuArray(hdr), ev)             # [W,H]
host_or = reverse(permutedims(tonemap_frame(hdr, ev)), dims = 2)              # [W,H]
println("ORIENT_SIZE=", size(dev_or) == size(host_or) == (8, 6))
println("ORIENT_MAXDEV=", bytedev(dev_or, host_or))
twin = Matrix{RGBA{N0f8}}(undef, 8, 6)
ExtGL._tonemap_orient!(twin, hdr, ev)                                         # the C2 host twin
println("ORIENT_VS_TWIN_MAXDEV=", bytedev(dev_or, twin))
using Random; Random.seed!(11)
hdr2  = rand(Float32, 4, 96, 72) .* 6.0f0
dev2  = Ext.tonemap_oriented_kernel_to_matrix(CuArray(hdr2), 0.0f0)
host2 = reverse(permutedims(tonemap_frame(hdr2, 0.0f0)), dims = 2)
println("ORIENT_MAXDEV_RAND=", bytedev(dev2, host2))

# 2. ZERO STEADY-STATE DEVICE ALLOCATION (window + ovrtx): a warmed GPU present! -> 0 device bytes.
OM.activate!(warmup = 16)
fig = Figure(); ax = LScene(fig[1,1]; show_axis = false)
mesh!(ax, Rect3f(Point3f(-1), Vec3f(2)); color = :orange)
session = OM.interactive_display(fig; size = (320, 240), gpu_direct = true)   # force GPU-direct
# Single-task / single-context: manual present! is the SOLE driver (mirrors m6_bench_test).
off(session.tick_listener)
session.glscreen.close_after_renderloop = false
GLMakie.stop_renderloop!(session.glscreen)
yield()
session.gpu_state === nothing || OM.gpu_unregister!(session)
for _ in 1:4; OM.present!(session, Val(:gpu)); end        # warm: JIT kernel + register + first oriented alloc
@assert session.blitter == :gpu "GPU present! fell back to CPU — the device-alloc gate would be vacuous"
@assert session.gpu_state !== nothing "GPU present! did not register the GL texture"
println("ALLOC_BLITTER=", session.blitter)
println("ALLOC_STATE_SET=", session.gpu_state !== nothing)
# The cached oriented device buffer holds a real (non-black) RTX frame.
ori = Array(session.gpu_state.oriented)
println("ORIENTED_NONBLACK=", count(c -> (Float32(red(c))+Float32(green(c))+Float32(blue(c)))>0.1, ori))
# Steady-state device allocation per GPU present! tick (two measurements — both must be 0).
a1 = CUDA.@allocated OM.present!(session, Val(:gpu))
a2 = CUDA.@allocated OM.present!(session, Val(:gpu))
println("DEV_ALLOC1=", a1)
println("DEV_ALLOC2=", a2)
OM.close(session)
println("OK_C3_PRESENT")
"""

include(joinpath(@__DIR__, "..", "helpers.jl"))
@testset "C3 GPU present! — fused oriented kernel + zero steady-state device alloc (subprocess, CUDA+GL)" begin
    # ovrtx has a known INTERMITTENT startup crash (GeometryGroup::attachToContext) that can kill
    # the child before it renders; retry until it reaches the allocation measurement so the hard
    # @tests aren't flaky on that crash.
    _, out = run_ovrtx_subprocess(_C3_PRESENT_PROG; timeout = 600, retries = 4,
                                  ready_marker = "DEV_ALLOC1=")
    contains(out, "OK_C3_PRESENT") || @info "C3 present output" out
    @test contains(out, "EXT_LOADED=true")
    @test contains(out, "CUDA_FUNCTIONAL=true")
    @test contains(out, "OK_C3_PRESENT")                 # subprocess completed all work (no mid-run death)

    # 1. Oriented kernel ≈ host orient chain ≈ the _tonemap_orient! twin: every byte within
    #    ≤1 LSB (device __nv_powf vs host libm powf; exact 0 on the A5000 but not portable —
    #    a real kernel/orientation bug produces large, structured deviations).
    @test contains(out, "ORIENT_SIZE=true")              # oriented output is [W,H]
    for key in ("ORIENT_MAXDEV", "ORIENT_VS_TWIN_MAXDEV", "ORIENT_MAXDEV_RAND")
        m = match(Regex("$key=(\\d+)"), out)
        @test m !== nothing && parse(Int, m.captures[1]) <= 1
    end

    # 2. Genuine GPU path (no vacuous pass via CPU fallback) + a real frame reached the buffer.
    @test contains(out, "ALLOC_BLITTER=gpu")
    @test contains(out, "ALLOC_STATE_SET=true")
    m_nb = match(r"ORIENTED_NONBLACK=(\d+)", out)
    @test m_nb !== nothing && parse(Int, m_nb.captures[1]) > 1000

    # 3. No FRAME-SCALED device allocation per warmed GPU present! tick (the C3 gate).
    #    Measured 0 bytes on CUDA.jl 6.2.0; the ≤256-byte allowance tolerates incidental
    #    pool bookkeeping a future CUDA.jl might add (unsafe_wrap / launch config / events)
    #    while still failing the pre-fix chain, which allocated three full display buffers
    #    (hundreds of KB at 320×240, MBs at 4K) every tick.
    m1 = match(r"DEV_ALLOC1=(\d+)", out)
    m2 = match(r"DEV_ALLOC2=(\d+)", out)
    @test m1 !== nothing && m2 !== nothing
    if m1 !== nothing && m2 !== nothing
        @test parse(Int, m1.captures[1]) <= 256
        @test parse(Int, m2.captures[1]) <= 256
    end
end
