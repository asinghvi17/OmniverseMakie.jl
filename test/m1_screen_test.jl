using Test

# M1.1 subprocess: load OmniverseMakie, activate!, build a Screen for a fresh Scene,
# assert size and active backend, then close cleanly.
const _M1_SCREEN_PROG = """
using OmniverseMakie

OmniverseMakie.activate!()

scene = Scene(size = (800, 600))
screen = OmniverseMakie.Screen(scene)
@assert size(screen) == (800, 600) "expected (800, 600), got \$(size(screen))"
@assert Makie.current_backend() === OmniverseMakie "wrong backend: \$(Makie.current_backend())"

close(screen)
println("OK")
"""

@testset "M1.1 Screen lifecycle (subprocess)" begin
    exitcode, output = run_ovrtx_subprocess(_M1_SCREEN_PROG)
    @test exitcode == 0
    @test contains(output, "OK")
end
