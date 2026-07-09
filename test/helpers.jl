# Shared subprocess test harness for OmniverseMakie.
# Provides run_ovrtx_subprocess plus shared scaffolding: the ovrtx-startup
# retry loop, the IndeX-libs default, and the pixel-inspection prelude.

const _HELPER_OVRTX_LIB = get(ENV, "OVRTX_LIBRARY_PATH", "")
const _HELPER_REPO_ROOT = joinpath(@__DIR__, "..")
const _HELPER_USDA = get(ENV, "OM_USDA",
    joinpath(@__DIR__, "fixtures", "ovrtx_feature_scene.usda"))
# NVIDIA IndeX ext libs dir.  Env-overridable; default is the dev-box location.
# Volume subprocess tests forward this via `env=` and skip when it's absent.
const _HELPER_INDEX_LIBS = get(ENV, "OMNIVERSEMAKIE_INDEX_LIBS",
    "/home/juliahub/.local/share/ov/data/exts/v2/omni.index.libs-1287db94366cf6fe")

"""
    PROG_PIXEL_HELPERS :: String

Prelude spliced into subprocess progs (`\$(PROG_PIXEL_HELPERS)`) that inspect
a rendered `Makie.colorbuffer`.  Defines the shared lit-pixel helpers and
thresholds:

  * `lum(c)`            — r+g+b luminance of one pixel;
  * `nonblack(img)`     — count of strictly non-black pixels (any channel > 0);
  * `lit_centroid(img)` — `(H, nb, crow, ccol)`: image height, lit-pixel
    count, and the mean (row, col) of pixels brighter than `LUM_MIN`.  Row
    INCREASES DOWNWARD (top-left origin), so world −Z projects to a LARGER
    row;
  * `region_nonblack(img, c0, c1)` — count of pixels with `lum > LUM_MIN`
    over columns [c0,c1];
  * `region_red(img, c0, c1)`      — red-DOMINANT pixel count
    (r > g+0.15 && r > b+0.15) over columns [c0,c1];
  * `region_diff(a, b, c0, c1)`    — mean PER-CHANNEL abs difference over
    columns [c0,c1] of two same-size images (per-channel so it is not blind
    to equal-luminance swaps like red↔blue);
  * `LUM_MIN = 0.04f0`  — r+g+b luminance above which a pixel counts as "lit".

Loops are wrapped in FUNCTIONS: progs run at top level, where Julia's
soft-scope rules break bare `for`-loop accumulators.  Pixels are read via
`.r/.g/.b` fields (a real `RGBA` colorbuffer element), so no
`using ColorTypes` is required in the prog.
"""
const PROG_PIXEL_HELPERS = raw"""
const LUM_MIN = 0.04f0
lum(c) = Float32(c.r) + Float32(c.g) + Float32(c.b)
nonblack(img) = count(c -> (Float32(c.r) + Float32(c.g) + Float32(c.b)) > 0, img)
function lit_centroid(img)
    H, W = size(img)
    sr = 0.0; sc = 0.0; nb = 0
    for h in 1:H, w in 1:W
        cc = img[h, w]
        if (Float32(cc.r) + Float32(cc.g) + Float32(cc.b)) > LUM_MIN
            sr += h; sc += w; nb += 1
        end
    end
    return (H = H, nb = nb, crow = nb > 0 ? sr / nb : -1.0, ccol = nb > 0 ? sc / nb : -1.0)
end
function region_nonblack(img, c0, c1)
    H, W = size(img); n = 0
    for h in 1:H, w in c0:min(c1, W)
        lum(img[h, w]) > LUM_MIN && (n += 1)
    end
    return n
end
function region_red(img, c0, c1)
    H, W = size(img); n = 0
    for h in 1:H, w in c0:min(c1, W)
        c = img[h, w]
        Float32(c.r) > Float32(c.g) + 0.15f0 && Float32(c.r) > Float32(c.b) + 0.15f0 && (n += 1)
    end
    return n
end
function region_diff(a, b, c0, c1)
    H, W = size(a); s = 0.0; n = 0
    for h in 1:H, w in c0:min(c1, W)
        p = a[h, w]; q = b[h, w]
        s += abs(Float32(p.r) - Float32(q.r)) + abs(Float32(p.g) - Float32(q.g)) +
             abs(Float32(p.b) - Float32(q.b)); n += 1
    end
    return n == 0 ? 0.0 : s / n
end
"""

# X credentials for GLMakie-tier subprocesses.  ENV wins; else use the newest
# `.mutter-Xwaylandauth.*` cookie — mutter regenerates it each Wayland login,
# so a hardcoded path rots on the next reboot and kills every GL/CUDA test
# with an opaque GLFW init error.  No cookie at all: warn once and forward
# nothing (offscreen tests still pass; GL tests fail fast at GLFW init).
const _HELPER_RUNTIME_DIR = get(ENV, "XDG_RUNTIME_DIR", "/run/user/$(Base.Libc.getuid())")
function _discover_xauthority()
    haskey(ENV, "XAUTHORITY") && return ENV["XAUTHORITY"]
    candidates = filter(isfile,
        [joinpath(_HELPER_RUNTIME_DIR, f) for f in
         (isdir(_HELPER_RUNTIME_DIR) ? readdir(_HELPER_RUNTIME_DIR) : String[])
         if startswith(f, ".mutter-Xwaylandauth.")])
    # newest cookie
    isempty(candidates) || return candidates[argmax(mtime.(candidates))]
    @warn """no Xwayland auth cookie found under $(_HELPER_RUNTIME_DIR) — GLMakie-tier
             subprocess tests will fail at GL context creation.  Export DISPLAY and
             XAUTHORITY (and XDG_RUNTIME_DIR if non-standard) to fix.""" maxlog = 1
    return nothing
end
const _HELPER_XAUTHORITY = _discover_xauthority()

# One attempt of `run_ovrtx_subprocess`: write `prog` to a temp `.jl`, run as
# a child `julia --project=<repo>` under the standard ovrtx env, reap with the
# truthful-exit watchdog, and return `(exitcode, output)`.  The public wrapper
# below adds the `ready_marker` retry loop.
function _run_ovrtx_once(prog::String; timeout::Int, kill_grace::Real, env)
    script = tempname() * ".jl"
    exitcode = -1
    output = ""
    try
        open(script, "w") do io
            print(io, prog)
        end
        # Stack test/ project after the main project so GLMakie (and other
        # test weakdeps) are loadable without being hard deps of
        # OmniverseMakie itself.
        _test_dir = joinpath(_HELPER_REPO_ROOT, "test")
        # "@" = active project (from --project); _test_dir = test env
        # (has GLMakie)
        _load_path = join(["@", _test_dir, "@v#.#", "@stdlib"], ":")
        env_pairs = Pair{String,String}[
            "OM_USDA"                   => _HELPER_USDA,
            "PATH"                      => get(ENV, "PATH", ""),
            "HOME"                      => get(ENV, "HOME", ""),
            # GLMakie needs a real X display for GL context creation.  Forward
            # the Xwayland display vars: ENV first, else the discovered newest
            # mutter cookie (see _discover_xauthority above).
            "DISPLAY"                   => get(ENV, "DISPLAY", ":0"),
            "XDG_RUNTIME_DIR"           => _HELPER_RUNTIME_DIR,
            # GLMakie cannot precompile without a display available.  AUTO=0
            # skips startup auto-precompile so the subprocess loads
            # OmniverseMakie cleanly.
            "JULIA_PKG_PRECOMPILE_AUTO" => "0",
            # When CUDA.jl is loaded it dlopens the forward-compat libcuda
            # from CUDA_Driver_jll, colliding with the system driver used by
            # ovrtx's carb.cudainterop → createDevices fails with "driver API
            # result: 3" (CUDA_ERROR_NOT_INITIALIZED).  Force the system
            # driver so both share one libcuda; no-op without CUDA.jl.
            "JULIA_CUDA_USE_COMPAT"     => "false",
            # Stack the test project so GLMakie (a weakdep of OmniverseMakie)
            # is loadable by subprocess programs that do `using GLMakie`.
            "JULIA_LOAD_PATH"           => _load_path,
        ]
        isempty(_HELPER_OVRTX_LIB) ||
            push!(env_pairs, "OVRTX_LIBRARY_PATH" => _HELPER_OVRTX_LIB)
        # Only forward XAUTHORITY when a cookie actually exists (ENV or
        # discovered) — forwarding a dead path produces an opaque GL error a
        # minute later instead of GLFW's clear init failure.
        _HELPER_XAUTHORITY === nothing ||
            push!(env_pairs, "XAUTHORITY" => _HELPER_XAUTHORITY)
        # Forward caller-supplied env pairs (e.g. OMNIVERSEMAKIE_INDEX_LIBS):
        # a single `name=>value` Pair or an iterable of Pairs.  They append to
        # (and can override) the base set; anything else is absent from the
        # child.
        for (k, v) in (env isa Pair ? (env,) : env)
            push!(env_pairs, String(k) => String(v))
        end
        cmd = setenv(`julia --project=$(_HELPER_REPO_ROOT) $script`, env_pairs)
        out = IOBuffer()
        err = IOBuffer()
        p = run(pipeline(cmd; stdout=out, stderr=err); wait=false)
        # Watchdog: wait up to `timeout` s for a clean exit; timedwait returns
        # :timed_out on expiry.  Then escalate SIGTERM → grace → SIGKILL so a
        # wedged child that swallows SIGTERM is still reaped.
        timed_out = timedwait(() -> !process_running(p), float(timeout)) === :timed_out
        if timed_out && process_running(p)
            kill(p)                                   # SIGTERM: ask it to stop
            if timedwait(() -> !process_running(p), float(kill_grace)) === :timed_out
                # still alive: force it
                process_running(p) && kill(p, Base.SIGKILL)
            end
        end
        wait(p)  # ALWAYS reap before reading exit state/output
        # Truthful exit code (table in the docstring): 0 only on real success;
        # watchdog timeout → -124 (real signals stop at 64, so no collision
        # with -signal); uncaught signal N → -N; else the child's own code.
        exitcode = success(p) ? 0 :
                   timed_out  ? -124 :
                   p.termsignal != 0 ? -p.termsignal : p.exitcode
        output = String(take!(out))
        errtext = String(take!(err))
        isempty(errtext) || @info "subprocess stderr" text=errtext
    finally
        isfile(script) && rm(script)
    end
    return (exitcode, output)
end

"""
    run_ovrtx_subprocess(prog::String; timeout=300, kill_grace=10, env=(),
                         retries=1, ready_marker="") -> (exitcode, output)

Write `prog` to a temp `.jl` file, run it as a child `julia --project=<repo>`
process with the standard ovrtx environment, wait up to `timeout` seconds for
a clean exit, collect stdout, and return `(exitcode, stdout_text)`.  The temp
file is always removed in a `finally` block.

On timeout the watchdog escalates SIGTERM → wait `kill_grace` s → SIGKILL (so
even a GPU-wedged child that ignores SIGTERM is reaped), then ALWAYS `wait`s
the child before reading its exit code and output.  The returned `exitcode`
is therefore TRUTHFUL — callers can assert `exitcode == 0`:

  * `0`       — the child exited 0 (real success);
  * `-124`    — the WATCHDOG timed out and killed it (negated GNU-timeout
                convention; distinct from `-9`, which means something ELSE
                SIGKILLed the child, e.g. the OOM killer);
  * `-N`      — the child died from uncaught signal `N` (e.g. -11 segfault);
  * otherwise — the child's own nonzero exit code.

`env` forwards extra `name => value` pairs (a single `Pair` or an iterable of
Pairs) into the child process — used for opt-in vars such as
`OMNIVERSEMAKIE_INDEX_LIBS`.  `setenv` REPLACES the child environment, so vars
neither in the base env set nor forwarded via `env` do NOT leak into the child
(the disabled-path volume test relies on this to exercise the "no volume env"
branch).

The subprocess LOAD_PATH stacks the test project (which has GLMakie in [deps])
after the main project (which has it only in [weakdeps]).  This lets
subprocess programs do `using OmniverseMakie, GLMakie` — GLMakie is found via
the test project — while `using OmniverseMakie` alone still doesn't load it.

`retries` / `ready_marker`: re-run `prog` until `ready_marker` appears in its
stdout, up to `retries` attempts, returning the LAST attempt's
`(exitcode, output)`.  The default `retries=1` / empty `ready_marker` is a
single run (an empty marker short-circuits the loop).  Retries absorb ovrtx's
intermittent pre-render startup crash (see src/binding/signals.jl).

CONVENTION — pick an EARLY marker, not the final OK sentinel: the marker
should be the first line the child prints AFTER its first successful render,
BEFORE any in-child pixel `@assert`.  Then only a startup death is retried; a
NOISE-MARGINAL pixel assert that fails after the marker fails ONCE, honestly.
With a final-OK marker the retry loop would re-roll a flaky assert until it
passes — converting real flakiness into a silent green.  (Progs with no
in-child asserts, where the parent judges printed values, may keep a final OK
marker.)
"""
function run_ovrtx_subprocess(prog::String; timeout::Int=300, kill_grace::Real=10, env=(),
                              retries::Int=1, ready_marker::AbstractString="")
    exitcode = -1
    output = ""
    for _ in 1:retries
        exitcode, output = _run_ovrtx_once(prog; timeout, kill_grace, env)
        (isempty(ready_marker) || contains(output, ready_marker)) && break
    end
    return (exitcode, output)
end
