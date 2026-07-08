module OVRTX_jll

using Downloads
using Pkg.Artifacts
using SHA
using p7zip_jll

export artifact_dir, ovrtx_root, ovrtx_bin, libovrtx_dynamic

const _PKG_UUID = Base.UUID("1f3d2a3f-048b-4987-93b1-45f346dd13cd")

const _RELEASE_VERSION = "0.3.0.312915"
const _RELEASE_TAG = "v0.3.0"
const _RELEASE_BASE = "https://github.com/NVIDIA-Omniverse/ovrtx/releases/download/$(_RELEASE_TAG)"

function _asset_for_host()
    if Sys.islinux() && Sys.ARCH === :x86_64
        return (
            tree = Base.SHA1("b12569b960445ddaddce8779d60e224dc642cd17"),
            sha256 = "5569e44b18d2d39f23f374c9352dac9c87b8115892209c243c98732085f1d5f9",
            url = "$(_RELEASE_BASE)/ovrtx%40$(_RELEASE_VERSION).cec773e1.manylinux_2_35_x86_64.zip",
        )
    elseif Sys.islinux() && Sys.ARCH === :aarch64
        return (
            tree = Base.SHA1("02f5a1813034a0ae7c9a976a2243b911532b5b82"),
            sha256 = "c0236bac497f720c251485891d72b1a57e75ca38ad5aa2350adcd3d948e44056",
            url = "$(_RELEASE_BASE)/ovrtx%40$(_RELEASE_VERSION).cec773e1.manylinux_2_35_aarch64.zip",
        )
    elseif Sys.iswindows() && Sys.ARCH === :x86_64
        return (
            tree = Base.SHA1("82023550c188bb5aed522d0ba676540bcbab2546"),
            sha256 = "7fe420790fcd4c0a8609cadba73c7bb03a30fa47cd4ab7f130e3cf92a972063a",
            url = "$(_RELEASE_BASE)/ovrtx%40$(_RELEASE_VERSION).cec773e1.windows-x86_64.zip",
        )
    end
    error("OVRTX_jll: unsupported platform $(Sys.KERNEL)-$(Sys.ARCH). Official ovrtx $(_RELEASE_VERSION) C archives exist for Linux x86_64, Linux aarch64, and Windows x86_64.")
end

function _download_verify_extract!(dir::AbstractString, asset)
    mktempdir() do tmp
        archive = joinpath(tmp, "ovrtx.zip")
        Downloads.download(asset.url, archive)
        got = bytes2hex(open(sha256, archive))
        got == asset.sha256 ||
            error("OVRTX_jll: SHA-256 mismatch for $(asset.url): expected $(asset.sha256), got $got")
        if Sys.isunix()
            # ovrtx's Linux release archives contain many symlinks.  Info-ZIP
            # preserves them; 7z rejects several as "dangerous links".
            run(`unzip -q $(archive) -d $(dir)`)
        else
            run(`$(p7zip_jll.p7zip()) x -bd -y -o$(dir) $(archive)`)
        end
    end
    return nothing
end

function _ensure_ovrtx_artifact()
    asset = _asset_for_host()
    artifact_exists(asset.tree) && return artifact_path(asset.tree)
    actual = create_artifact() do dir
        _download_verify_extract!(dir, asset)
    end
    if actual != asset.tree
        remove_artifact(actual)
        error("OVRTX_jll: extracted artifact tree hash mismatch for $(asset.url): expected $(asset.tree), got $(actual)")
    end
    return artifact_path(asset.tree)
end

const artifact_dir = _ensure_ovrtx_artifact()
const ovrtx_root = artifact_dir

function _first_existing(paths::AbstractVector{<:AbstractString}, what::AbstractString)
    for path in paths
        ispath(path) && return path
    end
    error("OVRTX_jll: could not locate $what in artifact at $(artifact_dir). Tried: $(join(paths, ", "))")
end

const ovrtx_bin = _first_existing([
    joinpath(ovrtx_root, "bin"),
    joinpath(ovrtx_root, "ovrtx", "bin"),
], "the ovrtx bin directory")

const libovrtx_dynamic = _first_existing([
    joinpath(ovrtx_bin, "libovrtx-dynamic.so"),
    joinpath(ovrtx_bin, "ovrtx-dynamic.dll"),
], "libovrtx-dynamic")

end # module
