#!/usr/bin/env bash
# Full-color NVIDIA IndeX volume rendering through a Kit runtime — the thing
# standalone ovrtx cannot do (IndeX Direct is grayscale scalar-only; see
# README.md).  Renders the composite ext's torus.vdb twice (gray vs colored
# transfer function), captures both headlessly, and chroma-checks that the
# colormap is actually honored.
#
# Requires a built Kit app with the IndeX composite extension chain resolved
# (default: the DSX blueprint's kit-cae build) and its extension cache
# materialized under ~/.local/share/ov/data/exts/v2.  GPU renders serialize
# on the shared lock like every other job in this repo.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
KIT_RELEASE_DIR="${KIT_RELEASE_DIR:-$HOME/temp/omniverse-dsx-blueprint-for-ai-factories/deps/kit-cae/_build/linux-x86_64/release}"
WORK="${WORK:-$(mktemp -d)}"

[ -x "$KIT_RELEASE_DIR/kit/kit" ] || {
    echo "no kit runtime at KIT_RELEASE_DIR=$KIT_RELEASE_DIR" >&2; exit 1; }

# The sample volume ships inside the composite ext (hash suffix varies per box).
VDB=$(ls "$HOME"/.local/share/ov/data/exts/v2/omni.rtx.index_composite-*/data/tests/volumes/torus.vdb 2>/dev/null | head -1)
[ -n "$VDB" ] || { echo "omni.rtx.index_composite ext (torus.vdb) not in the ext cache" >&2; exit 1; }

# Stage variants: identical but for the transfer function.
sed "s|@VDB_PATH@|@$VDB@|" "$HERE/torus_colormap.usda.in" > "$WORK/torus_color.usda"
sed -e 's|custom float4\[\] rgbaPoints = .*|custom float4[] rgbaPoints = [(0.27, 0.27, 0.27, 0), (0.63, 0.63, 0.63, 0.32), (0.5, 0.5, 0.5, 0.5)]|' \
    -e 's|custom float\[\] xPoints = .*|custom float[] xPoints = [0, 0.07, 1]|' \
    "$WORK/torus_color.usda" > "$WORK/torus_gray.usda"

# Kit's RTX scene renderer dlopens the MDL SDK (libneuray.so), which needs
# libGLU.so.1 + GLVND.  Missing system-wide -> extract locally, no root.
# (Symptom without it: "Failed to add Hydra engine ... Invalid sync scope".)
if ! ldconfig -p | grep -q "libGLU.so.1"; then
    if [ ! -f "$WORK/libglu/usr/lib/x86_64-linux-gnu/libGLU.so.1" ]; then
        echo "libGLU.so.1 missing system-wide; extracting locally into $WORK/libglu"
        mkdir -p "$WORK/libglu" && cd "$WORK/libglu"
        apt-get download libglu1-mesa libopengl0 libglvnd0
        for f in *.deb; do dpkg-deb -x "$f" .; done
        cd - >/dev/null
    fi
    export LD_LIBRARY_PATH="$WORK/libglu/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export PROBE_JOBS="$WORK/torus_gray.usda>$WORK/out_gray.png;$WORK/torus_color.usda>$WORK/out_color.png"
export PROBE_WARM_FRAMES="${PROBE_WARM_FRAMES:-240}"

# Minimal from-parts launch: the bare kit kernel + only the extensions the
# composite path needs (deps auto-resolve from the app's ext folders).  The
# full editor app works too (`./omni.cae.kit.sh --no-window ...`) but takes
# longer and its teardown stalls headless.  The rtx.index.overrideSubdivision*
# pair mirrors the CAE app's "Necessary for IndeX" block (without it the
# volume renders with heavy banding).
cd "$KIT_RELEASE_DIR"
flock -w 3600 /tmp/omniversemakie-gpu.lock -c "timeout 900 \
    ./kit/kit --empty --no-window \
    --ext-folder exts --ext-folder extscache \
    --enable omni.kit.mainwindow \
    --enable omni.kit.viewport.window \
    --enable omni.kit.viewport.utility \
    --enable omni.hydra.rtx \
    --enable omni.rtx.index_composite \
    --enable omni.kit.exec.core \
    --/app/asyncRendering=false \
    --/omni.kit.plugin/syncUsdLoads=true \
    --/rtx/index/compositeEnabled=true \
    --/rtx/index/overrideSubdivisionMode=\"kd_tree\" \
    --/rtx/index/overrideSubdivisionPartCount=1 \
    --exec $HERE/probe.py" 2>&1 | tee "$WORK/kit.log" | grep -aE "\[probe\]|PROBE_OK|PROBE_FAIL" || true

grep -aq "PROBE_OK" "$WORK/kit.log" || {
    echo "probe failed; full log: $WORK/kit.log" >&2; exit 1; }

julia --project="$REPO/examples" "$HERE/analyze.jl" "$WORK/out_gray.png" "$WORK/out_color.png"
echo "frames: $WORK/out_gray.png $WORK/out_color.png"
