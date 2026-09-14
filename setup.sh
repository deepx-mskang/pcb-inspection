#!/bin/bash
# One-time bootstrap for a fresh machine.
#   OS packages -> DEEPX runtime (dx_rt) -> Python packages -> demo assets
# After this finishes, start the demo with:  bash run.sh
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")" && pwd -P)"
. "$APP_DIR/_python.sh"

echo "[1/4] OS packages"
# libgl1 / libglib2.0-0 are the shared libs OpenCV and Qt load at import time
OS_PKGS="git python3-pip python3-venv libgl1 libglib2.0-0"
if ! command -v apt-get >/dev/null 2>&1; then
    echo "      apt-get not found -- install python3-pip plus the OpenGL/glib runtime libs yourself"
else
    need=""
    for p in $OS_PKGS; do dpkg -s "$p" >/dev/null 2>&1 || need="$need $p"; done
    if [ -z "$need" ]; then
        # Nothing to do -- so re-running setup.sh on a configured machine never needs sudo.
        echo "      already installed"
    else
        echo "     $need (needs sudo)"
        sudo apt-get update -qq
        sudo apt-get install -y -qq $need
        echo "      ok"
    fi
fi

echo "[2/4] DEEPX runtime (dx_engine + dxrt-cli)"
if PY="$(dx_python)"; then
    # Never reinstall over a working runtime -- a stray dx_rt install has
    # clobbered libdxrt-bin on this project before.
    echo "      ok -- $PY"
else
    DXRT="${DX_RUNTIME_DIR:-}"
    if [ -z "$DXRT" ]; then
        d="$APP_DIR"
        while [ "$d" != "/" ]; do
            [ -f "$d/dx-runtime/install.sh" ] && { DXRT="$d/dx-runtime"; break; }
            [ -f "$d/install.sh" ] && [ -d "$d/dx_rt" ] && { DXRT="$d"; break; }
            d="$(dirname "$d")"
        done
    fi
    if [ -z "$DXRT" ]; then
        # Nothing local to build from -- fetch the runtime. Submodules use
        # relative URLs, so they follow whichever protocol this clone used.
        command -v git >/dev/null 2>&1 || { echo "ERROR: git is required to fetch the runtime"; exit 1; }
        DXRT="$APP_DIR/dx-runtime"
        echo "      cloning ${DX_RUNTIME_URL:-https://github.com/DEEPX-AI/dx-runtime.git} (takes a while)"
        git clone --recursive "${DX_RUNTIME_URL:-https://github.com/DEEPX-AI/dx-runtime.git}" "$DXRT"
    fi
    echo "      building/installing from $DXRT (driver + dx_rt + firmware, needs sudo)"
    sudo -v
    bash "$DXRT/install.sh" --runtime-only
    PY="$(dx_python)" || {
        echo "ERROR: dx_engine still not importable after install -- check the log above."
        exit 1
    }
    echo "      ok -- $PY"
fi

# Record it: a copied-out install cannot find the dx-runtime venv by walking up.
echo "$PY" > "$APP_DIR/.dx_python"

echo "[3/4] Python packages -> $PY"
# A venv takes the packages directly; a system interpreter needs --user.
if "$PY" -c 'import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)'; then
    PIP_TARGET=""
else
    PIP_TARGET="--user"
fi
if ! "$PY" -m pip install $PIP_TARGET -q -r "$APP_DIR/requirements.txt"; then
    echo "ERROR: pip install failed. On PEP 668 systems (Ubuntu 24.04+) use a venv:"
    echo "  python3 -m venv $APP_DIR/venv && bash $0"
    exit 1
fi
echo "      ok"

echo "[4/4] demo assets"
# A distributed copy already carries these; this only fills them in when the
# session is rebuilt inside the dx-all-suite checkout.
SUITE=""
d="$APP_DIR"
while [ "$d" != "/" ]; do
    [ -d "$d/dx-runtime" ] && [ -d "$d/dx-compiler" ] && { SUITE="$d"; break; }
    d="$(dirname "$d")"
done
CH2="$SUITE/dx-runtime/dx_app/dx-agent-dev/20260825-224353_claude_opus5_patchcore_inference"
CH1C="$SUITE/dx-compiler/dx-agent-dev/20260831-092911_claude_opus5_yolo26n_deeppcb_compile"
CH3C="$SUITE/dx-compiler/dx-agent-dev/20260831-145134_claude_opus5_yolo26m_soldef_compile"

vendor() {  # vendor <dest> <source>
    [ -e "$APP_DIR/$1" ] && return 0
    [ -n "$SUITE" ] && [ -e "$2" ] || { echo "      MISSING: $1"; return 1; }
    mkdir -p "$(dirname "$APP_DIR/$1")"
    cp -r "$2" "$APP_DIR/$1"
    echo "      vendored $1"
}
models_missing=0
missing=0
vendor vendor/common "$SUITE/dx-runtime/dx_app/src/python_example/common" || missing=1
vendor vendor/patchcore "$CH2/patchcore" || missing=1
vendor vendor/preproc.py "$CH2/preproc.py" || missing=1
vendor models/bank_pcb1.npz "$CH2/bank_pcb1.npz" || missing=1
vendor models/bank_pcb1.meta.json "$CH2/bank_pcb1.meta.json" || missing=1
vendor models/ch1_deeppcb_yolo26n.dxnn "$CH1C/runs/deeppcb/weights/best_deepx_model/best.dxnn" || models_missing=1
vendor models/ch3_soldef_yolo26m.dxnn "$CH3C/runs/deeppcb/weights/best_deepx_model/best.dxnn" || models_missing=1
vendor models/ch2_patchcore_wrn50.dxnn "$CH2/model/wrn50_patchcore.dxnn" || models_missing=1
# Datasets: copy from a sibling compile session when this is an in-suite
# rebuild, otherwise rebuild them from the original public datasets. A copy
# unpacked from the distribution tar already has them, so neither path runs.
for ds in ch1_images:"$CH1C/dataset/images/val" ch3_images:"$CH3C/dataset/images/val"; do
    vendor "data/${ds%%:*}" "${ds#*:}" 2>/dev/null || true
done
need_fetch=""
for ds in ch1_images ch3_images ch2_visa; do
    [ -d "$APP_DIR/data/$ds" ] || need_fetch="$need_fetch --only $ds"
done
if [ -n "$need_fetch" ]; then
    echo "      downloading from the original public datasets:$need_fetch"
    "$PY" "$APP_DIR/fetch_datasets.py" $need_fetch || missing=1
fi
find "$APP_DIR/vendor" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
if [ "$models_missing" -ne 0 ]; then
    echo "      the .dxnn models are not in git (too large) -- they ship in the"
    echo "      distribution tar; unpack it over this directory, or copy models/ from a machine that has it"
fi
[ "$missing" -eq 0 ] && [ "$models_missing" -eq 0 ] && echo "      ok" \
    || echo "      WARNING: assets above are missing -- the demo will not start"

echo
command -v dxrt-cli >/dev/null 2>&1 && dxrt-cli --status 2>&1 | grep -E "Device [0-9]+|FW version" | head -3 || true
# The NPU is reached through dxrtd; the demo fails with "dxrt service is not
# running" if it is down, and `disabled` means it will not come back on boot --
# which matters most on an unattended booth machine.
if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files dxrt.service >/dev/null 2>&1; then
    [ "$(systemctl is-active dxrt 2>/dev/null)" = "active" ] \
        || echo "      dxrt.service is NOT running -- start it:  sudo systemctl start dxrt"
    [ "$(systemctl is-enabled dxrt 2>/dev/null)" = "enabled" ] \
        || echo "      dxrt.service is NOT enabled at boot -- for a booth PC:  sudo systemctl enable dxrt"
fi
echo "setup complete. next: bash run.sh"
