#!/bin/bash
# Start the 4-channel PCB inspection demo. Run `bash setup.sh` once first.
#
#   1/2/3  switch the live cell's model     p  pause/resume every channel
#   f      fullscreen toggle                q  quit (or click the X)
#
# Verification mode:  bash run.sh --headless --duration 10 --switch-test --save-dir demo_out
# No camera:          bash run.sh --no-live
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")" && pwd -P)"
. "$APP_DIR/_python.sh"

PY="$(dx_python)" || {
    echo "ERROR: no Python here can import dx_engine. Run: bash $APP_DIR/setup.sh"
    exit 1
}

for f in models/ch1_deeppcb_yolo26n.dxnn models/ch2_patchcore_wrn50.dxnn \
         models/ch3_soldef_yolo26m.dxnn vendor/common vendor/patchcore data; do
    [ -e "$APP_DIR/$f" ] || { echo "ERROR: $f is missing -- run: bash $APP_DIR/setup.sh"; exit 1; }
done

cd "$APP_DIR"
exec "$PY" grid_viewer.py "$@"
