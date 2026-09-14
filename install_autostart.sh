#!/bin/bash
# Start the demo automatically when the booth PC logs into its desktop.
#
#   bash install_autostart.sh              install
#   bash install_autostart.sh --uninstall  remove
#   bash install_autostart.sh --status     show what is installed
#
# Uses the XDG autostart directory rather than a systemd unit: this is a GUI
# app that needs the desktop session's DISPLAY, and ~/.config/autostart needs
# no sudo and no session-target wiring.
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")" && pwd -P)"
ENTRY="$HOME/.config/autostart/dx-pcb-demo.desktop"

case "${1:-}" in
--uninstall)
    rm -f "$ENTRY" && echo "removed $ENTRY"
    exit 0
    ;;
--status)
    if [ -f "$ENTRY" ]; then echo "installed: $ENTRY"; echo; cat "$ENTRY"
    else echo "not installed ($ENTRY)"; fi
    exit 0
    ;;
"") ;;
*)  echo "usage: $0 [--uninstall|--status]"; exit 2 ;;
esac

mkdir -p "$(dirname "$ENTRY")"
cat > "$ENTRY" <<EOF
[Desktop Entry]
Type=Application
Name=DX-M1 PCB AOI demo
Comment=4-channel PCB inspection demo on the DEEPX DX-M1 NPU
Exec=bash $APP_DIR/booth.sh
Path=$APP_DIR
Terminal=false
X-GNOME-Autostart-enabled=true
EOF

echo "installed $ENTRY"
echo "  runs: bash $APP_DIR/booth.sh   (supervised -- restarts on crash, honours a clean quit)"
echo "  log:  $APP_DIR/booth.log"
echo "  undo: bash $0 --uninstall"
