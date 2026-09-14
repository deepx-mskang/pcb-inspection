#!/bin/bash
# Booth supervisor: keep the demo on screen unattended.
#
# A clean quit (q / Esc / the X button) exits for good -- an operator closing
# the window should not have it reappear. A crash restarts, but only so many
# times in a row: a demo that dies instantly on every launch (missing model,
# wedged NPU) must stop and leave the error on screen instead of spinning.
#
#   bash booth.sh              supervise the demo
#   bash booth.sh --no-live    pass any run.sh argument straight through
set -uo pipefail
APP_DIR="$(cd "$(dirname "$0")" && pwd -P)"
LOG="${BOOTH_LOG:-$APP_DIR/booth.log}"
MAX_FAST_FAILS="${BOOTH_MAX_FAST_FAILS:-5}"   # consecutive crashes within MIN_UPTIME
MIN_UPTIME="${BOOTH_MIN_UPTIME:-30}"          # seconds; shorter than this counts as a fast fail

fast_fails=0
while true; do
    started=$(date +%s)
    echo "[booth] $(date '+%F %T') starting" | tee -a "$LOG"
    bash "$APP_DIR/run.sh" "$@" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    uptime=$(( $(date +%s) - started ))

    if [ "$rc" -eq 0 ]; then
        echo "[booth] $(date '+%F %T') clean exit -- not restarting" | tee -a "$LOG"
        exit 0
    fi
    if [ "$uptime" -ge "$MIN_UPTIME" ]; then
        fast_fails=0          # it ran a while, so this is a one-off crash
    else
        fast_fails=$((fast_fails + 1))
    fi
    if [ "$fast_fails" -ge "$MAX_FAST_FAILS" ]; then
        echo "[booth] $(date '+%F %T') exited $rc within ${uptime}s, $fast_fails times in a row" | tee -a "$LOG"
        echo "[booth] giving up -- fix the error above, then run: bash $APP_DIR/booth.sh" | tee -a "$LOG"
        exit "$rc"
    fi
    echo "[booth] $(date '+%F %T') exited $rc after ${uptime}s -- restarting in 3s ($fast_fails/$MAX_FAST_FAILS)" | tee -a "$LOG"
    sleep 3
done
