# Shared interpreter resolver, sourced by setup.sh and run.sh.
#
# The demo must run on the Python that can import dx_engine. That is normally
# the dx-runtime venv, NOT the system python3 -- the runtime installer puts its
# binding in that venv, so `python3 grid_viewer.py` fails with
# ModuleNotFoundError even on a machine whose NPU is perfectly healthy.
#
# Search order:
#   $DX_PYTHON            explicit override
#   ./.dx_python          what setup.sh resolved on THIS machine (see below)
#   ./venv, ./.venv       a venv created next to the demo
#   ../**/venv-dx-runtime the dx-runtime venv, walking up
#   python3               system interpreter
#
# The recorded ./.dx_python matters for a copied-out install: walking up from
# /opt/pcb-demo never reaches the dx-runtime checkout, so without the record the
# demo could not find its own interpreter. The record is always re-validated,
# so a stale one from another machine simply falls through to the search.
dx_python() {
    local c d cands="${DX_PYTHON:-}"
    [ -f "$APP_DIR/.dx_python" ] && cands="$cands $(cat "$APP_DIR/.dx_python")"
    cands="$cands $APP_DIR/venv/bin/python $APP_DIR/.venv/bin/python"
    d="$APP_DIR"
    while [ "$d" != "/" ]; do
        cands="$cands $d/venv-dx-runtime/bin/python $d/dx-runtime/venv-dx-runtime/bin/python"
        d="$(dirname "$d")"
    done
    for c in $cands $(command -v python3 || true); do
        [ -x "$c" ] || continue
        "$c" -c "import dx_engine" >/dev/null 2>&1 && { echo "$c"; return 0; }
    done
    return 1
}
