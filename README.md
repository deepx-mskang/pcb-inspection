# 4-channel PCB inspection demo — DEEPX DX-M1

Four AOI channels on one NPU, in one 2x2 window. Three replay a dataset; the
fourth runs a live USB camera against whichever of the three models you pick.

```
┌─ CH1 DeepPCB ────────────┬─ CH2 VisA ───────────────┐
│ yolo26n                  │ PatchCore (WRN50-2)      │
│ wiring defects, 6 classes│ anomaly score, PASS/FAIL │
├─ CH3 SolDef_AI ──────────┼─ LIVE ───────────────────┤
│ yolo26m                  │ USB camera               │
│ solder defects, 5 classes│ 1/2/3 switches the model │
└──────────────────────────┴──────────────────────────┘
```

All three models stay resident, so switching the live cell is a handle swap,
not a reload.

## Run it

```bash
bash setup.sh     # once per machine
bash run.sh       # opens fullscreen
```

| key | |
|---|---|
| `1` `2` `3` | switch the live cell's model |
| `p` | pause every channel — hold a frame while explaining |
| `f` | fullscreen toggle |
| `q` / `Esc` / click `X` | quit |

Other modes:

```bash
bash run.sh --no-live                 # no camera on this machine
bash run.sh --camera 2                # a different camera index
bash run.sh --headless --duration 10 --switch-test --save-dir demo_out
bash booth.sh                         # supervised: restarts on crash
bash install_autostart.sh             # start on desktop login (--uninstall)
```

## The code

`grid_viewer.py` is the whole demo — one file, ~640 lines.

| | |
|---|---|
| `YoloDatasetChannel` | CH1 / CH3 — replays a dataset through a yolo26 detector |
| `PatchCoreDatasetChannel` | CH2 — VisA frames through PatchCore kNN scoring |
| `LiveChannel` | the live cell; holds all 3 engines, `set_active()` switches |
| `FpsMeter`, `compose_board`, `_draw_*` | 2x2 board, title bar, verdict banners |
| `run_qt` | PySide6 window (imported lazily, so `--headless` needs no Qt) |

Each channel is a thread that writes its latest rendered frame into a shared
slot; the Qt timer composites whatever is there. `dx_engine.run()` releases the
GIL, so threads are enough — no IPC. See [NOTES.md](NOTES.md) for the
measurements behind that and the other design choices.

Everything under `vendor/` is **upstream code reused unmodified** — the shared
dx_app framework (`common`) and CH2's PatchCore package. The demo calls exactly
two things from it, `LetterboxPreprocessor` and `YOLOv8Postprocessor`; the rest
comes along because the framework's `__init__` re-exports all 22 tasks' worth of
processors. Nothing in `vendor/` was edited, and you should not need to read it.

| file | |
|---|---|
| `grid_viewer.py` | the demo |
| `setup.sh` | bootstrap: OS packages → DEEPX runtime → Python packages → assets |
| `run.sh` | resolve interpreter, check assets, launch |
| `booth.sh` | supervisor: restart on crash, honour a clean quit |
| `install_autostart.sh` | XDG autostart entry pointing at `booth.sh` |
| `fetch_datasets.py` | rebuild a missing `data/` slice from the public sources |
| `_python.sh` | interpreter resolver shared by setup/run |
| `vendor/` | upstream `common`, `patchcore`, `preproc.py` |

## What setup.sh does

OS packages (apt) → DEEPX runtime → Python packages → demo assets. Re-running it
on a configured machine is a no-op and needs no sudo.

If no Python on the box can `import dx_engine`, it clones and builds
[dx-runtime](https://github.com/DEEPX-AI/dx-runtime) (override with
`DX_RUNTIME_URL` / `DX_RUNTIME_DIR`); that step needs sudo.

**Which Python runs the demo**: whichever one can `import dx_engine` — normally
the dx-runtime venv, *not* the system `python3`. The runtime installer puts its
binding in that venv, so a bare `python3 grid_viewer.py` fails with
`ModuleNotFoundError` on a machine whose NPU is perfectly healthy. `_python.sh`
resolves it; `DX_PYTHON=/path/to/python` overrides.

**The NPU needs `dxrt.service` running.** If it is down the demo exits with
`dxrt service is not running`. On a booth machine also enable it at boot:

```bash
sudo systemctl start dxrt
sudo systemctl enable dxrt     # otherwise it will not come back after a reboot
```

## Models and datasets

**Models** (3 × `.dxnn`, 96 MB) are **not in git** — they ship in the
distribution tar. Without them `setup.sh` says so and points at it.

**Datasets** are slices of public sets, rebuilt on demand by `fetch_datasets.py`.
setup.sh runs it only for a slice missing from the path the demo reads, so a copy
unpacked from the tar never downloads anything.

| slice | source | what is taken |
|---|---|---|
| `data/ch1_images` (500) | [DeepPCB](https://github.com/tangsanli5201/DeepPCB), MIT | official `PCBData/test.txt` split, `_test` image of each pair |
| `data/ch3_images` (86) | [SolDef_AI](https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection) ([paper](https://www.mdpi.com/2504-4494/8/3/117)) | val 20 % of a seed-0 shuffle — the split the model trained against |
| `data/ch2_visa` (100+200) | [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar), 1.9 GB | pcb1: all `Anomaly` + first 200 sorted `Normal` — the range CH2 indexes |

Rebuilding into a scratch directory reproduced byte-identical images. Only
images are fetched; the loops never read labels. SolDef_AI has no
credential-free download — the script uses the `kaggle` CLI if configured,
otherwise it prints the page URL and the path to unzip into.

## Troubleshooting

| symptom | |
|---|---|
| `dxrt service is not running` | `sudo systemctl start dxrt` |
| `ModuleNotFoundError: dx_engine` | `bash setup.sh`, or set `DX_PYTHON` |
| `NO CAMERA` in the live cell | check the USB cable — it retries every 3 s; the other 3 channels keep running |
| models missing | unpack the distribution tar over this directory |
| demo dies instantly, repeatedly | `booth.sh` gives up after 5 tries; the error is in `booth.log` |

A corrupt `.dxnn` **segfaults inside the native runtime** — no Python error is
raised, so nothing in the process can report it. `booth.sh` is the mitigation.

## Requirements

numpy, opencv-python, Pillow, PySide6 (`requirements.txt`) plus the DEEPX
runtime (`dx_engine` + `dxrt-cli`), which is not a pip package — `setup.sh`
installs it.
