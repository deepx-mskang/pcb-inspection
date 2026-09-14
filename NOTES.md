# Engineering notes — 4-channel PCB inspection demo

Why this demo is built the way it is: measurements, the problems they
exposed, and the fixes. Kept out of `README.md` so that stays a
get-it-running document. Every number here came from a run on the demo
host, not an estimate.

## Exhibition-visibility pass (2026-09-02)

Booth feedback after the first live run with a real camera: CH2's PASS/DEFECT
was hard to read at a glance, and CH3 looked like "generic boxes" rather than
a defect finder. Fixed by unifying all 3 detector cells around one visual
language (`_draw_verdict_banner` / `_draw_defects` in `grid_viewer.py`):

- **Solid-color status bar + thick border**, scaled to image size, instead of
  a thin colored outline + small text — reads as PASS/DEFECT from a few
  meters away, the way a real AOI station's screen does.
- **CH3's "good" class (a properly soldered joint, not a defect) is no longer
  drawn** — only real defect classes get boxed, each with a bold label
  resolved by `class_id` (`YOLOv8Postprocessor` never fills
  `DetectionResult.class_name`, only `class_id` — the original
  `DetectionVisualizer` looked labels up internally; the replacement renderer
  here does the same lookup explicitly). CH1 and CH3 now also show the
  OK/DEFECT status bar, so all 3 detector cells read the same way.

## Resource pass (2026-09-02) — 683% CPU / 2.9 GB → 176% / 1.1 GB

Measured on the 20-core demo host, headless 4-channel run sampled at t=20s:

| | before | after |
|---|---|---|
| CPU | 683% | **176%** |
| RSS | 2931 MB | **1134 MB** |
| threads | 79 | **31** |

Four causes, all host-side — **none of them in the model or the NPU**. Per-stage
measurement put postprocessing at 0.12 ms (CH1) / 0.04 ms (CH3) per tick and NPU
inference at 0.08–0.14 cores (i.e. blocked waiting), so compiler-side options
(PPU, graph optimisation) had nothing to gain here. YOLO26 does not support PPU
anyway — its NMS-free `[1,300,6]` head already does on-chip what PPU exists to
offload.

1. **Thread-pool oversubscription** (biggest CPU win). cv2 and OpenBLAS each
   default to a 20-thread pool per operation; with 4 channels running
   concurrently, synchronisation cost dominated the actual work. Capping both to
   4 (`GRID_NUM_THREADS`, default 4) is not a latency trade — it is *faster*:

   | stage | 20 threads | 4 threads |
   |---|---|---|
   | CH2 `score_map` (BLAS GEMM) | 10.4 ms / 19.7 cores | 9.8 ms / 4.0 cores |
   | CH3 letterbox (2560×1440) | 10.1 ms / 11.6 cores | **5.1 ms** / 2.1 cores |

2. **`ctx.original_image` pinned every source frame** (biggest RAM win).
   `LetterboxPreprocessor.process()` unconditionally does
   `ctx.original_image = input_image.copy()` (for colour restoration in
   enhancement/denoising models). Detection postprocessing never reads it. Once
   the tick cache below started retaining one ctx per frame, that pinned all 86
   CH3 sources — 900 MB of invisible overhead (RSS grew 1147 MB for 150 MB of
   arrays). Cleared at preload; RSS per channel now matches its array footprint.

3. **Redundant per-tick work on a static dataset.** The dataset loops replay a
   fixed set of images but re-letterboxed them on every tick (11.4 ms CPU/tick on
   CH3) and kept full-resolution frames in RAM (200 × 10.5 MB = 2.06 GB). Preload
   now decodes + letterboxes **once**, keeping only the 640×640 NPU input and a
   cell-sized frame to draw on. CH3's list also stopped tiling 86 images into 200
   slots (duplicate decoded frames, no added variety).

4. **CH2 decoded each image twice** — `backend.embed(path)` decodes with PIL, and
   the display path called `cv2.imread` on the same file again (4.9 ms/tick of
   pure duplicate work). The display copy is now decoded once at cell size and
   cached per distinct path.

Plus: the live cell had **no frame-rate cap**, so it ran at whatever the QHD
sensor delivered (~30 fps of decode+infer+render that no booth visitor can
perceive). Now capped at 15 fps.

**Framework finding worth fixing upstream**: `letterbox_preprocessor.py:88` copies
the entire input frame into the context on *every* call, for every task, including
the detection tasks that never read it. For a 2560×1440 source at 8 fps that is
~84 MB/s of pointless memcpy in any app using this preprocessor. Making
`original_image` opt-in would benefit every detection app in the suite. (Not filed
as KB drift: the KB documents `PreprocessContext` only as a return type and makes
no claim about its memory behaviour, so there is no KB statement that reality
contradicts — this is a missing-documentation/implementation issue, not drift.)

## Why threads, not processes

`dx_engine.run()` is a native call that releases the GIL for most of its
execution time, and earlier concurrency measurements (3-6 resident engines,
~76-98MB RSS each, no init contention) hold for threads exactly as they would
for processes. So each cell is one `threading.Thread` writing its latest
rendered frame into a shared numpy slot under a lock — no IPC, no shared
memory, no queues.

## Why the live cell is switchable, not fixed to one model

The alternative was picking one of the 3 models to always run live and
leaving the other two as dataset loops only. Rejected because it makes the
live cell arbitrary — there's no principled reason CH1 deserves the camera
over CH2 or CH3. Instead **all 3 engines are loaded up front** (measured:
~76MB RSS each, no contention loading 3 simultaneously) and pressing 1/2/3
swaps which one processes the live frame — a handle swap, not a reload.

## Self-contained / portable

Verified with the suite's copy-out gate: the entire session directory was
copied to `/tmp` (outside `dx-all-suite`) and run there with no suite-relative
path resolved — `grid_viewer.py --headless --switch-test` completed cleanly
(see session.log). This required vendoring, via `setup.sh`:

- `vendor/common/` — the shared IFactory framework (`LetterboxPreprocessor`,
  `YOLOv8Postprocessor`) from `dx_app/src/python_example/common`
- `vendor/patchcore/` + `vendor/preproc.py` — CH2's feature-extraction/memory-bank package,
  copied whole from the CH2 app session (not imported in place)
- `models/*.dxnn` (3 compiled models) + `models/bank_pcb1.{npz,meta.json}`
  (CH2's memory bank)
- `data/ch1_images/` (500 DeepPCB val images), `data/ch3_images/` (86
  SolDef_AI val images) — the exact dataset-loop source images
- `data/ch2_visa/` — a vendored **subset** of the raw VisA dataset (100
  Anomaly + first 200 Normal images, ~77MB vs. the 287MB full set): CH2's
  dataset-loop cell only ever indexes into the first 200 positions of each
  class list, so this is the exact slice actually used, not an arbitrary trim

`grid_viewer.py` resolves these vendor-first (one `sys.path` entry for
`vendor/`, plus `data/...`, before falling back to sibling suite sessions).
Upstream code was moved under `vendor/` on 2026-09-14 so the top level of the
directory is purely this demo's own files -- see README.

## Verified issues fixed in this session

- **`self._stop` shadowed `threading.Thread`'s private `_stop()` method** —
  `Thread.join()` calls it internally; overwriting it with a
  `threading.Event()` made `join()` raise `TypeError: 'Event' object is not
  callable`. Renamed to `self._stop_event` throughout (`YoloDatasetChannel`,
  `PatchCoreDatasetChannel`, `LiveChannel`). This was also the root cause of
  an intermittent `terminate called without an active exception` on exit
  (a worker thread was still inside a native `dx_engine.run()` call when
  interpreter teardown began, because `join()` itself was crashing rather
  than actually waiting).
- Added `for w in workers: w.join(timeout=2.0)` in `main()`'s `finally` block
  so every worker actually leaves its loop before `cv2.destroyAllWindows()`
  and process exit — confirmed by 3 repeated clean headless runs with no
  `terminate` warning.
- CH2's PASS/DEFECT banner text (plain green/red on video) was low-contrast;
  both the dataset-loop cell and the live cell now draw a black outline pass
  before the colored text (`cv2.putText` twice at increasing thickness).

## Live-switch verification

`--switch-test` (headless only) cycles the live cell through models 1→2→3,
saving one frame per model (`live_ch1.jpg`, `live_ch2.jpg`, `live_ch3.jpg`).
Confirmed all 3 frames are distinct (different md5, different per-frame mean
pixel value) — i.e. `set_active()` actually changes which of the 3 resident
engines processes the live feed, not just which label is drawn.

## Verification evidence

- 3x repeated `--headless --duration 6` runs, full stdout+stderr captured:
  zero occurrences of `terminate`, `TypeError`, or `Traceback`.
- `--switch-test` run: all 3 live-model frames rendered and confirmed distinct.
- Copy-out test: session copied to `/tmp` (outside dx-all-suite), ran
  `--headless --switch-test` there — clean, no suite-relative path touched.
- `session.log` in this directory is the real captured output of `setup.sh` +
  a syntax check + a full `--headless --switch-test` run.

