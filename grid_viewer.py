#!/usr/bin/env python3
"""4-cell multichannel grid viewer.

Orchestration layer that SyncRunner doesn't cover (it assumes one model).
Each channel below reuses the IFactory components already built and verified
for CH1/CH2/CH3 -- only the "run one tick, write the latest rendered frame to
a shared slot" loop is new. Threads, not processes: dx_engine.run() is a
native call that releases the GIL for most of its time, and the earlier
concurrency benchmarks (3-6 resident engines, ~76-98MB RSS each) hold
regardless of thread vs process, so there is no IPC to build.

Grid layout:
    [ CH1 dataset loop ] [ CH2 dataset loop ]
    [ CH3 dataset loop ] [ LIVE: USB cam, 1/2/3 switches model ]
"""
from __future__ import annotations

import glob
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

# Thread-pool caps MUST be set before numpy/cv2 are imported -- OpenBLAS reads
# these at load time. On a 20-core host both cv2 and OpenBLAS default to a
# 20-thread pool per operation, and with 4 channels running concurrently the
# synchronisation cost dominates the actual work. Measured, single stage in
# isolation:
#   CH2 score_map (BLAS GEMM):     20 thr -> 10.4 ms wall / 19.7 cores
#                                   4 thr ->  9.8 ms wall /  4.0 cores
#   CH3 letterbox (2560x1440):     20 thr -> 10.1 ms wall / 11.6 cores
#                                   4 thr ->  5.1 ms wall /  2.1 cores
# i.e. capping is not a latency/CPU trade -- it is faster AND ~5x cheaper.
_THREADS = os.environ.get("GRID_NUM_THREADS", "4")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(int(_THREADS))

_HERE = Path(__file__).resolve().parent


def _suite_root(start: Path) -> Path:
    d = start
    while d != d.parent:
        if (d / "dx-runtime").is_dir() and (d / "dx-compiler").is_dir():
            return d
        d = d.parent
    raise RuntimeError("could not locate the dx-all-suite root above " + str(start))


# Upstream code this demo reuses unmodified lives in vendor/ -- the shared
# dx_app framework (`common`) and CH2's PatchCore package (`patchcore`,
# `preproc`). One sys.path entry covers all three, and keeping them under a
# single directory is what lets the top level stay purely this demo's own code.
# setup.sh populates it; the fallback is for in-place dev before it has run.
_VENDOR = _HERE / "vendor"
if (_VENDOR / "common").is_dir():
    sys.path.insert(0, str(_VENDOR))
else:
    _suite = _suite_root(_HERE)
    sys.path.insert(0, str(_suite / "dx-runtime/dx_app/src/python_example"))
    sys.path.insert(0, str(_suite / "dx-runtime/dx_app/dx-agent-dev"
                           / "20260825-224353_claude_opus5_patchcore_inference"))

from common.processors import LetterboxPreprocessor, YOLOv8Postprocessor  # noqa: E402
from dx_engine import InferenceEngine  # noqa: E402

CELL_W, CELL_H = 480, 360
PRELOAD_N = 200

# ---------------------------------------------------------------------------
# Paths -- this session's compiled models are always local (models/, vendored
# at session creation). Dataset images and the CH2 bank prefer the vendored
# ./data and ./models copies (setup.sh); if setup.sh has not run yet, fall
# back to the sibling sessions' data for in-place development.
# ---------------------------------------------------------------------------
CH1_DXNN = str(_HERE / "models" / "ch1_deeppcb_yolo26n.dxnn")
CH1_CLASSES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]

CH3_DXNN = str(_HERE / "models" / "ch3_soldef_yolo26m.dxnn")
CH3_CLASSES = ["good", "exc_solder", "spike", "no_good", "poor_solder"]

CH2_DXNN = str(_HERE / "models" / "ch2_patchcore_wrn50.dxnn")


def _vendored_or_suite(vendored_rel: str, suite_rel: str) -> str:
    v = _HERE / vendored_rel
    if v.exists():
        return str(v)
    return str(_suite_root(_HERE) / suite_rel)


CH1_IMAGES = _vendored_or_suite(
    "data/ch1_images",
    "dx-compiler/dx-agent-dev/20260831-092911_claude_opus5_yolo26n_deeppcb_compile/dataset/images/val")
CH3_IMAGES = _vendored_or_suite(
    "data/ch3_images",
    "dx-compiler/dx-agent-dev/20260831-145134_claude_opus5_yolo26m_soldef_compile/dataset/images/val")
CH2_BANK = _vendored_or_suite(
    "models/bank_pcb1.npz",
    "dx-runtime/dx_app/dx-agent-dev/20260825-224353_claude_opus5_patchcore_inference/bank_pcb1.npz")
CH2_META = _vendored_or_suite(
    "models/bank_pcb1.meta.json",
    "dx-runtime/dx_app/dx-agent-dev/20260825-224353_claude_opus5_patchcore_inference/bank_pcb1.meta.json")
_visa_vendored = _HERE / "data" / "ch2_visa"
CH2_VISA_ROOT = str(_visa_vendored) if _visa_vendored.exists() else str(
    Path.home() / "m1" / "datasets" / "visa" / "pcb1" / "Data" / "Images")


def _preload_ticks(pattern: str, n: int, pre: "LetterboxPreprocessor",
                   cell: tuple[int, int]) -> list[tuple]:
    """Decode + letterbox each dataset image ONCE, keeping only per-tick inputs.

    The dataset loop replays a fixed set of images, so letterboxing them on
    every tick recomputes a constant (measured 11.4 ms CPU/tick on CH3's
    2560x1440 source). Keeping the full-resolution frame is pure waste too:
    200 CH3 frames at 10.5 MB each is 2.06 GB, and only two derived things are
    ever used -- the 640x640 letterboxed NPU input and a cell-sized frame to
    draw on.

    Distinct paths only, capped at n: the old code tiled a short list up to n
    (CH3 has 86 images -> 200 slots), duplicating decoded frames in RAM for no
    added variety. The caller cycles with `i % len(frames)` instead.
    """
    paths = sorted(glob.glob(pattern))[:n]
    if not paths:
        raise FileNotFoundError(pattern)
    ticks = []
    for p in paths:
        bgr = cv2.imread(p)
        letterboxed, ctx = pre.process(bgr)
        # LetterboxPreprocessor stores a full copy of the source frame on the
        # context (`ctx.original_image = input_image.copy()`, for colour
        # restoration in enhancement/denoising models). The detection
        # postprocessor never reads it -- it only needs pad/scale/original
        # dimensions -- and holding one ctx per cached tick would pin every
        # source frame in RAM (CH3: 86 x 10.5 MB = 900 MB). Drop it.
        ctx.original_image = None
        ctx.normalized_input = None
        h, w = bgr.shape[:2]
        ticks.append((np.ascontiguousarray(letterboxed, dtype=np.uint8), ctx,
                      cv2.resize(bgr, cell), cell[0] / w, cell[1] / h))
    return ticks


OK_COLOR = (0, 170, 0)
DEFECT_COLOR = (0, 0, 255)


def _draw_verdict_banner(bgr: np.ndarray, ok: bool, text: str) -> np.ndarray:
    """Solid-color status bar + thick border, scaled to image size.

    Exhibition-visibility fix: a thin colored border and small text (the
    original CH2 style) reads as noise on a booth monitor from a few meters
    away. A full-width solid bar with large bold text reads as PASS/FAIL at a
    glance, the same way a real AOI station's screen does.
    """
    h, w = bgr.shape[:2]
    scale = max(w, h) / 640.0
    colour = OK_COLOR if ok else DEFECT_COLOR
    border = max(6, int(14 * scale))
    cv2.rectangle(bgr, (0, 0), (w - 1, h - 1), colour, border)
    bar_h = max(36, int(h * 0.13))
    cv2.rectangle(bgr, (0, 0), (w, bar_h), colour, cv2.FILLED)
    # Shrink to fit rather than letting the text run off the cell edge (a long
    # defect-class list did exactly that on CH1).
    font_scale = max(0.8, 1.0 * scale)
    thickness = max(2, int(3 * scale))
    while font_scale > 0.4:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if tw <= w - 20:
            break
        font_scale -= 0.05
        thickness = max(1, int(thickness * 0.9))
    cv2.putText(bgr, text, (10, int(bar_h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return bgr


def _draw_defects(bgr: np.ndarray, results, labels: list[str], normal_classes: frozenset,
                  prefix: str = "", box_scale: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """Detection rendering tuned for 'is this a defect finder' legibility.

    The generic DetectionVisualizer gives every class a random color and a
    small thin label -- fine for a general demo, but on SolDef_AI the "good"
    class (a properly soldered joint, not a defect) gets boxed identically to
    real defects, so the cell reads as "random boxes" rather than "finds
    solder defects". Normal-class boxes are dropped entirely here; only
    defects are drawn, in a bold, high-contrast style, plus the same
    OK/DEFECT status bar used for CH2 so all three detector cells share one
    visual language.

    YOLOv8Postprocessor only fills DetectionResult.class_id, never
    class_name (that lookup happens inside DetectionVisualizer, which this
    function replaces) -- so the label list is resolved here by class_id.

    box_scale maps postprocessor coordinates (always in the ORIGINAL source
    image's frame, via the letterbox ctx) onto `bgr` when `bgr` is a resized
    copy -- the dataset channels now draw straight onto a cell-sized frame
    rather than drawing at full resolution and downscaling afterwards.
    """
    def name_of(d) -> str:
        return labels[d.class_id] if d.class_id < len(labels) else f"class_{d.class_id}"

    h, w = bgr.shape[:2]
    scale = max(w, h) / 640.0
    sx, sy = box_scale
    defects = [d for d in results if name_of(d) not in normal_classes]
    box_th = max(2, int(4 * scale))
    font_scale = max(0.6, 0.8 * scale)
    text_th = max(1, int(2 * scale))
    for d in defects:
        x1, y1 = int(d.box[0] * sx), int(d.box[1] * sy)
        x2, y2 = int(d.box[2] * sx), int(d.box[3] * sy)
        name = name_of(d)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), DEFECT_COLOR, box_th)
        label = f"{name} {d.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_th)
        ty = y1 - 8 if y1 - 8 - th > 0 else y1 + th + 8
        cv2.rectangle(bgr, (x1, ty - th - 6), (x1 + tw + 6, ty + 4), DEFECT_COLOR, cv2.FILLED)
        cv2.putText(bgr, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), text_th, cv2.LINE_AA)
    if defects:
        names = ", ".join(sorted({name_of(d) for d in defects}))
        text = f"{prefix}DEFECT: {names}"
    else:
        text = f"{prefix}OK"
    return _draw_verdict_banner(bgr, ok=not defects, text=text)


class YoloDatasetChannel(threading.Thread):
    """One dataset-loop cell driven by a yolo26 detector (CH1 or CH3)."""

    def __init__(self, name: str, dxnn_path: str, image_glob: str, classes: list[str],
                 pause_event: threading.Event, normal_classes: frozenset = frozenset(),
                 fps: float = 8.0):
        super().__init__(daemon=True)
        self.name = name
        self.classes = classes
        self.normal_classes = normal_classes
        self.ie = InferenceEngine(dxnn_path)
        info = self.ie.get_input_tensors_info()[0]
        self.in_w, self.in_h = info["shape"][2], info["shape"][1]
        self.pre = LetterboxPreprocessor(self.in_w, self.in_h)
        self.post = YOLOv8Postprocessor(self.in_w, self.in_h,
                                        {"num_classes": len(classes), "conf_threshold": 0.25})
        self.ticks = _preload_ticks(image_glob, PRELOAD_N, self.pre, (CELL_W, CELL_H))
        self.period = 1.0 / fps
        self.latest = np.zeros((CELL_H, CELL_W, 3), np.uint8)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pause_event = pause_event
        self.processed = 0  # ticks completed, for the status bar's FPS readout

    def get_frame(self) -> np.ndarray:
        with self._lock:
            return self.latest

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        i = 0
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.05)
                continue
            t0 = time.time()
            letterboxed, ctx, disp, sx, sy = self.ticks[i % len(self.ticks)]
            i += 1
            raw = self.ie.run([letterboxed[None, ...]])
            results = self.post.process(raw, ctx)
            # Draw straight onto the cell-sized copy: postprocessor boxes are in
            # the original image's frame, so they scale by (sx, sy).
            rendered = _draw_defects(disp.copy(), results, self.classes, self.normal_classes,
                                     prefix=f"{self.name} ", box_scale=(sx, sy))
            with self._lock:
                self.latest = rendered
                self.processed += 1
            dt = time.time() - t0
            if dt < self.period:
                time.sleep(self.period - dt)


class PatchCoreDatasetChannel(threading.Thread):
    """CH2's cell: PatchCore anomaly detection, PASS/DEFECT banner + bbox."""

    def __init__(self, dxnn_path: str, bank_path: str, meta_path: str,
                 visa_root: str, pause_event: threading.Event, fps: float = 5.0):
        super().__init__(daemon=True)
        import json
        from patchcore.backend import DxnnBackend
        from patchcore.features import build_patch_embedding
        from patchcore.memorybank import MemoryBank

        self._embed = build_patch_embedding
        self.backend = DxnnBackend(dxnn_path)
        self.bank = MemoryBank.load(bank_path)
        self.thr = json.loads(Path(meta_path).read_text())["threshold"]

        normal = sorted(glob.glob(f"{visa_root}/Normal/*"))
        anomaly = sorted(glob.glob(f"{visa_root}/Anomaly/*"))
        mix = []
        for i in range(PRELOAD_N):
            mix.append(anomaly[i % len(anomaly)] if i % 3 == 0 else normal[i % len(normal)])
        # backend.embed() re-decodes the file itself (PIL), and the old loop then
        # decoded the SAME file a second time with cv2.imread just to draw on it
        # -- 4.9 ms of pure duplicate JPEG decode per tick. Decode the display
        # copy once here, at cell size, and cache it per distinct path.
        cache: dict[str, np.ndarray] = {}
        for p in mix:
            if p not in cache:
                cache[p] = cv2.resize(cv2.imread(p), (CELL_W, CELL_H))
        self.ticks = [(p, cache[p]) for p in mix]
        self.period = 1.0 / fps
        self.latest = np.zeros((CELL_H, CELL_W, 3), np.uint8)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pause_event = pause_event
        self.processed = 0  # ticks completed, for the status bar's FPS readout

    def get_frame(self) -> np.ndarray:
        with self._lock:
            return self.latest

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        i = 0
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.05)
                continue
            t0 = time.time()
            path, disp = self.ticks[i % len(self.ticks)]
            i += 1
            f2, f3 = self.backend.embed(path)
            smap = self.bank.score_map(self._embed(f2, f3))
            score = float(smap.max())
            verdict = "PASS" if score < self.thr else "DEFECT"

            banner = f"CH2 {verdict} {score:.2f}/{self.thr:.2f}"
            rendered = _draw_verdict_banner(disp.copy(), ok=(verdict == "PASS"), text=banner)
            with self._lock:
                self.latest = rendered
                self.processed += 1
            dt = time.time() - t0
            if dt < self.period:
                time.sleep(self.period - dt)


class LiveChannel(threading.Thread):
    """The 4th cell: USB camera, switchable among the 3 trained models.

    All 3 engines are loaded up front so switching is a handle swap, not a
    reload -- confirmed safe: 3 resident InferenceEngine instances in one
    process add ~76MB RSS each with no init contention (measured earlier).
    """

    def __init__(self, pause_event: threading.Event, camera_index: int = 0,
                 fps: float = 15.0):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)
        self._next_reopen = 0.0
        # Without a cap this loop ran as fast as the camera delivered frames
        # (a QHD sensor at 30 fps = 30 full decode+infer+render cycles/s), which
        # a wall display cannot show and a booth visitor cannot perceive.
        self.period = 1.0 / fps
        self.active = 1  # 1=CH1, 2=CH2, 3=CH3

        self._init_yolo(1, CH1_DXNN, CH1_CLASSES, frozenset())
        self._init_yolo(3, CH3_DXNN, CH3_CLASSES, frozenset({"good"}))
        self._init_patchcore()

        self.latest = np.zeros((CELL_H, CELL_W, 3), np.uint8)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pause_event = pause_event
        self.processed = 0  # ticks completed, for the status bar's FPS readout

    def _init_yolo(self, tag: int, dxnn_path: str, classes: list[str], normal_classes: frozenset) -> None:
        ie = InferenceEngine(dxnn_path)
        info = ie.get_input_tensors_info()[0]
        in_w, in_h = info["shape"][2], info["shape"][1]
        setattr(self, f"ie{tag}", ie)
        setattr(self, f"pre{tag}", LetterboxPreprocessor(in_w, in_h))
        setattr(self, f"post{tag}", YOLOv8Postprocessor(
            in_w, in_h, {"num_classes": len(classes), "conf_threshold": 0.10}))
        setattr(self, f"labels{tag}", classes)
        setattr(self, f"normal{tag}", normal_classes)

    def _init_patchcore(self) -> None:
        import json
        from patchcore.backend import DxnnBackend
        from patchcore.features import build_patch_embedding
        from patchcore.memorybank import MemoryBank
        self._embed = build_patch_embedding
        self.ie2 = DxnnBackend(CH2_DXNN)
        self.bank2 = MemoryBank.load(CH2_BANK)
        self.thr2 = json.loads(Path(CH2_META).read_text())["threshold"]

    def set_active(self, n: int) -> None:
        if n in (1, 2, 3):
            self.active = n

    def get_frame(self) -> np.ndarray:
        with self._lock:
            return self.latest

    def stop(self) -> None:
        self._stop_event.set()

    def _run_yolo(self, tag: int, bgr: np.ndarray, label: str) -> np.ndarray:
        pre, post, ie, labels, normal = (getattr(self, f"pre{tag}"), getattr(self, f"post{tag}"),
                                         getattr(self, f"ie{tag}"), getattr(self, f"labels{tag}"),
                                         getattr(self, f"normal{tag}"))
        letterboxed, ctx = pre.process(bgr)
        raw = ie.run([letterboxed[None, ...].astype(np.uint8)])
        results = post.process(raw, ctx)
        return _draw_defects(bgr.copy(), results, labels, normal, prefix=f"{label} ")

    def _run_patchcore(self, bgr: np.ndarray) -> np.ndarray:
        # Same geometry as CH2's preproc.load_hwc_u8: resize to 256x256 (aspect
        # ignored, matching training), then center-crop 224 -- NOT a
        # square-crop-then-resize, which would change the effective framing.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        off = (256 - 224) // 2
        crop = resized[off:off + 224, off:off + 224]
        # DxnnBackend.embed() is file-path-only (it re-decodes from disk), but
        # the live frame is already in memory, so this uses the same
        # InferenceEngine handle directly rather than round-tripping through a
        # temp file. No edit to the CH2 session's backend.py.
        outs = self.ie2.ie.run([np.ascontiguousarray(crop[None, ...].astype(np.uint8))])
        mapped = [self.ie2._nchw(o) for o in outs]
        f2 = next(m for m in mapped if m.shape[1] == 512)
        f3 = next(m for m in mapped if m.shape[1] == 1024)
        smap = self.bank2.score_map(self._embed(f2, f3))
        score = float(smap.max())
        verdict = "PASS" if score < self.thr2 else "DEFECT"
        banner = f"LIVE CH2 {verdict} {score:.2f}"
        return _draw_verdict_banner(bgr.copy(), ok=(verdict == "PASS"), text=banner)

    def _publish_no_camera(self) -> None:
        frame = np.full((CELL_H, CELL_W, 3), 24, np.uint8)
        cv2.rectangle(frame, (0, 0), (CELL_W - 1, CELL_H - 1), (0, 200, 255), 8)
        for text, y, scale in (("NO CAMERA", 150, 1.2),
                               (f"/dev/video{self.camera_index} not delivering frames", 190, 0.5),
                               ("check the USB cable -- retrying", 215, 0.5)):
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            cv2.putText(frame, text, ((CELL_W - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (0, 200, 255), 2, cv2.LINE_AA)
        with self._lock:
            self.latest = frame

    def run(self) -> None:
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.05)
                continue
            t0 = time.time()
            ok, bgr = self.cap.read()
            if not ok:
                # An unplugged or busy camera used to leave this cell black
                # forever with no explanation, which at a booth just looks like
                # the demo is broken. Say so, and keep trying to reopen.
                self._publish_no_camera()
                if time.time() >= self._next_reopen:
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.camera_index)
                    self._next_reopen = time.time() + 3.0
                time.sleep(0.3)
                continue
            active = self.active
            try:
                if active == 1:
                    rendered = self._run_yolo(1, bgr, "LIVE CH1")
                elif active == 3:
                    rendered = self._run_yolo(3, bgr, "LIVE CH3")
                else:
                    rendered = self._run_patchcore(bgr)
            except Exception as e:  # noqa: BLE001
                rendered = bgr.copy()
                cv2.putText(rendered, f"live error: {e}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            rendered = cv2.resize(rendered, (CELL_W, CELL_H))
            with self._lock:
                self.latest = rendered
                self.processed += 1
            dt = time.time() - t0
            if dt < self.period:
                time.sleep(self.period - dt)


# ---------------------------------------------------------------------------
# Booth chrome -- ported from dx-demo's yolo-multi demo
# (apps/yolo-multi/python/yolo_multi_demo.py). That app composites everything
# into one cv2 board and pushes it to a QLabel, exactly like this one, so the
# title bar / exit button / LIVE pill port over directly.
# ---------------------------------------------------------------------------
TITLE_H = 56
BOARD_W = CELL_W * 2
BOARD_H = TITLE_H + CELL_H * 2
EXIT_BTN = (BOARD_W - 32 - 8, (TITLE_H - 28) // 2, 32, 28)   # x, y, w, h
BRAND_W = 430
MODEL_LABEL = {1: "yolo26n", 2: "PatchCore", 3: "yolo26m"}


class FpsMeter:
    """Sliding-window FPS over the channels' completed-tick counters.

    Same shape as the C++/Qt demo's meter: sample the monotonically increasing
    per-channel counters, keep the deltas inside a time window, divide by the
    window's real span.
    """

    def __init__(self, channels, window_s: float = 5.0):
        self._channels = channels
        self._window = window_s
        self._last = [c.processed for c in channels]
        self._samples: "deque" = deque()
        self.fps = 0.0

    def tick(self) -> None:
        now = time.monotonic()
        delta = 0
        for i, ch in enumerate(self._channels):
            cur = ch.processed
            delta += max(0, cur - self._last[i])
            self._last[i] = cur
        self._samples.append((now, delta))
        cutoff = now - self._window
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
        if len(self._samples) > 1:
            span = self._samples[-1][0] - self._samples[0][0]
            total = sum(d for _, d in self._samples)
            self.fps = total / span if span > 0 else 0.0


def _draw_live_badge(board: np.ndarray, cell_x: int, cell_y: int) -> None:
    """dx-demo's 'Live' pill: a red dot plus the word, top-right of the tile."""
    text = "LIVE"
    fs, th = 0.5, 1
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    pad, dot_r, gap, margin = 7, 4, 5, 8
    pill_w = pad + dot_r * 2 + gap + tw + pad
    pill_h = pad + tht + pad
    x1 = cell_x + CELL_W - margin - pill_w
    y1 = cell_y + margin
    cv2.rectangle(board, (x1, y1), (x1 + pill_w, y1 + pill_h), (32, 32, 32), cv2.FILLED)
    cv2.rectangle(board, (x1, y1), (x1 + pill_w, y1 + pill_h), (200, 200, 200), 1)
    cv2.circle(board, (x1 + pad + dot_r, y1 + pill_h // 2), dot_r, (0, 0, 255), cv2.FILLED)
    cv2.putText(board, text, (x1 + pad + dot_r * 2 + gap, y1 + pad + tht),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)


def _draw_title_bar(board: np.ndarray, fps: float, active: int | None,
                    paused: bool, show_exit: bool) -> None:
    cv2.rectangle(board, (0, 0), (BRAND_W, TITLE_H), (0, 0, 200), cv2.FILLED)
    cv2.putText(board, "DX-M1  PCB AOI  -  4ch", (16, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(board, (BRAND_W, 0), (BOARD_W, TITLE_H), (0, 0, 0), cv2.FILLED)
    live_txt = f"LIVE: {MODEL_LABEL[active]}" if active else "LIVE: off"
    right = f"NPU {fps:5.1f} inf/s     {live_txt}"
    if paused:
        right = "PAUSED  |  " + right
    cv2.putText(board, right, (BRAND_W + 16, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 210, 255) if paused else (235, 235, 235), 2, cv2.LINE_AA)
    if show_exit:
        x, y, w, h = EXIT_BTN
        cv2.rectangle(board, (x, y), (x + w, y + h), (60, 60, 60), cv2.FILLED)
        cv2.rectangle(board, (x, y), (x + w, y + h), (200, 200, 200), 1)
        pad = 8
        cv2.line(board, (x + pad, y + pad), (x + w - pad, y + h - pad),
                 (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(board, (x + w - pad, y + pad), (x + pad, y + h - pad),
                 (255, 255, 255), 2, cv2.LINE_AA)


def compose_board(ch1, ch2, ch3, live, fps: float, paused: bool,
                  show_exit: bool = True) -> np.ndarray:
    """The 2x2 grid with the booth title bar on top."""
    live_frame = live.get_frame() if live else np.zeros((CELL_H, CELL_W, 3), np.uint8)
    grid = np.vstack([np.hstack([ch1.get_frame(), ch2.get_frame()]),
                      np.hstack([ch3.get_frame(), live_frame])])
    board = np.zeros((BOARD_H, BOARD_W, 3), np.uint8)
    board[TITLE_H:, :] = grid
    _draw_title_bar(board, fps, live.active if live else None, paused, show_exit)
    if live:
        _draw_live_badge(board, CELL_W, TITLE_H + CELL_H)
    return board


def run_qt(ch1, ch2, ch3, live, workers, pause_event, fps_meter) -> int:
    """PySide6 shell: the board is a cv2-composited numpy frame pushed into a
    QLabel on a timer -- the same approach dx-demo's yolo-multi demo uses.

    Imported here rather than at module scope so `--headless` verification
    still runs on a host with no Qt installed.
    """
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import QApplication, QLabel, QMainWindow

    class DemoWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("DX-M1 PCB AOI - 4 channel demo")
            self.label = QLabel()
            self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.label.setStyleSheet("background-color: #101010;")
            self.setCentralWidget(self.label)
            self.resize(BOARD_W, BOARD_H)
            self._board = compose_board(ch1, ch2, ch3, live, 0.0, False)
            self._closing = False
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._refresh)
            self.timer.start(33)

        def _refresh(self):
            fps_meter.tick()
            if not pause_event.is_set():
                self._board = compose_board(ch1, ch2, ch3, live, fps_meter.fps, False)
            else:
                # Repaint only the bar so the PAUSED state is visible while the
                # channel images stay frozen on the held frame.
                _draw_title_bar(self._board, fps_meter.fps,
                                live.active if live else None, True, True)
            h, w, ch = self._board.shape
            img = QImage(self._board.data, w, h, ch * w, QImage.Format.Format_BGR888)
            self.label.setPixmap(QPixmap.fromImage(img.copy()))

        def _board_point(self, pos):
            pix = self.label.pixmap()
            if pix is None or pix.isNull():
                return None
            sx = self._board.shape[1] / pix.width()
            sy = self._board.shape[0] / pix.height()
            off_x = self.label.x() + (self.label.width() - pix.width()) // 2
            off_y = self.label.y() + (self.label.height() - pix.height()) // 2
            return (pos.x() - off_x) * sx, (pos.y() - off_y) * sy

        def mousePressEvent(self, event):
            point = self._board_point(event.position().toPoint())
            if point is None:
                return
            x, y, w, h = EXIT_BTN
            if x <= point[0] <= x + w and y <= point[1] <= y + h:
                self.close()

        def keyPressEvent(self, event):
            key = event.key()
            if key in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
                self.close()
            elif key == Qt.Key.Key_P:
                if pause_event.is_set():
                    pause_event.clear()
                else:
                    pause_event.set()
            elif key == Qt.Key.Key_F:
                self.showNormal() if self.isFullScreen() else self.showFullScreen()
            elif live and key in (Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3):
                live.set_active({Qt.Key.Key_1: 1, Qt.Key.Key_2: 2, Qt.Key.Key_3: 3}[key])

        def closeEvent(self, event):
            if self._closing:
                event.accept()
                return
            self._closing = True
            self.timer.stop()
            for w in workers:
                w.stop()
            for w in workers:
                w.join(timeout=2.0)
            event.accept()

    app = QApplication(sys.argv)
    win = DemoWindow()
    win.showFullScreen()
    return app.exec()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true",
                    help="no imshow window; run for --duration seconds and save frames (verification mode)")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--no-live", action="store_true",
                    help="skip the USB-camera cell (for hosts with no camera)")
    ap.add_argument("--switch-test", action="store_true",
                    help="headless only: cycle the live cell through models 1/2/3 and save one frame each")
    ap.add_argument("--camera", type=int, default=0, help="USB camera index (default 0)")
    args = ap.parse_args()

    print("[grid] starting channels...", flush=True)
    pause_event = threading.Event()
    try:
        ch1 = YoloDatasetChannel("CH1 DeepPCB", CH1_DXNN, f"{CH1_IMAGES}/*.jpg", CH1_CLASSES, pause_event)
        ch3 = YoloDatasetChannel("CH3 SolDef", CH3_DXNN, f"{CH3_IMAGES}/*.jpg", CH3_CLASSES, pause_event,
                                 normal_classes=frozenset({"good"}))
        ch2 = PatchCoreDatasetChannel(CH2_DXNN, CH2_BANK, CH2_META, CH2_VISA_ROOT, pause_event)
        live = None if args.no_live else LiveChannel(pause_event, camera_index=args.camera)
    except Exception as e:  # noqa: BLE001
        # A bare traceback sends a booth operator hunting through Python frames.
        # Print the real error and a short checklist -- deliberately WITHOUT
        # guessing a cause: concurrent access is not one (two copies of this
        # demo were measured running side by side fine), and a corrupt .dxnn
        # segfaults in the native runtime rather than arriving here at all.
        print(f"\n[grid] startup failed: {e}\n", file=sys.stderr)
        print("  Check, in this order:", file=sys.stderr)
        print("    ls models/ data/                # assets present?  if not: bash setup.sh", file=sys.stderr)
        print("    systemctl is-active dxrt        # NPU service up?  if not: sudo systemctl start dxrt", file=sys.stderr)
        print("    dxrt-cli --status               # NPU visible?     if not: cold boot", file=sys.stderr)
        print("    ps -ef | grep [g]rid_viewer     # another copy up? (allowed, but uses the device)",
              file=sys.stderr)
        return 1

    workers = [ch1, ch2, ch3] + ([live] if live else [])
    for w in workers:
        w.start()
    print(f"[grid] all channels started ({len(workers)} threads).", flush=True)

    meter = FpsMeter([c for c in (ch1, ch2, ch3, live) if c])

    try:
        if args.headless:
            print(f"[grid] headless verification: running {args.duration}s", flush=True)
            out = Path(args.save_dir) if args.save_dir else None
            if out:
                out.mkdir(parents=True, exist_ok=True)
            if live and args.switch_test:
                # Spot-check that all 3 resident models actually respond to
                # set_active() -- not just that CH1 (the default) renders.
                per_model = max(args.duration / 3.0, 1.5)
                for n in (1, 2, 3):
                    live.set_active(n)
                    print(f"[grid] live switch -> model {n}, settling {per_model:.1f}s", flush=True)
                    deadline = time.time() + per_model
                    while time.time() < deadline:
                        time.sleep(0.25)
                        meter.tick()
                    if out:
                        cv2.imwrite(str(out / f"live_ch{n}.jpg"), live.get_frame())
            else:
                deadline = time.time() + args.duration
                while time.time() < deadline:
                    time.sleep(0.25)
                    meter.tick()
            # Same board the Qt shell shows, so the verification image proves
            # the booth chrome renders too, not just the four cells.
            grid = compose_board(ch1, ch2, ch3, live, meter.fps, False, show_exit=True)
            if out:
                cv2.imwrite(str(out / "grid.jpg"), grid)
                cv2.imwrite(str(out / "ch1.jpg"), ch1.get_frame())
                cv2.imwrite(str(out / "ch2.jpg"), ch2.get_frame())
                cv2.imwrite(str(out / "ch3.jpg"), ch3.get_frame())
                if live:
                    cv2.imwrite(str(out / "live.jpg"), live.get_frame())
                print(f"[grid] saved frames -> {out}", flush=True)
            print("[grid] RESULT: PASS" if grid.any() else "[grid] RESULT: FAIL (blank frame)", flush=True)
        else:
            print("[grid] keys: 1/2/3 = switch live model, p = pause/resume "
                  "(for explaining at a booth), f = fullscreen toggle, "
                  "q/Esc = quit (or click the X in the title bar)", flush=True)
            return run_qt(ch1, ch2, ch3, live, workers, pause_event, meter)
    finally:
        for w in workers:
            w.stop()
        # Wait for each worker to actually leave its loop before the
        # interpreter starts tearing down -- signalling stop() and returning
        # immediately let a worker still be inside a native dx_engine.run()
        # call when module teardown began, which raised a bare "terminate
        # called without an active exception" (observed empirically; fixed by
        # this join, not by guessing).
        for w in workers:
            w.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
