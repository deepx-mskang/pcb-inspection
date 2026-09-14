#!/usr/bin/env python3
"""Backbone backends.

Both backends take an image PATH and return (layer2, layer3) as NCHW float32,
applying the identical geometric preprocessing. They differ only in the input
convention each artifact declares:

  ONNX : float32 NCHW, raw [0,255]  -- normalization is inside the graph
  DXNN : uint8   NHWC, raw [0,255]  -- dxcom rewrites the input to uint8 NHWC
                                       and folds the normalization onto the NPU

Keeping the geometry identical is what makes the accuracy comparison fair: any
remaining gap is quantization, not preprocessing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preproc import load_nhwc_u8, load_raw_nchw  # noqa: E402


class Backend:
    name = "base"

    def embed(self, image_path: str) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class OnnxBackend(Backend):
    name = "onnx"

    def __init__(self, model_path: str):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(model_path,
                                         providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self._order = [o.name for o in self.sess.get_outputs()]

    def embed(self, image_path):
        outs = self.sess.run(None, {self.input_name: load_raw_nchw(image_path)})
        by_name = dict(zip(self._order, outs))
        return by_name["layer2"], by_name["layer3"]


class DxnnBackend(Backend):
    """DX-M1 backend.

    Outputs are matched by channel count rather than by index, because the
    compiled graph does not guarantee the ONNX output order.
    """

    name = "dxnn"

    def __init__(self, model_path: str):
        from dx_engine import InferenceEngine
        self.ie = InferenceEngine(model_path)

    @staticmethod
    def _nchw(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        if a.ndim == 3:
            a = a[None, ...]
        if a.shape[1] in (512, 1024):
            return a
        if a.shape[-1] in (512, 1024):
            return np.ascontiguousarray(a.transpose(0, 3, 1, 2))
        raise ValueError(f"cannot identify channel axis in {a.shape}")

    def embed(self, image_path):
        outs = self.ie.run([np.ascontiguousarray(load_nhwc_u8(image_path))])
        mapped = [self._nchw(o) for o in outs]
        f2 = next((m for m in mapped if m.shape[1] == 512), None)
        f3 = next((m for m in mapped if m.shape[1] == 1024), None)
        if f2 is None or f3 is None:
            raise RuntimeError(
                f"expected 512ch and 1024ch outputs, got {[m.shape for m in mapped]}")
        return f2, f3


def make_backend(kind: str, model_path: str) -> Backend:
    if kind == "onnx":
        return OnnxBackend(model_path)
    if kind == "dxnn":
        return DxnnBackend(model_path)
    raise ValueError(f"unknown backend: {kind}")
