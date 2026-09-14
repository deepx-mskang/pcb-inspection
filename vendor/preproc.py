#!/usr/bin/env python3
"""Canonical PatchCore preprocessing.

IMPORTANT -- normalization lives INSIDE the exported graph, not here.

dxcom rewrites the compiled model's input to uint8 NHWC [0,255] regardless of
the ONNX input being float32 NCHW, and folds the leading arithmetic into the
NPU. So the graph is exported taking RAW [0,255] pixels and doing /255 and the
ImageNet normalize itself. This module therefore only performs the GEOMETRIC
transform (resize + center crop), which must be identical for every backend.
"""
import numpy as np
from PIL import Image

RESIZE = 256
CROP = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_hwc_u8(path: str) -> np.ndarray:
    """Geometric transform only -> (224, 224, 3) uint8 RGB."""
    img = Image.open(path).convert("RGB").resize((RESIZE, RESIZE), Image.BILINEAR)
    off = (RESIZE - CROP) // 2
    img = img.crop((off, off, off + CROP, off + CROP))
    return np.asarray(img, dtype=np.uint8)


def load_raw_chw(path: str) -> np.ndarray:
    """(3, 224, 224) float32 in [0,255] -- the ONNX graph's expected input."""
    return np.ascontiguousarray(
        load_hwc_u8(path).astype(np.float32).transpose(2, 0, 1))


def load_raw_nchw(path: str) -> np.ndarray:
    return load_raw_chw(path)[None, ...]


def load_nhwc_u8(path: str) -> np.ndarray:
    """(1, 224, 224, 3) uint8 -- the DXNN model's expected input."""
    return load_hwc_u8(path)[None, ...]
