#!/usr/bin/env python3
"""PatchCore patch-embedding construction.

Turns the backbone's two feature maps into one locally-aware patch embedding
per spatial position, exactly as in the PatchCore paper:

  1. bilinear-upsample layer3 to layer2's resolution
  2. concatenate along channels          -> (1, 1536, 28, 28)
  3. 3x3 average pool with padding 1     -> local neighbourhood aggregation
  4. flatten spatial positions           -> (784, 1536)

Everything here runs on the host; only the backbone runs on the NPU.
"""
from __future__ import annotations

import numpy as np


def _upsample_bilinear(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize of an NCHW array, matching torch align_corners=False.

    The exact-2x case is the one PatchCore always hits (layer3 14x14 -> layer2
    28x28) and its interpolation weights are the constants 0.25/0.75, so it is
    done with plain slicing. That is ~3x faster than the general fancy-index
    path and matters: this runs once per board per channel, and the demo grid
    runs many channels at once.
    """
    n, c, h, w = x.shape
    if (h, w) == (out_h, out_w):
        return x
    if (out_h, out_w) == (2 * h, 2 * w):
        return _upsample_exact_2x(x)
    return _upsample_general(x, out_h, out_w)


def _upsample_exact_2x(x: np.ndarray) -> np.ndarray:
    n, c, h, w = x.shape
    a = np.empty((n, c, 2 * h, w), x.dtype)
    a[:, :, 0] = x[:, :, 0]
    a[:, :, -1] = x[:, :, -1]
    a[:, :, 1:-1:2] = 0.75 * x[:, :, :-1] + 0.25 * x[:, :, 1:]
    a[:, :, 2:-1:2] = 0.25 * x[:, :, :-1] + 0.75 * x[:, :, 1:]
    b = np.empty((n, c, 2 * h, 2 * w), x.dtype)
    b[:, :, :, 0] = a[:, :, :, 0]
    b[:, :, :, -1] = a[:, :, :, -1]
    b[:, :, :, 1:-1:2] = 0.75 * a[:, :, :, :-1] + 0.25 * a[:, :, :, 1:]
    b[:, :, :, 2:-1:2] = 0.25 * a[:, :, :, :-1] + 0.75 * a[:, :, :, 1:]
    return b


def _upsample_general(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    n, c, h, w = x.shape
    ys = np.clip((np.arange(out_h, dtype=np.float32) + 0.5) * (h / out_h) - 0.5, 0, h - 1)
    xs = np.clip((np.arange(out_w, dtype=np.float32) + 0.5) * (w / out_w) - 0.5, 0, w - 1)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0).astype(np.float32)[None, None, :, None]
    wx = (xs - x0).astype(np.float32)[None, None, None, :]
    top = x[:, :, y0, :][:, :, :, x0] * (1 - wx) + x[:, :, y0, :][:, :, :, x1] * wx
    bot = x[:, :, y1, :][:, :, :, x0] * (1 - wx) + x[:, :, y1, :][:, :, :, x1] * wx
    return top * (1 - wy) + bot * wy


def _avgpool3x3(x: np.ndarray) -> np.ndarray:
    """3x3 average pool, stride 1, padding 1 (zero-padded, count includes pad)."""
    n, c, h, w = x.shape
    p = np.zeros((n, c, h + 2, w + 2), dtype=x.dtype)
    p[:, :, 1:-1, 1:-1] = x
    out = np.zeros_like(x)
    for dy in range(3):
        for dx in range(3):
            out += p[:, :, dy:dy + h, dx:dx + w]
    return out / 9.0


def build_patch_embedding(f2: np.ndarray, f3: np.ndarray) -> np.ndarray:
    """(1,512,28,28) + (1,1024,14,14) -> (784, 1536) float32."""
    if f2.ndim != 4 or f3.ndim != 4:
        raise ValueError(f"expected NCHW arrays, got {f2.shape} and {f3.shape}")
    f2 = f2.astype(np.float32, copy=False)
    f3 = f3.astype(np.float32, copy=False)
    h, w = f2.shape[2], f2.shape[3]
    f3u = _upsample_bilinear(f3, h, w)
    cat = np.concatenate([f2, f3u], axis=1)
    cat = _avgpool3x3(cat)
    n, c, hh, ww = cat.shape
    return cat.reshape(n, c, hh * ww)[0].T.copy()          # (H*W, C)


def embedding_grid(f2: np.ndarray) -> tuple[int, int]:
    return int(f2.shape[2]), int(f2.shape[3])
