#!/usr/bin/env python3
"""PatchCore memory bank: greedy coreset selection + kNN anomaly scoring.

PatchCore has no gradient training step. "Fitting" the model means collecting
patch embeddings of defect-free boards and keeping a representative subset.
That makes it the only strong anomaly-detection option on a GPU-less host.
"""
from __future__ import annotations

import numpy as np


def greedy_coreset(x: np.ndarray, n_select: int, proj_dim: int = 128,
                   seed: int = 0) -> np.ndarray:
    """k-center-greedy subset selection, returning selected indices.

    Selection runs in a Johnson-Lindenstrauss random projection (as in the
    PatchCore reference implementation) so the O(n * k) distance updates stay
    cheap; the returned indices address the ORIGINAL full-dimension rows.
    """
    n = x.shape[0]
    if n_select >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    if proj_dim and proj_dim < d:
        proj = rng.normal(0.0, 1.0 / np.sqrt(proj_dim), size=(d, proj_dim)).astype(np.float32)
        y = x @ proj
    else:
        y = x
    sq = (y * y).sum(1)

    start = int(rng.integers(n))
    selected = [start]
    min_d = sq + sq[start] - 2.0 * (y @ y[start])
    min_d[start] = -1.0
    for _ in range(n_select - 1):
        nxt = int(np.argmax(min_d))
        selected.append(nxt)
        nd = sq + sq[nxt] - 2.0 * (y @ y[nxt])
        np.minimum(min_d, nd, out=min_d)
        min_d[nxt] = -1.0
    return np.asarray(selected, dtype=np.int64)


class MemoryBank:
    """Nearest-neighbour anomaly scorer over coreset-selected normal patches."""

    def __init__(self, bank: np.ndarray, grid: tuple[int, int]):
        self.bank = np.ascontiguousarray(bank.astype(np.float32))
        self.bank_sq = (self.bank * self.bank).sum(1)
        self.grid = grid

    @classmethod
    def fit(cls, embeddings: np.ndarray, grid: tuple[int, int],
            ratio: float = 0.01, seed: int = 0) -> "MemoryBank":
        n_select = max(1, int(round(embeddings.shape[0] * ratio)))
        idx = greedy_coreset(embeddings, n_select, seed=seed)
        return cls(embeddings[idx], grid)

    def score_map(self, emb: np.ndarray) -> np.ndarray:
        """Per-patch nearest-neighbour distance, reshaped to the feature grid."""
        emb = emb.astype(np.float32, copy=False)
        d2 = ((emb * emb).sum(1)[:, None]
              + self.bank_sq[None, :]
              - 2.0 * (emb @ self.bank.T))
        np.maximum(d2, 0.0, out=d2)
        nn = np.sqrt(d2.min(axis=1))
        return nn.reshape(self.grid)

    @staticmethod
    def image_score(score_map: np.ndarray) -> float:
        return float(score_map.max())

    def save(self, path: str) -> None:
        np.savez_compressed(path, bank=self.bank, grid=np.asarray(self.grid))

    @classmethod
    def load(cls, path: str) -> "MemoryBank":
        z = np.load(path)
        return cls(z["bank"], tuple(int(v) for v in z["grid"]))
