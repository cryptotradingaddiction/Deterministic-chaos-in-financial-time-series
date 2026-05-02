#!/usr/bin/env python3
"""Block-permutation surrogates for time-series hypothesis testing."""

from __future__ import annotations

import numpy as np


def load_series_1d(path: str) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim != 1:
        data = data[:, 0].ravel()
    return np.asarray(data, dtype=float)


def generate_permuted_samples(
    original: np.ndarray,
    n_samples: int,
    *_args,
    **_kwargs,
) -> list[np.ndarray]:
    """
    Surrogates via block permutation:
    1) split the original series into N contiguous blocks of equal size (plus tail),
    2) permute block order with randperm-like index shuffle,
    3) concatenate blocks back.

    This preserves short-range dynamics inside blocks while destroying long-range ordering.
    Optional kwargs:
      - n_blocks (int): number of blocks (default 100).
    """
    arr = np.asarray(original, dtype=float).ravel()
    n = len(arr)
    if n == 0:
        return [arr.copy() for _ in range(max(0, int(n_samples)))]

    n_blocks = int(_kwargs.get("n_blocks", 100))
    n_blocks = max(2, min(n_blocks, n))
    block_size = n // n_blocks
    if block_size <= 0:
        block_size = 1
        n_blocks = n

    blocks = []
    start = 0
    for _ in range(n_blocks - 1):
        end = min(start + block_size, n)
        blocks.append(arr[start:end])
        start = end
    blocks.append(arr[start:n])  # tail block (possibly larger)

    out: list[np.ndarray] = []
    for _ in range(n_samples):
        perm = np.random.permutation(len(blocks))
        out.append(np.concatenate([blocks[i] for i in perm]).copy())
    return out
