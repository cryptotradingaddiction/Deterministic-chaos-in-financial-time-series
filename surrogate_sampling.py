#!/usr/bin/env python3
"""Sampling helpers for time-series hypothesis testing."""

from __future__ import annotations

import numpy as np


def load_series_1d(path: str) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim != 1:
        data = data[:, 0].ravel()
    return np.asarray(data, dtype=float)


def stationary_bootstrap_samples(
    original: np.ndarray,
    n_samples: int,
    mean_block_length: float | None = None,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Generate stationary-bootstrap pseudo-series.

    The Politis-Romano stationary bootstrap draws blocks with geometrically
    distributed lengths. At every output position a new block starts with
    probability p = 1 / mean_block_length; otherwise the source index advances
    by one modulo n. If `mean_block_length` is not supplied, sqrt(n) is used as
    a conservative data-dependent default.
    """
    arr = np.asarray(original, dtype=float).ravel()
    n = len(arr)
    n_samples = max(0, int(n_samples))
    if n == 0:
        return [arr.copy() for _ in range(n_samples)]

    if mean_block_length is None or not np.isfinite(mean_block_length) or mean_block_length <= 0:
        mean_block_length = max(2.0, float(np.sqrt(n)))
    mean_block_length = max(1.0, min(float(mean_block_length), float(n)))
    restart_prob = 1.0 / mean_block_length

    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    for _ in range(n_samples):
        sample = np.empty(n, dtype=float)
        src = int(rng.integers(0, n))
        for i in range(n):
            if i == 0 or rng.random() < restart_prob:
                src = int(rng.integers(0, n))
            else:
                src = (src + 1) % n
            sample[i] = arr[src]
        out.append(sample)
    return out
