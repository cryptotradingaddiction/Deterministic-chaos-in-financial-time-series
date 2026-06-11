#!/usr/bin/env python3
"""LRU caches for attractor viewer data and embeddings."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from attractor_core import delay_embedding, load_logreturns, make_surrogate_series


class _LRU:
    def __init__(self, max_items: int = 32):
        self._max = max_items
        self._data: OrderedDict[Any, Any] = OrderedDict()

    def get(self, key: Any):
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: Any, value: Any) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()


_series_cache = _LRU(16)
_embed_cache = _LRU(48)
_surr_cache = _LRU(24)
_pca_cache = _LRU(24)


def _series_path_mtime(symbol: str, config) -> float:
    from config_loader import get_data_dir, pipeline_logreturn_files, prefer_liquidity_cut

    data_dir = Path(get_data_dir(config))
    filename = next(
        (f for f in pipeline_logreturn_files(ext="csv", config=config) if f.startswith(symbol)),
        None,
    )
    if filename is None:
        return 0.0
    path = Path(prefer_liquidity_cut(str(data_dir / filename)))
    return path.stat().st_mtime if path.is_file() else 0.0


def series_key(symbol: str, config, start_frac: float, end_frac: float, test_cap: int | None) -> tuple:
    return (
        symbol,
        round(start_frac, 4),
        round(end_frac, 4),
        test_cap,
        _series_path_mtime(symbol, config),
    )


def get_series(symbol: str, config, *, start_frac: float, end_frac: float, test_cap: int | None) -> np.ndarray:
    key = series_key(symbol, config, start_frac, end_frac, test_cap)
    hit = _series_cache.get(key)
    if hit is not None:
        return hit
    arr = load_logreturns(
        symbol, config, max_points=test_cap, start_frac=start_frac, end_frac=end_frac,
    )
    _series_cache.put(key, arr)
    return arr


def get_embedding(series: np.ndarray, tau: int, m: int, series_key_tuple: tuple) -> np.ndarray:
    key = (series_key_tuple, tau, m)
    hit = _embed_cache.get(key)
    if hit is not None:
        return hit
    emb = delay_embedding(series, m=m, tau=tau)
    _embed_cache.put(key, emb)
    return emb


def get_surrogate(series: np.ndarray, series_key_tuple: tuple, seed: int = 0) -> np.ndarray:
    key = (series_key_tuple, seed)
    hit = _surr_cache.get(key)
    if hit is not None:
        return hit
    surr = make_surrogate_series(series, seed=seed)
    _surr_cache.put(key, surr)
    return surr


def get_pca_view(embedded: np.ndarray, embed_key: tuple) -> np.ndarray:
    key = (embed_key, "pca", embedded.shape)
    hit = _pca_cache.get(key)
    if hit is not None:
        return hit
    from sklearn.decomposition import PCA

    n_comp = 3 if embedded.shape[1] >= 3 else 2
    pts = PCA(n_components=n_comp).fit_transform(embedded)
    _pca_cache.put(key, pts)
    return pts


def clear_all() -> None:
    _series_cache.clear()
    _embed_cache.clear()
    _surr_cache.clear()
    _pca_cache.clear()
