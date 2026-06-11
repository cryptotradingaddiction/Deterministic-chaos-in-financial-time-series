#!/usr/bin/env python3
"""Recurrence plot construction and RQA helpers for the desktop viewer."""

from __future__ import annotations

import numpy as np

from hypothesis_config import RQA_KEYS, RQA_RADIUS_PERCENTILE_DEFAULT
from invariants_rqa import (
    compute_percentile_radius,
    compute_pyrqa_metrics,
    compute_rqa_trend,
    embed_series,
    tisean_theiler_min_diagonal_k,
)


def subsample_display_indices(n: int, max_points: int = 2500) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, max_points, dtype=int).astype(int)


def diagonal_recurrence_profile(
    embedded: np.ndarray,
    radius: float,
    *,
    min_k: int,
    max_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Recurrence density along off-LOI diagonals (for TREND visualization)."""
    n = embedded.shape[0]
    if n < 3:
        return np.empty(0), np.empty(0)
    min_k = max(1, int(min_k))
    if max_k is None:
        max_k = min(max(min_k + 1, n // 10), n - 1)
    max_k = max(min_k + 1, min(int(max_k), n - 1))
    xs, ys = [], []
    radius = float(radius)
    for k in range(min_k, max_k + 1):
        diff = embedded[:-k] - embedded[k:]
        if diff.size == 0:
            continue
        recurrent = np.sqrt(np.sum(diff * diff, axis=1)) <= radius
        xs.append(float(k))
        ys.append(float(np.mean(recurrent)))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def build_recurrence_matrix(
    embedded: np.ndarray,
    radius: float,
    *,
    theiler_w: int,
    display_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Boolean recurrence matrix for display (subsampled).

    Returns (matrix, index_map, theiler_corrector).
    """
    if embedded.shape[0] < 2:
        return np.zeros((0, 0), dtype=np.uint8), np.arange(0), 1
    idx = display_indices if display_indices is not None else np.arange(embedded.shape[0], dtype=int)
    sub = embedded[idx]
    n = sub.shape[0]
    radius = float(radius)
    theiler_corr = tisean_theiler_min_diagonal_k(theiler_w)

    diff = sub[:, None, :] - sub[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    recurrence = (dist <= radius).astype(np.uint8)

    for i in range(n):
        j_lo = max(0, i - theiler_corr + 1)
        j_hi = min(n, i + theiler_corr)
        recurrence[i, j_lo:j_hi] = 2

    return recurrence, idx, theiler_corr


def compute_rqa_metrics_custom_m(
    series: np.ndarray,
    delay: int,
    theiler: int,
    radius: float,
    m: int,
) -> dict[str, float]:
    """PyRQA metrics with configurable embedding dimension m."""
    from pyrqa.analysis_type import Classic
    from pyrqa.computation import RQAComputation
    from pyrqa.metric import EuclideanMetric
    from pyrqa.neighbourhood import FixedRadius
    from pyrqa.settings import Settings
    from pyrqa.time_series import TimeSeries

    try:
        ts = TimeSeries(series, embedding_dimension=int(m), time_delay=int(delay))
        pyrqa_theiler = tisean_theiler_min_diagonal_k(theiler)
        settings = Settings(
            ts,
            analysis_type=Classic,
            neighbourhood=FixedRadius(float(radius)),
            similarity_measure=EuclideanMetric,
            theiler_corrector=pyrqa_theiler,
        )
        computation = RQAComputation.create(settings, verbose=False)
        result = computation.run()
        trend = compute_rqa_trend(
            series, delay=delay, radius=radius, min_k=pyrqa_theiler, m=m,
        )
        return {
            "RR": float(result.recurrence_rate),
            "DET": float(result.determinism),
            "LAM": float(result.laminarity),
            "MAXLINE": float(result.longest_diagonal_line),
            "ENTR": float(result.entropy_diagonal_lines),
            "TT": float(result.trapping_time),
            "TREND": float(trend),
        }
    except Exception:
        return {k: float("nan") for k in RQA_KEYS}


def resolve_radius(
    series: np.ndarray,
    *,
    delay: int,
    m: int,
    mode: str,
    manual_radius: float,
    percentile: float,
) -> float:
    if mode == "manual":
        return float(manual_radius)
    r = compute_percentile_radius(
        series, delay=delay, m=m, percentile=percentile,
    )
    if np.isfinite(r):
        return float(r)
    return float(manual_radius)


def format_recurrence_stats(
    symbol: str,
    *,
    tau: int,
    m: int,
    radius: float,
    theiler_w: int,
    n_series: int,
    n_embedded: int,
    n_display: int,
    metrics: dict[str, float],
    theiler_corrector: int,
) -> str:
    lines = [
        f"Recurrence analysis — {symbol}",
        f"Series length N = {n_series}",
        f"Embedded points = {n_embedded}  (m={m}, tau={tau})",
        f"Display matrix = {n_display} x {n_display}",
        f"Radius epsilon = {radius:.8g}",
        f"Theiler W = {theiler_w}  (PyRQA corrector = {theiler_corrector})",
        "",
        "RQA metrics (PyRQA, full series):",
    ]
    for key in RQA_KEYS:
        val = metrics.get(key, float("nan"))
        lines.append(f"  {key:8s} = {val:.6g}" if np.isfinite(val) else f"  {key:8s} = n/a")
    return "\n".join(lines)


def compute_recurrence_bundle(
    series: np.ndarray,
    *,
    tau: int,
    m: int,
    radius: float,
    theiler_w: int,
    max_display: int = 2500,
) -> dict:
    """All data needed to paint the recurrence viewer."""
    embedded = embed_series(series, delay=tau, m=m)
    n_emb = embedded.shape[0]
    disp_idx = subsample_display_indices(n_emb, max_display)
    matrix, idx_map, theiler_corr = build_recurrence_matrix(
        embedded, radius, theiler_w=theiler_w, display_indices=disp_idx,
    )
    diag_k, diag_density = diagonal_recurrence_profile(
        embedded, radius, min_k=theiler_corr,
    )
    metrics = compute_rqa_metrics_custom_m(series, tau, theiler_w, radius, m)

    phase2d = embedded[:, [0, min(1, m - 1)]] if m >= 2 else embedded

    return {
        "embedded": embedded,
        "matrix": matrix,
        "display_indices": idx_map,
        "diag_k": diag_k,
        "diag_density": diag_density,
        "metrics": metrics,
        "phase2d": phase2d,
        "theiler_corrector": theiler_corr,
        "radius": float(radius),
        "n_embedded": n_emb,
        "n_display": matrix.shape[0],
    }
