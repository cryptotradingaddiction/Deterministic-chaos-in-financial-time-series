#!/usr/bin/env python3
"""Cao-method false-nearest-neighbor visualization (dual 2D projections)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors

from attractor_core import coord_label
from cao_ import find_nonzero_neighbor


@dataclass
class CaoFnnExemplars:
    """Exemplar true/false neighbor pairs from Cao's a_i(m) ratios."""

    m: int
    tau: int
    embedded_m: np.ndarray
    nn_index: np.ndarray
    nn_distance: np.ndarray
    distance_m_plus_1: np.ndarray
    abs_diff_new: np.ndarray
    a_i: np.ndarray
    E_m: float
    true_i: int
    true_j: int
    false_i: int
    false_j: int
    a_i_true: float
    a_i_false: float
    proj_a: tuple[int, int]
    proj_b: tuple[int, int]

    def format_stats(self, symbol: str) -> str:
        z = self.proj_a[1]
        lines = [
            f"Cao FNN — {symbol}",
            f"Embedding dimension m = {self.m}, delay tau = {self.tau}",
            f"E({self.m}) = {self.E_m:.4f}  (mean a_i(m) = d_(m+1) / d_m, Chebyshev)",
            "",
            "True neighbors (filled): NN pair with a_i(m) ~ 1",
            f"  indices ({self.true_i}, {self.true_j}),  a_i = {self.a_i_true:.4f}",
            f"  d_m = {self.nn_distance[self.true_i]:.6g},  d_(m+1) = {self.distance_m_plus_1[self.true_i]:.6g}",
            "",
            "False neighbors (hollow): NN pair with large a_i(m)",
            f"  indices ({self.false_i}, {self.false_j}),  a_i = {self.a_i_false:.4f}",
            f"  d_m = {self.nn_distance[self.false_i]:.6g},  d_(m+1) = {self.distance_m_plus_1[self.false_i]:.6g}",
            f"  |Delta x(t+{self.m * self.tau})| = {self.abs_diff_new[self.false_i]:.6g}",
            "",
            f"Left panel: dim {self.proj_a[0]} vs dim {z}",
            f"Right panel: dim {self.proj_b[0]} vs dim {z}",
        ]
        return "\n".join(lines)


def cao_embed_matrix(data: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Takens matrix at dimension m (same layout as cao_.py)."""
    x = np.asarray(data, dtype=np.float64).ravel()
    n_valid = len(x) - m * tau
    if n_valid < 2:
        raise ValueError(f"Series too short for Cao m={m}, tau={tau}.")
    x_m = np.zeros((n_valid, m), dtype=np.float64)
    for k in range(m):
        x_m[:, k] = x[k * tau : n_valid + k * tau]
    return x_m


def cao_neighbor_ratios(data: np.ndarray, m: int, tau: int) -> dict:
    """
    Per-point Cao ratios a_i(m) using Chebyshev NN in R^m and lift to m+1.
    Matches calculate_for_m() in cao_.py.
    """
    x = np.asarray(data, dtype=np.float64).ravel()
    x_m = cao_embed_matrix(x, m, tau)
    n_valid = x_m.shape[0]

    nn_model = NearestNeighbors(n_neighbors=2, metric="chebyshev", algorithm="kd_tree", n_jobs=1)
    nn_model.fit(x_m)
    distances, indices = nn_model.kneighbors(x_m, return_distance=True)
    nn_distance = distances[:, 1].copy()
    nn_index = indices[:, 1].copy()

    zero_mask = nn_distance == 0.0
    if np.any(zero_mask):
        for i in np.where(zero_mask)[0]:
            new_dist, new_idx = find_nonzero_neighbor(nn_model, x_m[i : i + 1], start_k=3, max_k=100)
            nn_distance[i] = new_dist
            nn_index[i] = new_idx

    next_coord = x[m * tau : n_valid + m * tau]
    nn_next_coord = x[nn_index + m * tau]
    abs_diff_new = np.abs(next_coord - nn_next_coord)
    distance_m_plus_1 = np.maximum(nn_distance, abs_diff_new)

    with np.errstate(divide="ignore", invalid="ignore"):
        a_i = distance_m_plus_1 / nn_distance
    finite = a_i[np.isfinite(a_i)]
    e_m = float(np.mean(finite)) if finite.size else float("nan")

    return {
        "embedded_m": x_m,
        "nn_index": nn_index,
        "nn_distance": nn_distance,
        "distance_m_plus_1": distance_m_plus_1,
        "abs_diff_new": abs_diff_new,
        "a_i": a_i,
        "E_m": e_m,
    }


def _pick_exemplar_indices(
    a_i: np.ndarray,
    nn_distance: np.ndarray,
    nn_index: np.ndarray,
    *,
    false_threshold: float = 2.0,
) -> tuple[int, int, int, int, float, float]:
    finite = np.isfinite(a_i) & (nn_distance > 0)
    if not np.any(finite):
        return 0, int(nn_index[0]), 0, int(nn_index[0]), float("nan"), float("nan")

    dist_ok = nn_distance > np.percentile(nn_distance[finite], 5)

    true_pool = np.where(finite & dist_ok & (a_i >= 1.0) & (a_i < 1.8))[0]
    if true_pool.size == 0:
        true_pool = np.where(finite & dist_ok & (a_i < false_threshold))[0]
    if true_pool.size == 0:
        true_pool = np.where(finite)[0]
    true_i = int(true_pool[np.argmin(a_i[true_pool])])
    true_j = int(nn_index[true_i])

    false_pool = np.where(finite & dist_ok & (a_i >= false_threshold))[0]
    if false_pool.size == 0:
        false_pool = np.where(finite & dist_ok)[0]
        false_i = int(false_pool[np.argmax(a_i[false_pool])])
    else:
        false_i = int(false_pool[np.argmax(a_i[false_pool])])
    false_j = int(nn_index[false_i])

    if true_i == false_i and finite.sum() > 2:
        alt = np.where(finite & dist_ok & (np.arange(len(a_i)) != true_i))[0]
        if alt.size:
            false_i = int(alt[np.argmax(a_i[alt])])
            false_j = int(nn_index[false_i])

    return true_i, true_j, false_i, false_j, float(a_i[true_i]), float(a_i[false_i])


def build_cao_fnn_exemplars(
    data: np.ndarray,
    m: int,
    tau: int,
    *,
    proj_a: tuple[int, int] | None = None,
    proj_b: tuple[int, int] | None = None,
) -> CaoFnnExemplars:
    """Compute Cao ratios and pick true/false exemplar pairs for dual-panel plot."""
    if m < 2:
        raise ValueError("Cao FNN view requires m >= 2.")
    ratios = cao_neighbor_ratios(data, m, tau)
    x_m = ratios["embedded_m"]
    if m < 3:
        z = 1
        proj_a = proj_a or (0, 1)
        proj_b = proj_b or (0, 1)
    else:
        z = min(2, m - 1)
        proj_a = proj_a or (0, z)
        proj_b = proj_b or (1, z)

    for name, pair in (("proj_a", proj_a), ("proj_b", proj_b)):
        h, v = pair
        if not (0 <= h < m and 0 <= v < m):
            raise ValueError(f"{name} dims out of range for m={m}: {pair}")

    true_i, true_j, false_i, false_j, a_true, a_false = _pick_exemplar_indices(
        ratios["a_i"], ratios["nn_distance"], ratios["nn_index"],
    )

    return CaoFnnExemplars(
        m=m,
        tau=tau,
        embedded_m=x_m,
        nn_index=ratios["nn_index"],
        nn_distance=ratios["nn_distance"],
        distance_m_plus_1=ratios["distance_m_plus_1"],
        abs_diff_new=ratios["abs_diff_new"],
        a_i=ratios["a_i"],
        E_m=ratios["E_m"],
        true_i=true_i,
        true_j=true_j,
        false_i=false_i,
        false_j=false_j,
        a_i_true=a_true,
        a_i_false=a_false,
        proj_a=proj_a,
        proj_b=proj_b,
    )


def axis_pair_labels(dim_h: int, dim_v: int, tau: int) -> tuple[str, str]:
    return coord_label(dim_h, tau), coord_label(dim_v, tau)
