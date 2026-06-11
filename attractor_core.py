#!/usr/bin/env python3
"""Computational helpers for the interactive Takens attractor viewer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config_loader import (
    PIPELINE_SYMBOLS,
    get_data_dir,
    pipeline_logreturn_files,
    prefer_liquidity_cut,
)

M_MIN = 2
M_MAX = 60
TAU_MAX = 2000


def coord_label(dim_index: int, tau: int) -> str:
    return "x(t)" if dim_index == 0 else f"x(t+{dim_index * tau})"


def max_feasible_tau(series_n: int, m: int) -> int:
    if series_n < 2:
        return 0
    if m <= 1:
        return series_n - 1
    return max(0, (series_n - 1) // (m - 1))


def delay_embedding(data: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Vectorized Takens embedding (float32 internally for speed)."""
    x = np.asarray(data, dtype=np.float64).ravel()
    n = x.size
    max_delay = (m - 1) * tau
    if max_delay >= n:
        raise ValueError(f"Series too short for m={m}, tau={tau} (need > {max_delay} points).")
    num_points = n - max_delay
    cols = [x[i * tau : i * tau + num_points] for i in range(m)]
    return np.column_stack(cols)


def load_logreturns(
    symbol: str,
    config=None,
    *,
    max_points: int | None = None,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
) -> np.ndarray:
    from config_loader import load_config

    cfg = config or load_config()
    data_dir = Path(get_data_dir(cfg))
    filename = next(
        (f for f in pipeline_logreturn_files(ext="csv", config=cfg) if f.startswith(symbol)),
        None,
    )
    if filename is None:
        raise FileNotFoundError(f"No pipeline CSV registered for symbol {symbol}.")
    path = prefer_liquidity_cut(str(data_dir / filename))
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    x = df["log_return"].astype(str).str.replace(",", ".", regex=False).astype(float)
    x = x[np.isfinite(x)].to_numpy(dtype=float)
    n = len(x)
    if n == 0:
        return x
    i0 = int(np.clip(np.floor(start_frac * (n - 1)), 0, n - 1))
    i1 = int(np.clip(np.ceil(end_frac * (n - 1)), i0 + 1, n))
    x = x[i0:i1]
    if max_points is not None and len(x) > max_points:
        x = x[-max_points:]
    return x


def subsample_indices(n: int, max_display: int) -> np.ndarray:
    if n <= max_display:
        return np.arange(n, dtype=int)
    stride = int(np.ceil(n / max_display))
    return np.arange(0, n, stride, dtype=int)


def dim_triple_sequence(m: int) -> list[tuple[int, int, int]]:
    """Consecutive Takens triples: (0,1,2), (1,2,3), ..."""
    if m < 3:
        return [(0, 1, 0)]
    return [(k, k + 1, k + 2) for k in range(m - 2)]


def project_embedding(
    embedded: np.ndarray,
    mode: str,
    view_dims: tuple[int, int, int],
    tau: int,
) -> tuple[np.ndarray, list[str], str]:
    """
    Return (points Nx2 or Nx3, axis labels, projection kind '2d'|'3d').
    mode: 'manual' | 'pca'
    """
    m = embedded.shape[1]

    if mode == "pca" and m >= 2:
        from sklearn.decomposition import PCA

        n_comp = 3 if m >= 3 else 2
        pts = PCA(n_components=n_comp).fit_transform(embedded)
        labels = [f"PC{i + 1}" for i in range(n_comp)]
        return pts, labels, "3d" if n_comp == 3 else "2d"

    if m < 3:
        d0, d1 = view_dims[0], view_dims[1]
        pts = embedded[:, [d0, d1]]
        return pts, [coord_label(d0, tau), coord_label(d1, tau)], "2d"

    dims = list(view_dims)
    pts = embedded[:, dims]
    return pts, [coord_label(d, tau) for d in dims], "3d"


def pca_labels() -> list[str]:
    return ["PC1", "PC2", "PC3"]


def build_tau_m_heatmap(
    series_n: int,
    m_values: np.ndarray,
    tau_values: np.ndarray,
    metric: str = "embedded_n",
    series: np.ndarray | None = None,
) -> np.ndarray:
    """Grid shape (len(m_values), len(tau_values)); NaN where infeasible."""
    m_arr = np.asarray(m_values, dtype=np.int64)[:, None]
    t_arr = np.asarray(tau_values, dtype=np.int64)[None, :]
    en = series_n - (m_arr - 1) * t_arr

    if metric in {"embedded_n", "log_embedded_n"}:
        grid = en.astype(np.float64)
        grid[en <= 1] = np.nan
        if metric == "log_embedded_n":
            with np.errstate(invalid="ignore"):
                grid = np.log10(grid)
        return grid

    grid = np.full((len(m_values), len(tau_values)), np.nan, dtype=float)
    if series is None:
        return grid

    # Heavy metrics: coarse stride over grid to keep UI responsive.
    stride_m = max(1, len(m_values) // 12)
    stride_t = max(1, len(tau_values) // 12)
    for i in range(0, len(m_values), stride_m):
        m = int(m_values[i])
        for j in range(0, len(tau_values), stride_t):
            tau = int(tau_values[j])
            if en[i, j] <= 1:
                continue
            try:
                emb = delay_embedding(series, m=m, tau=tau)
            except ValueError:
                continue
            if metric == "mean_corr" and emb.shape[1] >= 2:
                c = np.corrcoef(emb.T)
                off = c[np.triu_indices(c.shape[0], k=1)]
                val = float(np.nanmean(np.abs(off)))
            elif metric == "path_length":
                view = emb[:, : min(3, m)]
                steps = np.diff(view, axis=0)
                val = float(np.sum(np.linalg.norm(steps, axis=1)))
            else:
                continue
            grid[i, j] = val
            if stride_m > 1 or stride_t > 1:
                i_sl = slice(i, min(i + stride_m, len(m_values)))
                j_sl = slice(j, min(j + stride_t, len(tau_values)))
                grid[i_sl, j_sl] = val
    return grid


def recurrence_density_2d(pts2d: np.ndarray, percentile: float = 5.0, max_points: int = 4000) -> np.ndarray:
    """Per-point recurrence count within a distance radius (2D projection)."""
    n = pts2d.shape[0]
    if n < 2:
        return np.zeros(n, dtype=float)
    idx = subsample_indices(n, max_points) if n > max_points else np.arange(n)
    sub = pts2d[idx]
    diff = sub[:, None, :] - sub[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    positive = dist[dist > 0]
    if positive.size == 0:
        return np.zeros(n, dtype=float)
    radius = float(np.percentile(positive, percentile))
    counts_sub = (dist <= radius).sum(axis=1).astype(float)
    if len(idx) == n:
        return counts_sub
    full = np.zeros(n, dtype=float)
    full[idx] = counts_sub
    return full


def local_expansion_proxy(
    embedded: np.ndarray,
    tau: int,
    *,
    theiler: int | None = None,
    k_neighbors: int = 8,
    max_points: int = 2500,
) -> dict[str, float | np.ndarray]:
    """
    Rosenstein-style rough λ₁ proxy (subsampled for speed on long series).
    """
    theiler = max(1, int(theiler or tau))
    pts = np.asarray(embedded, dtype=np.float64)
    n = pts.shape[0]
    if n < theiler + 10:
        return {"rosenstein_rate": float("nan"), "per_point": np.full(n, np.nan)}

    if n > max_points:
        idx = subsample_indices(n, max_points)
        pts_work = pts[idx]
        idx_map = idx
    else:
        pts_work = pts
        idx_map = np.arange(n, dtype=int)

    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min(k_neighbors + 4, len(pts_work) - 1), algorithm="kd_tree")
    nn.fit(pts_work)
    rates = np.full(len(pts_work), np.nan, dtype=float)
    for wi, i_orig in enumerate(idx_map):
        i = int(i_orig)
        if i + 1 >= n:
            continue
        dists, inds = nn.kneighbors(pts_work[wi : wi + 1], return_distance=True)
        for j_idx, wj in enumerate(inds[0]):
            j = int(idx_map[wj])
            if j == i or abs(i - j) <= theiler or j + 1 >= n:
                continue
            d0 = max(float(dists[0][j_idx]), 1e-18)
            d1 = float(np.linalg.norm(pts[i + 1] - pts[j + 1]))
            if d1 > 0:
                rates[wi] = (1.0 / tau) * np.log(d1 / d0)
                break
    valid = rates[np.isfinite(rates)]
    lam = float(np.median(valid)) if valid.size else float("nan")
    per_point = np.full(n, np.nan, dtype=float)
    per_point[idx_map] = rates
    return {"rosenstein_rate": lam, "per_point": per_point}


def make_surrogate_series(series: np.ndarray, seed: int = 0) -> np.ndarray:
    from hypothesis_surrogates import generate_single_surrogate

    rng = np.random.default_rng(seed)
    return generate_single_surrogate(series, rng)


def select_lod_indices(
    n: int,
    *,
    zoom_level: float,
    auto_lod: bool,
    manual_cap: int | None,
    view_mask: np.ndarray | None = None,
    gl_mode: bool = False,
) -> np.ndarray:
    """Choose trajectory indices for drawing (LOD + optional view-box mask)."""
    base = np.arange(n, dtype=int)
    if view_mask is not None and view_mask.shape[0] == n:
        masked = base[view_mask]
        if masked.size >= 100:
            base = masked
    if manual_cap is not None and base.size > manual_cap:
        return base[subsample_indices(base.size, manual_cap)]
    if auto_lod and manual_cap is None:
        if gl_mode:
            budget = int(min(n, max(12000, base.size / max(zoom_level, 0.35))))
        else:
            budget = int(min(n, max(6000, base.size / max(zoom_level, 0.25))))
        if base.size > budget:
            return base[subsample_indices(base.size, budget)]
    return base


def axis_limits_with_margin(values: np.ndarray, margin_frac: float = 0.05) -> tuple[float, float]:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        pad = max(abs(lo), 1e-12) * 0.1
        return lo - pad, hi + pad
    pad = (hi - lo) * margin_frac
    return lo - pad, hi + pad


@dataclass
class PhaseSpaceStats:
    symbol: str
    tau: int
    m: int
    series_n: int
    embedded_n: int
    points_lost: int
    series_mean: float
    series_std: float
    series_min: float
    series_max: float
    coord_labels: list[str]
    coord_mean: list[float]
    coord_std: list[float]
    coord_min: list[float]
    coord_max: list[float]
    coord_range: list[float]
    bbox_volume: float
    path_length: float
    mean_step: float
    max_step: float
    corr_01: float
    corr_02: float
    corr_12: float
    autocorr_tau: float
    display_n: int
    subsampled: bool
    view_dims: tuple[int, int, int]
    max_tau_feasible: int
    all_dim_mean: list[float]
    all_dim_std: list[float]
    rosenstein_rate: float = float("nan")
    window_note: str = ""

    def format_text(self) -> str:
        lines = [
            f"=== {self.symbol} phase-space statistics ===",
            "",
            "Input series (log-returns)",
            f"  N           : {self.series_n}",
            f"  mean        : {self.series_mean:.8f}",
            f"  std         : {self.series_std:.8f}",
            f"  min / max   : {self.series_min:.8f} / {self.series_max:.8f}",
        ]
        if self.window_note:
            lines.append(f"  window      : {self.window_note}")
        lines.extend(
            [
                "",
                "Takens embedding",
                f"  m           : {self.m}",
                f"  tau         : {self.tau}",
                f"  max tau     : {self.max_tau_feasible}  (for this N, m)",
                f"  embedded N  : {self.embedded_n}",
                f"  lost delay  : {self.points_lost}  (N - (m-1)*tau)",
                f"  autocorr@tau: {self.autocorr_tau:.6f}",
                f"  view dims   : {self.view_dims[0]}, {self.view_dims[1]}, {self.view_dims[2]}",
                f"  Rosenstein 1-step : {self.rosenstein_rate:.6f}  (NN expansion/tau; not thesis LLE)",
                "",
                "View coordinates",
            ]
        )
        for i, label in enumerate(self.coord_labels):
            lines.extend(
                [
                    f"  {label}",
                    f"    mean      : {self.coord_mean[i]:.8f}",
                    f"    std       : {self.coord_std[i]:.8f}",
                    f"    range     : {self.coord_range[i]:.8f}",
                    f"    min / max : {self.coord_min[i]:.8f} / {self.coord_max[i]:.8f}",
                ]
            )
        lines.extend(
            [
                "",
                "Trajectory geometry (view path)",
                f"  path length : {self.path_length:.8f}",
                f"  mean step   : {self.mean_step:.8f}",
                f"  max step    : {self.max_step:.8f}",
                f"  bbox volume : {self.bbox_volume:.6e}",
                "",
                "Pairwise correlations (view axes)",
                f"  corr(dim{self.view_dims[0]},dim{self.view_dims[1]}) : {self.corr_01:.6f}",
                f"  corr(dim{self.view_dims[0]},dim{self.view_dims[2]}) : {self.corr_02:.6f}",
                f"  corr(dim{self.view_dims[1]},dim{self.view_dims[2]}) : {self.corr_12:.6f}",
                "",
                f"All {self.m} embedding dimensions (mean / std)",
            ]
        )
        for k in range(self.m):
            lines.append(
                f"  dim {k} {coord_label(k, self.tau):>14} : "
                f"{self.all_dim_mean[k]:.8f} / {self.all_dim_std[k]:.8f}"
            )
        lines.extend(
            [
                "",
                "Display",
                f"  points drawn: {self.display_n}"
                + (" (FULL trajectory)" if not self.subsampled else f" / {self.embedded_n} (capped/LOD)"),
            ]
        )
        return "\n".join(lines)


def compute_phase_space_stats(
    symbol: str,
    series: np.ndarray,
    embedded: np.ndarray,
    tau: int,
    m: int,
    display_n: int,
    subsampled: bool,
    view_dims: tuple[int, int, int],
    *,
    view_pts: np.ndarray | None = None,
    window_note: str = "",
    coord_labels: list[str] | None = None,
) -> PhaseSpaceStats:
    if view_pts is None:
        if m < 3:
            view_pts = embedded[:, [view_dims[0], view_dims[1]]]
        else:
            view_pts = embedded[:, list(view_dims)]
    if view_pts.shape[1] == 2:
        pts3 = np.column_stack([view_pts, np.zeros(view_pts.shape[0])])
    else:
        pts3 = view_pts[:, :3]

    steps = np.diff(pts3, axis=0)
    step_len = np.linalg.norm(steps, axis=1) if steps.size else np.array([0.0])

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 2:
            return float("nan")
        c = np.corrcoef(a, b)[0, 1]
        return float(c) if np.isfinite(c) else float("nan")

    if len(series) > tau and np.std(series) > 0:
        autocorr = float(np.corrcoef(series[:-tau], series[tau:])[0, 1])
    else:
        autocorr = float("nan")

    ranges = pts3.max(axis=0) - pts3.min(axis=0)
    if coord_labels is not None:
        labels = coord_labels
    else:
        labels = [coord_label(k, tau) for k in view_dims[: view_pts.shape[1]]]
    lam = local_expansion_proxy(embedded, tau)["rosenstein_rate"]

    return PhaseSpaceStats(
        symbol=symbol,
        tau=tau,
        m=m,
        series_n=int(len(series)),
        embedded_n=int(embedded.shape[0]),
        points_lost=int(len(series) - embedded.shape[0]),
        series_mean=float(np.mean(series)),
        series_std=float(np.std(series, ddof=1)) if len(series) > 1 else 0.0,
        series_min=float(np.min(series)),
        series_max=float(np.max(series)),
        coord_labels=labels[:3] if pts3.shape[1] == 3 else labels[:2],
        coord_mean=[float(np.mean(view_pts[:, i])) for i in range(view_pts.shape[1])],
        coord_std=[
            float(np.std(view_pts[:, i], ddof=1)) if view_pts.shape[0] > 1 else 0.0
            for i in range(view_pts.shape[1])
        ],
        coord_min=[float(np.min(view_pts[:, i])) for i in range(view_pts.shape[1])],
        coord_max=[float(np.max(view_pts[:, i])) for i in range(view_pts.shape[1])],
        coord_range=[float(ranges[i]) for i in range(min(3, len(ranges)))],
        bbox_volume=float(np.prod(np.maximum(ranges, 1e-18))),
        path_length=float(np.sum(step_len)),
        mean_step=float(np.mean(step_len)) if step_len.size else 0.0,
        max_step=float(np.max(step_len)) if step_len.size else 0.0,
        corr_01=_corr(pts3[:, 0], pts3[:, 1]),
        corr_02=_corr(pts3[:, 0], pts3[:, 2]) if pts3.shape[1] >= 3 else float("nan"),
        corr_12=_corr(pts3[:, 1], pts3[:, 2]) if pts3.shape[1] >= 3 else float("nan"),
        autocorr_tau=autocorr,
        display_n=int(display_n),
        subsampled=subsampled,
        view_dims=view_dims,
        max_tau_feasible=max_feasible_tau(len(series), m),
        all_dim_mean=[float(np.mean(embedded[:, k])) for k in range(m)],
        all_dim_std=[
            float(np.std(embedded[:, k], ddof=1)) if embedded.shape[0] > 1 else 0.0
            for k in range(m)
        ],
        rosenstein_rate=lam,
        window_note=window_note,
    )
