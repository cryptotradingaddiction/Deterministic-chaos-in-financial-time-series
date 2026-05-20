"""Kantz lyap_k S(t) parsing and OLS Lyapunov extraction."""

import logging
import warnings

import numpy as np

from hypothesis_config import M_LYAP, MIN_LYAP_LINEAR_POINTS, lyap_min_neighbors
from invariants_correlation import _pearson_abs

logger = logging.getLogger(__name__)


def _best_linear_slope_window(x, y, min_points=MIN_LYAP_LINEAR_POINTS):
    """Find the best linear window of an S(t) curve and fit OLS with std error.

    Window rule: longest contiguous segment with ``|rho| >= 0.99``, else
    fallback to the segment with the largest ``|rho|``. The line is
    ``S(t) = slope * t + intercept`` on ``t in [t_lo, t_hi]``.

    Returns
    -------
    (slope, t_lo, t_hi, intercept, std_err)
        ``std_err`` is the OLS standard error of the slope from
        ``np.polyfit(..., cov=True)``. NaN when the fit is degenerate
        (n <= deg+1, residual variance zero, or polyfit raises).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < min_points:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    x = x[:n]
    y = y[:n]
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    n = len(x)
    if n < min_points:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    R2_THRESHOLD = 0.99
    win_thresh = None  # (start, width, slope, intercept)
    best_len_thresh = 0
    win_fb = None
    best_rho_fallback = -np.inf

    # Window search: prefer the longest strictly-linear segment (|rho| >= 0.99)
    # and fall back to the highest-|rho| segment when none is "strictly" linear.
    for width in range(min_points, n + 1):
        for start in range(0, n - width + 1):
            xs = x[start:start + width]
            ys = y[start:start + width]
            rho = _pearson_abs(xs, ys)
            if not np.isfinite(rho):
                continue
            try:
                coeffs = np.polyfit(xs, ys, 1)
                slope = float(coeffs[0])
                intercept = float(coeffs[1])
            except Exception:
                continue
            if not np.isfinite(slope):
                continue
            if rho >= R2_THRESHOLD and width > best_len_thresh:
                best_len_thresh = width
                win_thresh = (start, width, slope, intercept)
            if rho > best_rho_fallback:
                best_rho_fallback = rho
                win_fb = (start, width, slope, intercept)

    win = win_thresh if win_thresh is not None else win_fb
    if win is None:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    start, width, slope, intercept = win
    t_lo = float(x[start])
    t_hi = float(x[start + width - 1])

    # OLS with covariance gives the standard error of the slope. Skip when the
    # fit has no residual degrees of freedom (width = deg + 1 = 2 is enough for
    # a slope but produces a singular covariance scaling factor).
    std_err = np.nan
    if width > 2:
        xs = x[start:start + width]
        ys = y[start:start + width]
        try:
            with warnings.catch_warnings():
                # polyfit warns when residuals are tiny / matrix ill-conditioned;
                # we already gate via the rho threshold so the warning is noise.
                warnings.simplefilter("ignore")
                coeffs, cov = np.polyfit(xs, ys, 1, cov=True)
            slope_cov = float(coeffs[0])
            intercept_cov = float(coeffs[1])
            if np.isfinite(slope_cov):
                slope = slope_cov
                intercept = intercept_cov
            if np.ndim(cov) == 2 and cov.shape == (2, 2):
                var_slope = float(cov[0, 0])
                if np.isfinite(var_slope) and var_slope >= 0.0:
                    std_err = float(np.sqrt(var_slope))
        except Exception:
            pass

    return (float(slope), t_lo, t_hi, float(intercept), std_err)


def _best_linear_slope(x, y, min_points=MIN_LYAP_LINEAR_POINTS):
    """Slope-only convenience wrapper around :func:`_best_linear_slope_window`."""
    slope, _t0, _t1, _b, _se = _best_linear_slope_window(x, y, min_points)
    return slope


def _parse_lyap_blocks(lyap_file, dim=M_LYAP):
    """Parse epsilon blocks for ``dim`` from lyap_k output.

    Returns a list of dicts ``[{'eps': float, 'n_neighbors': int, 'data': ndarray}]``
    where ``data[:, 0]`` is iteration t and ``data[:, 1]`` is S(t).
    ``n_neighbors`` is the median of column 3 across rows of the block
    (how many neighbors contributed to each averaged S(t) point).
    """
    blocks = []
    current_dim = None
    current_eps = np.nan
    current_rows = []

    def _flush():
        if current_dim == dim and current_rows:
            arr = np.array(current_rows, dtype=float)
            n_nbrs = int(np.median(arr[:, 2])) if arr.shape[1] >= 3 else 0
            blocks.append(
                {
                    "eps": current_eps,
                    "n_neighbors": n_nbrs,
                    "data": arr[:, :2],
                }
            )

    try:
        with open(lyap_file, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped.startswith("#epsilon"):
                    _flush()
                    current_dim = None
                    current_eps = np.nan
                    current_rows = []
                    parts = stripped.split()
                    for i, tok in enumerate(parts):
                        tlow = tok.lower()
                        if tlow.startswith("#epsilon") or tlow.startswith("epsilon"):
                            if i + 1 < len(parts):
                                try:
                                    current_eps = float(parts[i + 1])
                                except ValueError:
                                    pass
                        if tlow.startswith("dim"):
                            if i + 1 < len(parts):
                                try:
                                    current_dim = int(float(parts[i + 1]))
                                except ValueError:
                                    pass
                    continue
                if stripped.startswith("#") or stripped == "":
                    continue
                if current_dim != dim:
                    continue
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        row = [float(p) for p in parts[:3]]
                        if len(row) == 2:
                            row.append(0.0)
                        current_rows.append(row)
                    except ValueError:
                        continue
        _flush()
    except Exception:
        pass
    return blocks


def extract_lle_ols(lyap_file, min_neighbors=None, dim=None):
    """LLE = OLS slope of the highest-quality epsilon block at the chosen *m*.

    For each ε-block at ``dim`` (default :data:`M_LYAP` = 3):

    1. Filter blocks with ``n_neighbors < min_neighbors`` (Kantz/Schreiber
       recommendation: too-few-neighbors blocks have noisy S(t)).
    2. Find the best linear window of S(t) via
       :func:`_best_linear_slope_window`.
    3. Fit OLS slope ± std_err on that window.
    4. Score quality = ``(t_hi - t_lo) / std_err`` — longer windows with
       smaller fit error rank higher (the book's "longest stable linear
       region" criterion, automated).

    Returns
    -------
    (slope_lambda, std_err_lambda, n_usable_blocks)
        ``std_err_lambda`` is the OLS standard error of the *selected*
        block's slope. This matches the uncertainty reported in
        Hegger-Kantz-Schreiber 1999. Returns ``(NaN, NaN, 0)`` when no
        block produces a finite (slope, std_err) pair.

    The median and spread of slopes across all usable blocks are also logged
    at INFO level as a robustness check.
    """
    if min_neighbors is None:
        min_neighbors = lyap_min_neighbors()
    if dim is None:
        dim = M_LYAP

    blocks = _parse_lyap_blocks(lyap_file, dim=dim)
    if not blocks:
        return np.nan, np.nan, 0

    def _fit_block(blk):
        data = blk["data"]
        if data.shape[0] < 3:
            return None
        slope, t_lo, t_hi, _b, std_err = _best_linear_slope_window(
            data[:, 0], data[:, 1]
        )
        if not (np.isfinite(slope) and np.isfinite(std_err) and std_err > 0.0):
            return None
        if not (np.isfinite(t_lo) and np.isfinite(t_hi) and t_hi > t_lo):
            return None
        quality = (t_hi - t_lo) / std_err
        return (quality, float(slope), float(std_err), float(blk["eps"]),
                float(t_lo), float(t_hi))

    candidates = []
    for blk in blocks:
        if blk["n_neighbors"] < min_neighbors:
            logger.debug(
                "lyap block eps=%.6g skipped: n_neighbors=%d < %d",
                blk["eps"], blk["n_neighbors"], min_neighbors,
            )
            continue
        fitted = _fit_block(blk)
        if fitted is not None:
            candidates.append(fitted)

    # Fallback: relax neighbor filter only if no block survives it.
    if not candidates:
        logger.warning(
            "extract_lle_ols: no block passed n_neighbors>=%d in %s; "
            "relaxing neighbour filter.",
            min_neighbors, lyap_file,
        )
        for blk in blocks:
            fitted = _fit_block(blk)
            if fitted is not None:
                candidates.append(fitted)

    if not candidates:
        return np.nan, np.nan, 0

    candidates.sort(reverse=True)  # by quality desc
    best_quality, best_slope, best_std_err, best_eps, best_t_lo, best_t_hi = candidates[0]

    all_slopes = np.array([c[1] for c in candidates], dtype=float)
    median_slope = float(np.median(all_slopes))
    spread = float(np.std(all_slopes, ddof=1)) if all_slopes.size > 1 else 0.0
    logger.info(
        "lle ols: best eps=%.6g slope=%.6g +/- %.3g "
        "(window t=[%.3g,%.3g], quality=%.3g); "
        "median across %d blocks = %.6g, spread = %.3g",
        best_eps, best_slope, best_std_err,
        best_t_lo, best_t_hi, best_quality,
        len(candidates), median_slope, spread,
    )

    return float(best_slope), float(best_std_err), int(len(candidates))


def extract_lle_mean_std(lyap_file, min_neighbors=None):
    """Backward-compatible alias for :func:`extract_lle_ols`.

    Note: the second return value is now the **OLS standard error of the
    selected block's slope**, not the spread across blocks. The signature
    ``(value, sd, n)`` is unchanged so callers in
    :mod:`invariants_compute` and :mod:`plot_lyap_k_output` still work.
    """
    return extract_lle_ols(lyap_file, min_neighbors=min_neighbors)
