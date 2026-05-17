"""Kantz lyap_k S(t) parsing and LLE extraction."""

import logging

import numpy as np

from hypothesis_config import M_LYAP, MIN_LYAP_LINEAR_POINTS, lyap_min_neighbors
from invariants_correlation import _pearson_abs

logger = logging.getLogger(__name__)


def _best_linear_slope_window(x, y, min_points=MIN_LYAP_LINEAR_POINTS):
    """Return (slope, t_lo, t_hi, intercept) for the Kantz S(t) linear window.

    Same window rule as `_best_linear_slope`: longest contiguous segment with
    |rho| >= 0.99, else fallback to the segment with largest |rho|.
    The line is S(t) = slope * t + intercept over t in [t_lo, t_hi].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < min_points:
        return (np.nan, np.nan, np.nan, np.nan)
    x = x[:n]
    y = y[:n]
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    n = len(x)
    if n < min_points:
        return (np.nan, np.nan, np.nan, np.nan)

    R2_THRESHOLD = 0.99
    win_thresh = None  # (start, width, slope, intercept)
    best_len_thresh = 0
    win_fb = None
    best_rho_fallback = -np.inf

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
        return (np.nan, np.nan, np.nan, np.nan)
    start, width, slope, intercept = win
    t_lo = float(x[start])
    t_hi = float(x[start + width - 1])
    return (float(slope), t_lo, t_hi, float(intercept))


def _best_linear_slope(x, y, min_points=MIN_LYAP_LINEAR_POINTS):
    """Slope from the longest contiguous window with |r| above threshold.

    Threshold-based: take the longest window with |r| >= 0.99 rather than
    chasing the single maximum |r| window, which can land in saturation.
    Falls back to max |r| if no window meets threshold.
    """
    slope, _t0, _t1, _b = _best_linear_slope_window(x, y, min_points)
    return slope


def _parse_lyap_blocks(lyap_file, dim=M_LYAP):
    """Parse ALL epsilon blocks for given dim from lyap_k output.

    Returns list of dicts: [{'eps': float, 'n_neighbors': int, 'data': ndarray(T,2)}]
    data[:,0] = iteration t, data[:,1] = S(t)
    n_neighbors = median of column 3 across rows (how many neighbors contributed).
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


def extract_lle_mean_std(lyap_file, min_neighbors=None):
    if min_neighbors is None:
        min_neighbors = lyap_min_neighbors()
    """LLE as median slope across all usable epsilon blocks for m=3.

    Faithful to the TISEAN paper (Hegger, Kantz, Schreiber 1999, Fig. on CO laser):
    the Lyapunov exponent is the slope of S(t) in the region where curves for
    different epsilon values overlap and grow linearly. Using the median slope
    across multiple epsilon blocks operationalizes the 'overlap' criterion
    without requiring manual visual inspection.

    Blocks with median n_neighbors < min_neighbors are excluded because their
    inner sum in S(t) is dominated by fluctuations (too few neighbors to average).
    This corresponds to the paper's recommendation to exclude reference points
    with very few neighbors.

    Returns:
        (median_slope, std_slope, n_usable_blocks)
        std_slope is the spread across epsilon blocks - a real uncertainty estimate.
        Returns (NaN, NaN, 0) if no usable blocks found.
    """
    blocks = _parse_lyap_blocks(lyap_file, dim=M_LYAP)
    if not blocks:
        return np.nan, np.nan, 0

    slopes = []
    for blk in blocks:
        if blk["n_neighbors"] < min_neighbors:
            logger.debug(
                "lyap block eps=%.6g skipped: n_neighbors=%d < %d",
                blk["eps"],
                blk["n_neighbors"],
                min_neighbors,
            )
            continue
        data = blk["data"]
        if data.shape[0] < 3:
            continue
        slope = _best_linear_slope(data[:, 0], data[:, 1])
        if np.isfinite(slope):
            slopes.append(slope)

    # Fallback: if all blocks fail the neighbor filter, use them all with a warning
    if not slopes:
        logger.warning(
            "extract_lle_mean_std: no block passed n_neighbors>=%d in %s; using all blocks",
            min_neighbors,
            lyap_file,
        )
        for blk in blocks:
            data = blk["data"]
            if data.shape[0] < 3:
                continue
            slope = _best_linear_slope(data[:, 0], data[:, 1])
            if np.isfinite(slope):
                slopes.append(slope)

    if not slopes:
        return np.nan, np.nan, 0

    arr = np.array(slopes, dtype=float)
    return (
        float(np.median(arr)),
        float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan,
        int(len(arr)),
    )