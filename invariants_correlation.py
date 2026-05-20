"""Takens plateau and Ellner correlation-dimension estimators."""

import logging

import numpy as np

from hypothesis_config import M_D2, MIN_PLATEAU_POINTS
from tisean_io import extract_tagged_block

logger = logging.getLogger(__name__)

# Number of leading / trailing samples excluded from the candidate plateau
# window. The smallest epsilon points are dominated by discretization noise on
# C(r); the largest are dominated by finite-sample saturation. Keeping the
# search away from those endpoints prevents the plateau picker from latching
# onto an accidentally-flat edge region.
DEFAULT_PLATEAU_EDGE_MARGIN = 2

# Weight of the length bonus inside the plateau score. The bonus uses sqrt so
# longer windows are clearly preferred but the marginal gain saturates near the
# full available range, leaving room for the flatness penalty to win.
PLATEAU_LENGTH_WEIGHT = 0.5


def select_plateau_values(rows, min_points=MIN_PLATEAU_POINTS,
                          edge_margin=DEFAULT_PLATEAU_EDGE_MARGIN):
    """Select a stable scaling/plateau window from (epsilon, value) rows.

    Returns ``(y_values, r_min, r_max)`` where ``y_values`` are the plateau
    values sorted by ln(epsilon) and ``r_min`` / ``r_max`` are the epsilon
    end-points of the selected window (in the original linear scale).

    Returns ``(empty, NaN, NaN)`` when fewer than ``min_points`` usable rows are
    available. Returning the whole short set was previously misleading: no real
    plateau detection had happened, but the caller treated the bounds as valid.

    ``edge_margin`` excludes the first and last ``edge_margin`` samples from the
    candidate window. The smallest-epsilon points are biased by C(r)
    discretization and the largest-epsilon points by saturation, so a plateau
    placed at the very edge is rarely physical. When the searchable interior
    cannot accommodate ``min_points``, the margin is relaxed and a warning is
    logged so the caller knows the choice was forced.
    """
    arr = np.asarray(rows, dtype=float)
    if arr.size == 0 or arr.ndim != 2 or arr.shape[1] < 2:
        return np.array([], dtype=float), np.nan, np.nan

    eps = arr[:, 0]
    values = arr[:, 1]
    mask = np.isfinite(eps) & np.isfinite(values) & (eps > 0.0) & (values > 0.0)
    eps = eps[mask]
    values = values[mask]
    if values.size == 0:
        return np.array([], dtype=float), np.nan, np.nan

    # Sort by log scale because both local slopes and Takens estimates are
    # interpreted visually/methodologically as functions of ln(r).
    order = np.argsort(np.log(eps))
    eps_sorted = eps[order]
    x = np.log(eps_sorted)
    y = values[order]
    n = y.size
    if n < min_points:
        logger.warning(
            "select_plateau_values: only %d usable points (need >= %d); "
            "plateau detection skipped.", n, min_points,
        )
        return np.array([], dtype=float), np.nan, np.nan

    # Restrict the candidate window to the interior so discretization artefacts
    # at the smallest/largest epsilon cannot anchor the choice. Relax the margin
    # only if the interior would otherwise be too short for ``min_points``.
    eff_margin = max(0, int(edge_margin))
    if n - 2 * eff_margin < min_points:
        eff_margin = max(0, (n - min_points) // 2)
        logger.warning(
            "select_plateau_values: relaxed edge_margin to %d "
            "(n=%d, min_points=%d).", eff_margin, n, min_points,
        )
    i_lo = eff_margin
    i_hi = n - eff_margin

    # Score: flat + low-spread + reasonably long. The length term uses sqrt so
    # the marginal benefit of extending the window saturates near the full
    # available interior, preventing the optimum from always being the largest
    # interior window. Weight 0.5 (was 0.10) makes the bonus comparable to the
    # flatness / spread penalties for typical financial dimensions.
    best_score = -np.inf
    best_ij = (i_lo, i_hi)
    interior = max(1, i_hi - i_lo)
    for i in range(i_lo, i_hi - min_points + 1):
        for j in range(i + min_points, i_hi + 1):
            xs = x[i:j]
            ys = y[i:j]
            # mean_abs guards against division by zero. For correlation
            # dimensions the mean cannot reach zero, but keeping the guard means
            # an accidentally near-zero mean (e.g. a sign-flipped curve segment)
            # produces a very large rel_slope/rel_sd and is therefore rejected.
            mean_abs = abs(float(np.mean(ys))) + 1e-12
            try:
                slope, _ = np.polyfit(xs, ys, 1)
            except Exception:
                continue
            rel_slope = abs(float(slope)) / mean_abs
            rel_sd = float(np.std(ys, ddof=1)) / mean_abs if ys.size > 1 else np.inf
            length_bonus = np.sqrt((j - i) / interior)
            score = PLATEAU_LENGTH_WEIGHT * length_bonus - rel_slope - rel_sd
            if score > best_score:
                best_score = score
                best_ij = (i, j)

    i, j = best_ij
    if i == i_lo or j == i_hi:
        logger.warning(
            "select_plateau_values: optimum touches search edge "
            "(i=%d, j=%d, i_lo=%d, i_hi=%d, n=%d) — scaling may extend "
            "beyond the d2 grid.", i, j, i_lo, i_hi, n,
        )
    return y[i:j], float(eps_sorted[i]), float(eps_sorted[j - 1])


def _pearson_abs(x, y):
    """Absolute Pearson correlation with guards for degenerate windows.

    Used by :mod:`invariants_lyapunov` to score linear-window candidates in the
    Kantz S(t) curve. Kept here so the dimension and Lyapunov modules share one
    implementation of the same numerical guard.
    """
    if len(x) < 2 or len(y) < 2:
        return np.nan
    sx = float(np.std(x, ddof=1))
    sy = float(np.std(y, ddof=1))
    if sx <= 0.0 or sy <= 0.0:
        return np.nan
    return abs(float(np.corrcoef(x, y)[0, 1]))


def _mean_sd_n(values):
    """Return mean, sample SD, and finite-count for an estimator vector."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, 0
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if values.size > 1 else np.nan
    return mean, sd, int(values.size)


def extract_takens_plateau(takens_file, dim=M_D2):
    """Find a stable plateau on d_2^(T)(r') for embedding m=`dim`.

    Returns `(mean_val, sd_val, n_val, r_min, r_max)`:
      * `mean_val`/`sd_val`/`n_val` summarise the Takens values inside the plateau;
      * `r_min`/`r_max` are the radius end-points of that plateau (eq. 8.78 in the
        book). `r_min`/`r_max` are NaN if no plateau could be found.
    """
    # c2t produces d_2^(T)(r') as a curve:
    #
    #   d_2^(T)(r') = < log(r' / r_ij) >^(-1)                         (8.75)
    #
    # and, equivalently in terms of the correlation integral,
    #
    #   d_2^(T)(r') = C^(m)(r') / integral_0^{r'} C^(m)(r) / r dr       (8.76)
    #
    # The book explicitly notes that r' is a free scale parameter. Rather than
    # fixing a single r' by hand, it recommends computing the full curve and
    # choosing a plateau in d_2^(T)(r') versus ln(r'). That is exactly what
    # select_plateau_values() does below.
    #
    # The returned mean is the standalone TAKENS invariant. The returned r_min
    # and r_max are not arbitrary bounds: they are the selected plateau interval,
    # later reused by compute_ellner_from_c2() for equation (8.78).
    #
    # Note on uncertainty:
    # Equation (8.77) gives a theoretical standard deviation for a single Takens
    # estimate at a fixed r' in terms of the number of close pairs M_C(r'). Here
    # we report the empirical SD of the plateau values instead. That SD is used
    # as an orientation for plateau stability; the formal hypothesis-test SD is
    # computed across the B stationary-bootstrap invariant estimates.
    rows = extract_tagged_block(takens_file, dim=dim, tag="#m")
    if rows.size == 0:
        return np.nan, np.nan, 0, np.nan, np.nan
    y, r_min, r_max = select_plateau_values(rows)
    if y.size == 0:
        return np.nan, np.nan, 0, np.nan, np.nan
    mean_val, sd_val, n_val = _mean_sd_n(y)
    return mean_val, sd_val, n_val, r_min, r_max


def compute_ellner_from_c2(c2_file, r_min, r_max, dim=M_D2):
    """Ellner extension of Takens' estimator (book eq. 8.78).

    `d_2^(E) = [C^(m)(r_max) - C^(m)(r_min)] / integral_{r_min}^{r_max} C(r)/r dr`.
    `r_min`/`r_max` are typically taken from the Takens plateau auto-detected on
    the d_2^(T)(r') curve so that they bracket the linear scaling region.
    Returns NaN if the integral cannot be evaluated.
    """
    # Ellner's extension is introduced after the book explains a weakness of the
    # raw Takens estimator: d_2^(T)(r') averages over all distances below r'.
    # If the correlation integral contains several scaling regions, including
    # every distance from 0 to r' can blur the desired region. Ellner's formula
    # solves this by restricting the estimator to a finite scaling interval:
    #
    #   d_2^(E) =
    #       [ C^(m)(r_max) - C^(m)(r_min) ]
    #       / integral_{r_min}^{r_max} C^(m)(r) / r dr                 (8.78)
    #
    # In this implementation, r_min and r_max are not hand-selected. They come
    # from the plateau of d_2^(T)(r') detected in extract_takens_plateau(). Thus
    # TAKENS and ELLNER are tied to the same observed scaling region.
    #
    # Equation (8.79) gives a standard-deviation expression for Ellner's estimate
    # using M_C(r_max) and a theta-like ratio. We do not use that theoretical
    # formula as the hypothesis-test uncertainty; instead the test uses the
    # empirical SD across stationary-bootstrap ELLNER estimates, which is the
    # quantity requested by the current supervisor-aligned testing procedure.
    #
    # Numerically, we interpolate C(r_min) and C(r_max), because the detected
    # plateau endpoints need not coincide exactly with .c2 grid points.
    rows = extract_tagged_block(c2_file, dim=dim, tag="#dim")
    if rows.size == 0:
        return np.nan
    if not (np.isfinite(r_min) and np.isfinite(r_max)) or r_min <= 0.0 or r_max <= r_min:
        return np.nan
    r = rows[:, 0]
    c = rows[:, 1]
    # The integral C(r)/r and the logarithmic scale interpretation require
    # strictly positive radii and strictly positive correlation integrals.
    finite = np.isfinite(r) & np.isfinite(c) & (r > 0.0) & (c > 0.0)
    r = r[finite]
    c = c[finite]
    if r.size < 2:
        return np.nan
    order = np.argsort(r)
    r = r[order]
    c = c[order]
    # Restrict integration to the Takens-selected plateau interval. This is the
    # key Ellner idea: use only the finite scaling region, not all distances from
    # zero up to a single r'.
    mask = (r >= r_min) & (r <= r_max)
    if int(mask.sum()) < 2:
        return np.nan
    r_sel = r[mask]
    c_sel = c[mask]
    # Interpolate C at r_min, r_max against the FULL sorted grid (not the
    # masked subset) so the endpoints use the two nearest grid points on each
    # side. Interpolating against r_sel would clamp to r_sel[0] / r_sel[-1]
    # whenever r_min / r_max fall strictly between grid points, biasing both
    # boundary values toward the included subset.
    c_max = float(np.interp(r_max, r, c))
    c_min = float(np.interp(r_min, r, c))
    if not (np.isfinite(c_max) and np.isfinite(c_min)) or c_max <= c_min:
        return np.nan
    # d2.exe uses an exponentially spaced epsilon grid (``-#100`` steps with
    # multiplicative factor). Trapezoidal integration in linear r over-weights
    # the large-r side. Substitute u = ln r so the integrand
    # C(r)/r * dr becomes C(r) * d(ln r), which is the natural form on a
    # log-spaced grid and matches the way the plateau is detected in ln r.
    log_r_sel = np.log(r_sel)
    _trapz = getattr(np, "trapezoid", np.trapz)
    integral = float(_trapz(c_sel, log_r_sel))
    if not np.isfinite(integral) or integral <= 0.0:
        return np.nan
    return float((c_max - c_min) / integral)