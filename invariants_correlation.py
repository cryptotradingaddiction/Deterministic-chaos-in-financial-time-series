"""Takens plateau and Ellner correlation-dimension estimators."""

import numpy as np

from hypothesis_config import M_D2, MIN_PLATEAU_POINTS
from tisean_io import extract_tagged_block


def select_plateau_values(rows, min_points=MIN_PLATEAU_POINTS):
    """Select a stable scaling/plateau window from (epsilon, value) rows.

    Returns `(y_values, r_min, r_max)` where `y_values` are the plateau values
    sorted by ln(epsilon) and `r_min`, `r_max` are the epsilon end-points of the
    selected window (in the original linear scale). `r_min` / `r_max` are NaN if
    no usable rows were supplied.
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
        return y, float(eps_sorted[0]), float(eps_sorted[-1])

    # Window scoring favours flat, low-variance, and reasonably long regions.
    #
    # Book connection:
    # After equation (8.77), the text recommends treating d_2^(T)(r') as a
    # function of scale r' and estimating the correlation dimension from a
    # plateau in the graph d_2^(T)(r') versus ln(r'), instead of choosing a single
    # ad hoc r'. This function operationalizes that instruction. The score is
    # deliberately transparent: penalize slope and relative spread, add a small
    # bonus for longer intervals.
    best_score = -np.inf
    best_ij = (0, n)
    for i in range(0, n - min_points + 1):
        for j in range(i + min_points, n + 1):
            xs = x[i:j]
            ys = y[i:j]
            mean_abs = abs(float(np.mean(ys))) + 1e-12
            try:
                slope, _ = np.polyfit(xs, ys, 1)
            except Exception:
                continue
            rel_slope = abs(float(slope)) / mean_abs
            rel_sd = float(np.std(ys, ddof=1)) / mean_abs if ys.size > 1 else np.inf
            length_bonus = (j - i) / n
            score = 0.10 * length_bonus - rel_slope - rel_sd
            if score > best_score:
                best_score = score
                best_ij = (i, j)

    i, j = best_ij
    return y[i:j], float(eps_sorted[i]), float(eps_sorted[j - 1])


def _pearson_abs(x, y):
    """Absolute Pearson correlation with guards for degenerate windows."""
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
    c_max = float(np.interp(r_max, r_sel, c_sel))
    c_min = float(np.interp(r_min, r_sel, c_sel))
    if not (np.isfinite(c_max) and np.isfinite(c_min)) or c_max <= c_min:
        return np.nan
    # Trapezoidal integration is adequate here because d2.exe already gives a
    # dense, ordered grid of radius values. NumPy renamed trapz -> trapezoid in
    # recent versions, so use the new function when available.
    integrand = c_sel / r_sel
    _trapz = getattr(np, "trapezoid", np.trapz)
    integral = float(_trapz(integrand, r_sel))
    if not np.isfinite(integral) or integral <= 0.0:
        return np.nan
    return float((c_max - c_min) / integral)