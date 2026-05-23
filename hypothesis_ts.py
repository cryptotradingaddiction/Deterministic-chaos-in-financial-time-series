"""Stationary-bootstrap test statistic TS = (boot_mean - reference) / boot_sd.

The reference value can be any independent draw from the same length-N
distribution as the original series: a random permutation (``surr``), a
Gaussian with matched first / second moments (``normal``), or a scaled
Student-t reference (``t<dof>``). Each comparison answers a slightly
different question — *"is the structured-resampling invariant far from
the invariant under this particular null?"* — and each gets its own
TS / p-value / decision triple.

The p-value is two-sided under a Student-t distribution with ``df = B - 1``
degrees of freedom, where ``B`` is the number of stationary-bootstrap
replicates. This matches the MATLAB convention
``p = 2 * (1 - tcdf(abs(TS), B-1))`` requested by the supervisor.
"""

import numpy as np
from scipy import stats

from hypothesis_config import DEFAULT_TS_THRESHOLD


def invariant_bootstrap_ts_test(
    boot_mean: float,
    boot_sd: float,
    reference_value: float,
    n_bootstrap: int | None = None,
    threshold: float = DEFAULT_TS_THRESHOLD,
) -> tuple[float, float, float, str]:
    """TS = (boot_mean − reference) / boot_sd with two-sided Student-t p-value.

    Parameters
    ----------
    boot_mean, boot_sd
        Centre and sample SD of the stationary-bootstrap distribution of the
        invariant on the original series.
    reference_value
        Invariant value computed on the comparison series (one of ``surr``,
        ``normal``, ``t<dof>``).
    n_bootstrap
        Number of finite bootstrap invariant values used to compute
        ``boot_mean`` / ``boot_sd``. Used as ``df = n_bootstrap − 1`` for the
        Student-t p-value. ``None`` (or ≤ 1) leaves the p-value as ``NaN``
        — useful for downstream parsers that need a column even when the
        bootstrap was skipped.
    threshold
        Absolute-TS threshold for the ``reject H0`` / ``fail to reject H0``
        decision. The thesis uses ``3``.

    Returns
    -------
    (ts, abs_ts, p_value, decision)
        ``ts`` and ``abs_ts`` are NaN when any input is non-finite or
        ``boot_sd ≤ 0`` (the decision string then carries the reason).
        ``p_value`` is NaN when ``n_bootstrap`` is not usable.
    """
    if not (np.isfinite(boot_mean) and np.isfinite(boot_sd) and np.isfinite(reference_value)):
        return np.nan, np.nan, np.nan, "insufficient data"
    if boot_sd <= 0.0:
        return np.nan, np.nan, np.nan, "no sd"
    ts = float((boot_mean - reference_value) / boot_sd)
    abs_ts = abs(ts)

    # Two-sided p-value under a Student-t distribution with df = B − 1.
    # ``stats.t.sf(x, df) == 1 − tcdf(x, df)`` but is more numerically stable
    # in the upper tail than ``1 - cdf(x)``, which matters for the 3-sigma
    # decision region (typical p-values are 1e-3 .. 1e-5).
    pvalue = np.nan
    if n_bootstrap is not None:
        try:
            df = int(n_bootstrap) - 1
        except (TypeError, ValueError):
            df = -1
        if df >= 1:
            pvalue = float(2.0 * stats.t.sf(abs_ts, df=df))

    decision = "reject H0" if abs_ts > threshold else "fail to reject H0"
    return ts, abs_ts, pvalue, decision
