"""Stationary-bootstrap test statistic TS = (boot_mean - reshuffle) / boot_sd."""

import numpy as np

from hypothesis_config import DEFAULT_TS_THRESHOLD


def invariant_bootstrap_ts_test(
    boot_mean: float,
    boot_sd: float,
    resh_value: float,
    threshold: float = DEFAULT_TS_THRESHOLD,
) -> tuple[float, float, str]:
    """TS=(boot_mean-reshuffle)/boot_sd with decision |TS| > threshold.

    The bootstrap distribution represents structured pseudo-series from the
    original data. The reshuffled value represents independent random ordering.
    A large absolute TS means the invariant under structured resampling is far
    from the invariant under full temporal destruction.
    """
    if not (np.isfinite(boot_mean) and np.isfinite(boot_sd) and np.isfinite(resh_value)):
        return np.nan, np.nan, "insufficient data"
    if boot_sd <= 0.0:
        return np.nan, np.nan, "no sd"
    ts = float((boot_mean - resh_value) / boot_sd)
    abs_ts = abs(ts)
    decision = "reject H0" if abs_ts > threshold else "fail to reject H0"
    return ts, abs_ts, decision