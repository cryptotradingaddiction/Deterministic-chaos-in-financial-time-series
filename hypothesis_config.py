"""Shared constants for hypothesis / invariant pipeline."""

import os
# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
#
# The constants below are intentionally kept near the top of the file because
# they define the methodological contract used by the batch scripts:
#
# * DEFAULT_BOOTSTRAP_SAMPLES controls the number of stationary-bootstrap
#   pseudo-series used to estimate the empirical centre/spread of an invariant.
# * DEFAULT_TS_THRESHOLD is the supervisor-approved "three sigma" decision rule
#   for the bootstrap test statistic.
# * M_D2 and M_LYAP fix the embedding dimension used by the current thesis
#   workflow. The Windows batch files are expected to use matching values.
#
# Keeping these values centralized avoids hidden drift between Python
# recomputation, TISEAN command-line calls, and the generated result summaries.

DEFAULT_BOOTSTRAP_SAMPLES = 100
DEFAULT_TS_THRESHOLD = 3.0
DEFAULT_STATIONARY_BLOCK_MEAN = 0.0  # <=0 means sqrt(n)
DEFAULT_HYPOTHESIS_SEED = 0

# Degrees of freedom for t-distribution reference series
T_DOF = 3.5

# Active invariant embedding dimension. d2.exe sweeps m = 1 .. D2_DIAGNOSTIC_M_MAX,
# but TAKENS/ELLNER values are extracted from the m = M_D2 block only.
M_D2 = 3
# d2.exe runs over m = 1 .. D2_DIAGNOSTIC_M_MAX. Used by tisean_io.run_d2 (-M1,X)
# and matched by correlation_dimension.bat (EMBED=1,X) so plots cover all m.
D2_DIAGNOSTIC_M_MAX = 10
# Active Lyapunov embedding dimension. lyap_k sweeps m = M_LYAP .. M_LYAP_DIAGNOSTIC_MAX
# in the .bat path (single per-coin run), but the OLS slope estimate and TS test
# always use the m = M_LYAP block. Multi-m blocks are kept for the gnuplot panels
# that show m-independence of the linear S(t) region.
M_LYAP = 3
M_LYAP_DIAGNOSTIC_MAX = 10
MIN_LYAP_LINEAR_POINTS = 3  # was 5; financial S(t) linear region is ~1-2 iterations
MIN_LYAP_NEIGHBORS = 10
# Short-series test mode (DCH_TEST_MODE) lyap_k overrides via env.
DEFAULT_DCH_LYAP_STEPS_TEST = 200       # -n reference points (was 40 — see lyap_k_steps)
DEFAULT_DCH_LYAP_ITERATIONS_TEST = 30   # -s iterations / S(t) curve length
DEFAULT_DCH_LYAP_MIN_NEIGHBORS_TEST = 3
DEFAULT_LYAP_ITERATIONS = 100           # production -s default (S(t) curve length)


def _is_test_mode_env() -> bool:
    return os.environ.get("DCH_TEST_MODE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def lyap_k_steps() -> int:
    """Reference points for ``lyap_k -n`` (env ``DCH_LYAP_STEPS``).

    TISEAN's ``-n`` is the *number of reference points* used to average S(t)
    over the trajectory, not the curve length. See :func:`lyap_k_iterations`
    for the ``-s`` curve length.

    Defaults: 500 in production, 200 in ``DCH_TEST_MODE=true`` (short series).
    """
    raw = os.environ.get("DCH_LYAP_STEPS", "").strip()
    if raw:
        try:
            return max(5, int(float(raw)))
        except ValueError:
            pass
    return DEFAULT_DCH_LYAP_STEPS_TEST if _is_test_mode_env() else 500


def lyap_k_iterations() -> int:
    """Forward iterations for ``lyap_k -s`` (env ``DCH_LYAP_ITERATIONS``).

    ``-s`` controls the length of the Kantz S(t) curve. A previous version of
    this project incorrectly passed the per-Python neighbor floor here, which
    truncated S(t) to ~10 points and made the linear region undetectable.

    Defaults: 100 in production, 30 in ``DCH_TEST_MODE=true`` (short series).
    Test mode runs typically have N≈100 phase-space points after embedding;
    asking for ``-s 100`` then fails inside lyap_k (insufficient trajectory).
    """
    raw = os.environ.get("DCH_LYAP_ITERATIONS", "").strip()
    if raw:
        try:
            return max(5, int(float(raw)))
        except ValueError:
            pass
    return (
        DEFAULT_DCH_LYAP_ITERATIONS_TEST
        if _is_test_mode_env()
        else DEFAULT_LYAP_ITERATIONS
    )


def lyap_min_neighbors() -> int:
    """Minimum neighbours for LLE block filter (env ``DCH_LYAP_MIN_NEIGHBORS``).

    Applied in Python (``invariants_lyapunov.extract_lle_ols``) after parsing
    lyap_k output. **Not** a lyap_k CLI flag.
    """
    raw = os.environ.get("DCH_LYAP_MIN_NEIGHBORS", "").strip()
    if raw:
        try:
            return max(1, int(float(raw)))
        except ValueError:
            pass
    return MIN_LYAP_NEIGHBORS


DEFAULT_RQA_RADIUS = 0.005
RQA_EMBEDDING_DIM = 3

# Percentile-based recurrence threshold (per `rqa_tran.pdf`):
# fix the radius as a quantile of pairwise distances between embedded state
# vectors so the recurrence rate stays roughly invariant across coins,
# embedding dimensions and window lengths. Default 4% ~ RR ~ 4%.
RQA_RADIUS_PERCENTILE_DEFAULT = 4.0
# Embedded matrix is subsampled to this many vectors before pdist to keep
# memory bounded (pdist is O(N^2)). 5000 vectors -> ~12.5M pairs (~100 MB).
RQA_RADIUS_MAX_VECTORS = 5000
RQA_RADIUS_SAMPLE_SEED = 0
D2_EPSILON_STEPS = 100
D2_PAIR_LIMIT = 0
MIN_PLATEAU_POINTS = 8

# The metric registry controls CLI validation, null/reference recomputation, and
# bootstrap testing. Only metrics in BOOTSTRAP_TEST_METRICS get a TS decision.
#
# The two sets below are *defaults* used when ``--rqa_bootstrap=on`` (the
# project default). When the user passes ``--rqa_bootstrap=off``, RQA scalar
# metrics are demoted to original-series-only just like the legacy behaviour.
# Switching is handled in ``main()`` so the in-file constants stay declarative.
RQA_KEYS = ("RR", "DET", "LAM", "MAXLINE", "ENTR", "TT", "TREND")
ALL_METRICS = ("TAKENS", "ELLNER", "LLE", *RQA_KEYS)
DIM_LLE_METRICS = ("TAKENS", "ELLNER", "LLE")
NULL_SERIES_METRICS = {*DIM_LLE_METRICS, *RQA_KEYS}
BOOTSTRAP_TEST_METRICS = {*DIM_LLE_METRICS, *RQA_KEYS}
