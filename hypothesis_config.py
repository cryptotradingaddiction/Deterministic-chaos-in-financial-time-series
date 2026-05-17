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

# Must match -M1,3 in correlation_dimension.bat
M_D2 = 3
M_LYAP = 3
MIN_LYAP_LINEAR_POINTS = 3  # was 5; financial S(t) linear region is ~1-2 iterations
MIN_LYAP_NEIGHBORS = 10
# Short-series test mode (DCH_TEST_MODE): lyap_k -n / -s overrides via env.
DEFAULT_DCH_LYAP_STEPS_TEST = 40
DEFAULT_DCH_LYAP_MIN_NEIGHBORS_TEST = 3


def lyap_k_steps() -> int:
    """Iteration count for lyap_k ``-n`` (env ``DCH_LYAP_STEPS``, default 500)."""
    raw = os.environ.get("DCH_LYAP_STEPS", "").strip()
    if raw:
        try:
            return max(5, int(float(raw)))
        except ValueError:
            pass
    return 500


def lyap_min_neighbors() -> int:
    """Minimum neighbours for LLE block filter (env ``DCH_LYAP_MIN_NEIGHBORS``)."""
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
