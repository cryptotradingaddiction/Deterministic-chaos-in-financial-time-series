#!/usr/bin/env python3
"""Hypothesis package facade: thin CLI entry + re-exports for batch scripts and tools."""

from hypothesis_cli import main, metric_names_for_scope, parse_metrics_list

# Config / registry
from hypothesis_config import (  # noqa: F401
    ALL_METRICS,
    BOOTSTRAP_TEST_METRICS,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_HYPOTHESIS_SEED,
    DEFAULT_RQA_RADIUS,
    DEFAULT_STATIONARY_BLOCK_MEAN,
    DEFAULT_TS_THRESHOLD,
    DIM_LLE_METRICS,
    M_D2,
    M_LYAP,
    MIN_LYAP_LINEAR_POINTS,
    MIN_LYAP_NEIGHBORS,
    MIN_PLATEAU_POINTS,
    NULL_SERIES_METRICS,
    RQA_EMBEDDING_DIM,
    RQA_KEYS,
    RQA_RADIUS_MAX_VECTORS,
    RQA_RADIUS_PERCENTILE_DEFAULT,
    T_DOF,
)

# Surrogates / TS
from hypothesis_surrogates import (  # noqa: F401
    generate_normal_series,
    generate_single_surrogate,
    generate_t_series,
)
from hypothesis_ts import invariant_bootstrap_ts_test  # noqa: F401

# Invariants + TISEAN
from invariants_compute import compute_invariants  # noqa: F401
from invariants_correlation import compute_ellner_from_c2, extract_takens_plateau  # noqa: F401
from invariants_lyapunov import (  # noqa: F401
    _best_linear_slope,
    _best_linear_slope_window,
    _parse_lyap_blocks,
    extract_lle_mean_std,
)
from invariants_rqa import (  # noqa: F401
    compute_percentile_radius,
    compute_pyrqa_metrics,
    compute_rqa_trend,
    embed_series,
    format_rqa_radius,
    tisean_theiler_min_diagonal_k,
)
from tisean_io import extract_tagged_block, load_data, resolve_tool, run_c2t, run_d2, run_lyap_k  # noqa: F401

__all__ = [
    "main",
    "parse_metrics_list",
    "metric_names_for_scope",
    "compute_invariants",
    "compute_percentile_radius",
    "format_rqa_radius",
    "extract_lle_mean_std",
    "_parse_lyap_blocks",
    "_best_linear_slope",
    "_best_linear_slope_window",
]

if __name__ == "__main__":
    main()
