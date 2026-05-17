#!/usr/bin/env python3
"""
Cross-check τ, W, embedding dimension *m*, normalization policy, and bootstrap defaults.

Run before a full hypothesis batch or after editing ``_per_coin_settings.bat``::

    py -3 audit_invariant_parameters.py

Exit code 0 = all invariant-path checks passed; 1 = at least one issue reported.
"""
from __future__ import annotations

import sys

from config_loader import PIPELINE_SYMBOLS, audit_invariant_parameters
from hypothesis_config import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_STATIONARY_BLOCK_MEAN,
    M_D2,
    M_LYAP,
    RQA_EMBEDDING_DIM,
)


def main() -> int:
    print("DCh invariant parameter audit")
    print("=" * 60)
    print()
    print("Formal invariants (TAKENS/ELLNER, LLE, RQA hypothesis path)")
    print(f"  embedding m     : {M_D2} (d2/c2t), {M_LYAP} (lyap_k), {RQA_EMBEDDING_DIM} (PyRQA/recurr)")
    print(f"  tau, W per coin : _per_coin_settings.bat (W_D2_* := TAU_D2_* after theilers_w.bat)")
    print(f"  PyRQA Theiler   : W+1 via tisean_theiler_min_diagonal_k (matches TISEAN |i-j|<=W)")
    print(f"  series scale    : raw log-returns (liquidity-cut .dat when present; no z-score)")
    print(f"  bootstrap B     : default {DEFAULT_BOOTSTRAP_SAMPLES} (env DCH_BOOTSTRAP_SAMPLES)")
    blk = (
        "sqrt(n)"
        if DEFAULT_STATIONARY_BLOCK_MEAN <= 0
        else str(DEFAULT_STATIONARY_BLOCK_MEAN)
    )
    print(f"  block length    : {blk} (env DCH_STATIONARY_BLOCK_MEAN or --stationary_block_mean)")
    print()
    print("Intentional differences (not errors)")
    print("  cao_.py         : m = 1..d_max (dimension selection diagnostic)")
    print("  2dc.py          : per-axis min-max to [0,1] before box counting")
    print("  phase_2D/3D     : visualization only (m=2 or m=3)")
    print("  tau_w.py        : separate decorrelation heuristic (not used in invariants)")
    print()

    issues = audit_invariant_parameters(symbols=PIPELINE_SYMBOLS)
    if not issues:
        print("OK: all checks passed for", ", ".join(PIPELINE_SYMBOLS))
        return 0

    print(f"FOUND {len(issues)} issue(s):")
    for item in issues:
        print(f"  - {item}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
