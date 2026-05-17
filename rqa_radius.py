#!/usr/bin/env python3
"""
Compute the recurrence radius for TISEAN ``recurr.exe`` and PyRQA from embedded data.

**Role in the pipeline**

``RQA.bat`` calls this script before ``recurr.exe`` so the recurrence plot and
downstream ``rqa_values.py`` / ``hypothesis.py`` share one threshold per coin:

1. Load the (possibly liquidity-cut) log-return series.
2. Embed with delay ``--delay`` and dimension ``--m`` (default **m = 3**).
3. Take the **percentile** of pairwise Euclidean distances between embedded
   vectors (default **4th percentile**, see ``rqa_tran.pdf`` / ``hypothesis``).
4. Print a single formatted number on **stdout** for the batch file to capture
   into ``COIN_RAD_EFF`` and ``recurr.exe -r<value>``.

**Fallback**

If the percentile is non-finite or ≤ 0 (degenerate series, too few points),
``--fallback`` is used — typically ``RAD_RQA_<sym>`` from ``_per_coin_settings.bat``.

**Output format**

Stdout is produced by :func:`hypothesis.format_rqa_radius` so run IDs and TISEAN
flags stay consistent (no scientific notation surprises in ``.bat`` parsing).
"""

from __future__ import annotations

import argparse

import numpy as np

from hypothesis import (
    RQA_EMBEDDING_DIM,
    RQA_RADIUS_MAX_VECTORS,
    RQA_RADIUS_PERCENTILE_DEFAULT,
    compute_percentile_radius,
    format_rqa_radius,
)
from surrogate_sampling import load_series_1d


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print one recurrence radius (percentile of embedded distances)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to 1-column .dat or similar (same files as RQA.bat).",
    )
    parser.add_argument(
        "--delay",
        type=int,
        required=True,
        help="Embedding delay tau (TAU_RQA_<sym> from per-coin settings).",
    )
    parser.add_argument(
        "--fallback",
        type=float,
        default=np.nan,
        help="RAD_RQA_<sym> used when percentile radius cannot be computed.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=RQA_RADIUS_PERCENTILE_DEFAULT,
        help="Percentile of pairwise distances (default 4.0).",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=RQA_EMBEDDING_DIM,
        help="Embedding dimension (fixed at 3 in this project).",
    )
    parser.add_argument(
        "--max_vectors",
        type=int,
        default=RQA_RADIUS_MAX_VECTORS,
        help="Subsample cap for distance matrix (memory / speed).",
    )
    args = parser.parse_args()

    data = load_series_1d(args.input)
    radius = compute_percentile_radius(
        data,
        delay=args.delay,
        m=args.m,
        percentile=args.percentile,
        max_vectors=args.max_vectors,
    )
    if not (np.isfinite(radius) and radius > 0.0):
        radius = args.fallback

    if not (np.isfinite(radius) and radius > 0.0):
        raise SystemExit("rqa_radius.py: could not compute radius and fallback is invalid")

    # Single token for cmd / RQA.bat variable substitution.
    print(format_rqa_radius(radius))


if __name__ == "__main__":
    main()
