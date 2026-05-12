#!/usr/bin/env python3
"""CLI helper for the percentile-based RQA recurrence threshold.

Prints one value to stdout: the effective radius formatted exactly as the BAT
pipeline should use it in run IDs and `recurr.exe -r`.
"""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--delay", type=int, required=True)
    parser.add_argument("--fallback", type=float, default=np.nan)
    parser.add_argument("--percentile", type=float, default=RQA_RADIUS_PERCENTILE_DEFAULT)
    parser.add_argument("--m", type=int, default=RQA_EMBEDDING_DIM)
    parser.add_argument("--max_vectors", type=int, default=RQA_RADIUS_MAX_VECTORS)
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

    print(format_rqa_radius(radius))


if __name__ == "__main__":
    main()
