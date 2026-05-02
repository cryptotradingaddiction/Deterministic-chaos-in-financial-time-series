#!/usr/bin/env python3
"""
Test normality of all generated surrogate series.

Surrogates are generated with block permutation (same helper as hypothesis.py).
For each surrogate, a normality test p-value is computed and compared to alpha.
"""

import argparse
import sys

import numpy as np
import scipy.stats as stats

from surrogate_sampling import generate_permuted_samples, load_series_1d


def test_normality(series: np.ndarray, alpha: float) -> tuple[float, bool, str]:
    """Return (p_value, is_normal, method_used)."""
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan"), False, "insufficient_n"

    # D'Agostino-Pearson requires n >= 8.
    if n >= 8:
        _stat, p_val = stats.normaltest(x)
        method = "normaltest"
    else:
        _stat, p_val = stats.shapiro(x)
        method = "shapiro"

    is_normal = bool(np.isfinite(p_val) and (p_val >= alpha))
    return float(p_val), is_normal, method


def main() -> int:
    parser = argparse.ArgumentParser(description="Check normality for all surrogates.")
    parser.add_argument("--input", required=True, help="Path to 1D input series (.dat).")
    parser.add_argument("--b", type=int, default=100, help="Number of surrogates (default: 100).")
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=100,
        help="Number of contiguous blocks for block permutation (default: 100).",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05).")
    args = parser.parse_args()

    original = load_series_1d(args.input)
    surrogates = generate_permuted_samples(original, args.b, n_blocks=args.n_blocks)

    passed = 0
    failed_indices = []
    p_values = []
    method_used = "normaltest"

    for idx, surr in enumerate(surrogates, start=1):
        p_val, is_normal, method = test_normality(surr, args.alpha)
        p_values.append(p_val)
        method_used = method
        if is_normal:
            passed += 1
        else:
            failed_indices.append(idx)

    total = len(surrogates)
    all_normal = (passed == total and total > 0)

    finite_p = [p for p in p_values if np.isfinite(p)]
    p_min = min(finite_p) if finite_p else float("nan")
    p_med = float(np.median(finite_p)) if finite_p else float("nan")
    p_max = max(finite_p) if finite_p else float("nan")

    print("Surrogate normality check")
    print(f"Input file         : {args.input}")
    print(f"Surrogates (B)     : {total}")
    print(f"Block count        : {args.n_blocks}")
    print(f"Alpha              : {args.alpha}")
    print(f"Test method        : {method_used} (fallback shapiro for very short series)")
    print(f"Passed             : {passed}/{total}")
    print(f"Failed             : {total - passed}/{total}")
    print(f"p-value stats      : min={p_min:.4e}, median={p_med:.4e}, max={p_max:.4e}")
    if failed_indices:
        preview = ",".join(map(str, failed_indices[:20]))
        suffix = "..." if len(failed_indices) > 20 else ""
        print(f"Failed surrogate # : {preview}{suffix}")

    print(f"ALL_SURROGATES_NORMAL: {'YES' if all_normal else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
