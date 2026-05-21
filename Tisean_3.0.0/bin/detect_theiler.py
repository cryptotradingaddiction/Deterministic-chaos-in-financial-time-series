"""Theiler-window detector based on the Kantz/Schreiber lower bound.

The Theiler window ``W`` (denoted ``tau_0`` in the literature) is computed
from the textbook bound (eq. 8.85)

    W = ceil( tau_a * (2 / N) ** (2 / m) )

where:

* ``tau_a`` (decorrelation time) is the first non-positive lag of the
  autocorrelation function produced by TISEAN ``corr.exe``; the 1/e crossing
  is available as an alternative via ``--decor 1e``;
* ``N`` is the length of the analysed series (``--N`` or counted from
  ``--data``);
* ``m`` is the embedding dimension used downstream (``--m``).

Because ``(2/N)^(2/m) << 1`` for any realistic ``N``, the analytic bound is
essentially trivially satisfied. The script therefore relies on
``--floor_at_tau`` -- the practical recommendation from the same reference
("in many practical computations setting tau_0 = tau used for the phase-space
reconstruction is sufficient") -- to make ``W`` a meaningful Theiler window.

A space-time separation plot from ``stp.exe`` is no longer used to decide
``W``; the script still parses it when supplied so the per-band saturation
value (``W_stp``) is stored in the report file for diagnostic comparison with
the formula.

Stdout contains exactly one integer (``W_final``) so a Windows batch can
ingest it with ``set /p`` via a temporary file. When ``--report <path>`` is
provided, all components plus optional metadata are written there as
``KEY=VALUE`` lines.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# STP parsing (diagnostic only)
# ---------------------------------------------------------------------------


def _load_blocks(filename: str) -> list[list[tuple[int, float]]]:
    blocks: list[list[tuple[int, float]]] = []
    current: list[tuple[int, float]] = []
    try:
        fh = open(filename, "r", encoding="ascii", errors="replace")
    except OSError as exc:
        print(f"[detect_theiler] cannot open STP file {filename}: {exc}", file=sys.stderr)
        return blocks
    with fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                if current:
                    blocks.append(current)
                    current = []
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                dt = int(float(parts[0]))
                value = float(parts[1])
            except ValueError:
                continue
            current.append((dt, value))
        if current:
            blocks.append(current)
    return blocks


def _detect_block_w(
    pairs: list[tuple[int, float]],
    threshold: float,
    smooth: int,
    check: int,
) -> int:
    if len(pairs) < max(10, check + 1):
        return 0
    pairs = sorted(pairs, key=lambda r: r[0])
    dts = np.asarray([p[0] for p in pairs], dtype=int)
    vals = np.asarray([p[1] for p in pairs], dtype=float)

    if smooth >= 2:
        kernel = np.ones(smooth, dtype=float) / float(smooth)
        smoothed = np.convolve(vals, kernel, mode="same")
    else:
        smoothed = vals.copy()

    tail_start = int(len(smoothed) * 0.8)
    tail_start = min(max(tail_start, 1), len(smoothed) - 1)
    asymptote = float(np.median(smoothed[tail_start:]))
    if not np.isfinite(asymptote) or asymptote <= 0.0:
        return 0
    limit = threshold * asymptote

    horizon = len(smoothed) - check
    for i in range(horizon):
        if smoothed[i] < limit:
            continue
        if np.all(smoothed[i + 1 : i + 1 + check] >= limit):
            return int(dts[i])

    for i in range(len(smoothed)):
        if smoothed[i] >= limit:
            return int(dts[i])
    return 0


def _aggregate(values: Iterable[int], how: str) -> int:
    vals = [int(v) for v in values if int(v) > 0]
    if not vals:
        return 0
    how = (how or "max").strip().lower()
    if how == "min":
        return int(min(vals))
    if how == "median":
        return int(round(float(np.median(vals))))
    if how == "mean":
        return int(round(float(np.mean(vals))))
    return int(max(vals))


def detect_stp_w(
    stp_file: str,
    threshold: float = 0.95,
    smooth: int = 5,
    check: int = 10,
    aggregate: str = "max",
    verbose: bool = True,
) -> int:
    """Diagnostic-only: per-band saturation W from a TISEAN stp.exe output."""
    blocks = _load_blocks(stp_file)
    if not blocks:
        if verbose:
            print(f"[detect_theiler] no STP blocks parsed from {stp_file}", file=sys.stderr)
        return 0

    per_block: list[int] = []
    for idx, block in enumerate(blocks):
        w = _detect_block_w(block, threshold=threshold, smooth=smooth, check=check)
        per_block.append(w)
        if verbose:
            frac = (idx + 1) / float(len(blocks))
            print(
                f"[detect_theiler] STP block {idx:>3d} (fraction~{frac:.2%}): W={w}",
                file=sys.stderr,
            )
    return _aggregate(per_block, aggregate)


# ---------------------------------------------------------------------------
# ACF parsing: decorrelation time τ_a
# ---------------------------------------------------------------------------


def _read_acf(acf_file: str) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    try:
        fh = open(acf_file, "r", encoding="ascii", errors="replace")
    except OSError as exc:
        print(f"[detect_theiler] cannot open ACF file {acf_file}: {exc}", file=sys.stderr)
        return rows
    with fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                lag = int(float(parts[0]))
                val = float(parts[1])
            except ValueError:
                continue
            rows.append((lag, val))
    return rows


def _normalise_acf(rows: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Rescale ρ(0) → 1 if the file does not already store it that way."""
    rho0 = None
    for lag, val in rows:
        if lag == 0:
            rho0 = val
            break
    if rho0 is None or abs(rho0) < 1e-30 or abs(rho0 - 1.0) < 1e-6:
        return rows
    return [(lag, val / rho0) for lag, val in rows]


def decorrelation_time(
    acf_file: str,
    method: str = "acf_zero",
    verbose: bool = True,
) -> int:
    """Return ``tau_a`` (positive integer lag) using the requested ACF criterion.

    ``method == 'acf_zero'``   first lag >= 1 with rho(k) <= 0
    ``method == '1e'``         first lag >= 1 with rho(k) <= 1/e ~ 0.3679
    Returns 0 when the criterion is never reached or the file is empty.
    """
    rows = _normalise_acf(_read_acf(acf_file))
    rows = [(lag, val) for lag, val in rows if lag >= 1]
    rows.sort(key=lambda r: r[0])
    if not rows:
        if verbose:
            print(f"[detect_theiler] empty ACF in {acf_file}", file=sys.stderr)
        return 0

    threshold = 0.0 if method == "acf_zero" else math.exp(-1.0)
    for lag, val in rows:
        if val <= threshold:
            if verbose:
                print(
                    f"[detect_theiler] decorrelation_time ({method}): "
                    f"tau_a = {lag} (rho={val:.4g} <= {threshold:.4g})",
                    file=sys.stderr,
                )
            return int(lag)

    if verbose:
        print(
            f"[detect_theiler] ACF never reaches {method} threshold "
            f"within {len(rows)} lags",
            file=sys.stderr,
        )
    return 0


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def count_nonblank_lines(data_file: str) -> int:
    try:
        fh = open(data_file, "r", encoding="ascii", errors="replace")
    except OSError:
        return 0
    n = 0
    with fh:
        for raw in fh:
            if raw.strip():
                n += 1
    return n


def _write_report(path: str, items: dict[str, object]) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="ascii", errors="replace", newline="\r\n") as fh:
            for key, value in items.items():
                fh.write(f"{key}={value}\n")
    except OSError as exc:
        print(f"[detect_theiler] cannot write report {path}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Theiler-window formula
# ---------------------------------------------------------------------------


def theiler_from_formula(tau_a: int, n: int, m: int, verbose: bool = True) -> int:
    """``W = ceil(tau_a * (2 / N) ** (2 / m))`` with sanity checks.

    Implements eq. 8.85 from the reference verbatim. The textbook gives the
    strict inequality ``W > tau_a * (2/N)^(2/m)``; using ``ceil`` returns the
    smallest integer that satisfies the bound. Returns 0 when any input is
    non-positive or the raw value is non-finite/<=0 (formula undefined there).

    Note: for any realistic ``N`` this bound is essentially trivial -- the
    raw value falls well below 1 -- so the binding rule in practice is
    ``--floor_at_tau`` (set ``W = tau`` per the same textbook).
    """
    if tau_a <= 0 or n <= 1 or m <= 0:
        if verbose:
            print(
                f"[detect_theiler] invalid formula inputs: tau_a={tau_a}, N={n}, m={m}",
                file=sys.stderr,
            )
        return 0
    raw = float(tau_a) * (2.0 / float(n)) ** (2.0 / float(m))
    if not math.isfinite(raw) or raw <= 0.0:
        return 0
    w = int(math.ceil(raw))
    if verbose:
        print(
            f"[detect_theiler] formula: W = ceil({tau_a} * (2/{n})^(2/{m})) "
            f"= ceil({raw:.6g}) = {w}",
            file=sys.stderr,
        )
    return int(w)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Theiler window W from the Kantz/Schreiber formula "
            "W = ceil(tau_a * (2/N)^(2/m)) (eq. 8.85). Use --floor_at_tau to "
            "enforce the practical lower bound W >= tau. "
            "Stdout contains a single integer."
        )
    )
    parser.add_argument(
        "--acf",
        required=True,
        help="Path to corr.exe ACF output used to compute the decorrelation time tau_a.",
    )
    parser.add_argument(
        "--decor",
        choices=("acf_zero", "1e"),
        default="acf_zero",
        help=(
            "Definition of the decorrelation time tau_a: 'acf_zero' = first lag with rho(k) <= 0 "
            "(default), '1e' = first lag with rho(k) <= 1/e."
        ),
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to the data file; when --N is not given, N is the non-blank line count.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=0,
        help="Series length N. When omitted, --data is used to count non-blank lines.",
    )
    parser.add_argument(
        "--m",
        type=int,
        required=True,
        help="Embedding dimension m used downstream (d2/lyap_k/PyRQA).",
    )
    parser.add_argument(
        "--tau",
        type=int,
        default=None,
        help=(
            "Optional embedding delay tau. Used only for reporting and (with --floor_at_tau) "
            "to enforce W >= tau -- a practical lower bound recommended by Kantz/Schreiber."
        ),
    )
    parser.add_argument(
        "--floor_at_tau",
        action="store_true",
        help="Guarantee W >= --tau when the formula returns a smaller value.",
    )
    parser.add_argument(
        "--stp",
        default=None,
        help="Optional path to stp.exe output. Used only for diagnostic W_stp in the report.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Diagnostic STP saturation threshold (default 0.95; ignored for the formula).",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Diagnostic STP smoothing window (default 5).",
    )
    parser.add_argument(
        "--check",
        type=int,
        default=10,
        help="Diagnostic STP consecutive-step count (default 10).",
    )
    parser.add_argument(
        "--aggregate",
        default="max",
        choices=("max", "min", "median", "mean"),
        help="Diagnostic STP per-band aggregator (default 'max').",
    )
    parser.add_argument(
        "--fallback",
        type=int,
        default=0,
        help="Returned when the formula cannot be computed (default 0).",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional KEY=VALUE report path (TAU_A, N, M, W_FORMULA, W_STP, W_FINAL, ...).",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Optional symbol label written to the report (e.g. BTCUSD).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress diagnostics on stderr.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)

    verbose = not args.quiet

    n_pts = int(args.N) if args.N > 0 else (
        count_nonblank_lines(args.data) if args.data else 0
    )

    tau_a = decorrelation_time(args.acf, method=args.decor, verbose=verbose)

    w_formula = theiler_from_formula(tau_a=tau_a, n=n_pts, m=int(args.m), verbose=verbose)

    w_stp = 0
    if args.stp:
        w_stp = detect_stp_w(
            args.stp,
            threshold=args.threshold,
            smooth=max(1, int(args.smooth)),
            check=max(1, int(args.check)),
            aggregate=args.aggregate,
            verbose=verbose,
        )

    w_final = int(w_formula) if w_formula > 0 else int(args.fallback)
    floored = False
    # Project rule (theilers_w.bat): W equals the embedding delay tau used in
    # phase-space reconstruction — not tau_a, not W_stp, not the raw formula alone.
    if args.floor_at_tau and args.tau is not None and int(args.tau) > 0:
        w_final = int(args.tau)
        floored = True
        if verbose:
            print(
                f"[detect_theiler] W := tau={args.tau} "
                f"(formula={w_formula}, W_stp_diag={w_stp})",
                file=sys.stderr,
            )

    if args.report:
        report = {
            "TAU_A": int(tau_a),
            "N": int(n_pts),
            "M": int(args.m),
            "W_FORMULA": int(w_formula),
            "W_STP": int(w_stp),
            "W_FINAL": int(w_final),
            "DECOR_METHOD": args.decor,
            "FLOORED_AT_TAU": "true" if floored else "false",
        }
        if args.symbol:
            report["SYMBOL"] = args.symbol
        if args.tau is not None:
            report["TAU"] = int(args.tau)
        report["ACF_FILE"] = args.acf
        if args.stp:
            report["STP_FILE"] = args.stp
        if args.data:
            report["DATA_FILE"] = args.data
        _write_report(args.report, report)

    print(int(w_final))
    return 0


if __name__ == "__main__":
    sys.exit(main())
