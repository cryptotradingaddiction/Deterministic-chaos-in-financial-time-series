#!/usr/bin/env python3
"""
Capacity dimension diagnostic via TISEAN boxcount.exe (partition method).

Uses Renyi entropy order Q=0 (capacity / box-count dimension D_0). For each
embedding dimension m, estimates d_c as the slope of H_0(m, epsilon) vs ln(1/epsilon)
from the TISEAN partition grid — no custom NumPy box-count heuristics.
"""

from __future__ import annotations

import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from config_loader import (
    ensure_dir,
    get_data_dir,
    get_results_dir,
    load_config,
    pipeline_logreturn_files,
    prefer_liquidity_cut,
    tau_for_symbol_from_mutual,
)
from report_helper import Reporter, append_summary_row
from tisean_io import parse_boxcount_blocks, run_boxcount

SUMMARY_FILE = "_2dc_summary.txt"
SERIES_COL_W = 46

SUMMARY_HEADER = (
    f"{'series_id':<{SERIES_COL_W}} {'tau':>4} {'best_m':>6} {'d_c':>10} {'+/-95CI':>10} "
    f"{'R^2':>8} {'note':<40}"
)

files = pipeline_logreturn_files(ext="csv")

m_values = [2, 3, 4, 5, 10]
BOXCOUNT_Q = 0.0
BOXCOUNT_EPS_STEPS = 40
MIN_WINDOW_POINTS = 8
MIN_R2_FOR_TRUST = 0.98


def tau_for_path(file_path: str, config=None) -> int:
    sym = os.path.basename(file_path).split("_")[0]
    return tau_for_symbol_from_mutual(sym, config)


def _boxcount_input_options(file_path: str) -> tuple[int, int | None]:
    """Return (skip_rows, column) for boxcount -c / -x."""
    if file_path.lower().endswith(".csv"):
        return 1, 2
    return 0, None


def select_best_scaling_window(x_vals, y_vals, min_points=8):
    """Pick the best contiguous linear window on ln(1/eps) vs H_Q."""
    n = len(x_vals)
    if n < min_points:
        return None

    best = None
    best_score = -np.inf
    for i in range(0, n - min_points + 1):
        for j in range(i + min_points, n + 1):
            xs = x_vals[i:j]
            ys = y_vals[i:j]
            fit = linregress(xs, ys)
            r2 = fit.rvalue**2
            score = r2 + 0.05 * ((j - i) / n)
            if score > best_score:
                best_score = score
                best = {
                    "start": i,
                    "end": j,
                    "slope": fit.slope,
                    "intercept": fit.intercept,
                    "r2": r2,
                    "stderr": fit.stderr if fit.stderr is not None else np.nan,
                }
    return best


def estimate_dc_from_boxcount_block(block: np.ndarray, embed_m: int) -> dict | None:
    """
    Estimate D_Q(m) from one boxcount block (TISEAN partition method).

    TISEAN documents the *differential* entropy (column 3) slope vs ln(1/epsilon)
    as D_Q(m). For Q=0 (capacity) this plateau is often flat in epsilon; then the
    plateau mean is the incremental dimension at embedding m. For m=1 we use the
    slope of H_Q (column 2) on the middle epsilon band (coarse/fine ends trimmed).
    """
    if block.shape[0] < MIN_WINDOW_POINTS:
        return None

    eps = block[:, 0]
    hq = block[:, 1]
    diff = block[:, 2]

    if embed_m <= 1:
        valid = np.isfinite(eps) & np.isfinite(hq) & (eps > 0) & (hq > 0)
        if np.sum(valid) < MIN_WINDOW_POINTS:
            return None
        ln_inv_eps = np.log(1.0 / eps[valid])
        hq_valid = hq[valid]
        n = len(ln_inv_eps)
        i0, i1 = int(0.12 * n), max(int(0.72 * n), i0 + MIN_WINDOW_POINTS)
        xs, ys = ln_inv_eps[i0:i1], hq_valid[i0:i1]
        fit = linregress(xs, ys)
        r2 = fit.rvalue**2
        return {
            "ln_inv_eps": ln_inv_eps,
            "y_plot": hq_valid,
            "y_label": "H_Q (Q=0)",
            "selected": {"start": i0, "end": i1, "intercept": fit.intercept, "slope": fit.slope},
            "slope": float(fit.slope),
            "r2": r2,
            "stderr": fit.stderr if fit.stderr is not None else np.nan,
            "ci95": 1.96 * fit.stderr if fit.stderr is not None else np.nan,
            "method": "hq_slope",
        }

    valid = np.isfinite(eps) & np.isfinite(diff) & (eps > 0) & (diff > 0)
    if np.sum(valid) < MIN_WINDOW_POINTS:
        return None

    ln_inv_eps = np.log(1.0 / eps[valid])
    diff_valid = diff[valid]
    rel_std = float(np.std(diff_valid) / (np.mean(diff_valid) + 1e-12))

    if rel_std < 0.02:
        slope = float(np.mean(diff_valid))
        return {
            "ln_inv_eps": ln_inv_eps,
            "y_plot": diff_valid,
            "y_label": "dH_Q (m|m-1)",
            "selected": {"start": 0, "end": len(ln_inv_eps), "intercept": slope, "slope": 0.0},
            "slope": slope,
            "r2": 1.0,
            "stderr": float(np.std(diff_valid) / np.sqrt(len(diff_valid))),
            "ci95": 1.96 * float(np.std(diff_valid) / np.sqrt(len(diff_valid))),
            "method": "diff_plateau",
        }

    selected = select_best_scaling_window(ln_inv_eps, diff_valid, min_points=MIN_WINDOW_POINTS)
    if selected is None:
        return None
    return {
        "ln_inv_eps": ln_inv_eps,
        "y_plot": diff_valid,
        "y_label": "dH_Q (m|m-1)",
        "selected": selected,
        "slope": selected["slope"],
        "r2": selected["r2"],
        "stderr": selected["stderr"],
        "ci95": 1.96 * selected["stderr"] if np.isfinite(selected["stderr"]) else np.nan,
        "method": "diff_slope",
    }


def run_2dc_single(file_path: str, output_dir: str, tau: int) -> None:
    stem = os.path.splitext(os.path.basename(file_path))[0]
    series_id = (stem[:SERIES_COL_W] + " " * SERIES_COL_W)[:SERIES_COL_W].rstrip()

    rep = Reporter()
    rep.add(f"\nProcessing {stem}: {file_path} (tau={tau})")
    rep.add("Engine: TISEAN boxcount.exe (partition Renyi entropy, Q=0 -> capacity D_0)")
    rep.add(f"Embedding dimensions m = {m_values}")
    rep.add(f"Epsilon steps: {BOXCOUNT_EPS_STEPS}")
    rep.add("-" * 80)
    rep.add(f"{'m':>3} {'d_c':>10} {'+/-95CI':>10} {'R^2':>8} "
            f"{'window':>12} {'points':>8}  flags")
    rep.add("-" * 80)

    skip_rows, column = _boxcount_input_options(file_path)
    out_prefix = os.path.join(output_dir, f"{stem}_tau{tau}_boxcount")
    try:
        run_boxcount(
            file_path,
            tau,
            max(m_values),
            out_prefix,
            q=BOXCOUNT_Q,
            eps_steps=BOXCOUNT_EPS_STEPS,
            skip_rows=skip_rows,
            column=column,
        )
        blocks = parse_boxcount_blocks(out_prefix)
    except (OSError, subprocess.CalledProcessError) as exc:
        rep.add(f"boxcount failed: {exc}")
        print(f"  SKIP (boxcount): {file_path}: {exc}")
        return

    if not blocks:
        rep.add("boxcount produced no parseable blocks.")
        print(f"  SKIP (empty boxcount output): {file_path}")
        return

    dc_results = []
    r2_results = []
    ci95_results = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for m in m_values:
        block = blocks.get(m)
        if block is None:
            dc_results.append(np.nan)
            r2_results.append(np.nan)
            ci95_results.append(np.nan)
            rep.add(f"{m:>3} {'nan':>10} {'nan':>10} {'nan':>8} {'-':>12} {'-':>8}  missing block")
            continue

        est = estimate_dc_from_boxcount_block(block, m)
        if est is None:
            dc_results.append(np.nan)
            r2_results.append(np.nan)
            ci95_results.append(np.nan)
            rep.add(f"{m:>3} {'nan':>10} {'nan':>10} {'nan':>8} {'-':>12} {'-':>8}  no valid scaling window")
            continue

        slope = est["slope"]
        r2 = est["r2"]
        ci95 = est["ci95"]
        selected = est["selected"]
        dc_results.append(slope)
        r2_results.append(r2)
        ci95_results.append(ci95)

        flags = []
        if r2 < MIN_R2_FOR_TRUST:
            flags.append("LOW_R2")
        if est["method"] == "diff_plateau":
            flags.append("DIFF_PLATEAU")
        flags_str = ",".join(flags) if flags else "-"
        rep.add(
            f"{m:>3} {slope:>10.4f} {ci95:>10.4f} {r2:>8.4f} "
            f"{selected['start']:>5}:{selected['end']:<5} "
            f"{len(est['ln_inv_eps']):>3}/{len(est['ln_inv_eps']):<3}  {flags_str} [{est['method']}]"
        )

        line, = ax1.plot(
            est["ln_inv_eps"], est["y_plot"],
            marker="o", markersize=3, label=f"m={m}",
        )
        if est["method"] == "diff_plateau":
            ax1.axhline(slope, color=line.get_color(), linestyle="--", linewidth=1.5, alpha=0.8)
        else:
            ax1.plot(
                est["ln_inv_eps"][selected["start"]:selected["end"]],
                selected["intercept"] + slope * est["ln_inv_eps"][selected["start"]:selected["end"]],
                color=line.get_color(),
                linestyle="--",
                linewidth=2,
                alpha=0.8,
            )

    ax1.set_xlabel("ln(1/epsilon)")
    ax1.set_ylabel("TISEAN boxcount (Q=0): H_Q or dH_Q")
    ax1.set_title(f"{stem} TISEAN boxcount scaling (tau={tau})")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(m_values, dc_results, "ko-", label="Estimated d_c")
    ax2.plot(m_values, m_values, "r--", label="d_c = m (stochastic process)")
    ax2.set_xlabel("Embedding dimension m")
    ax2.set_ylabel("Capacity dimension d_c")
    ax2.set_title(f"{stem} d_c saturation (tau={tau})")

    reliable_idx = [
        i
        for i, (r2, dc, ci95) in enumerate(zip(r2_results, dc_results, ci95_results))
        if np.isfinite(r2) and np.isfinite(dc) and np.isfinite(ci95)
        and r2 >= MIN_R2_FOR_TRUST and dc > 0
    ]
    if reliable_idx:
        best_i = min(reliable_idx, key=lambda i: (ci95_results[i], -r2_results[i]))
        best_m = m_values[best_i]
        best_dc = dc_results[best_i]
        best_ci = ci95_results[best_i]
        ax2.scatter([best_m], [best_dc], color="blue", s=80, zorder=5, label="Best m")
        ax2.annotate(
            f"best m={best_m}\nd_c={best_dc:.3f}+/-{best_ci:.3f}",
            xy=(best_m, best_dc),
            xytext=(best_m + 1.0, best_dc + 0.15),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", color="blue", lw=1.2),
            fontsize=9, color="blue",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="blue", alpha=0.8),
        )
    else:
        ax2.text(
            0.03, 0.95,
            "No reliable best m (R^2 below threshold)",
            transform=ax2.transAxes,
            ha="left", va="top", fontsize=9, color="darkred",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="darkred", alpha=0.8),
        )
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    out_plot = os.path.join(output_dir, f"{stem}_2dc_capacity_dimension_tau{tau}.png")
    plt.savefig(out_plot, dpi=150)
    plt.close(fig)

    rep.add("-" * 80)
    if reliable_idx:
        rep.add(
            f"Best m: m={best_m}, d_c={best_dc:.4f} +/- {best_ci:.4f} (95% CI), "
            f"R^2={r2_results[best_i]:.4f}"
        )
        note = f"best m={best_m} (d_c={best_dc:.3f}+/-{best_ci:.3f})"
        summary_row = (
            f"{series_id:<{SERIES_COL_W}} {tau:>4d} {best_m:>6d} {best_dc:>10.4f} {best_ci:>10.4f} "
            f"{r2_results[best_i]:>8.4f} {note:<40}"
        )
    else:
        rep.add(f"No reliable best m (no fit reached R^2 >= {MIN_R2_FOR_TRUST}).")
        summary_row = (
            f"{series_id:<{SERIES_COL_W}} {tau:>4d} {'-':>6} {'nan':>10} {'nan':>10} "
            f"{'nan':>8} {'no reliable best m':<40}"
        )

    rep.add(f"Saved boxcount: {out_prefix}")
    rep.add(f"Saved plot: {out_plot}")
    out_txt = rep.write(output_dir, f"{stem}_2dc_tau{tau}_results.txt")
    print(f"Saved text report: {out_txt}")
    summary_path = append_summary_row(output_dir, SUMMARY_FILE, SUMMARY_HEADER, summary_row)
    print(f"Appended row to summary: {summary_path}")


if __name__ == "__main__":
    cfg = load_config()
    dd = get_data_dir(cfg)
    od = ensure_dir(os.path.join(get_results_dir(cfg), "2dc"))

    try:
        os.remove(os.path.join(od, SUMMARY_FILE))
    except OSError:
        pass

    for fn in files:
        fp = prefer_liquidity_cut(os.path.join(dd, fn))
        if not os.path.exists(fp):
            print(f"File not found, skipping: {fp}")
            continue
        run_2dc_single(fp, od, tau_for_path(fp, cfg))
