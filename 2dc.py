import numpy as np                   # Numerical operations (arrays, log, floor, clip, etc.)
import matplotlib.pyplot as plt      # Plotting — replicates Horák et al. (2003) obr. 8.3 visually
from scipy.stats import linregress   # OLS linear regression for slope of ln M(r) vs ln(1/r) — eq. (8.4)
import os                            # File path utilities

from config_loader import (
    load_config,
    get_data_dir,
    get_results_dir,
    ensure_dir,
    pipeline_logreturn_files,
    prefer_liquidity_cut,
    tau_for_symbol_from_mutual,      # tau from mutual information first-minimum criterion
)
from report_helper import Reporter, append_summary_row

# ── Summary file constants ─────────────────────────────────────────────────────
 
SUMMARY_FILE = "_2dc_summary.txt"
SERIES_COL_W = 46
 
SUMMARY_HEADER = (
    f"{'series_id':<{SERIES_COL_W}} {'tau':>4} {'best_m':>6} {'d_c':>10} {'+/-95CI':>10} "
    f"{'R^2':>8} {'note':<40}"
)

# ── Input file list ────────────────────────────────────────────────────────────
# Per-coin log-return CSVs (timestamp + value); centralized in config_loader.
files = pipeline_logreturn_files(ext="csv")

# ── Computation parameters ─────────────────────────────────────────────────────
 
m_values = [2, 3, 4, 5, 10]
# Horák et al. (2003): capacity dimension d_c is estimated in reconstructed phase space
# of increasing embedding dimension m. If d_c saturates below m, it indicates a
# finite-dimensional attractor; if d_c grows with m, the process is consistent with
# a stochastic system with no finite attractor (obr. 8.2).
 
r_values = np.logspace(np.log10(0.02), np.log10(0.5), 40)
# Box sizes r — the "jemnost dělení" (grid fineness) in Horák et al. (2003), p. [8.2].
# Log-spacing gives uniform density on the ln(1/r) axis, which is the x-axis of the
# scaling law plot ln M(r) vs ln(1/r) — see obr. 8.3.
# Range [0.02, 0.5] in normalised coordinates: too small r causes finite-size saturation
# (M(r) → N), too large r gives M(r) = 1; the scaling region lies between.
 
MIN_WINDOW_POINTS = 8
# Minimum points required in a candidate scaling window.
# Horák et al. (2003): capacity is read from the "slope of the linear part of ln M(r)
# vs ln(1/r)" — eq. (8.4). Fewer than 8 points cannot reliably define a linear region.
 
MIN_R2_FOR_TRUST = 0.98
# Horák et al. (2003): scaling according to M(r) ~ r^(-d_c) — eq. (8.5) — is assumed
# to hold for small but finite r. R^2 >= 0.98 enforces that the selected window actually
# conforms to this power-law assumption; below this threshold the slope is unreliable.
 


# ── Helper: resolve tau from filename ─────────────────────────────────────────
 
def tau_for_path(file_path: str, config=None) -> int:
    sym = os.path.basename(file_path).split("_")[0]  # Extract symbol (e.g. "BTCUSD")
    return tau_for_symbol_from_mutual(sym, config)
    # tau is the embedding delay from the mutual information first-minimum criterion,
    # shared across all invariants for consistency.
 
 
# ── Helper: load log-return column ────────────────────────────────────────────
 
def load_logreturns_column(file_path: str) -> np.ndarray:
    """Load log-return CSV (datetime, log_return) or single-column .dat."""
    low = file_path.lower()
    if low.endswith(".csv"):
        return np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=1)
    return np.loadtxt(file_path)


# ── Core: find the best linear scaling window ─────────────────────────────────
 
def select_best_scaling_window(x_vals, y_vals, min_points=8):
    """
    Exhaustively scan all contiguous sub-windows of the ln M(r) vs ln(1/r) curve
    and select the one that best represents the linear scaling region.
 
    Horák et al. (2003) obr. 8.3: capacity d_c is the slope of the linear part of
    ln M(r) vs ln(1/r). This function automates that visual identification by scoring
    every candidate window as:
 
        score = R^2 + 0.05 * (window_length / total_length)
 
    R^2 dominates — the window must be genuinely linear (eq. 8.5: M(r) ~ r^(-d_c)).
    The small length bonus breaks ties in favour of longer windows, consistent with
    using as much of the true scaling region as possible.
 
    Returns a dict {start, end, slope, intercept, r2, stderr} or None if too short.
    """
    n = len(x_vals)
    if n < min_points:
        return None                  # Cannot define any meaningful linear region
 
    best       = None
    best_score = -np.inf
 
    for i in range(0, n - min_points + 1):      # All possible window start positions
        for j in range(i + min_points, n + 1):  # All possible window end positions
            xs  = x_vals[i:j]
            ys  = y_vals[i:j]
            fit = linregress(xs, ys)             # OLS: slope = d_c estimate for this window
            r2  = fit.rvalue**2                  # R^2: how well the window obeys eq. (8.5)
            length_bonus = (j - i) / n
            score = r2 + 0.05 * length_bonus
            if score > best_score:
                best_score = score
                best = {
                    "start"    : i,
                    "end"      : j,
                    "slope"    : fit.slope,
                    # slope of ln M(r) vs ln(1/r) = d_c per eq. (8.4):
                    # d_c = -lim_{r->0} ln M(r) / ln r = lim_{r->0} ln M(r) / ln(1/r)
                    "intercept": fit.intercept,
                    "r2"       : r2,
                    "stderr"   : fit.stderr if fit.stderr is not None else np.nan,
                    # OLS standard error of slope; 95% CI = 1.96 * stderr
                }
    return best
 
 
# ── Main processing function for a single file ────────────────────────────────
 
def run_2dc_single(file_path: str, output_dir: str, tau: int) -> None:
    stem      = os.path.splitext(os.path.basename(file_path))[0]
    series_id = (stem[:SERIES_COL_W] + " " * SERIES_COL_W)[:SERIES_COL_W].rstrip()
 
    rep = Reporter()
    rep.add(f"\nProcessing {stem}: {file_path} (tau={tau})")
 
    try:
        x = load_logreturns_column(file_path)
    except OSError:
        print(f"  SKIP (cannot load): {file_path}")
        return
 
    rep.add(f"Series length N = {len(x)}")
    rep.add(f"Embedding dimensions m = {m_values}")
    rep.add(f"Box sizes r in [{r_values.min():.4f}, {r_values.max():.4f}] "
            f"({len(r_values)} values, log-spaced)")
    rep.add("-" * 80)
    rep.add(f"{'m':>3} {'d_c':>10} {'+/-95CI':>10} {'R^2':>8} "
            f"{'window':>12} {'points':>8}  flags")
    rep.add("-" * 80)
 
    dc_results   = []
    r2_results   = []
    ci95_results = []
 
    # Two subplots replicating the structure of Horák et al. (2003):
    #   ax1 — ln M(r) vs ln(1/r) scaling law plot (obr. 8.3)
    #   ax2 — d_c vs m saturation plot (obr. 8.2 concept)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
 
    # ── Loop over embedding dimensions ────────────────────────────────────────
    for m in m_values:
 
        # Number of delay vectors obtainable from the series.
        # Horák et al. (2003): phase space is reconstructed via Takens embedding
        # with dimension m and delay tau; each vector covers a span of (m-1)*tau steps.
        L = len(x) - (m - 1) * tau
        if L <= 5:                               # Degenerate case — skip
            dc_results.append(np.nan)
            r2_results.append(np.nan)
            ci95_results.append(np.nan)
            continue
 
        # Takens delay-embedding: Y[i] = [x(i), x(i+tau), ..., x(i+(m-1)*tau)]
        # This reconstructs the attractor in m-dimensional phase space (obr. 8.2).
        Y = np.array([x[i : i + m * tau : tau] for i in range(L)])
 
        # Min-max normalise each dimension to [0, 1].
        # Horák et al. (2003): the phase space is partitioned into boxes of size r
        # ("množiny o velikosti r") — normalisation ensures r is dimensionless and
        # directly comparable across dimensions and symbols.
        Y = (Y - Y.min(0)) / (Y.max(0) - Y.min(0) + 1e-12)  # 1e-12 guards against zero range
 
        # ── Box counting: compute M(r) for each r ─────────────────────────────
        # Horák et al. (2003): M(r) is the minimum number of boxes of size r needed
        # to cover the attractor — "minimální počet množin o velikosti r" (p. [8.1]).
        M_vals = []
        for r in r_values:
            n_bins = int(np.ceil(1.0 / r))
            # Assign each embedded point to a box by flooring its coordinates
            idx          = np.clip(np.floor(Y / r).astype(int), 0, n_bins - 1)
            unique_boxes = len(np.unique(idx, axis=0))  # Count non-empty boxes = M(r)
            M_vals.append(unique_boxes)
 
        M_vals = np.array(M_vals)               # Shape (40,)
 
        # ── Saturation filter ─────────────────────────────────────────────────
        # Horák et al. (2003): the scaling M(r) ~ r^(-d_c) — eq. (8.5) — holds only
        # for "konečné, avšak malé hodnoty měřítka r" (finite but small r).
        # When r is so small that almost every point occupies its own box,
        # M(r) saturates at N — this is a finite-sample artefact, not true geometry.
        saturation_limit = 0.15 * L             # Heuristic: M(r) > 15% of N signals saturation
        valid_indices    = M_vals < saturation_limit
        if sum(valid_indices) < 5:              # Fallback: keep only coarsest boxes (large r)
            valid_indices      = np.ones_like(M_vals, dtype=bool)
            valid_indices[15:] = False          # Discard the 25 finest r values
 
        # Convert to log-log coordinates for the linear fit — eq. (8.4):
        # d_c = lim_{r->0} ln M(r) / ln(1/r)  →  slope of ln M(r) vs ln(1/r)
        # This is exactly the x- and y-axis of Horák et al. (2003) obr. 8.3.
        ln_1_r_valid = np.log(1.0 / r_values)[valid_indices]   # x-axis: ln(1/r)
        ln_M_valid   = np.log(M_vals)[valid_indices]            # y-axis: ln M(r)
 
        # Identify the best linear scaling region (obr. 8.3: "slope of linear part")
        selected = select_best_scaling_window(ln_1_r_valid, ln_M_valid, min_points=MIN_WINDOW_POINTS)
        if selected is None:
            dc_results.append(np.nan)
            r2_results.append(np.nan)
            ci95_results.append(np.nan)
            rep.add(f"{m:>3} {'nan':>10} {'nan':>10} {'nan':>8} "
                    f"{'-':>12} {'-':>8}  no valid scaling window")
            continue
 
        slope     = selected["slope"]           # d_c estimate — slope of ln M(r) vs ln(1/r), eq. (8.4)
        intercept = selected["intercept"]
        r2        = selected["r2"]
        stderr    = selected["stderr"]
        ci95      = 1.96 * stderr if np.isfinite(stderr) else np.nan  # 95% CI half-width
 
        dc_results.append(slope)
        r2_results.append(r2)
        ci95_results.append(ci95)
 
        quality_flags = []
        if r2 < MIN_R2_FOR_TRUST:
            # R^2 < 0.98: the selected window does not conform well to eq. (8.5);
            # slope cannot be trusted as a reliable d_c estimate.
            quality_flags.append("LOW_R2")
        saturation_ratio = np.mean(~valid_indices)
        if saturation_ratio > 0.40:
            # More than 40% of r values were discarded as saturated — the usable
            # scaling range is narrow and the estimate may be fragile.
            quality_flags.append("SATURATION_HIGH")
        flags_str = ",".join(quality_flags) if quality_flags else "-"
 
        rep.add(
            f"{m:>3} {slope:>10.4f} {ci95:>10.4f} {r2:>8.4f} "
            f"{selected['start']:>5}:{selected['end']:<5} "
            f"{sum(valid_indices):>3}/{len(r_values):<3}  {flags_str}"
        )
 
        # ── Left plot: ln M(r) vs ln(1/r) — replicates Horák et al. (2003) obr. 8.3
        line, = ax1.plot(
            np.log(1.0 / r_values), np.log(M_vals),
            marker="o", markersize=3, label=f"m={m}"
        )
        # Overlay the fitted line: slope = d_c, this is the visual counterpart of
        # reading d_c from the "slope of linear part" as shown in obr. 8.3.
        ax1.plot(
            ln_1_r_valid[selected["start"] : selected["end"]],
            intercept + slope * ln_1_r_valid[selected["start"] : selected["end"]],
            color=line.get_color(),
            linestyle="--",
            linewidth=2,
            alpha=0.8,
        )
 
    # ── Left plot formatting ───────────────────────────────────────────────────
    ax1.set_xlabel("ln(1/r)")           # Horák et al. (2003) obr. 8.3 x-axis
    ax1.set_ylabel("ln M(r)")           # Horák et al. (2003) obr. 8.3 y-axis
    ax1.set_title(f"{stem} scaling law — d_c (tau={tau})")
    ax1.legend()
    ax1.grid(True)
 
    # ── Right plot: d_c vs m saturation ───────────────────────────────────────
    ax2.plot(m_values, dc_results, "ko-", label="Estimated d_c")
    ax2.plot(m_values, m_values,   "r--", label="d_c = m (stochastic process)")
    # Red dashed line (d_c = m): Horák et al. (2003) obr. 8.2 — for a stochastic
    # process there is no finite attractor, so d_c grows without bound with m.
    # Saturation of d_c below this line is evidence of a finite-dimensional attractor.
    ax2.set_xlabel("Embedding dimension m")
    ax2.set_ylabel("Capacity dimension d_c")
    ax2.set_title(f"{stem} d_c saturation (tau={tau})")
 
    # Best m: smallest 95% CI among all m that passed the R^2 threshold,
    # i.e. the most precise estimate where eq. (8.5) is actually satisfied.
    reliable_idx = [
        i
        for i, (r2, dc, ci95) in enumerate(zip(r2_results, dc_results, ci95_results))
        if np.isfinite(r2) and np.isfinite(dc) and np.isfinite(ci95) and r2 >= MIN_R2_FOR_TRUST
    ]
    if reliable_idx:
        best_i  = min(reliable_idx, key=lambda i: (ci95_results[i], -r2_results[i]))
        best_m  = m_values[best_i]
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
        # Horák et al. (2003): if no linear scaling region can be identified,
        # the capacity dimension cannot be reliably estimated — this is a known
        # limitation for noisy or stochastic financial time series.
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
    plt.close(fig)                       # Free memory — important in a 7-symbol batch loop
 
    # ── Finalise text report ───────────────────────────────────────────────────
    rep.add("-" * 80)
    if reliable_idx:
        rep.add(
            f"Best m: m={best_m}, d_c={best_dc:.4f} +/- {best_ci:.4f} (95% CI), "
            f"R^2={r2_results[best_i]:.4f}"
        )
        note        = f"best m={best_m} (d_c={best_dc:.3f}+/-{best_ci:.3f})"
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
 
    rep.add(f"Saved plot: {out_plot}")
    out_txt      = rep.write(output_dir, f"{stem}_2dc_tau{tau}_results.txt")
    print(f"Saved text report: {out_txt}")
    summary_path = append_summary_row(output_dir, SUMMARY_FILE, SUMMARY_HEADER, summary_row)
    print(f"Appended row to summary: {summary_path}")
 
 
# ── Entry point ────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    cfg = load_config()
    dd  = get_data_dir(cfg)
    od  = ensure_dir(os.path.join(get_results_dir(cfg), "2dc"))
 
    try:
        os.remove(os.path.join(od, SUMMARY_FILE))  # Remove stale summary from previous run
    except OSError:
        pass
 
    for fn in files:
        fp = prefer_liquidity_cut(os.path.join(dd, fn))
        if not os.path.exists(fp):
            print(f"File not found, skipping: {fp}")
            continue
        run_2dc_single(fp, od, tau_for_path(fp, cfg))
 