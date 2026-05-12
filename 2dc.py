import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

from config_loader import load_config, get_data_dir, get_results_dir, ensure_dir, prefer_liquidity_cut
from report_helper import Reporter, append_summary_row

SUMMARY_FILE = "_2dc_summary.txt"
SERIES_COL_W = 46
SUMMARY_HEADER = (
    f"{'series_id':<{SERIES_COL_W}} {'tau':>4} {'best_m':>6} {'d_c':>10} {'+/-95CI':>10} "
    f"{'R^2':>8} {'note':<40}"
)

files = [
    "BTCUSD_BITSTAMP_1h_complete_logreturns.csv",
    "ETHUSD_BITSTAMP_1h_complete_logreturns.csv",
    "LTCUSD_BITSTAMP_1h_complete_logreturns.csv",
    "XRPUSD_BITSTAMP_1h_complete_logreturns.csv",
    "LINKUSD_BITSTAMP_1h_complete_logreturns.csv",
    "DOGEUSD_BITSTAMP_1h_complete_logreturns.csv",
    "ADAUSD_BITSTAMP_1h_complete_logreturns.csv",
]

TAU_BY_SYMBOL = {
    "BTCUSD": 5,
    "ETHUSD": 5,
    "LTCUSD": 2,
    "XRPUSD": 2,
    "LINKUSD": 4,
    "DOGEUSD": 6,
    "ADAUSD": 2,
}
m_values = [2, 3, 4, 5, 10]
r_values = np.logspace(np.log10(0.02), np.log10(0.5), 40)
MIN_WINDOW_POINTS = 8
MIN_R2_FOR_TRUST = 0.98


def tau_for_path(file_path: str) -> int:
    sym = os.path.basename(file_path).split("_")[0]
    return TAU_BY_SYMBOL.get(sym, 3)


def load_logreturns_column(file_path: str) -> np.ndarray:
    """Cut log-return CSV (datetime, log_return) or single-column .dat."""
    low = file_path.lower()
    if low.endswith(".csv"):
        return np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=1)
    return np.loadtxt(file_path)


def select_best_scaling_window(x_vals, y_vals, min_points=8):
    """
    Select a linear scaling region by scanning all contiguous windows.
    Score prioritizes high R^2 and longer windows.
    """
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
            length_bonus = (j - i) / n
            score = r2 + 0.05 * length_bonus
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


def run_2dc_single(file_path: str, output_dir: str, tau: int) -> None:
    stem = os.path.splitext(os.path.basename(file_path))[0]
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

    dc_results = []
    r2_results = []
    ci95_results = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for m in m_values:
        L = len(x) - (m - 1) * tau
        if L <= 5:
            dc_results.append(np.nan)
            r2_results.append(np.nan)
            ci95_results.append(np.nan)
            continue

        Y = np.array([x[i : i + m * tau : tau] for i in range(L)])
        Y = (Y - Y.min(0)) / (Y.max(0) - Y.min(0) + 1e-12)

        M_vals = []
        for r in r_values:
            n_bins = int(np.ceil(1.0 / r))
            idx = np.clip(np.floor(Y / r).astype(int), 0, n_bins - 1)
            unique_boxes = len(np.unique(idx, axis=0))
            M_vals.append(unique_boxes)

        M_vals = np.array(M_vals)
        ln_1_r = np.log(1.0 / r_values)
        ln_M = np.log(M_vals)

        saturation_limit = 0.15 * L
        valid_indices = M_vals < saturation_limit
        if sum(valid_indices) < 5:
            valid_indices = np.ones_like(M_vals, dtype=bool)
            valid_indices[15:] = False

        ln_1_r_valid = ln_1_r[valid_indices]
        ln_M_valid = ln_M[valid_indices]
        selected = select_best_scaling_window(ln_1_r_valid, ln_M_valid, min_points=MIN_WINDOW_POINTS)
        if selected is None:
            dc_results.append(np.nan)
            r2_results.append(np.nan)
            ci95_results.append(np.nan)
            rep.add(f"{m:>3} {'nan':>10} {'nan':>10} {'nan':>8} "
                    f"{'-':>12} {'-':>8}  no valid scaling window")
            continue

        slope = selected["slope"]
        intercept = selected["intercept"]
        r2 = selected["r2"]
        stderr = selected["stderr"]
        ci95 = 1.96 * stderr if np.isfinite(stderr) else np.nan
        dc_results.append(slope)
        r2_results.append(r2)
        ci95_results.append(ci95)

        quality_flags = []
        if r2 < MIN_R2_FOR_TRUST:
            quality_flags.append("LOW_R2")
        saturation_ratio = np.mean(~valid_indices)
        if saturation_ratio > 0.40:
            quality_flags.append("SATURATION_HIGH")
        flags_str = ",".join(quality_flags) if quality_flags else "-"

        rep.add(
            f"{m:>3} {slope:>10.4f} {ci95:>10.4f} {r2:>8.4f} "
            f"{selected['start']:>5}:{selected['end']:<5} "
            f"{sum(valid_indices):>3}/{len(r_values):<3}  {flags_str}"
        )

        line, = ax1.plot(ln_1_r, ln_M, marker="o", markersize=3, label=f"m={m}")
        ax1.plot(
            ln_1_r_valid[selected["start"] : selected["end"]],
            intercept + slope * ln_1_r_valid[selected["start"] : selected["end"]],
            color=line.get_color(),
            linestyle="--",
            linewidth=2,
            alpha=0.8,
        )

    ax1.set_xlabel("ln(1/r)")
    ax1.set_ylabel("ln M(r)")
    ax1.set_title(f"{stem} scaling law — d_c (tau={tau})")
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
        if np.isfinite(r2) and np.isfinite(dc) and np.isfinite(ci95) and r2 >= MIN_R2_FOR_TRUST
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
            fontsize=9,
            color="blue",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="blue", alpha=0.8),
        )
    else:
        ax2.text(
            0.03,
            0.95,
            "No reliable best m (R^2 below threshold)",
            transform=ax2.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="darkred",
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
        run_2dc_single(fp, od, tau_for_path(fp))
