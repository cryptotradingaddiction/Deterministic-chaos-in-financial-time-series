#!/usr/bin/env python3
"""
Surrogate hypothesis testing for dynamical invariants (D2, K2, LLE, RQA).

Workflow:
    1. Generate ONE surrogate by random permutation (randperm) of the observed series.
       No block structure — individual elements shuffled.
    2. Generate two reference series of the same length using mu/sigma estimated from
       the original log-returns: Normal N(mu, sigma) and scaled t(df=3.5).
    3. Run all requested metrics on the original series. For null/reference series
       (surrogate, normal, t), run only metrics with a defined variance test (D2/K2).
    4. Where an invariant has multiple methodologically defined values, test
       equality of invariant SD/variance between original and one shuffled surrogate.
       D2/K2 use all values from the #dim=3 block; LLE and RQA use one value
       computed from the full time series.
    5. Report invariant values and series mu/sigma for all four series.
"""
import argparse
import glob
import logging
import os
import shutil
import subprocess
import warnings

import numpy as np
from scipy import stats as scipy_stats
from pyrqa.analysis_type import Classic
from pyrqa.computation import RQAComputation
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius
from pyrqa.settings import Settings
from pyrqa.time_series import TimeSeries

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from surrogate_sampling import load_series_1d

logger = logging.getLogger(__name__)

# Default significance level
DEFAULT_ALPHA = 0.01
# Degrees of freedom for t-distribution reference series
T_DOF = 3.5
# Must match -M1,3 in correlation_dimension.bat / correlation_entropy.bat
M_D2 = 3
M_LYAP = 3
DEFAULT_RQA_RADIUS = 0.005
RQA_EMBEDDING_DIM = 3

RQA_KEYS = ("RR", "DET", "LAM", "MAXLINE", "ENTR", "TT")
ALL_METRICS = ("D2", "K2", "LLE", *RQA_KEYS)
NULL_SERIES_METRICS = {"D2", "K2"}

PRED_EPSILON = 1e-5
PRED_TOLERANCE = 1e-2


# ---------------------------------------------------------------------------
# Surrogate and reference series generation
# ---------------------------------------------------------------------------

def generate_single_surrogate(data: np.ndarray) -> np.ndarray:
    """Random permutation of original series (randperm, no blocking)."""
    return np.random.permutation(data)


def generate_normal_series(mu: float, sigma: float, n: int) -> np.ndarray:
    """N(mu, sigma) series of length n."""
    return np.random.normal(mu, sigma, n)


def generate_t_series(mu: float, sigma: float, n: int, dof: float = T_DOF) -> np.ndarray:
    """t(dof) series scaled to (mu, sigma).

    t(dof) has zero mean and variance dof/(dof-2) for dof>2, so
    scale raw draws by sigma / sqrt(dof/(dof-2)) and shift by mu.
    """
    t_raw = np.random.standard_t(dof, n)
    t_sd = np.sqrt(dof / (dof - 2.0))
    return mu + sigma * (t_raw / t_sd)


# ---------------------------------------------------------------------------
# Hypothesis test: equality of invariant SD/variance
# ---------------------------------------------------------------------------

def invariant_sd_f_test(
    sd_orig: float,
    n_orig: int,
    sd_surr: float,
    n_surr: int,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float, str]:
    """Two-sided F-test of H0: Var(T_orig) == Var(T_surr)."""
    if n_orig < 2 or n_surr < 2:
        return np.nan, np.nan, "insufficient n"
    if not (np.isfinite(sd_orig) and np.isfinite(sd_surr) and sd_orig > 0 and sd_surr > 0):
        return np.nan, np.nan, "no sd"

    f_stat = float((sd_orig * sd_orig) / (sd_surr * sd_surr))
    df1, df2 = n_orig - 1, n_surr - 1
    lower_tail = float(scipy_stats.f.cdf(f_stat, df1, df2))
    upper_tail = float(scipy_stats.f.sf(f_stat, df1, df2))
    p = min(1.0, 2.0 * min(lower_tail, upper_tail))
    decision = "reject H0" if (np.isfinite(p) and p < alpha) else "fail to reject H0"
    return f_stat, p, decision


# ---------------------------------------------------------------------------
# TISEAN / RQA wrappers (unchanged from original)
# ---------------------------------------------------------------------------

def load_data(filename):
    return load_series_1d(filename)


def resolve_tool(tool_name):
    from_env = os.environ.get("TISEAN_BIN")
    if from_env:
        candidate = os.path.join(from_env, f"{tool_name}.exe")
        if os.path.exists(candidate):
            return candidate
    local_bin = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Tisean_3.0.0", "bin", f"{tool_name}.exe",
    )
    if os.path.exists(local_bin):
        return local_bin
    which = shutil.which(tool_name)
    if which:
        return which
    which_exe = shutil.which(f"{tool_name}.exe")
    if which_exe:
        return which_exe
    raise FileNotFoundError(f"TISEAN executable for '{tool_name}' was not found.")


def run_d2(data_file, delay, theiler, output_prefix):
    cmd = [resolve_tool("d2"), f"-d{delay}", f"-M1,{M_D2}", f"-t{theiler}",
           "-o", output_prefix, data_file]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_prefix + ".d2", output_prefix + ".h2"


def run_lyap_k(data_file, delay, output_file):
    cmd = [resolve_tool("lyap_k"), f"-d{delay}", f"-m{M_LYAP}", f"-M{M_LYAP}",
           "-n", "500", "-o", output_file, data_file]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def extract_dim_values(path, dim=M_D2, value_column_idx=1):
    """Read values from a specific TISEAN #dim block."""
    values = []
    current_dim = None
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#dim="):
                    try:
                        current_dim = int(stripped.split("=")[1].strip())
                    except (ValueError, IndexError):
                        current_dim = None
                    continue
                if stripped.startswith("#") or stripped.startswith("!"):
                    continue
                if current_dim != dim:
                    continue
                parts = stripped.split()
                if len(parts) <= value_column_idx:
                    continue
                try:
                    values.append(float(parts[value_column_idx]))
                except ValueError:
                    continue
    except Exception:
        return np.array([], dtype=float)
    return np.array(values, dtype=float)


def _mean_sd_n(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, 0
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if values.size > 1 else np.nan
    return mean, sd, int(values.size)


def extract_d2_mean_std(d2_file):
    """Mean/SD of local D2 slopes from the full #dim=3 block."""
    return _mean_sd_n(extract_dim_values(d2_file, dim=M_D2, value_column_idx=1))


def extract_k2_mean_std(h2_file):
    """Mean/SD of K2 ordinates from the full #dim=3 block."""
    return _mean_sd_n(extract_dim_values(h2_file, dim=M_D2, value_column_idx=1))


def _slope_fit(x, y, lo=2, hi=10):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    lo = min(lo, max(0, n - 3))
    hi = min(hi, n)
    if hi - lo < 2:
        return np.nan
    try:
        slope, _ = np.polyfit(x[lo:hi], y[lo:hi], 1)
        return float(slope)
    except Exception:
        return np.nan


def extract_lle_mean_std(lyap_file):
    """Largest Lyapunov estimate: slope of S(t) for the first m=3 epsilon block."""
    try:
        with open(lyap_file, "r", encoding="utf-8", errors="ignore") as handle:
            lines = handle.readlines()
        first_block = None
        current_block = []
        current_dim = None
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#epsilon"):
                if current_dim == M_LYAP and current_block:
                    first_block = np.array(current_block, dtype=float)
                    break
                current_dim = None
                current_block = []
                tokens = stripped.replace("=", " ").split()
                for idx, token in enumerate(tokens):
                    if token.lower() == "dim" and idx + 1 < len(tokens):
                        try:
                            current_dim = int(float(tokens[idx + 1]))
                        except ValueError:
                            current_dim = None
                        break
                continue
            if stripped.startswith("#") or stripped == "":
                continue
            if current_dim != M_LYAP:
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                current_block.append([float(parts[0]), float(parts[1])])
        if first_block is None and current_dim == M_LYAP and current_block:
            first_block = np.array(current_block, dtype=float)
        if first_block is None or first_block.size == 0 or first_block.ndim < 2:
            return np.nan, np.nan, 0
        slope = _slope_fit(first_block[:, 0], first_block[:, 1], lo=2, hi=10)
        if not np.isfinite(slope):
            return np.nan, np.nan, 0
        return slope, np.nan, 1
    except Exception:
        return np.nan, np.nan, 0


def compute_pyrqa_metrics(series, delay, theiler, radius=None):
    r_eff = float(DEFAULT_RQA_RADIUS if radius is None else radius)
    try:
        ts = TimeSeries(series, embedding_dimension=RQA_EMBEDDING_DIM, time_delay=delay)
        settings = Settings(ts, analysis_type=Classic,
                            neighbourhood=FixedRadius(r_eff),
                            similarity_measure=EuclideanMetric,
                            theiler_corrector=theiler)
        computation = RQAComputation.create(settings, verbose=False)
        result = computation.run()
        result.min_diagonal_line_length = 2
        result.min_vertical_line_length = 2
        return {
            "RR": float(result.recurrence_rate),
            "DET": float(result.determinism),
            "LAM": float(result.laminarity),
            "MAXLINE": float(result.longest_diagonal_line),
            "ENTR": float(result.entropy_diagonal_lines),
            "TT": float(result.trapping_time),
        }
    except Exception:
        return {k: np.nan for k in RQA_KEYS}


def compute_invariants(series_array, output_dir, label, delay, theiler,
                       metric_names, rqa_radius=None, series_std_fallback=np.nan):
    """Compute invariants for an in-memory series. Returns (mean_dict, sd_dict, n_dict).

    SD/N sources:
      D2/K2 — all second-column values in #dim=3 block
      LLE   — one slope from the linear part of S(t) for m=3
      RQA   — one metric value computed on the full time series
    """
    metric_names = list(metric_names)
    need_d2 = "D2" in metric_names
    need_k2 = "K2" in metric_names
    need_lle = "LLE" in metric_names
    need_rqa = any(k in metric_names for k in RQA_KEYS)

    prefix = os.path.join(output_dir, label)
    data_file = prefix + ".dat"
    np.savetxt(data_file, series_array)

    out = {k: np.nan for k in metric_names}
    out_std = {k: np.nan for k in metric_names}
    out_n = {k: 0 for k in metric_names}

    try:
        d2_file = h2_file = None
        if need_d2 or need_k2:
            d2_file, h2_file = run_d2(data_file, delay, theiler, prefix)
        if need_d2 and d2_file:
            mu, sg, nn = extract_d2_mean_std(d2_file)
            out["D2"], out_std["D2"] = mu, sg
            out_n["D2"] = nn
        if need_k2 and h2_file:
            mu, sg, nn = extract_k2_mean_std(h2_file)
            out["K2"], out_std["K2"] = mu, sg
            out_n["K2"] = nn
        if need_lle:
            lyap_file = prefix + "_lyap.txt"
            run_lyap_k(data_file, delay, lyap_file)
            mu, sg, nn = extract_lle_mean_std(lyap_file)
            out["LLE"], out_std["LLE"] = mu, sg
            out_n["LLE"] = nn
        if need_rqa:
            rqa_values = compute_pyrqa_metrics(series_array, delay, theiler, rqa_radius)
            for k in RQA_KEYS:
                if k in out:
                    value = float(rqa_values.get(k, np.nan))
                    out[k] = value
                    out_std[k] = np.nan
                    out_n[k] = 1 if np.isfinite(value) else 0
    except subprocess.CalledProcessError:
        logger.exception("TISEAN failed for label=%s", label)
    except Exception:
        logger.exception("Error computing invariants for label=%s", label)
    finally:
        for tmp in glob.glob(prefix + "*"):
            try:
                os.remove(tmp)
            except OSError:
                pass

    return out, out_std, out_n


def predictability_time(lle, eps=PRED_EPSILON, tol=PRED_TOLERANCE):
    if lle is None or not np.isfinite(lle) or lle <= 0:
        return np.nan
    return float((1.0 / lle) * np.log(tol / eps))


def parse_metrics_list(raw_metrics):
    tokens = [t.strip().upper() for t in str(raw_metrics).split(",") if t.strip()]
    if not tokens:
        raise ValueError("metrics list is empty")
    invalid = [t for t in tokens if t not in ALL_METRICS]
    if invalid:
        raise ValueError(f"unknown metric(s): {', '.join(invalid)}")
    return list(dict.fromkeys(tokens))


def metric_names_for_scope(_=None):
    return ["D2", "K2", "LLE", *RQA_KEYS]


def is_test_mode(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--delay", type=int, required=True)
    parser.add_argument("--theiler", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--test_mode", default="false")
    parser.add_argument("--metrics", default="full")
    parser.add_argument("--metrics_list", default="")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--rqa_radius", type=float, default=DEFAULT_RQA_RADIUS)
    args = parser.parse_args()

    if not (0.0 < args.alpha < 1.0):
        raise SystemExit("hypothesis.py: --alpha must be strictly between 0 and 1.")
    if args.rqa_radius <= 0.0:
        raise SystemExit("hypothesis.py: --rqa_radius must be positive.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    metric_names = (
        parse_metrics_list(args.metrics_list)
        if args.metrics_list.strip()
        else metric_names_for_scope(args.metrics)
    )

    print(f"  -> Surrogate hypothesis test (single surrogate): {args.base}")
    print(f"     tau={args.delay}, W={args.theiler}, alpha={args.alpha}, "
          f"metrics={','.join(metric_names)}")

    # ------------------------------------------------------------------
    # 1. Load original log-returns; estimate mu and sigma of the series
    # ------------------------------------------------------------------
    orig_data = load_data(args.input)
    n = len(orig_data)
    mu_r = float(np.mean(orig_data))
    sigma_r = float(np.std(orig_data, ddof=1))
    print(f"  -> Original series: n={n}, mu={mu_r:.6f}, sigma={sigma_r:.6f}")

    # ------------------------------------------------------------------
    # 2. Generate surrogate (randperm) and two reference series
    # ------------------------------------------------------------------
    surr_data = generate_single_surrogate(orig_data)
    norm_data = generate_normal_series(mu_r, sigma_r, n)
    t_data = generate_t_series(mu_r, sigma_r, n, dof=T_DOF)

    series_specs = [
        ("orig",   orig_data),
        ("surr",   surr_data),
        ("normal", norm_data),
        (f"t{T_DOF}", t_data),
    ]

    # ------------------------------------------------------------------
    # 3. Compute invariants. LLE/RQA are original-only because they are scalar here.
    # ------------------------------------------------------------------
    tmp_dir = os.path.join(args.output_dir, "tmp_hyp")
    os.makedirs(tmp_dir, exist_ok=True)

    results = {}   # label -> {metric: mean invariant value}
    stds    = {}   # label -> {metric: SD of invariant values}
    counts  = {}   # label -> {metric: n values used for SD}

    for label, series in series_specs:
        metrics_for_label = (
            metric_names
            if label == "orig"
            else [m for m in metric_names if m in NULL_SERIES_METRICS]
        )

        if metrics_for_label:
            print(f"  -> Computing invariants for series: {label} ({','.join(metrics_for_label)}) ...")
            inv_part, inv_std_part, inv_n_part = compute_invariants(
                series, tmp_dir, f"{args.base}_{label}",
                args.delay, args.theiler, metrics_for_label, args.rqa_radius,
                series_std_fallback=sigma_r,
            )
        else:
            print(f"  -> Skipping invariant recomputation for series: {label} (not needed for selected metrics).")
            inv_part, inv_std_part, inv_n_part = {}, {}, {}

        results[label] = {k: inv_part.get(k, np.nan) for k in metric_names}
        stds[label]    = {k: inv_std_part.get(k, np.nan) for k in metric_names}
        counts[label]  = {k: inv_n_part.get(k, 0) for k in metric_names}

    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    # ------------------------------------------------------------------
    # 4. F-test: equality of invariant SD/variance, original vs shuffled surrogate
    # ------------------------------------------------------------------
    f_stats   = {}
    p_values  = {}
    decisions = {}

    for metric in metric_names:
        f_stat, p, dec = invariant_sd_f_test(
            stds["orig"].get(metric, np.nan),
            counts["orig"].get(metric, 0),
            stds["surr"].get(metric, np.nan),
            counts["surr"].get(metric, 0),
            args.alpha,
        )
        f_stats[metric] = f_stat
        p_values[metric]  = p
        decisions[metric] = dec

    # ------------------------------------------------------------------
    # 5. Write output table
    # ------------------------------------------------------------------
    labels_ordered = [lbl for lbl, _ in series_specs]

    summary_file = os.path.join(args.output_dir, f"{args.base}_surrogate_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as fh:
        fh.write(f"Single-surrogate hypothesis test ({args.base})\n")
        fh.write(
            f"Parameters: tau={args.delay}, W={args.theiler}, alpha={args.alpha}, "
            f"metrics={','.join(metric_names)}, T_dof={T_DOF}\n"
        )
        fh.write(f"Original series: n={n}, mu_r={mu_r:.6f}, sigma_r={sigma_r:.6f}\n")
        fh.write(
            "Surrogate: randperm (full random permutation, no blocking).\n"
            "Normal:    N(mu_r, sigma_r) of same length.\n"
            f"t-series:  t(dof={T_DOF}) scaled to (mu_r, sigma_r) of same length.\n"
            "Null/reference invariants are recomputed only for D2/K2, where the variance test is defined.\n"
        )
        fh.write(
            "Test: two-sided F-test of invariant SD/variance, original vs shuffled surrogate.\n"
            "  H0: Var_orig(invariant values) = Var_surr(invariant values), alpha=0.01 by default.\n"
            "  Normal and t-series are reported as reference benchmarks, not as the main p-value pair.\n"
            "Invariant value sources:\n"
            "  D2/K2 — all second-column values from #dim=3 block\n"
            "  LLE   — one slope of S(t) from the first m=3 lyap_k epsilon block; SD/F-test is unavailable for n=1\n"
            "  RQA   — one value computed on the full time series; SD/F-test is unavailable for n=1\n\n"
        )

        # --- Series statistics header ---
        fh.write("Series statistics (mean and SD of each series):\n")
        fh.write(f"  {'series':<10}  {'mean':>12}  {'sd':>12}\n")
        fh.write(f"  {'-'*38}\n")
        for label, series in series_specs:
            fh.write(f"  {label:<10}  {np.mean(series):>12.6f}  {np.std(series, ddof=1):>12.6f}\n")
        fh.write("\n")

        # --- Invariant values table ---
        col_w = 12
        hdr_parts = [f"{'Invariant':<12}", f"{'orig_sd':>{col_w}}", f"{'surr_sd':>{col_w}}", f"{'n':>{6}}"]
        for lbl in labels_ordered:
            hdr_parts.append(f"{lbl:>{col_w}}")
        hdr_parts += [f"{'F':>{col_w}}", f"{'p-value':>{col_w}}", f"{'decision':<20}"]
        hdr = "  " + "  ".join(hdr_parts)
        fh.write(hdr + "\n")
        fh.write("  " + "-" * (len(hdr) - 2) + "\n")

        def _f(v):
            return f"{v:>{col_w}.4f}" if np.isfinite(v) else f"{'nan':>{col_w}}"

        print(f"\n  --- Invariant table ---")
        print(f"  {hdr.strip()}")
        print(f"  {'-' * (len(hdr) - 2)}")

        for metric in metric_names:
            so = stds["orig"].get(metric, np.nan)
            ss = stds["surr"].get(metric, np.nan)
            nn = counts["orig"].get(metric, 0)
            row_parts = [f"{metric:<12}", _f(so), _f(ss), f"{nn:>6}"]
            for lbl in labels_ordered:
                row_parts.append(_f(results[lbl][metric]))
            row_parts.append(_f(f_stats[metric]))
            row_parts.append(_f(p_values[metric]))
            row_parts.append(f"  {decisions[metric]:<20}")
            row = "  " + "  ".join(row_parts)
            fh.write(row + "\n")
            print(f"  {row.strip()}")

        # --- Conclusion ---
        fh.write(f"\nConclusion (alpha={args.alpha}):\n")
        print(f"\n  Conclusion (alpha={args.alpha}):")
        for metric in metric_names:
            line = f"  {metric:<10}: {decisions[metric]}"
            fh.write(line + "\n")
            print(line)

        # --- Predictability time (LLE) ---
        if "LLE" in metric_names:
            lle_orig = results["orig"].get("LLE", np.nan)
            T_orig = predictability_time(lle_orig)
            fh.write("\nPredictability time T (hours)\n")
            fh.write(f"  T = (1/lambda) * log(L/eps), eps={PRED_EPSILON:g}, L={PRED_TOLERANCE:g}\n")
            fh.write(f"  Original LLE : {lle_orig:.6f}\n")
            if np.isfinite(T_orig):
                fh.write(f"  Original T   : {T_orig:.2f} h ({T_orig/24:.2f} days)\n")
            else:
                fh.write("  Original T   : undefined (LLE <= 0)\n")

    print(f"\n  -> Summary written to {summary_file}")


if __name__ == "__main__":
    main()