#!/usr/bin/env python3
import argparse
import concurrent.futures
import glob
import os
import shutil
import subprocess
import warnings

import numpy as np
from pyrqa.analysis_type import Classic
from pyrqa.computation import RQAComputation
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius
from pyrqa.settings import Settings
from pyrqa.time_series import TimeSeries

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from surrogate_sampling import generate_permuted_samples, load_series_1d

FULL_B = 100
TEST_B = 100
M_D2 = 30
M_LYAP = 100
R_RECURR = 0.01
RQA_EMBEDDING_DIM = 3

RQA_KEYS = ("RR", "DET", "LAM", "MAXLINE", "ENTR", "TT")

METRICS_SCOPE_FULL = "full"
METRICS_SCOPE_D2_K2_LLE_RQA = "d2_k2_lle_rqa"
ALL_METRICS = ("D2", "K2", "LLE", *RQA_KEYS)


def metric_names_for_scope(_metrics_scope=None):
    """Default metric set for full-scope runs (argument kept for call-site compatibility)."""
    return ["D2", "K2", "LLE", *RQA_KEYS]


def parse_metrics_list(raw_metrics):
    tokens = [t.strip().upper() for t in str(raw_metrics).split(",") if t.strip()]
    if not tokens:
        raise ValueError("metrics list is empty")
    invalid = [t for t in tokens if t not in ALL_METRICS]
    if invalid:
        raise ValueError(f"unknown metric(s): {', '.join(invalid)}")
    return list(dict.fromkeys(tokens))


# Predictability time T = (1/lambda) * log(L / EPS) when lambda > 0.
# Same constants as predictability.py.
PRED_EPSILON = 1e-5
PRED_TOLERANCE = 1e-2


def is_test_mode(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_data(filename):
    return load_series_1d(filename)


def resolve_tool(tool_name):
    from_env = os.environ.get("TISEAN_BIN")
    if from_env:
        candidate = os.path.join(from_env, f"{tool_name}.exe")
        if os.path.exists(candidate):
            return candidate
    local_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tisean_3.0.0", "bin", f"{tool_name}.exe")
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
    cmd = [
        resolve_tool("d2"),
        f"-d{delay}",
        f"-M1,{M_D2}",
        f"-t{theiler}",
        "-o",
        output_prefix,
        data_file,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_prefix + ".d2", output_prefix + ".h2"


def run_c2t(c2_file, output_file):
    """TISEAN c2t is FORTRAN with character*72 file path truncation, so
    invoke it from c2_file's directory using short relative names."""
    work_dir = os.path.dirname(os.path.abspath(c2_file)) or "."
    rel_in = os.path.basename(c2_file)
    rel_out = os.path.relpath(os.path.abspath(output_file), work_dir)
    cmd = [resolve_tool("c2t"), "-o", rel_out, rel_in]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=work_dir)


def run_lyap_k(data_file, delay, output_file):
    cmd = [
        resolve_tool("lyap_k"),
        f"-d{delay}",
        "-m1",
        f"-M{M_LYAP}",
        "-r0.0005",
        "-R0.05",
        "-n",
        "500",
        "-o",
        output_file,
        data_file,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def run_c1(data_file, delay, theiler, output_prefix):
    """FORTRAN tool: cwd into the output prefix's parent and use a short -o name."""
    work_dir = os.path.dirname(os.path.abspath(output_prefix)) or "."
    rel_prefix = os.path.basename(output_prefix)
    rel_data = os.path.relpath(os.path.abspath(data_file), work_dir)
    cmd = [
        resolve_tool("c1"),
        f"-d{delay}",
        "-m2",
        "-M30",
        f"-t{theiler}",
        "-n500",
        "-o",
        rel_prefix,
        rel_data,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=work_dir)


def run_c2d(c1_source_file, output_file):
    """FORTRAN tool: cwd into source's directory, use short relative names."""
    work_dir = os.path.dirname(os.path.abspath(c1_source_file)) or "."
    rel_in = os.path.basename(c1_source_file)
    cmd = [resolve_tool("c2d"), "-a2", rel_in]
    with open(output_file, "w", encoding="utf-8") as out:
        subprocess.run(cmd, check=True, stdout=out, stderr=subprocess.PIPE, text=True, cwd=work_dir)


def run_boxcount(data_file, delay, output_file):
    primary = [
        resolve_tool("boxcount"),
        f"-d{delay}",
        "-M30",
        "-Q1.0",
        "-#20",
        "-o",
        output_file,
        data_file,
    ]
    try:
        subprocess.run(primary, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        fallback = [
            resolve_tool("boxcount"),
            f"-d{delay}",
            "-M1,30",
            "-Q1.0",
            "-#20",
            "-o",
            output_file,
            data_file,
        ]
        subprocess.run(fallback, check=True, capture_output=True, text=True)


def extract_d2(d2_file):
    try:
        data = np.loadtxt(d2_file)
        if data.size == 0 or data.ndim < 2:
            return np.nan
        slopes = data[:, 1]
        n = len(slopes)
        low, high = int(0.25 * n), int(0.75 * n)
        if low >= high:
            return np.nan
        return float(np.mean(slopes[low:high]))
    except Exception:
        return np.nan


def extract_k2(h2_file):
    try:
        data = np.loadtxt(h2_file)
        if data.size == 0 or data.ndim < 2:
            return np.nan
        k2_vals = data[:, 1]
        n = len(k2_vals)
        low, high = int(0.25 * n), int(0.75 * n)
        if low >= high:
            return np.nan
        return float(np.mean(k2_vals[low:high]))
    except Exception:
        return np.nan


def extract_lle(lyap_file):
    try:
        with open(lyap_file, "r", encoding="utf-8", errors="ignore") as handle:
            lines = handle.readlines()
        blocks = []
        current_block = []
        for line in lines:
            if line.startswith("#") or line.strip() == "":
                continue
            if "epsilon" in line:
                if current_block:
                    blocks.append(np.array(current_block))
                    current_block = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    current_block.append([float(parts[0]), float(parts[1])])
        if current_block:
            blocks.append(np.array(current_block))
        if not blocks:
            return np.nan
        data = blocks[-1]
        if data.size == 0 or data.ndim < 2:
            return np.nan
        t, s = data[:, 0], data[:, 1]
        if len(t) < 2:
            return np.nan
        n_fit = max(5, int(0.2 * len(t)))
        slope, _ = np.polyfit(t[:n_fit], s[:n_fit], 1)
        return float(slope)
    except Exception:
        return np.nan


def extract_middle_window_from_text(file_path, value_column_idx=1):
    values = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) > value_column_idx:
                    try:
                        values.append(float(parts[value_column_idx]))
                    except ValueError:
                        continue
        if not values:
            return np.nan
        arr = np.array(values, dtype=float)
        n = len(arr)
        low, high = int(0.25 * n), int(0.75 * n)
        if low >= high:
            return np.nan
        return float(np.mean(arr[low:high]))
    except Exception:
        return np.nan


def extract_d1(d1_file):
    return extract_middle_window_from_text(d1_file, value_column_idx=1)


def extract_k1(box_file):
    return extract_middle_window_from_text(box_file, value_column_idx=1)


def compute_pyrqa_metrics(series, delay, theiler):
    try:
        ts = TimeSeries(series, embedding_dimension=RQA_EMBEDDING_DIM, time_delay=delay)
        settings = Settings(
            ts,
            analysis_type=Classic,
            neighbourhood=FixedRadius(R_RECURR),
            similarity_measure=EuclideanMetric,
            theiler_corrector=theiler,
        )
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


def compute_original_invariants(orig_file, output_dir, base, delay, theiler, metric_names=None):
    metric_names = list(ALL_METRICS) if not metric_names else list(metric_names)
    need_d2 = "D2" in metric_names
    need_k2 = "K2" in metric_names
    need_lle = "LLE" in metric_names
    need_rqa = any(k in metric_names for k in RQA_KEYS)

    prefix = os.path.join(output_dir, f"{base}_orig")
    out = {k: np.nan for k in metric_names}
    d2_file = h2_file = None
    if need_d2 or need_k2:
        d2_file, h2_file = run_d2(orig_file, delay, theiler, prefix)
    if need_d2 and d2_file:
        out["D2"] = extract_d2(d2_file)
    if need_k2 and h2_file:
        out["K2"] = extract_k2(h2_file)
    if need_lle:
        lyap_file = prefix + "_lyap.txt"
        run_lyap_k(orig_file, delay, lyap_file)
        out["LLE"] = extract_lle(lyap_file)
    if need_rqa:
        original_series = load_data(orig_file)
        rqa_metrics = compute_pyrqa_metrics(original_series, delay, theiler)
        for k in RQA_KEYS:
            if k in out:
                out[k] = rqa_metrics.get(k, np.nan)
    return out


def process_single_bootstrap(args_tuple):
    i, sample, tmp_dir, delay, theiler, metric_names = args_tuple
    if not metric_names:
        metric_names = list(ALL_METRICS)
    need_d2 = "D2" in metric_names
    need_k2 = "K2" in metric_names
    need_lle = "LLE" in metric_names
    need_rqa = any(k in metric_names for k in RQA_KEYS)
    sample_file = os.path.join(tmp_dir, f"perm_{i:04d}.dat")
    np.savetxt(sample_file, sample)
    prefix = os.path.join(tmp_dir, f"perm_{i:04d}")
    result = {"i": i, **{k: np.nan for k in metric_names}}
    try:
        d2_file = h2_file = None
        if need_d2 or need_k2:
            d2_file, h2_file = run_d2(sample_file, delay, theiler, prefix)
            if need_d2:
                result["D2"] = extract_d2(d2_file)
            if need_k2:
                result["K2"] = extract_k2(h2_file)
        if need_lle:
            lyap_file = prefix + "_lyap.txt"
            run_lyap_k(sample_file, delay, lyap_file)
            result["LLE"] = extract_lle(lyap_file)
        if need_rqa:
            rqa_vals = compute_pyrqa_metrics(sample, delay, theiler)
            for k in RQA_KEYS:
                if k in result:
                    result[k] = rqa_vals.get(k, np.nan)
    except Exception:
        pass
    finally:
        for tmp in glob.glob(prefix + "*"):
            try:
                os.remove(tmp)
            except OSError:
                pass
        try:
            os.remove(sample_file)
        except OSError:
            pass
    return result


def safe_mean(arr):
    return float(np.mean(arr)) if len(arr) > 0 else np.nan


def safe_std(arr):
    return float(np.std(arr, ddof=1)) if len(arr) > 1 else (float(np.std(arr)) if len(arr) == 1 else np.nan)


# Empirical tail for p = (1 + #{...}) / (B + 1); see Theiler et al. style surrogate testing.
# D2/K2: typically test whether original is unusually small vs surrogates (lower tail).
# LLE: typically unusually large (upper tail). RQA scalars: omnibus two-sided combination.
METRIC_EMPIRICAL_TAIL = {
    "D2": "lower",
    "K2": "lower",
    "LLE": "upper",
    "RR": "two_sided",
    "DET": "two_sided",
    "LAM": "two_sided",
    "MAXLINE": "two_sided",
    "ENTR": "two_sided",
    "TT": "two_sided",
}


def empirical_surrogate_test(
    boot_dist: np.ndarray,
    orig_val: float,
    tail: str,
) -> tuple[float, float, float, str]:
    """
    Rank/count empirical p-value from B surrogate replicates (no Gaussian / t-assumption).

    Let T_1..T_B be surrogate metric values. Define:
      p_upper = (1 + #{T_i >= T_orig}) / (B + 1)
      p_lower = (1 + #{T_i <= T_orig}) / (B + 1)

    tail='upper' uses p_upper; 'lower' uses p_lower.
    tail='two_sided' uses min(1, 2 * min(p_upper, p_lower)) when direction is not fixed a priori.

    Descriptive (not used for p):
      z_sigma = (T_orig - mean(T)) / SD(T)  — sigma-score (Theiler-style).
      z_se    = (T_orig - mean(T)) / SE(T) with SE = SD/sqrt(B) — sensitive to B; for comparison only.

    Decision: reject H0 if p < 0.05.
    """
    bd = np.asarray(boot_dist, dtype=float)
    bd = bd[np.isfinite(bd)]
    b_reps = len(bd)
    if b_reps < 1 or not np.isfinite(orig_val):
        return float("nan"), float("nan"), float("nan"), "insufficient data"

    m = float(np.mean(bd))
    sd = float(np.std(bd, ddof=1)) if b_reps > 1 else 0.0
    se = sd / np.sqrt(b_reps) if b_reps > 0 else float("nan")

    if sd > 0:
        z_sigma = float((orig_val - m) / sd)
    elif np.isfinite(orig_val) and np.isfinite(m) and np.isclose(orig_val, m, rtol=0.0, atol=1e-12):
        z_sigma = 0.0
    else:
        z_sigma = float(np.sign(orig_val - m)) * float("inf")

    if np.isfinite(se) and se > 0:
        z_se = float((orig_val - m) / se)
    else:
        z_se = float("nan")

    ge = int(np.sum(bd >= orig_val))
    le = int(np.sum(bd <= orig_val))
    p_upper = (1 + ge) / (b_reps + 1)
    p_lower = (1 + le) / (b_reps + 1)

    if tail == "upper":
        p_val = p_upper
    elif tail == "lower":
        p_val = p_lower
    elif tail == "two_sided":
        p_val = min(1.0, 2.0 * min(p_upper, p_lower))
    else:
        raise ValueError(f"unknown tail mode: {tail!r}")

    decision = (
        "reject H0" if np.isfinite(p_val) and (p_val < 0.05) else "fail to reject H0"
    )
    return z_sigma, z_se, float(p_val), decision


def predictability_time(lle, eps=PRED_EPSILON, tol=PRED_TOLERANCE):
    """T = (1/lambda) * log(tol/eps) for lambda > 0; nan otherwise."""
    if lle is None or not np.isfinite(lle) or lle <= 0:
        return float("nan")
    return float((1.0 / lle) * np.log(tol / eps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--delay", type=int, required=True)
    parser.add_argument("--theiler", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--test_mode", default="false")
    parser.add_argument(
        "--metrics",
        choices=(METRICS_SCOPE_FULL, METRICS_SCOPE_D2_K2_LLE_RQA),
        default=METRICS_SCOPE_FULL,
        help="full or d2_k2_lle_rqa: D2, K2, LLE and RQA metrics.",
    )
    parser.add_argument(
        "--metrics_list",
        default="",
        help="Optional explicit comma-separated metric subset (e.g. D2,K2 or LLE or RR,DET). Overrides --metrics.",
    )
    parser.add_argument(
        "--surrogate_blocks",
        type=int,
        default=100,
        help="Number of contiguous blocks used for block-permutation surrogate generation (default 100).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    test_mode = is_test_mode(args.test_mode)
    b_reps = TEST_B if test_mode else FULL_B

    metric_names = parse_metrics_list(args.metrics_list) if args.metrics_list.strip() else metric_names_for_scope(args.metrics)

    metrics_label = ",".join(metric_names)
    print(f"  -> Starting permutation-surrogate hypothesis test for {args.base} (tau={args.delay}, W={args.theiler})")
    print(f"  -> Mode: {'TEST' if test_mode else 'FULL'}, surrogate replicates B={b_reps}, metrics={metrics_label}")
    print(
        f"  -> Surrogates: block permutation (N_blocks={args.surrogate_blocks}); "
        "inference: empirical p from surrogate ranks (Theiler-style); "
        f"z_sigma / z_SE(B={b_reps}) (column label) descriptive only; decision by p<0.05."
    )

    orig_data = load_data(args.input)
    n = len(orig_data)

    print(f"  -> Generating {b_reps} block-permuted surrogate samples...")
    permuted_samples = generate_permuted_samples(orig_data, b_reps, n_blocks=args.surrogate_blocks)

    tmp_dir = os.path.join(args.output_dir, "tmp_perm")
    os.makedirs(tmp_dir, exist_ok=True)

    boot = {name: np.full(b_reps, np.nan, dtype=float) for name in metric_names}

    print("  -> Launching parallel TISEAN + RQA execution...")
    tasks = [(i, permuted_samples[i], tmp_dir, args.delay, args.theiler, metric_names) for i in range(b_reps)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for count, res in enumerate(executor.map(process_single_bootstrap, tasks), 1):
            idx = res["i"]
            for metric in metric_names:
                boot[metric][idx] = res.get(metric, np.nan)
            if count % 25 == 0 or count == b_reps:
                print(f"    Completed {count}/{b_reps} surrogate iterations...")

    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    print("  -> Computing invariants for original data...")
    orig = compute_original_invariants(
        args.input, args.output_dir, args.base, args.delay, args.theiler, metric_names=metric_names
    )

    boot_clean = {k: arr[np.isfinite(arr)] for k, arr in boot.items()}
    z_sigma_scores: dict[str, float] = {}
    z_se_scores: dict[str, float] = {}
    p_values = {}
    decisions = {}
    for k in metric_names:
        tail_k = METRIC_EMPIRICAL_TAIL.get(k, "two_sided")
        zs, zse, p_val, dec = empirical_surrogate_test(boot_clean[k], orig[k], tail_k)
        z_sigma_scores[k] = zs
        z_se_scores[k] = zse
        p_values[k] = p_val
        decisions[k] = dec

    lle_orig = orig.get("LLE", float("nan"))
    T_orig = predictability_time(lle_orig)
    lle_boot = boot_clean.get("LLE", np.array([]))
    pos_lle_boot = lle_boot[(np.isfinite(lle_boot)) & (lle_boot > 0)] if len(lle_boot) else np.array([])
    if pos_lle_boot.size:
        T_boot = (1.0 / pos_lle_boot) * np.log(PRED_TOLERANCE / PRED_EPSILON)
        T_boot_mean = float(np.mean(T_boot))
        T_boot_lo = float(np.percentile(T_boot, 2.5))
        T_boot_hi = float(np.percentile(T_boot, 97.5))
        T_boot_n = int(pos_lle_boot.size)
    else:
        T_boot_mean = float("nan")
        T_boot_lo = float("nan")
        T_boot_hi = float("nan")
        T_boot_n = 0

    summary_file = os.path.join(args.output_dir, f"{args.base}_surrogate_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as handle:
        handle.write(f"Permutation surrogate hypothesis test ({args.base})\n")
        handle.write(
            f"Parameters: tau={args.delay}, W={args.theiler}, B={b_reps}, metrics={metrics_label}, "
            f"surrogate_blocks={args.surrogate_blocks}\n"
        )
        handle.write(f"Original data length: {n}\n")
        handle.write(f"Mode: {'TEST' if test_mode else 'FULL'}\n")
        handle.write("Surrogates: block permutation of observed series (contiguous blocks shuffled in random order).\n")
        handle.write(
            "Inference : empirical p-values from surrogate counts (+1 / B+1 correction); "
            "tails: D2,K2=lower; LLE=upper; RQA=two_sided (see METRIC_EMPIRICAL_TAIL in hypothesis.py). "
            "z_sigma = (orig-mean)/SD(surr) (Theiler-style, descriptive); "
            f"z_SE(B={b_reps}) = (orig-mean)/(SD/sqrt(B)) — SE uses this run’s B (descriptive only). "
            "Reject H0 if p < 0.05.\n\n"
        )
        handle.write(
            f"Invariant       Orig.    Mean(surr)  SD(surr)    z_sigma   z_SE(B={b_reps})    p-value    decision\n"
        )
        handle.write(
            "--------------------------------------------------------------------------------------------------------\n"
        )

        def _fmt_zcol(z: float) -> str:
            if np.isnan(z):
                return "      nan"
            return f"{z:>10.4f}"

        for metric in metric_names:
            bd = boot_clean[metric]
            mn = safe_mean(bd)
            sdb = safe_std(bd)
            zs = z_sigma_scores[metric]
            zse = z_se_scores[metric]
            pv = p_values[metric]
            ssig = _fmt_zcol(zs)
            sse = _fmt_zcol(zse)
            pp = f"{pv:>10.4e}" if np.isfinite(pv) else "      nan"
            handle.write(
                f"{metric:<14} {orig[metric]:>8.4f}  {mn:>10.4f}  {sdb:>10.4f}  {ssig}  {sse}  {pp}  "
                f"{decisions[metric]}\n"
            )
        handle.write("\nConclusion (decision by p-value < 0.05):\n")
        for metric in metric_names:
            handle.write(f"  {metric:<8}: {decisions[metric]}\n")

        if "LLE" in metric_names:
            handle.write("\nPredictability time T (hours)\n")
            handle.write("-----------------------------\n")
            handle.write(
                f"  Formula      : T = (1/lambda) * log(L/eps)  with eps={PRED_EPSILON:g}, L={PRED_TOLERANCE:g}\n"
            )
            handle.write(f"  Original LLE : {lle_orig:.6f}\n")
            if np.isfinite(T_orig):
                handle.write(f"  Original T   : {T_orig:.2f} hours ({T_orig/24:.2f} days)\n")
            else:
                handle.write("  Original T   : undefined (LLE <= 0 or non-finite -> not predictable in this model)\n")
            if T_boot_n > 0:
                handle.write(
                    "  Surrogate T  : mean={mean:.2f} h, 95% CI=[{lo:.2f}, {hi:.2f}] h, "
                    "based on {n_pos}/{n_tot} positive-LLE surrogates\n".format(
                        mean=T_boot_mean,
                        lo=T_boot_lo,
                        hi=T_boot_hi,
                        n_pos=T_boot_n,
                        n_tot=int(np.sum(np.isfinite(lle_boot))) if len(lle_boot) else 0,
                    )
                )
            else:
                handle.write("  Surrogate T  : no surrogate has lambda > 0 -> T undefined for surrogate distribution\n")
            handle.write(
                "  Note         : LLE itself is hypothesis-tested above; T is reported for predictability budgeting only.\n"
            )

    print(f"  -> Summary written to {summary_file}")


if __name__ == "__main__":
    main()
