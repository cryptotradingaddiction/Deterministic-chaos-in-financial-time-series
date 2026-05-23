#!/usr/bin/env python3
"""Hypothesis CLI (stationary-bootstrap TS tests)."""
import argparse
import logging
import os
import warnings

import numpy as np

from hypothesis_config import (
    ALL_METRICS,
    BOOTSTRAP_TEST_METRICS,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_HYPOTHESIS_SEED,
    DEFAULT_RQA_RADIUS,
    DEFAULT_STATIONARY_BLOCK_MEAN,
    DEFAULT_TS_THRESHOLD,
    DIM_LLE_METRICS,
    NULL_SERIES_METRICS,
    RQA_EMBEDDING_DIM,
    RQA_KEYS,
    RQA_RADIUS_PERCENTILE_DEFAULT,
    T_DOF,
)
from hypothesis_surrogates import (
    generate_normal_series,
    generate_single_surrogate,
    generate_t_series,
)
from hypothesis_ts import invariant_bootstrap_ts_test
from invariants_compute import compute_invariants
from surrogate_sampling import stationary_bootstrap_samples
from tisean_io import load_data

logger = logging.getLogger(__name__)


def _bootstrap_samples_default():
    raw = os.environ.get("DCH_BOOTSTRAP_SAMPLES", "").strip()
    if raw:
        try:
            return max(1, int(float(raw)))
        except ValueError:
            pass
    return DEFAULT_BOOTSTRAP_SAMPLES


def _stationary_block_mean_default():
    raw = os.environ.get("DCH_STATIONARY_BLOCK_MEAN", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return DEFAULT_STATIONARY_BLOCK_MEAN


def parse_metrics_list(raw_metrics):
    """Normalize and validate a comma-separated CLI metric list."""
    tokens = [t.strip().upper() for t in str(raw_metrics).split(",") if t.strip()]
    if not tokens:
        raise ValueError("metrics list is empty")
    invalid = [t for t in tokens if t not in ALL_METRICS]
    if invalid:
        raise ValueError(f"unknown metric(s): {', '.join(invalid)}")
    return list(dict.fromkeys(tokens))


def default_direct_run_metrics():
    """
    Metrics used when ``--metrics_list`` is omitted.

    Direct CLI runs default to Ellner only. Batch scripts pass an explicit
    ``--metrics_list`` for TAKENS, LLE, or RQA scopes.
    """
    return ["ELLNER"]


def metric_names_for_scope(scope=None):
    """
    Backward-compatible alias for :func:`default_direct_run_metrics`.

    The legacy ``--metrics`` flag is ignored; use ``--metrics_list`` instead.
    """
    del scope
    return default_direct_run_metrics()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # CLI contract
    # ------------------------------------------------------------------
    #
    # hypothesis.py is designed to be called from several Windows batch files
    # with different metric scopes:
    #   correlation_dimension.bat -> ELLNER by default, or TAKENS/ELLNER switch
    #   Lambda_max.bat           -> LLE
    #   RQA.bat                  -> RQA metrics; --rqa_radius_mode fixed + precomputed r
    #
    # Direct calls without --metrics_list default to ["ELLNER"]. That default is
    # deliberate: the current thesis dimension hypothesis is Ellner-first, while
    # TAKENS remains available through --metrics_list TAKENS or TAKENS,ELLNER.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--delay", type=int, required=True)
    parser.add_argument("--theiler", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--test_mode",
        default="false",
        help=(
            "When 'true', sets ``DCH_TEST_MODE`` in this process so downstream "
            "TISEAN wrappers (run_lyap_k, lyap_k_iterations, lyap_min_neighbors) "
            "pick the short-series defaults. Equivalent to setting the env var."
        ),
    )
    parser.add_argument("--metrics", default="full")
    parser.add_argument("--metrics_list", default="")
    parser.add_argument("--bootstrap_samples", type=int, default=_bootstrap_samples_default())
    parser.add_argument(
        "--stationary_block_mean",
        type=float,
        default=_stationary_block_mean_default(),
        help="Mean stationary-bootstrap block length. <=0 uses sqrt(n). Env: DCH_STATIONARY_BLOCK_MEAN.",
    )
    parser.add_argument("--ts_threshold", type=float, default=DEFAULT_TS_THRESHOLD)
    parser.add_argument("--rqa_radius", type=float, default=DEFAULT_RQA_RADIUS)
    parser.add_argument(
        "--rqa_radius_mode",
        choices=["percentile", "fixed"],
        default="percentile",
        help=(
            "How to select PyRQA recurrence radius. "
            "'percentile' (default): use the --rqa_percentile-th percentile of pairwise "
            "Euclidean distances between embedded state vectors of the analysed series; "
            "fall back to --rqa_radius if the percentile cannot be computed. "
            "'fixed': always use --rqa_radius."
        ),
    )
    parser.add_argument(
        "--rqa_percentile",
        type=float,
        default=RQA_RADIUS_PERCENTILE_DEFAULT,
        help="Percentile (in %%) of pairwise distances used as RQA radius when --rqa_radius_mode=percentile.",
    )
    parser.add_argument(
        "--rqa_bootstrap",
        choices=["on", "off"],
        default="on",
        help=(
            "Include PyRQA scalars (RR, DET, LAM, MAXLINE, ENTR, TT, TREND) in the "
            "stationary-bootstrap TS test (default 'on'). When 'on', the radius is "
            "locked to the value computed on the original series so every bootstrap, "
            "reshuffle, normal, and t-series RQA computation uses the same r. "
            "Switch to 'off' to keep RQA as an original-series-only scalar "
            "(legacy behaviour, no TS decision)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_HYPOTHESIS_SEED,
        help="RNG seed for reshuffle, normal/t reference series, and stationary bootstrap.",
    )
    args = parser.parse_args()

    # Propagate --test_mode into the environment so helpers in
    # ``hypothesis_config`` (lyap_k_steps, lyap_k_iterations, lyap_min_neighbors)
    # and any subprocess inherit the same short-series defaults. Without this
    # propagation, a direct ``hypothesis.py --test_mode true`` run on a sliced
    # .dat fell back to production lyap_k flags and could fail.
    if args.test_mode.strip().lower() in ("1", "true", "yes", "on"):
        os.environ["DCH_TEST_MODE"] = "true"

    # Fail early on invalid scalar parameters. These checks catch batch-script
    # wiring mistakes before a long TISEAN/bootstrap run starts.
    if args.rqa_radius <= 0.0:
        raise SystemExit("hypothesis.py: --rqa_radius must be positive.")
    if not (0.0 < args.rqa_percentile < 100.0):
        raise SystemExit("hypothesis.py: --rqa_percentile must be in (0, 100).")
    if args.bootstrap_samples < 1:
        raise SystemExit("hypothesis.py: --bootstrap_samples must be >= 1.")
    if args.ts_threshold <= 0:
        raise SystemExit("hypothesis.py: --ts_threshold must be positive.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve the metric scope after CLI validation. When a batch file passes an
    # explicit list, that list is authoritative; otherwise use the direct-run
    # default from default_direct_run_metrics().
    metric_names = (
        parse_metrics_list(args.metrics_list)
        if args.metrics_list.strip()
        else default_direct_run_metrics()
    )

    # Active bootstrap / reference scope. With --rqa_bootstrap=off, drop the RQA
    # keys from both sets so PyRQA stays an original-only scalar (legacy mode).
    rqa_in_bootstrap = (args.rqa_bootstrap == "on")
    if rqa_in_bootstrap:
        active_bootstrap = set(BOOTSTRAP_TEST_METRICS)
        active_null = set(NULL_SERIES_METRICS)
    else:
        active_bootstrap = set(DIM_LLE_METRICS)
        active_null = set(DIM_LLE_METRICS)

    print(f"  -> Stationary-bootstrap hypothesis test: {args.base}")
    print(f"     tau={args.delay}, W={args.theiler}, seed={args.seed}, "
          f"metrics={','.join(metric_names)}")
    print(
        f"     RQA bootstrap test: {args.rqa_bootstrap} "
        f"(radius is locked from the original series when 'on')."
    )
    block_mean_label = "sqrt(n)" if args.stationary_block_mean <= 0 else f"{args.stationary_block_mean:g}"
    print(
        f"     Bootstrap TS test: B={args.bootstrap_samples}, "
        f"stationary_block_mean={block_mean_label}, threshold=|TS|>{args.ts_threshold:g}"
    )
    if args.rqa_radius_mode == "percentile":
        print(
            f"     RQA radius mode: percentile (p={args.rqa_percentile:g}%, "
            f"m={RQA_EMBEDDING_DIM}, tau={args.delay}); "
            f"fallback fixed radius = {args.rqa_radius:g}"
        )
    else:
        print(f"     RQA radius mode: fixed (r={args.rqa_radius:g})")

    # ------------------------------------------------------------------
    # 1. Load original log-returns; estimate mu and sigma of the series
    # ------------------------------------------------------------------
    #
    # mu_r and sigma_r generate the Gaussian and Student-t descriptive references.
    orig_data = load_data(args.input)
    n = len(orig_data)
    mu_r = float(np.mean(orig_data))
    sigma_r = float(np.std(orig_data, ddof=1))
    print(f"  -> Original series: n={n}, mu={mu_r:.6f}, sigma={sigma_r:.6f}")

    # ------------------------------------------------------------------
    # 2. Generate surrogate (randperm) and two reference series
    # ------------------------------------------------------------------
    #
    # The reshuffled series is the actual null comparison in the TS statistic.
    # Normal and t(3.5) are kept as Step 0 descriptive benchmarks so the thesis
    # tables can compare the original against several noise shapes.
    #
    # Surrogates and the stationary bootstrap consume independent random
    # streams derived from the same ``--seed``. Using a single Generator for
    # both would mean the bootstrap consumes whatever bits the surrogate
    # block left over (still reproducible, but the two procedures share state
    # for no methodological reason). ``SeedSequence.spawn(2)`` gives two
    # statistically independent child streams from one seed.
    seed_surr, seed_boot = np.random.SeedSequence(args.seed).spawn(2)
    rng = np.random.default_rng(seed_surr)
    surr_data = generate_single_surrogate(orig_data, rng)
    norm_data = generate_normal_series(mu_r, sigma_r, n, rng)
    t_data = generate_t_series(mu_r, sigma_r, n, rng, dof=T_DOF)

    series_specs = [
        ("orig",   orig_data),
        ("surr",   surr_data),
        ("normal", norm_data),
        (f"t{T_DOF}", t_data),
    ]

    # ------------------------------------------------------------------
    # 3. Compute invariants for original and reference series
    # ------------------------------------------------------------------
    #
    # For null/reference series, recompute only metrics listed in
    # NULL_SERIES_METRICS. This prevents expensive and methodologically invalid
    # recomputation of scalar-only RQA values for synthetic references.
    tmp_dir = os.path.join(args.output_dir, "tmp_hyp")
    os.makedirs(tmp_dir, exist_ok=True)

    results = {}   # label -> {metric: mean invariant value}
    stds    = {}   # label -> {metric: SD of invariant values}
    counts  = {}   # label -> {metric: n values used for SD}
    rqa_radius_log = {}  # label -> {radius, source}
    bootstrap_metrics = [m for m in metric_names if m in active_bootstrap]
    bootstrap_values = {m: [] for m in bootstrap_metrics}
    bootstrap_mean = {m: np.nan for m in metric_names}
    bootstrap_sd = {m: np.nan for m in metric_names}
    bootstrap_n = {m: 0 for m in metric_names}

    # When RQA participates in the bootstrap test, the radius is computed once
    # on the original series and reused everywhere else. Comparing RR/DET on
    # different radii would be meaningless, so we capture orig's value and
    # switch later iterations to ``rqa_radius_mode='fixed'``. The pre-CLI
    # request stays in ``orig_rqa_mode`` / ``orig_rqa_radius`` for the orig
    # call only.
    orig_rqa_mode = args.rqa_radius_mode
    orig_rqa_radius = args.rqa_radius
    locked_rqa_mode = args.rqa_radius_mode
    locked_rqa_radius = args.rqa_radius

    for label, series in series_specs:
        # Original series gets the full requested scope. Synthetic reference
        # series get only the metrics that are meaningful for Step 0 and TS.
        metrics_for_label = (
            metric_names
            if label == "orig"
            else [m for m in metric_names if m in active_null]
        )

        if metrics_for_label:
            print(f"  -> Computing invariants for series: {label} ({','.join(metrics_for_label)}) ...")
            use_mode = orig_rqa_mode if label == "orig" else locked_rqa_mode
            use_radius = orig_rqa_radius if label == "orig" else locked_rqa_radius
            label_key = f"{args.base}_{label}"
            # For the original series, keep a copy of the lyap_k output so the
            # diagnostic LLE plot can be drawn from a guaranteed path even when
            # the upstream .bat (Lambda_max.bat) is not what called us.
            lyap_keep_path = None
            if label == "orig" and "LLE" in metrics_for_label:
                lyap_keep_path = os.path.join(
                    args.output_dir, f"{args.base}_orig_lyap.txt",
                )
            inv_part, inv_std_part, inv_n_part = compute_invariants(
                series, tmp_dir, label_key,
                args.delay, args.theiler, metrics_for_label, use_radius,
                series_std_fallback=sigma_r,
                rqa_radius_mode=use_mode,
                rqa_percentile=args.rqa_percentile,
                rqa_radius_log=rqa_radius_log,
                lyap_keep_path=lyap_keep_path,
            )

            # Lock the radius using whatever value was applied to the original
            # series so every subsequent reference and bootstrap call inherits
            # the same r. ``rqa_radius_log`` keys are the full ``label_key``s.
            if (
                label == "orig"
                and rqa_in_bootstrap
                and any(k in metrics_for_label for k in RQA_KEYS)
                and label_key in rqa_radius_log
                and np.isfinite(rqa_radius_log[label_key]["radius"])
                and rqa_radius_log[label_key]["radius"] > 0.0
            ):
                locked_rqa_radius = float(rqa_radius_log[label_key]["radius"])
                locked_rqa_mode = "fixed"
                print(
                    f"     RQA radius locked from orig for surrogates/bootstrap: "
                    f"r={locked_rqa_radius:.6g}"
                )
        else:
            print(f"  -> Skipping invariant recomputation for series: {label} (not needed for selected metrics).")
            inv_part, inv_std_part, inv_n_part = {}, {}, {}

        results[label] = {k: inv_part.get(k, np.nan) for k in metric_names}
        stds[label]    = {k: inv_std_part.get(k, np.nan) for k in metric_names}
        counts[label]  = {k: inv_n_part.get(k, 0) for k in metric_names}

    if bootstrap_metrics:
        # Stationary bootstrap preserves local time dependence by resampling
        # geometrically distributed blocks. Each pseudo-series is passed through
        # exactly the same invariant computation as the original.
        mean_block_length = None if args.stationary_block_mean <= 0 else args.stationary_block_mean
        print(
            f"  -> Stationary bootstrap for {','.join(bootstrap_metrics)}: "
            f"B={args.bootstrap_samples}, mean_block_length={block_mean_label}"
        )
        boot_series = stationary_bootstrap_samples(
            orig_data,
            args.bootstrap_samples,
            mean_block_length=mean_block_length,
            seed=seed_boot,
        )
        show_warnings = os.environ.get("DCH_HYPOTHESIS_WARNINGS", "").strip().lower() in (
            "1", "true", "yes", "on"
        )
        with warnings.catch_warnings():
            if not show_warnings:
                # Narrow filter: bootstrap repeats many short-series invariant runs.
                warnings.simplefilter("ignore", category=RuntimeWarning)
            for idx, boot_data in enumerate(boot_series, start=1):
                print(
                    f"     bootstrap {idx:03d}/{args.bootstrap_samples}: "
                    f"computing {','.join(bootstrap_metrics)} ..."
                )
                inv_part, _inv_std_part, _inv_n_part = compute_invariants(
                    boot_data,
                    tmp_dir,
                    f"{args.base}_statboot_{idx:03d}",
                    args.delay,
                    args.theiler,
                    bootstrap_metrics,
                    locked_rqa_radius,
                    series_std_fallback=sigma_r,
                    rqa_radius_mode=locked_rqa_mode,
                    rqa_percentile=args.rqa_percentile,
                )
                for metric in bootstrap_metrics:
                    value = inv_part.get(metric, np.nan)
                    if np.isfinite(value):
                        bootstrap_values[metric].append(float(value))

        # Convert the bootstrap cloud of invariant estimates into the robust
        # centre and spread used by the TS statistic.
        for metric in bootstrap_metrics:
            arr = np.asarray(bootstrap_values[metric], dtype=float)
            arr = arr[np.isfinite(arr)]
            bootstrap_n[metric] = int(arr.size)
            if arr.size:
                bootstrap_mean[metric] = float(np.mean(arr))
            if arr.size > 1:
                bootstrap_sd[metric] = float(np.std(arr, ddof=1))

    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    # ------------------------------------------------------------------
    # 4. Bootstrap TS test against THREE reference series
    # ------------------------------------------------------------------
    #
    # TAKENS / ELLNER / LLE (and bootstrap-enabled RQA scalars) receive a TS
    # decision per reference series: ``surr`` (random permutation),
    # ``normal`` (Gaussian with matched mu_r / sigma_r) and ``t<T_DOF>``
    # (scaled Student-t). Each reference is an independent realisation of a
    # different null hypothesis:
    #
    #   surr   -> "the invariant is just an artefact of the multi-set of
    #             values" (any ordering would produce the same number).
    #   normal -> "the invariant is what you would get from i.i.d. Gaussian
    #             noise with matched first two moments".
    #   t<DOF> -> same as normal but with the project's heavy-tailed reference
    #             (DOF = 3.5 matches financial log-returns far better than
    #             Gaussian).
    #
    # The bootstrap centre / SD are shared across all three tests — only the
    # reference value in the numerator changes — so the denominator df for the
    # Student-t p-value is the same ``B - 1`` in every cell.
    test_stats = {m: {} for m in metric_names}
    abs_test_stats = {m: {} for m in metric_names}
    p_values = {m: {} for m in metric_names}
    decisions = {m: {} for m in metric_names}

    ref_keys = ["surr", "normal", f"t{T_DOF}"]

    for metric in metric_names:
        if metric in active_bootstrap:
            bm = bootstrap_mean.get(metric, np.nan)
            bs = bootstrap_sd.get(metric, np.nan)
            bn = bootstrap_n.get(metric, 0)
            for ref_key in ref_keys:
                ref_val = results.get(ref_key, {}).get(metric, np.nan)
                ts, abs_ts, p_val, dec = invariant_bootstrap_ts_test(
                    bm, bs, ref_val,
                    n_bootstrap=bn,
                    threshold=args.ts_threshold,
                )
                test_stats[metric][ref_key] = ts
                abs_test_stats[metric][ref_key] = abs_ts
                p_values[metric][ref_key] = p_val
                decisions[metric][ref_key] = dec
        else:
            for ref_key in ref_keys:
                test_stats[metric][ref_key] = np.nan
                abs_test_stats[metric][ref_key] = np.nan
                p_values[metric][ref_key] = np.nan
                decisions[metric][ref_key] = "not bootstrap-tested"

    # ------------------------------------------------------------------
    # 5. summary
    # ------------------------------------------------------------------
    #
    # The text format is intentionally simple: print_results.py parses these
    # fixed-width rows to build per-folder aggregate summaries, while documents.py
    # uses those aggregates for Word tables.
    labels_ordered = [lbl for lbl, _ in series_specs]

    summary_file = os.path.join(args.output_dir, f"{args.base}_surrogate_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as fh:
        fh.write(f"Stationary-bootstrap hypothesis test ({args.base})\n")
        has_rqa_metrics = any(m in RQA_KEYS for m in metric_names)
        fh.write(
            f"Parameters: tau={args.delay}, W={args.theiler}, seed={args.seed}, "
            f"B={args.bootstrap_samples}, stationary_block_mean={block_mean_label}, "
            f"TS_threshold={args.ts_threshold:g}, metrics={','.join(metric_names)}, "
            f"T_dof={T_DOF}\n"
        )
        fh.write(f"Original series: n={n}, mu_r={mu_r:.6f}, sigma_r={sigma_r:.6f}\n")
        if args.rqa_radius_mode == "percentile":
            fh.write(
                f"RQA radius mode: percentile (p={args.rqa_percentile:g}%, m={RQA_EMBEDDING_DIM}, "
                f"tau={args.delay}); fallback fixed radius = {args.rqa_radius:g}\n"
            )
        else:
            fh.write(f"RQA radius mode: fixed (r={args.rqa_radius:g})\n")
        if rqa_radius_log:
            for lbl, info in rqa_radius_log.items():
                fh.write(
                    f"  RQA radius applied to {lbl}: r={info['radius']:.6g}  ({info['source']})\n"
                )
        fh.write(
            "Surrogate: randperm (full random permutation, no blocking).\n"
            "Normal:    N(mu_r, sigma_r) of same length.\n"
            f"t-series:  t(dof={T_DOF}) scaled to (mu_r, sigma_r) of same length.\n"
            f"RNG seed:  {args.seed} (reshuffle, reference series, stationary bootstrap).\n"
            "Stationary bootstrap: B pseudo-series from the original, with geometric block lengths.\n"
        )
        if has_rqa_metrics:
            rqa_mode_label = (
                "bootstrap-tested; recurrence radius locked from orig"
                if rqa_in_bootstrap
                else "original-series only (--rqa_bootstrap off)"
            )
            fh.write(
                f"Reference invariants are recomputed for the active metrics; RQA: {rqa_mode_label}.\n"
            )
        else:
            fh.write("Reference invariants are recomputed for the active metrics.\n")
        rqa_block = ""
        if has_rqa_metrics:
            rqa_block = (
                "  RQA   — PyRQA scalars (RR, DET, LAM, MAXLINE, ENTR, TT, TREND); radius locked from\n"
                "           orig when bootstrap-tested; Theiler W matches d2/lyap_k (-t, exclude |i-j|<=W).\n"
                if rqa_in_bootstrap
                else
                "  RQA   — one PyRQA value per metric on the full original series only.\n"
            )
        fh.write(
            "Test: TS=(mean_bootstrap(invariant) - invariant_reference)/SD_bootstrap(invariant).\n"
            f"  Decision: reject H0 if |TS| > {args.ts_threshold:g}; this indicates structure/memory as a prerequisite for chaos, not proof of chaos.\n"
            "  p-value: two-sided under Student-t with df = B - 1 (matches MATLAB '2*(1 - tcdf(|TS|, B-1))').\n"
            "  The TS test is repeated against THREE reference series per metric:\n"
            "    surr   = random permutation of the original (multi-set null)\n"
            "    normal = Gaussian N(mu_r, sigma_r) of same length\n"
            f"    t{T_DOF}   = scaled Student-t(nu={T_DOF}) of same length\n"
            "  Each reference is an independent null; the per-reference decision is reported below.\n"
            "Invariant value sources:\n"
            "  TAKENS — mean of the stable plateau of the c2t Takens-Theiler curve d_2^(T)(r') for m=3\n"
            "  ELLNER — Ellner extension (eq. 8.78) computed from .c2 over [r_min, r_max] auto-detected\n"
            "           from the same Takens plateau; reported SD is the Takens-plateau dispersion\n"
            "           (orientation of the interval quality).\n"
            "  LLE   — OLS slope of the highest-quality lyap_k S(t) block at m=3\n"
            "           (best ε-block by (window length / OLS std_err)); reported SD is the\n"
            "           OLS standard error of that slope, not a cross-block spread.\n"
            f"{rqa_block}\n"
        )

        # --- Series statistics header ---
        # These are statistics of the input series themselves, not uncertainty
        # estimates of the invariants.
        fh.write("Series statistics (mean and SD of each series):\n")
        fh.write(f"  {'series':<10}  {'mean':>12}  {'sd':>12}\n")
        fh.write(f"  {'-'*38}\n")
        for label, series in series_specs:
            fh.write(f"  {label:<10}  {np.mean(series):>12.6f}  {np.std(series, ddof=1):>12.6f}\n")
        fh.write("\n")

        col_w = 12

        def _f(v):
            return f"{v:>{col_w}.4f}" if np.isfinite(v) else f"{'nan':>{col_w}}"

        if any(m in active_bootstrap for m in metric_names):
            # Step 0 is a descriptive sanity check: original value against three
            # noise/reference constructions. The formal decision is still TS.
            fh.write("Step 0 invariant comparison against noise references:\n")
            fh.write(
                f"  {'Invariant':<12}  {'orig':>{col_w}}  {'resh':>{col_w}}  "
                f"{'normal':>{col_w}}  {f't{T_DOF}':>{col_w}}\n"
            )
            fh.write(f"  {'-'*66}\n")
            for metric in metric_names:
                if metric not in active_bootstrap:
                    continue
                fh.write(
                    f"  {metric:<12}  {_f(results['orig'].get(metric, np.nan))}  "
                    f"{_f(results['surr'].get(metric, np.nan))}  "
                    f"{_f(results['normal'].get(metric, np.nan))}  "
                    f"{_f(results[f't{T_DOF}'].get(metric, np.nan))}\n"
                )
            fh.write(
                "  Interpretation: compare original invariant values directly with reshuffle, Gaussian, and t(3.5) references.\n\n"
            )

        # --- Invariant × reference table (machine-parsed) ---
        # One row per (metric, reference). The reference column is the
        # primary key for downstream parsers in print_results.py /
        # documents.py — keep its label tokens ('surr', 'normal',
        # f't{T_DOF}') stable.
        def _pf(v, w=col_w):
            """Format a p-value: use scientific notation for very small values
            so the 3-sigma decision region (p ~ 1e-3 .. 1e-5) keeps precision
            inside a fixed-width column."""
            if not np.isfinite(v):
                return f"{'nan':>{w}}"
            if v < 1e-3:
                return f"{v:>{w}.3e}"
            return f"{v:>{w}.4f}"

        hdr_parts = [
            f"{'Invariant':<12}",
            f"{'ref':<8}",
            f"{'boot_mean':>{col_w}}",
            f"{'boot_sd':>{col_w}}",
            f"{'B':>{6}}",
            f"{'orig':>{col_w}}",
            f"{'ref_val':>{col_w}}",
            f"{'TS':>{col_w}}",
            f"{'abs_TS':>{col_w}}",
            f"{'p_value':>{col_w}}",
            f"{'decision':<26}",
        ]
        hdr = "  " + "  ".join(hdr_parts)
        fh.write(hdr + "\n")
        fh.write("  " + "-" * (len(hdr) - 2) + "\n")

        print(f"\n  --- Invariant × reference table ---")
        print(f"  {hdr.strip()}")
        print(f"  {'-' * (len(hdr) - 2)}")

        for metric in metric_names:
            bm = bootstrap_mean.get(metric, np.nan)
            bs = bootstrap_sd.get(metric, np.nan)
            bn = bootstrap_n.get(metric, 0)
            orig_v = results.get("orig", {}).get(metric, np.nan)
            for ref_key in ref_keys:
                ref_v = results.get(ref_key, {}).get(metric, np.nan)
                row_parts = [
                    f"{metric:<12}",
                    f"{ref_key:<8}",
                    _f(bm),
                    _f(bs),
                    f"{bn:>6}",
                    _f(orig_v),
                    _f(ref_v),
                    _f(test_stats[metric].get(ref_key, np.nan)),
                    _f(abs_test_stats[metric].get(ref_key, np.nan)),
                    _pf(p_values[metric].get(ref_key, np.nan)),
                    f"  {decisions[metric].get(ref_key, 'n/a'):<26}",
                ]
                row = "  " + "  ".join(row_parts)
                fh.write(row + "\n")
                print(f"  {row.strip()}")

        # --- Conclusion ---
        # Compact summary: per-metric decision against each reference, plus a
        # roll-up "any reference rejects" flag for quick visual scanning.
        fh.write(f"\nConclusion (TS threshold={args.ts_threshold:g}):\n")
        print(f"\n  Conclusion (TS threshold={args.ts_threshold:g}):")
        for metric in metric_names:
            per_ref = " | ".join(
                f"{rk}={decisions[metric].get(rk, 'n/a')}" for rk in ref_keys
            )
            any_reject = any(
                decisions[metric].get(rk) == "reject H0" for rk in ref_keys
            )
            rollup = "any reject" if any_reject else "no rejection"
            line = f"  {metric:<10}: {per_ref}   [{rollup}]"
            fh.write(line + "\n")
            print(line)

    if "LLE" in metric_names:
        try:
            from plot_lyap_k_output import plot_orig_lle_fit

            # Resolution order for the orig S(t) curves:
            # 1. ``lyap_keep_path`` written by compute_invariants for this run,
            # 2. ``{run_dir}/{base}_lyap.txt`` produced by Lambda_max.bat
            #    when this CLI was invoked from the .bat pipeline.
            run_dir = os.path.dirname(os.path.abspath(args.output_dir))
            cand_paths = [
                os.path.join(args.output_dir, f"{args.base}_orig_lyap.txt"),
                os.path.join(run_dir, f"{args.base}_lyap.txt"),
            ]
            cand_lyap = next((p for p in cand_paths if os.path.isfile(p)), None)
            if cand_lyap:
                out_png = os.path.join(args.output_dir, f"{args.base}_lyap_lle_fit.png")
                plot_orig_lle_fit(cand_lyap, out_png)
                print(f"  -> LLE diagnostic plot written to {out_png}")
            else:
                logger.warning(
                    "LLE plot skipped: no lyap_k output found in %s",
                    cand_paths,
                )
        except BaseException:
            logger.exception("Failed to write LLE diagnostic plot")

    print(f"\n  -> Summary written to {summary_file}")


if __name__ == "__main__":
    main()