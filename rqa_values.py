#!/usr/bin/env python3
"""
Batch RQA scalar extraction for all Bitstamp log-return series (PyRQA).

**Purpose**

Standalone companion to ``RQA.bat``: for each coin, load the active analysis
series (liquidity cut when present), compute recurrence quantification metrics
with **PyRQA**, and write ``<symbol>_rqa_metrics.txt`` under
``results/rqa_full`` or ``results/rqa_test_<N>``.

**Parameters (per coin)**

Read from ``Tisean_3.0.0/bin/_per_coin_settings.bat`` via :func:`config_loader.rqa_params_for_symbol`:

- ``TAU_RQA_<sym>`` — embedding delay (same τ as ``recurr.exe -d``).
- ``RAD_RQA_<sym>`` — **fallback** radius only; active radius is usually the
  **4th percentile** of embedded pairwise distances (aligned with ``rqa_radius.py``).
- ``W_D2_<sym>`` — Theiler window from ``theilers_w.bat``; passed to PyRQA as
  ``theiler_corrector`` after :func:`hypothesis.tisean_theiler_min_diagonal_k`.

**Metrics written**

``RR``, ``DET``, ``LAM``, ``MAXLINE``, ``ENTR``, ``TT``, plus custom ``TREND``
(diagonal-density slope vs lag, :func:`hypothesis.compute_rqa_trend`).

**Test mode**

When ``DCH_TEST_MODE`` is set, only the first ``DCH_TEST_POINTS`` rows are used
(same trim as ``RQA.bat`` / ``hypothesis.py --test_mode``).
"""

from __future__ import annotations

import os

import numpy as np

from config_loader import (
    dch_test_mode_from_env,
    dch_test_point_count,
    dch_test_results_tag,
    default_per_coin_settings_bat_path,
    ensure_dir,
    get_data_dir,
    get_results_dir,
    load_config,
    parse_per_coin_settings_bat,
    prefer_liquidity_cut,
    rqa_params_for_symbol,
)
from hypothesis_config import RQA_EMBEDDING_DIM, RQA_RADIUS_PERCENTILE_DEFAULT
from hypothesis import (
    RQA_RADIUS_MAX_VECTORS,
    compute_percentile_radius,
    compute_rqa_trend,
    format_rqa_radius,
    tisean_theiler_min_diagonal_k,
)
from pyrqa.analysis_type import Classic
from pyrqa.computation import RQAComputation
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius
from pyrqa.settings import Settings
from pyrqa.time_series import TimeSeries

# Active threshold: percentile of embedded pairwise Euclidean distances (rqa_tran.pdf).
# RAD_RQA_<sym> in _per_coin_settings.bat is fallback when percentile fails.
RADIUS_PERCENTILE = RQA_RADIUS_PERCENTILE_DEFAULT

# Default coin list (must match FILES= in RQA.bat and other pipeline scripts).
_DEFAULT_LOGRETURN_FILES = [
    "BTCUSD_BITSTAMP_1h_complete_logreturns.dat",
    "ETHUSD_BITSTAMP_1h_complete_logreturns.dat",
    "LTCUSD_BITSTAMP_1h_complete_logreturns.dat",
    "XRPUSD_BITSTAMP_1h_complete_logreturns.dat",
    "LINKUSD_BITSTAMP_1h_complete_logreturns.dat",
    "DOGEUSD_BITSTAMP_1h_complete_logreturns.dat",
    "ADAUSD_BITSTAMP_1h_complete_logreturns.dat",
]


def _load_dat_series(path: str) -> list[float]:
    """
    Read a TISEAN-style 1-column .dat file (or CSV-like second column).

    Skips blank lines; tolerates comma-separated rows by taking the last float field.
    """
    data: list[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data.append(float(stripped))
            except ValueError:
                parts = stripped.split(",")
                if len(parts) > 1:
                    try:
                        data.append(float(parts[1]))
                    except ValueError:
                        continue
    return data


def main() -> None:
    config = load_config()
    data_dir = get_data_dir(config)
    test_mode = dch_test_mode_from_env()
    out_root_name = f"rqa_{dch_test_results_tag()}" if test_mode else "rqa_full"
    output_root = ensure_dir(os.path.join(get_results_dir(config), out_root_name))

    # Per-coin tau, fallback radius, Theiler W (shared with d2 / lyap_k / hypothesis).
    bat_path = default_per_coin_settings_bat_path()
    per_coin = parse_per_coin_settings_bat(bat_path)
    if not os.path.isfile(bat_path):
        print(f"[WARN] Per-coin settings not found: {bat_path}")
        print("       Using defaults tau=3, r=0.005, W=0 (same fallbacks as RQA.bat).")
    elif not per_coin:
        print(f"[WARN] No assignments parsed from {bat_path}")

    for filename in _DEFAULT_LOGRETURN_FILES:
        input_path = prefer_liquidity_cut(os.path.join(data_dir, filename))
        symbol = filename.split("_")[0]
        tau, config_radius, theiler_w = rqa_params_for_symbol(symbol, per_coin)

        if not os.path.exists(input_path):
            print(f"Error: The file {input_path} was not found. Skipping.")
            continue

        print(f"\nLoading data from: {input_path}")
        data = _load_dat_series(input_path)

        # Match RQA.bat: test mode uses first DCH_TEST_POINTS rows only.
        if test_mode:
            data = data[: dch_test_point_count()]

        n_pts = len(data)
        if n_pts < 20:
            print(f"Skipping {symbol}: not enough points.")
            continue

        data_arr = np.asarray(data, dtype=float)

        # Dynamic radius (preferred) vs static RAD_RQA_<sym> from .bat.
        percentile_radius = compute_percentile_radius(
            data_arr,
            delay=tau,
            m=RQA_EMBEDDING_DIM,
            percentile=RADIUS_PERCENTILE,
            max_vectors=RQA_RADIUS_MAX_VECTORS,
        )
        if np.isfinite(percentile_radius) and percentile_radius > 0.0:
            radius = float(percentile_radius)
            radius_source = f"percentile({RADIUS_PERCENTILE:g}%)"
        else:
            radius = float(config_radius)
            radius_source = "config fallback"

        # Map TISEAN Theiler W to PyRQA diagonal corrector (see hypothesis_config).
        theiler_eff = max(0, int(theiler_w))
        pyrqa_theiler = tisean_theiler_min_diagonal_k(theiler_eff)
        print(
            f"Processing {symbol}: N={n_pts}, tau={tau}, "
            f"r={radius:.6g} ({radius_source}), "
            f"W={theiler_eff} (TISEAN -t; PyRQA/TREND corrector={pyrqa_theiler}), "
            f"m={RQA_EMBEDDING_DIM}; config radius={config_radius:g}"
        )

        # Classic RQA with fixed radius neighbourhood in embedded space.
        time_series = TimeSeries(
            data_arr,
            embedding_dimension=RQA_EMBEDDING_DIM,
            time_delay=tau,
        )
        settings = Settings(
            time_series,
            analysis_type=Classic,
            neighbourhood=FixedRadius(radius),
            similarity_measure=EuclideanMetric,
            theiler_corrector=pyrqa_theiler,
        )
        computation = RQAComputation.create(settings, verbose=False)
        result = computation.run()

        # Custom diagonal-structure trend (not a built-in PyRQA scalar).
        trend = compute_rqa_trend(
            data_arr,
            delay=tau,
            radius=radius,
            min_k=pyrqa_theiler,
            m=RQA_EMBEDDING_DIM,
        )

        print(f"\n--- RQA RESULTS ({symbol}) ---")
        print(f"RR       = {result.recurrence_rate:.6f}")
        print(f"DET      = {result.determinism:.6f}")
        print(f"LAM      = {result.laminarity:.6f}")
        print(f"MAXLINE  = {result.longest_diagonal_line}")
        print(f"ENTR     = {result.entropy_diagonal_lines:.6f}")
        print(f"TT       = {result.trapping_time:.6f}")
        print(f"TREND    = {trend:.6f}")

        # Run folder naming matches RQA.bat / recurr output layout.
        radius_id = format_rqa_radius(radius)
        run_id = f"run2_tau{tau}_r{radius_id}"
        out_dir = os.path.join(output_root, f"{symbol}_{run_id}")
        os.makedirs(out_dir, exist_ok=True)

        out_txt = os.path.join(out_dir, f"{symbol}_rqa_metrics.txt")
        with open(out_txt, "w", encoding="utf-8") as out:
            out.write(f"RQA RESULTS ({symbol})\n")
            out.write(
                f"# PyRQA params: tau={tau}, radius={radius:.10g} ({radius_source}), "
                f"config_radius={config_radius:g}, m={RQA_EMBEDDING_DIM}, "
                f"W={theiler_eff} (per-coin W_D2 from theilers_w.bat), "
                f"percentile={RADIUS_PERCENTILE:g}%, "
                f"settings_file=_per_coin_settings.bat\n"
            )
            out.write(f"RR={result.recurrence_rate:.6f}\n")
            out.write(f"DET={result.determinism:.6f}\n")
            out.write(f"LAM={result.laminarity:.6f}\n")
            out.write(f"MAXLINE={result.longest_diagonal_line}\n")
            out.write(f"ENTR={result.entropy_diagonal_lines:.6f}\n")
            out.write(f"TT={result.trapping_time:.6f}\n")
            out.write(f"TREND={trend:.6f}\n")
        print(f"Saved metrics: {out_txt}")


if __name__ == "__main__":
    main()
