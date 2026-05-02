"""
Calculation of the empirical time window tau_w according to formula (8.30) from literature:
    tau_w ~= sqrt( 3 * mean(x**2) / mean(dx) )
where:
    mean(x**2) = mean of squared signal
    mean(dx)   = mean of the numerical derivative over the whole time series
"""

import numpy as np
from scipy.signal import find_peaks
import os
from config_loader import load_config, get_data_dir, get_results_dir, ensure_dir, prefer_liquidity_cut
from report_helper import Reporter, append_summary_row

SUMMARY_FILE = "_tau_w_summary.txt"
SUMMARY_HEADER = (
    f"{'Symbol':<10} {'N':>10} {'mean(x^2)':>14} {'mean(dx)':>14} "
    f"{'tau_w':>10} {'tau_p':>10} {'final_tau_w':>14}"
)


config = load_config()
data_dir = get_data_dir(config)
output_dir = ensure_dir(os.path.join(get_results_dir(config), "tau_w"))
files = [
    "BTCUSD_BITSTAMP_1h_complete_logreturns.dat",
    "ETHUSD_BITSTAMP_1h_complete_logreturns.dat",
    "LTCUSD_BITSTAMP_1h_complete_logreturns.dat",
    "XRPUSD_BITSTAMP_1h_complete_logreturns.dat",
    "LINKUSD_BITSTAMP_1h_complete_logreturns.dat",
    "DOGEUSD_BITSTAMP_1h_complete_logreturns.dat",
    "ADAUSD_BITSTAMP_1h_complete_logreturns.dat",
]

# Reset the aggregated summary so each script run gets a clean table.
try:
    os.remove(os.path.join(output_dir, SUMMARY_FILE))
except FileNotFoundError:
    pass

for filename in files:
    file_path = prefer_liquidity_cut(os.path.join(data_dir, filename))
    symbol = filename.split("_")[0]
    if not os.path.exists(file_path):
        print(f"File not found, skipping: {file_path}")
        continue

    x = np.loadtxt(file_path)
    mean_x2 = float(np.mean(x ** 2))
    dx = np.gradient(x)
    mean_dx = float(np.mean(dx))
    radicand = 3 * mean_x2 / mean_dx if mean_dx != 0 else np.nan
    tau_w = float(np.sqrt(radicand)) if np.isfinite(radicand) and radicand > 0 else float("nan")

    peaks, _ = find_peaks(x)
    if len(peaks) > 1:
        peak_distances = np.diff(peaks)
        tau_p = float(np.mean(peak_distances))
        peak_warning = ""
    else:
        tau_p = 0.0
        peak_warning = "Warning - not enough peaks to determine tau_p."

    final_tau_w = max(tau_w, tau_p) if np.isfinite(tau_w) else tau_p

    rep = Reporter()
    rep.add("=" * 80)
    rep.add(f"tau_w analysis - {symbol}")
    rep.add(f"Input file : {file_path}")
    rep.add(f"Series len : {len(x)}")
    rep.add("=" * 80)
    if peak_warning:
        rep.add(f"{symbol}: {peak_warning}")
    rep.add(f"--- Results ({symbol}) ---")
    rep.add(f"mean(x^2) = {mean_x2:.6e}")
    rep.add(f"mean(dx)  = {mean_dx:.6e}")
    if np.isfinite(tau_w):
        rep.add(f"Semi-empirical tau_w ~= {tau_w:.4f}")
    else:
        rep.add("Semi-empirical tau_w is undefined (radicand <= 0 or non-finite).")
    rep.add(f"Average period tau_p = {tau_p:.4f}")
    rep.add(f"Final tau_w = max(tau_w, tau_p) = {final_tau_w:.4f}")

    out_txt = rep.write(output_dir, f"{symbol}_tau_w_results.txt")
    print(f"Saved text report: {out_txt}")

    tau_w_str = f"{tau_w:.4f}" if np.isfinite(tau_w) else "nan"
    summary_row = (
        f"{symbol:<10} {len(x):>10d} {mean_x2:>14.6e} {mean_dx:>14.6e} "
        f"{tau_w_str:>10} {tau_p:>10.4f} {final_tau_w:>14.4f}"
    )
    summary_path = append_summary_row(
        output_dir, SUMMARY_FILE, SUMMARY_HEADER, summary_row
    )
    print(f"Appended row to summary: {summary_path}")
