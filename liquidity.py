# Find the optimal start point for you data, if the monthly percentage of zero returns is below 1 %.
# This is done since crypto_all_data.py dowlnoads since times, where liquidity was low,
# causing logarithmic returns to be zero values, which might cause issues for all other computations

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from config_loader import load_config, get_data_dir, get_results_dir, ensure_dir

# ==============================================================================
# CONFIGURATION SECTION
# ==============================================================================

# Example files in centralized data folder:
# C:\DCh\data\BTCUSD_BITSTAMP_1h_complete_logreturns.csv
# C:\DCh\data\ETHUSD_BITSTAMP_1h_complete_logreturns.csv
# C:\DCh\data\LTCUSD_BITSTAMP_1h_complete_logreturns.csv
# C:\DCh\data\XRPUSD_BITSTAMP_1h_complete_logreturns.csv
# C:\DCh\data\LINKUSD_BITSTAMP_1h_complete_logreturns.csv
# C:\DCh\data\ADAUSD_BITSTAMP_1h_complete_logreturns.csv


config = load_config()
data_dir = get_data_dir(config)
output_dir = ensure_dir(os.path.join(get_results_dir(config), "liquidity"))
files = [
    "BTCUSD_BITSTAMP_1h_complete_logreturns.csv",
    "ETHUSD_BITSTAMP_1h_complete_logreturns.csv",
    "LTCUSD_BITSTAMP_1h_complete_logreturns.csv",
    "XRPUSD_BITSTAMP_1h_complete_logreturns.csv",
    "LINKUSD_BITSTAMP_1h_complete_logreturns.csv",
    "DOGEUSD_BITSTAMP_1h_complete_logreturns.csv",
    "ADAUSD_BITSTAMP_1h_complete_logreturns.csv",
]

window_size = 720
tolerance = 1.0
create_liquidity_cut_files = True
create_backup_before_cut = True
analysis_end = pd.Timestamp("2026-05-02 20:00:00")


def cut_dataset_from_optimal_start(df, optimal_start, analysis_end):
    """
    Keep rows from the first acceptable-liquidity timestamp up to the analysis cutoff.
    """
    return df[(df.index >= optimal_start) & (df.index <= analysis_end)].copy()


def save_cut_dataset(cut_df, source_file_path, symbol, make_backup=True):
    """
    Save the cut dataset to the sibling *_logreturns_cut.csv file.
    """
    if make_backup:
        backup_path = source_file_path + ".bak"
        if not os.path.exists(backup_path):
            with open(source_file_path, "rb") as src, open(backup_path, "wb") as dst:
                dst.write(src.read())
            print(f"{symbol}: backup created -> {backup_path}")
        else:
            print(f"{symbol}: backup already exists -> {backup_path}")

    output_df = cut_df.reset_index()
    output_df["datetime_str"] = output_df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    output_df = output_df.drop(columns=["datetime", "is_zero", "zero_pct"], errors="ignore")
    cut_csv_path = source_file_path.replace("_logreturns.csv", "_logreturns_cut.csv")
    output_df.to_csv(cut_csv_path, index=False)
    print(f"{symbol}: cut CSV saved -> {cut_csv_path}")


def save_cut_dat_file(source_dat_path, rows_to_skip, rows_to_keep, symbol, make_backup=True):
    """
    Cut the matching .dat file by keeping the same row window selected in the CSV.
    """
    if not os.path.exists(source_dat_path):
        print(f"{symbol}: matching .dat file not found, skipping cut -> {source_dat_path}")
        return

    if make_backup:
        backup_path = source_dat_path + ".bak"
        if not os.path.exists(backup_path):
            with open(source_dat_path, "rb") as src, open(backup_path, "wb") as dst:
                dst.write(src.read())
            print(f"{symbol}: .dat backup created -> {backup_path}")
        else:
            print(f"{symbol}: .dat backup already exists -> {backup_path}")

    with open(source_dat_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    cut_lines = lines[rows_to_skip:rows_to_skip + rows_to_keep]
    cut_dat_path = source_dat_path.replace("_logreturns.dat", "_logreturns_cut.dat")
    with open(cut_dat_path, "w", encoding="utf-8") as f:
        f.writelines(cut_lines)
    print(f"{symbol}: cut .dat saved -> {cut_dat_path}")

for filename in files:
    file_path = os.path.join(data_dir, filename)
    symbol = filename.split("_")[0]
    if not os.path.exists(file_path):
        print(f"File not found, skipping: {file_path}")
        continue

    df = pd.read_csv(file_path)
    # Input timestamps are in ISO-like order: YYYY-MM-DD HH:MM:SS.
    # Using explicit format avoids parser ambiguity warnings and is faster.
    df['datetime'] = pd.to_datetime(df['datetime_str'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
    df = df.dropna(subset=['datetime'])
    df = df.sort_values('datetime').set_index('datetime')
    df['is_zero'] = (df['log_return'] == 0.0).astype(int)
    df['zero_pct'] = df['is_zero'].rolling(window=window_size).mean() * 100
    valid_data = df[(df['zero_pct'] < tolerance) & (df.index <= analysis_end)]

    if not valid_data.empty:
        optimal_start = valid_data.index[0]
        print(
            f"{symbol}: optimal start date = {optimal_start.date()} "
            f"(zero_pct < {tolerance}%), analysis end = {analysis_end}"
        )
        if create_liquidity_cut_files:
            rows_to_skip = int((df.index < optimal_start).sum())
            cut_df = cut_dataset_from_optimal_start(df, optimal_start, analysis_end)
            rows_to_keep = int(len(cut_df))
            if rows_to_keep == 0:
                print(f"{symbol}: no rows remain after applying analysis cutoff {analysis_end}.")
                continue
            save_cut_dataset(
                cut_df=cut_df,
                source_file_path=file_path,
                symbol=symbol,
                make_backup=create_backup_before_cut,
            )
            dat_path = file_path.replace("_logreturns.csv", "_logreturns.dat")
            save_cut_dat_file(
                source_dat_path=dat_path,
                rows_to_skip=rows_to_skip,
                rows_to_keep=rows_to_keep,
                symbol=symbol,
                make_backup=create_backup_before_cut,
            )
    else:
        optimal_start = None
        print(f"{symbol}: market never reached required liquidity threshold.")

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['zero_pct'], color='purple', label='Zero-return % (30-day avg)')
    plt.axhline(y=tolerance, color='red', linestyle='--', label=f'Threshold ({tolerance}%)')
    if optimal_start is not None:
        plt.axvline(x=optimal_start, color='green', linestyle='-', linewidth=2, label=f'Optimal start: {optimal_start.date()}')
    plt.axvline(x=analysis_end, color='orange', linestyle='--', linewidth=2, label=f'Analysis end: {analysis_end.date()}')
    plt.title(f'{symbol} Liquidity Analysis: When to start Cao\'s method?')
    plt.ylabel('% of zero hourly returns')
    plt.xlabel('Time')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_plot = os.path.join(output_dir, f"{symbol}_liquidity_zero_returns.png")
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"Saved plot: {out_plot}")
