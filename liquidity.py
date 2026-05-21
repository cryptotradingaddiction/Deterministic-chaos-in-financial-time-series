# Find the optimal start point for your data if the monthly percentage of zero returns
# is below 1 %. This is done since crypto_data_all.py downloads from times when liquidity
# was low, causing logarithmic returns to be zero values, which might cause issues for
# all other computations.
#
# Cut window is controlled by config.yaml -> liquidity (see config.example.yaml).
# Optional LOCAL_LIQUIDITY_PATCH below merges on top of YAML for quick experiments.

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from config_loader import (
    ensure_dir,
    get_data_dir,
    get_liquidity_settings,
    get_results_dir,
    load_config,
    pipeline_logreturn_files,
)

# Merged into settings from config.yaml (same keys as under `liquidity:`).
# Example:
#   LOCAL_LIQUIDITY_PATCH = {
#       "mode": "fixed",
#       "fixed_tail_points": 17520,
#   }
LOCAL_LIQUIDITY_PATCH = {}

# Per-coin log-return CSVs (timestamp + value column). Suffix and symbol list
# come from ``pipeline_logreturn_files`` so adding a coin is a one-line edit
# in ``config_loader.PIPELINE_SYMBOLS``. Resolved paths live under
# ``get_data_dir(config)`` regardless of project location.
FILES = pipeline_logreturn_files(ext="csv")


def cut_dataset_from_liquidity_start(df, optimal_start, analysis_end):
    """Keep rows from the first acceptable-liquidity timestamp through analysis_end (inclusive).

    If ``analysis_end`` is None, the slice runs through the last row of ``df``.
    """
    end_ts = analysis_end if analysis_end is not None else df.index.max()
    return df[(df.index >= optimal_start) & (df.index <= end_ts)].copy()


def cut_dataset_last_n_rows(df, n):
    """Keep the last ``n`` rows of ``df`` (must be sorted chronologically)."""
    n = int(n)
    if n <= 0:
        return df.iloc[0:0].copy()
    if len(df) <= n:
        return df.copy()
    return df.iloc[-n:].copy()


def save_cut_dataset(cut_df, source_file_path, symbol, make_backup=True):
    """Save the cut dataset to the sibling *_logreturns_cut.csv file."""
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
    """Cut the matching .dat file by keeping the same row window selected in the CSV."""
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

    cut_lines = lines[rows_to_skip : rows_to_skip + rows_to_keep]
    cut_dat_path = source_dat_path.replace("_logreturns.dat", "_logreturns_cut.dat")
    with open(cut_dat_path, "w", encoding="utf-8") as f:
        f.writelines(cut_lines)
    print(f"{symbol}: cut .dat saved -> {cut_dat_path}")


def rows_window_for_cut(df, cut_df):
    """Return (rows_to_skip, rows_to_keep) for .dat alignment with full sorted ``df``."""
    if cut_df is None or cut_df.empty:
        return 0, 0
    first_ts = cut_df.index[0]
    rows_to_skip = int((df.index < first_ts).sum())
    rows_to_keep = int(len(cut_df))
    return rows_to_skip, rows_to_keep


def main():
    config = load_config()
    liq = get_liquidity_settings(config)
    liq.update(LOCAL_LIQUIDITY_PATCH)

    mode = liq["mode"]
    window_size = int(liq["window_size"])
    tolerance = float(liq["tolerance"])
    create_cut_files = bool(liq["create_cut_files"])
    create_backup = bool(liq["create_backup_before_cut"])

    ae_raw = liq.get("analysis_end")
    analysis_end = pd.Timestamp(ae_raw) if ae_raw not in (None, "", False) else None

    tail_n = int(liq.get("fixed_tail_points", 17520))

    if mode == "fixed":
        if tail_n < 1:
            raise SystemExit("liquidity.fixed_tail_points must be >= 1 for mode 'fixed'.")

    data_dir = get_data_dir(config)
    output_dir = ensure_dir(os.path.join(get_results_dir(config), "liquidity"))

    for filename in FILES:
        file_path = os.path.join(data_dir, filename)
        symbol = filename.split("_")[0]
        if not os.path.exists(file_path):
            print(f"File not found, skipping: {file_path}")
            continue

        df = pd.read_csv(file_path)
        df["datetime"] = pd.to_datetime(
            df["datetime_str"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )
        df = df.dropna(subset=["datetime"])
        df = df.sort_values("datetime").set_index("datetime")
        df["is_zero"] = (df["log_return"] == 0.0).astype(int)
        df["zero_pct"] = df["is_zero"].rolling(window=window_size).mean() * 100

        optimal_start = None
        cut_df = None
        plot_right_end = None

        if mode == "fixed":
            cut_df = cut_dataset_last_n_rows(df, tail_n)
            plot_right_end = cut_df.index.max() if not cut_df.empty else None
            if cut_df.empty:
                print(f"{symbol}: fixed window (last {tail_n} points) is empty for this file.")
            else:
                optimal_start = cut_df.index[0]
                print(
                    f"{symbol}: fixed tail cut — last {len(cut_df)} rows "
                    f"({cut_df.index[0]} … {cut_df.index[-1]})"
                )
        else:
            end_cap = analysis_end if analysis_end is not None else df.index.max()
            plot_right_end = end_cap
            valid_data = df[(df["zero_pct"] < tolerance) & (df.index <= end_cap)]
            if not valid_data.empty:
                optimal_start = valid_data.index[0]
                end_label = analysis_end if analysis_end is not None else "series end"
                print(
                    f"{symbol}: optimal start = {optimal_start} "
                    f"(zero_pct < {tolerance}%), window end = {end_label}"
                )
                cut_df = cut_dataset_from_liquidity_start(df, optimal_start, analysis_end)
            else:
                print(f"{symbol}: market never reached required liquidity threshold (before end cap).")

        if cut_df is not None and not cut_df.empty and create_cut_files:
            rows_to_skip, rows_to_keep = rows_window_for_cut(df, cut_df)
            if rows_to_keep == 0:
                print(f"{symbol}: no rows remain after cut.")
            else:
                save_cut_dataset(
                    cut_df=cut_df,
                    source_file_path=file_path,
                    symbol=symbol,
                    make_backup=create_backup,
                )
                dat_path = file_path.replace("_logreturns.csv", "_logreturns.dat")
                save_cut_dat_file(
                    source_dat_path=dat_path,
                    rows_to_skip=rows_to_skip,
                    rows_to_keep=rows_to_keep,
                    symbol=symbol,
                    make_backup=create_backup,
                )

        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df["zero_pct"], color="purple", label="Zero-return % (rolling)")
        plt.axhline(y=tolerance, color="r", linestyle="--", label=f"Threshold ({tolerance}%)")
        if optimal_start is not None:
            plt.axvline(
                x=optimal_start,
                color="green",
                linestyle="-",
                linewidth=2,
                label=f"Cut start: {optimal_start}",
            )
        if plot_right_end is not None:
            plt.axvline(
                x=plot_right_end,
                color="orange",
                linestyle="--",
                linewidth=2,
                label=f"Cut end cap: {plot_right_end}",
            )
        title_mode = "fixed (last N points)" if mode == "fixed" else "liquidity"
        plt.title(f"{symbol} — {title_mode} window")
        plt.ylabel("% of zero hourly returns")
        plt.xlabel("Time")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_plot = os.path.join(output_dir, f"{symbol}_liquidity_zero_returns.png")
        plt.savefig(out_plot, dpi=150)
        plt.close()
        print(f"Saved plot: {out_plot}")


if __name__ == "__main__":
    main()
