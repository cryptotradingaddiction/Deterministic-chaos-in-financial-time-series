import os

from config_loader import (
    default_per_coin_settings_bat_path,
    ensure_dir,
    get_data_dir,
    get_results_dir,
    load_config,
    parse_per_coin_settings_bat,
    prefer_liquidity_cut,
    rqa_params_for_symbol,
)
from pyrqa.analysis_type import Classic
from pyrqa.computation import RQAComputation
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius
from pyrqa.settings import Settings
from pyrqa.time_series import TimeSeries

# Same embedding dimension as `hypothesis.compute_pyrqa_metrics` and `RQA.bat` (EMBED_DIM=3).
RQA_EMBEDDING_DIM = 3


def main():
    config = load_config()
    data_dir = get_data_dir(config)
    test_mode = str(os.environ.get("DCH_TEST_MODE", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    out_root_name = "rqa_test_2000" if test_mode else "rqa_full"
    output_root = ensure_dir(os.path.join(get_results_dir(config), out_root_name))

    bat_path = default_per_coin_settings_bat_path()
    per_coin = parse_per_coin_settings_bat(bat_path)
    if not os.path.isfile(bat_path):
        print(f"[WARN] Per-coin settings not found: {bat_path}")
        print("       Using defaults tau=3, r=0.01, W=0 (same fallbacks as RQA.bat).")
    elif not per_coin:
        print(f"[WARN] No assignments parsed from {bat_path}")

    files = [
        "BTCUSD_BITSTAMP_1h_complete_logreturns.dat",
        "ETHUSD_BITSTAMP_1h_complete_logreturns.dat",
        "LTCUSD_BITSTAMP_1h_complete_logreturns.dat",
        "XRPUSD_BITSTAMP_1h_complete_logreturns.dat",
        "LINKUSD_BITSTAMP_1h_complete_logreturns.dat",
        "DOGEUSD_BITSTAMP_1h_complete_logreturns.dat",
        "ADAUSD_BITSTAMP_1h_complete_logreturns.dat",
    ]

    for filename in files:
        input_path = prefer_liquidity_cut(os.path.join(data_dir, filename))
        symbol = filename.split("_")[0]
        tau, radius, theiler_w = rqa_params_for_symbol(symbol, per_coin)

        if not os.path.exists(input_path):
            print(f"Error: The file {input_path} was not found. Skipping.")
            continue

        data = []
        print(f"\nLoading data from: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
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

        # Align length with RQA.bat: TEST uses first 2000 points; FULL uses entire series.
        if test_mode:
            data = data[:2000]

        n_pts = len(data)
        print(
            f"Processing {symbol}: N={n_pts}, tau={tau}, r={radius}, "
            f"W={theiler_w}, m={RQA_EMBEDDING_DIM} (from _per_coin_settings.bat)"
        )
        if n_pts < 20:
            print(f"Skipping {symbol}: not enough points.")
            continue

        time_series = TimeSeries(
            data,
            embedding_dimension=RQA_EMBEDDING_DIM,
            time_delay=tau,
        )
        settings = Settings(
            time_series,
            analysis_type=Classic,
            neighbourhood=FixedRadius(radius),
            similarity_measure=EuclideanMetric,
            theiler_corrector=theiler_w,
        )
        computation = RQAComputation.create(settings, verbose=False)
        result = computation.run()
        result.min_diagonal_line_length = 2
        result.min_vertical_line_length = 2

        print(f"\n--- RQA RESULTS ({symbol}) ---")
        print(f"RR       = {result.recurrence_rate:.6f}")
        print(f"DET      = {result.determinism:.6f}")
        print(f"LAM      = {result.laminarity:.6f}")
        print(f"MAXLINE  = {result.longest_diagonal_line}")
        print(f"ENTR     = {result.entropy_diagonal_lines:.6f}")
        print(f"TT       = {result.trapping_time:.6f}")

        run_id = f"run2_tau{tau}_r{radius}"
        out_dir = os.path.join(output_root, f"{symbol}_{run_id}")
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        out_txt = os.path.join(out_dir, f"{symbol}_rqa_metrics.txt")
        with open(out_txt, "w", encoding="utf-8") as out:
            out.write(f"RQA RESULTS ({symbol})\n")
            out.write(
                f"# PyRQA params: tau={tau}, radius={radius}, m={RQA_EMBEDDING_DIM}, "
                f"Theiler_W={theiler_w}, settings_file=_per_coin_settings.bat\n"
            )
            out.write(f"RR={result.recurrence_rate:.6f}\n")
            out.write(f"DET={result.determinism:.6f}\n")
            out.write(f"LAM={result.laminarity:.6f}\n")
            out.write(f"MAXLINE={result.longest_diagonal_line}\n")
            out.write(f"ENTR={result.entropy_diagonal_lines:.6f}\n")
            out.write(f"TT={result.trapping_time:.6f}\n")
        print(f"Saved metrics: {out_txt}")


if __name__ == "__main__":
    main()
