import glob
import os
from config_loader import load_config, get_data_dir, get_results_dir, ensure_dir, prefer_liquidity_cut
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation

def main():
    config = load_config()
    data_dir = get_data_dir(config)
    test_mode = str(os.environ.get("DCH_TEST_MODE", "false")).strip().lower() in {"1", "true", "yes", "y", "on"}
    out_root_name = "rqa_test_2000" if test_mode else "rqa_full"
    output_root = ensure_dir(os.path.join(get_results_dir(config), out_root_name))
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
                    # Backward-compatible fallback for CSV-like rows: use second column.
                    parts = stripped.split(",")
                    if len(parts) > 1:
                        try:
                            data.append(float(parts[1]))
                        except ValueError:
                            continue

        data = data[:1000]
        print(f"Processing {symbol} with {len(data)} data points...")
        if len(data) < 20:
            print(f"Skipping {symbol}: not enough points.")
            continue

        time_series = TimeSeries(data, embedding_dimension=5, time_delay=2)
        settings = Settings(
            time_series,
            analysis_type=Classic,
            neighbourhood=FixedRadius(0.01),
            similarity_measure=EuclideanMetric,
            theiler_corrector=1,
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

        run_dirs = sorted(glob.glob(os.path.join(output_root, f"{symbol}_run2_*")))
        out_dir = run_dirs[0] if run_dirs else output_root
        out_txt = os.path.join(out_dir, f"{symbol}_rqa_metrics.txt")
        with open(out_txt, "w", encoding="utf-8") as out:
            out.write(f"RQA RESULTS ({symbol})\n")
            out.write(f"RR={result.recurrence_rate:.6f}\n")
            out.write(f"DET={result.determinism:.6f}\n")
            out.write(f"LAM={result.laminarity:.6f}\n")
            out.write(f"MAXLINE={result.longest_diagonal_line}\n")
            out.write(f"ENTR={result.entropy_diagonal_lines:.6f}\n")
            out.write(f"TT={result.trapping_time:.6f}\n")
        print(f"Saved metrics: {out_txt}")

if __name__ == "__main__":
    main()
