import re
import subprocess
import tempfile
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config_loader import load_config, get_data_dir, get_results_dir, ensure_dir, prefer_liquidity_cut


def resolve_lyap_k_exe():
    """Path to TISEAN lyap_k (Kantz algorithm)."""
    root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(root, "Tisean_3.0.0", "bin", "lyap_k.exe"),
        os.path.join(root, "Tisean_3.0.0", "bin", "lyap_k"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    from_env = os.environ.get("TISEAN_BIN")
    if from_env:
        p = os.path.join(from_env, "lyap_k.exe")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "TISEAN lyap_k not found. Expected Tisean_3.0.0/bin/lyap_k.exe next to predictability.py "
        "or TISEAN_BIN pointing at the bin directory."
    )


def run_kantz_lyap_k(data_1d: np.ndarray, m: int, tau: int, out_txt: str) -> None:
    """
    Kantz largest Lyapunov via TISEAN lyap_k (same family as hypothesis.bat).
    Fixed embedding dimension m, delay tau; writes multi-block .lyap text file.
    """
    exe = resolve_lyap_k_exe()
    fd, tmp_dat = tempfile.mkstemp(suffix=".dat", text=False)
    os.close(fd)
    try:
        np.savetxt(tmp_dat, data_1d, fmt="%.18e")
        cmd = [
            exe,
            f"-d{tau}",
            f"-m{m}",
            f"-M{m}",
            "-r0.0005",
            "-R0.05",
            "-#5",
            "-n",
            "500",
            "-s",
            "100",
            "-o",
            out_txt,
            tmp_dat,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    finally:
        try:
            os.remove(tmp_dat)
        except OSError:
            pass


def parse_kantz_first_block_for_dim(path: str, target_dim: int):
    """
    lyap_k output: blocks starting with '#epsilon= ... dim= k', then rows:
    iteration, log(stretching factor), count.
    Returns (iterations, log_curve) for the first block matching target_dim
    (first length scale when several blocks share the same dim).
    """
    block_dim = None
    rows = []

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            ls = line.strip()
            if ls.startswith("#"):
                if block_dim == target_dim and rows:
                    arr = np.array(rows, dtype=float)
                    return arr[:, 0], arr[:, 1]
                rows = []
                md = re.search(r"dim=\s*(\d+)", ls)
                block_dim = int(md.group(1)) if md else None
                continue
            if block_dim != target_dim:
                continue
            parts = ls.split()
            if len(parts) >= 2:
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass

    if block_dim == target_dim and rows:
        arr = np.array(rows, dtype=float)
        return arr[:, 0], arr[:, 1]
    return None


# =====================================================================
# 1. SETUP AND DATA LOADING
# =====================================================================

SETTINGS_BY_SYMBOL = {
    "BTCUSD": {"m": 4, "tau": 2},
    "ETHUSD": {"m": 4, "tau": 2},
    "LTCUSD": {"m": 4, "tau": 4},
    "XRPUSD": {"m": 4, "tau": 3},
    "LINKUSD": {"m": 4, "tau": 4},
    "DOGEUSD": {"m": 4, "tau": 3},
    "ADAUSD": {"m": 4, "tau": 2},
}
epsilon = 1e-5  # Initial precision/error (for calculating T)
L = 1e-2        # Maximum tolerated error (for calculating T)

FILES = [
    "BTCUSD_BITSTAMP_1h_complete_logreturns.csv",
    "ETHUSD_BITSTAMP_1h_complete_logreturns.csv",
    "LTCUSD_BITSTAMP_1h_complete_logreturns.csv",
    "XRPUSD_BITSTAMP_1h_complete_logreturns.csv",
    "LINKUSD_BITSTAMP_1h_complete_logreturns.csv",
    "DOGEUSD_BITSTAMP_1h_complete_logreturns.csv",
    "ADAUSD_BITSTAMP_1h_complete_logreturns.csv",
]

def main():
    config = load_config()
    data_dir = get_data_dir(config)
    output_dir = ensure_dir(os.path.join(get_results_dir(config), "predictability"))

    for filename in FILES:
        file_path = prefer_liquidity_cut(os.path.join(data_dir, filename))
        symbol = filename.split("_")[0]
        if not os.path.exists(file_path):
            print(f"File not found, skipping: {file_path}")
            continue

        params = SETTINGS_BY_SYMBOL.get(symbol, {"m": 4, "tau": 4})
        m = params["m"]
        tau = params["tau"]

        print(f"\nLoading {symbol} from {file_path} (m={m}, tau={tau})...")
        df = pd.read_csv(file_path)
        data = df["log_return"].astype(str).str.replace(",", ".", regex=False).astype(float).values
        data = data[np.isfinite(data)]
        if len(data) < 20:
            print(f"Skipping {symbol}: not enough data.")
            continue

        lyap_txt = os.path.join(output_dir, f"_{symbol}_kantz_work.lyap.txt")
        print("Executing Kantz algorithm (TISEAN lyap_k)... (this might take a while)")
        try:
            run_kantz_lyap_k(data, m, tau, lyap_txt)
        except subprocess.CalledProcessError as e:
            print(f"Skipping {symbol}: lyap_k failed: {e.stderr or e}")
            continue
        except FileNotFoundError as e:
            print(e)
            raise SystemExit(1)

        parsed = parse_kantz_first_block_for_dim(lyap_txt, m)
        if parsed is None:
            print(f"Skipping {symbol}: no lyap_k block found for dim={m}.")
            continue

        x_axis, divergence_curve = parsed
        fit_start = 2
        fit_end = min(10, len(divergence_curve))
        if len(divergence_curve) <= fit_start or fit_end - fit_start < 2:
            print(f"Skipping {symbol}: divergence curve too short for fit.")
            continue

        slope, intercept = np.polyfit(x_axis[fit_start:fit_end], divergence_curve[fit_start:fit_end], 1)
        lambda_kantz = slope

        print(f"==================================================")
        print(f"{symbol} Lyapunov Exponent (Kantz / TISEAN lyap_k): lambda = {lambda_kantz:.6f}")
        print(f"==================================================")

        if lambda_kantz <= 0:
            print("Exponent is negative or zero - system is not chaotic, T makes no sense.")
        else:
            T = (1 / lambda_kantz) * math.log(L / epsilon)
            print(f"{symbol} Predictability Time (T): {T:.2f} hours ({T/24:.2f} days)")

        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, divergence_curve, "b-o", label="Kantz S(t) (log stretch)", markersize=4)
        fit_line = (slope * x_axis) + intercept
        plt.plot(x_axis[fit_start:fit_end], fit_line[fit_start:fit_end], "r-", linewidth=3, label=f"Linear fit (lambda={lambda_kantz:.4f})")
        plt.axvspan(fit_start, fit_end, color="red", alpha=0.1, label="Selected scaling region")
        plt.title(f"{symbol} Kantz method (TISEAN lyap_k), m={m}, tau={tau}")
        plt.xlabel("Time steps forward (t)")
        plt.ylabel("Logarithm of stretching factor (Lyap curve)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        out_plot = os.path.join(output_dir, f"{symbol}_predictability_kantz_m{m}_tau{tau}.png")
        plt.savefig(out_plot, dpi=150)
        plt.close()
        print(f"Saved plot: {out_plot}")


if __name__ == "__main__":
    main()
