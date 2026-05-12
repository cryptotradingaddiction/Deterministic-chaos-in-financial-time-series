import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from config_loader import load_config, get_data_dir, get_results_dir, ensure_dir, prefer_liquidity_cut

# ==========================================================
# 1. SETTING THE DELAY PARAMETER
# ==========================================================
# Individual tau values per symbol (can be tuned independently).
TAU_BY_SYMBOL = {
    "BTCUSD": 5,
    "ETHUSD": 5,
    "LTCUSD": 2,
    "XRPUSD": 2,
    "LINKUSD": 4,
    "DOGEUSD": 6,
    "ADAUSD": 2,
}

FILES = [
    "BTCUSD_BITSTAMP_1h_complete_logreturns.csv",
    "ETHUSD_BITSTAMP_1h_complete_logreturns.csv",
    "LTCUSD_BITSTAMP_1h_complete_logreturns.csv",
    "XRPUSD_BITSTAMP_1h_complete_logreturns.csv",
    "LINKUSD_BITSTAMP_1h_complete_logreturns.csv",
    "DOGEUSD_BITSTAMP_1h_complete_logreturns.csv",
    "ADAUSD_BITSTAMP_1h_complete_logreturns.csv",
]

config = load_config()
data_dir = get_data_dir(config)
output_dir = ensure_dir(os.path.join(get_results_dir(config), "phase_2d"))

for filename in FILES:
    file_path = prefer_liquidity_cut(os.path.join(data_dir, filename))
    symbol = filename.split("_")[0]
    tau = TAU_BY_SYMBOL.get(symbol, 4)
    if not os.path.exists(file_path):
        print(f"File not found, skipping: {file_path}")
        continue

    print(f"\nProcessing {symbol}: {file_path}")
    df = pd.read_csv(file_path)
    x = df["log_return"].astype(str).str.replace(",", ".", regex=False).astype(float)
    x = x[np.isfinite(x)].values
    print(f"Total series length: {len(x)} points")

    if len(x) <= tau:
        print(f"Skipping {symbol}: too few points for tau={tau}.")
        continue

    x_t = x[:-tau]
    x_t_plus_tau = x[tau:]

    plt.figure(figsize=(8, 8))
    plt.scatter(x_t, x_t_plus_tau, s=1, color='darkblue', alpha=0.3)
    plt.title(f"{symbol} 2D phase-space projection ($\\tau$={tau})", fontsize=14)
    plt.xlabel('$x(t)$ (Current return)', fontsize=12)
    plt.ylabel(f'$x(t + {tau})$ (Delayed return)', fontsize=12)
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    out_plot = os.path.join(output_dir, f"{symbol}_phase2d_tau{tau}.png")
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"Saved plot: {out_plot}")
