import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from config_loader import load_config, get_data_dir, get_results_dir, ensure_dir, prefer_liquidity_cut

# ==========================================================
# 1. SETTING RECONSTRUCTION PARAMETERS
# ==========================================================
M = 3        # Embedding dimension (must be 3 for a 3D graph)
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

# ==========================================================
# 3. PHASE SPACE RECONSTRUCTION (DELAYED COORDINATES)
# ==========================================================
def delay_embedding(data, m, tau):
    """
    Creates a matrix of delayed coordinates for phase space reconstruction.
    """
    N = len(data)
    # Total length of the "window" for one point in phase space
    max_delay = (m - 1) * tau
    
    if max_delay >= N:
        raise ValueError("Time series is too short for the specified m and tau.")
        
    # Number of valid points we can create
    num_points = N - max_delay
    
    # Preallocation of the matrix for points in phase space
    embedded = np.zeros((num_points, m))
    
    for i in range(m):
        # Coordinates: x(t), x(t+tau), x(t+2*tau)...
        embedded[:, i] = data[i * tau : i * tau + num_points]
        
    return embedded

config = load_config()
data_dir = get_data_dir(config)
output_dir = ensure_dir(os.path.join(get_results_dir(config), "phase_3d"))

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

    try:
        phase_space = delay_embedding(x, M, tau)
    except ValueError as e:
        print(f"Skipping {symbol}: {e}")
        continue

    print(f"Created {phase_space.shape[0]} points in {M}D space.")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(phase_space[:, 0], phase_space[:, 1], phase_space[:, 2], color='blue', linewidth=0.3, alpha=0.7)
    ax.set_title(f"{symbol} phase-space reconstruction (m={M}, $\\tau$={tau})")
    ax.set_xlabel('$x(t)$')
    ax.set_ylabel(f'$x(t + {tau})$')
    ax.set_zlabel(f'$x(t + {2*tau})$')
    plt.tight_layout()
    out_plot = os.path.join(output_dir, f"{symbol}_phase3d_tau{tau}_m{M}.png")
    plt.savefig(out_plot, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_plot}")
