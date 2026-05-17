import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# [Inference] The following line imports custom configuration and utility functions 
# from a local module named 'config_loader'. I cannot verify the internal workings of this module.
from config_loader import load_config, get_data_dir, get_results_dir, ensure_dir, prefer_liquidity_cut, tau_for_symbol_from_mutual

# ==========================================================
# 1. DELAY τ from mutual information (``mutual/_mi_summary.txt``); see config_loader.
# ==========================================================

# List of target CSV filenames containing log return data for various cryptocurrency pairs.
FILES = [
    "BTCUSD_BITSTAMP_1h_complete_logreturns.csv",
    "ETHUSD_BITSTAMP_1h_complete_logreturns.csv",
    "LTCUSD_BITSTAMP_1h_complete_logreturns.csv",
    "XRPUSD_BITSTAMP_1h_complete_logreturns.csv",
    "LINKUSD_BITSTAMP_1h_complete_logreturns.csv",
    "DOGEUSD_BITSTAMP_1h_complete_logreturns.csv",
    "ADAUSD_BITSTAMP_1h_complete_logreturns.csv",
]

# [Inference] Load the configuration parameters into a data structure.
config = load_config()

# [Inference] Retrieve the directory path where the source data files are stored.
data_dir = get_data_dir(config)

# [Inference] Construct the output directory path for phase 2D results and ensure it exists, creating it if necessary.
output_dir = ensure_dir(os.path.join(get_results_dir(config), "phase_2d"))

# Iterate over each filename defined in the FILES list.
for filename in FILES:
    # [Inference] Retrieve the actual file path, potentially applying a liquidity cut logic defined in the custom module.
    file_path = prefer_liquidity_cut(os.path.join(data_dir, filename))
    
    # Extract the base symbol (e.g., 'BTCUSD') by splitting the filename string at the first underscore.
    symbol = filename.split("_")[0]
    
    # [Inference] Retrieve the specific delay value (tau) for the current symbol, calculated previously via mutual information.
    tau = tau_for_symbol_from_mutual(symbol, config)
    
    # Check if the generated file path actually exists on the filesystem.
    if not os.path.exists(file_path):
        # Print a warning and skip to the next file if the current one is missing.
        print(f"File not found, skipping: {file_path}")
        continue

    # Announce the beginning of processing for the current symbol and its corresponding file.
    print(f"\nProcessing {symbol}: {file_path}")
    
    # Read the CSV file into a Pandas DataFrame.
    df = pd.read_csv(file_path)
    
    # Extract the 'log_return' column.
    # Convert it to string format to safely replace any comma decimal separators with period decimal separators.
    # Finally, cast the cleaned string data to floating-point numbers.
    x = df["log_return"].astype(str).str.replace(",", ".", regex=False).astype(float)
    
    # Filter the array to retain only finite values (removing NaN or infinite values).
    x = x[np.isfinite(x)].values
    print(f"Total series length: {len(x)} points")

    # Verify that the time series contains more data points than the delay parameter tau.
    if len(x) <= tau:
        # Skip plotting if there are insufficient data points to create the delayed series.
        print(f"Skipping {symbol}: too few points for tau={tau}.")
        continue

    # Construct the base time series (current return) by slicing the array from the start up to the tau index from the end.
    x_t = x[:-tau]
    
    # Construct the delayed time series (delayed return) by slicing the array from the tau index to the end.
    x_t_plus_tau = x[tau:]

    # Initialize a matplotlib figure with specific dimensions (8x8 inches).
    plt.figure(figsize=(8, 8))
    
    # Create a scatter plot of x(t) versus x(t + tau).
    # Use a small point size (s=1), a dark blue color, and alpha transparency (0.3) to visualize dense data clusters.
    plt.scatter(x_t, x_t_plus_tau, s=1, color='darkblue', alpha=0.3)
    
    # Set the plot title, embedding the tau value and using LaTeX formatting for the tau symbol.
    plt.title(f"{symbol} 2D phase-space projection ($\\tau$={tau})", fontsize=14)
    
    # Set the x-axis and y-axis labels.
    plt.xlabel('$x(t)$ (Current return)', fontsize=12)
    plt.ylabel(f'$x(t + {tau})$ (Delayed return)', fontsize=12)
    
    # Force the axes to have an equal aspect ratio, ensuring the geometric scale is identical for both axes.
    plt.axis('equal')
    
    # Add a semi-transparent dashed grid to the plot for easier reading of coordinate values.
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Adjust subplot parameters to give specified padding and prevent clipping of labels.
    plt.tight_layout()
    
    # Define the output file path for the generated plot.
    out_plot = os.path.join(output_dir, f"{symbol}_phase2d_tau{tau}.png")
    
    # Save the figure to the filesystem with a resolution of 150 dots per inch.
    plt.savefig(out_plot, dpi=150)
    
    # Close the current figure to free up memory before the next iteration.
    plt.close()
    
    # Confirm that the plot was successfully saved.
    print(f"Saved plot: {out_plot}")