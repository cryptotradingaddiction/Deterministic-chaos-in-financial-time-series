import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Import Axes3D specifically to register the '3d' projection for matplotlib. 
# The '# noqa: F401' comment tells linters (like flake8) to ignore the "imported but unused" warning.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

# [Inference] Import configuration and utility functions from a local custom module. 
# These handle environment paths, data preprocessing preferences, and dynamic parameter retrieval.
from config_loader import (
    load_config,
    get_data_dir,
    get_results_dir,
    ensure_dir,
    pipeline_logreturn_files,
    prefer_liquidity_cut,
    tau_for_symbol_from_mutual,
)

# ==========================================================
# 1. DELAY τ from mutual information (``mutual/_mi_summary.txt``); see config_loader.
# ==========================================================

# Define the embedding dimension 'm'. 
# Since we want to visualize the phase space in a 3D graph, we set M to 3.
# This means each point in the reconstructed phase space will have 3 coordinates.
M = 3        

# Per-coin log-return CSVs (timestamp + value); centralized in config_loader.
FILES = pipeline_logreturn_files(ext="csv")

# ==========================================================
# 3. PHASE SPACE RECONSTRUCTION (DELAYED COORDINATES)
# ==========================================================
def delay_embedding(data, m, tau):
    """
    Creates a matrix of delayed coordinates for phase space reconstruction
    based on Takens' embedding theorem.
    
    Parameters:
    data (numpy.ndarray): The 1D time series data array.
    m (int): The embedding dimension (number of axes in the phase space).
    tau (int): The time delay index used to separate coordinates.
    
    Returns:
    numpy.ndarray: A 2D array of shape (num_points, m) representing the embedded phase space.
    """
    # Total number of observations in the original 1D time series
    N = len(data)
    
    # Calculate the maximum delay required for the last coordinate.
    # For m=3, the coordinates are x(t), x(t+tau), and x(t+2*tau). 
    # Therefore, the maximum offset from the base index is 2*tau (which is (m - 1) * tau).
    max_delay = (m - 1) * tau
    
    # Ensure we have enough data points to create at least one phase space vector.
    if max_delay >= N:
        raise ValueError("Time series is too short for the specified m and tau.")
        
    # The number of valid rows we can generate without going out of bounds of the array.
    num_points = N - max_delay
    
    # Preallocate an empty matrix to hold the reconstructed phase space.
    # Dimensions are (number of valid points, embedding dimension).
    # Preallocation is computationally faster than appending rows dynamically.
    embedded = np.zeros((num_points, m))
    
    # Populate the columns of the matrix.
    for i in range(m):
        # For each dimension i, slice the original data shifted by i*tau.
        # i=0: x(t)            -> slices data[0 : num_points]
        # i=1: x(t + tau)      -> slices data[tau : tau + num_points]
        # i=2: x(t + 2*tau)    -> slices data[2*tau : 2*tau + num_points]
        embedded[:, i] = data[i * tau : i * tau + num_points]
        
    return embedded

# [Inference] Load the project's configuration dictionary.
config = load_config()

# [Inference] Get the base directory where the CSV datasets are located.
data_dir = get_data_dir(config)

# [Inference] Construct the output directory for the 3D phase plots, creating the folder if it does not exist.
output_dir = ensure_dir(os.path.join(get_results_dir(config), "phase_3d"))

# Iterate through each CSV file to process its data.
for filename in FILES:
    # [Inference] Retrieve the final file path (possibly applying logic to use a liquidity-cut version of the data).
    file_path = prefer_liquidity_cut(os.path.join(data_dir, filename))
    
    # Extract the asset symbol from the filename (e.g., 'BTCUSD' from 'BTCUSD_BITSTAMP...').
    symbol = filename.split("_")[0]
    
    # [Inference] Dynamically retrieve the optimal time delay (tau) for this specific symbol 
    # calculated in a previous step (usually via mutual information algorithms).
    tau = tau_for_symbol_from_mutual(symbol, config)
    
    # Check for the existence of the file to prevent FileNotFoundError during read.
    if not os.path.exists(file_path):
        print(f"File not found, skipping: {file_path}")
        continue

    print(f"\nProcessing {symbol}: {file_path}")
    
    # Load the CSV data into a Pandas DataFrame.
    df = pd.read_csv(file_path)
    
    # Data Cleaning: 
    # 1. Access the 'log_return' column.
    # 2. Cast to string to safely replace European-style decimal commas with dots.
    # 3. Cast back to float for numerical processing.
    x = df["log_return"].astype(str).str.replace(",", ".", regex=False).astype(float)
    
    # Keep only finite, valid numerical values (dropping any NaNs or Infinities).
    x = x[np.isfinite(x)].values
    print(f"Total series length: {len(x)} points")

    # Attempt to reconstruct the 3D phase space using our delay_embedding function.
    try:
        phase_space = delay_embedding(x, M, tau)
    except ValueError as e:
        # If the time series is shorter than the required maximum delay, catch the error and skip this asset.
        print(f"Skipping {symbol}: {e}")
        continue

    # Log the successful creation and dimensions of the resulting matrix.
    print(f"Created {phase_space.shape[0]} points in {M}D space.")

    # Initialize a matplotlib figure with a specific size (10x8 inches).
    fig = plt.figure(figsize=(10, 8))
    
    # Add a 3-dimensional subplot to the figure using the registered '3d' projection.
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the phase space trajectory as a continuous line in 3D.
    # phase_space[:, 0] is the X axis (current return)
    # phase_space[:, 1] is the Y axis (return delayed by tau)
    # phase_space[:, 2] is the Z axis (return delayed by 2*tau)
    # We use a very thin line (linewidth=0.3) and slight transparency (alpha=0.7) to reveal dense attractor structures.
    ax.plot(phase_space[:, 0], phase_space[:, 1], phase_space[:, 2], color='blue', linewidth=0.3, alpha=0.7)
    
    # Set the main title of the plot, including the embedding dimension M and dynamically retrieved tau value.
    ax.set_title(f"{symbol} phase-space reconstruction (m={M}, $\\tau$={tau})")
    
    # Label the 3 axes using LaTeX formatting to denote the temporal shifts.
    ax.set_xlabel('$x(t)$')
    ax.set_ylabel(f'$x(t + {tau})$')
    ax.set_zlabel(f'$x(t + {2*tau})$')
    
    # Automatically adjust subplot parameters to give appropriate padding.
    plt.tight_layout()
    
    # Construct the file path where the generated plot will be saved.
    out_plot = os.path.join(output_dir, f"{symbol}_phase3d_tau{tau}_m{M}.png")
    
    # Save the 3D plot to the disk as a high-resolution PNG image (150 dots per inch).
    plt.savefig(out_plot, dpi=150)
    
    # Explicitly close the specific figure to free up memory before the loop proceeds to the next asset.
    plt.close(fig)
    print(f"Saved plot: {out_plot}")