#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Cao's method for determining the minimum embedding dimension.
- Parallel dimension processing (multiprocessing)
- Fast neighbor search using tree structure (sklearn KD-tree)
- Preserves the absolutely exact mathematical calculation according to the original paper:
  Cao, L. (1997). Practical method for determining the minimum embedding dimension of a scalar time series.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os
import time
import logging
import multiprocessing as mp
from functools import partial
from config_loader import load_config, get_data_dir, get_results_dir, ensure_dir, prefer_liquidity_cut
from report_helper import Reporter, append_summary_row

SUMMARY_FILE = "_cao_summary.txt"
SUMMARY_HEADER = (
    f"{'Symbol':<10} {'tau':>4} {'m_optimal':>10} {'E1@m_opt':>10} "
    f"{'E1@m_opt+1':>12} {'E2_avg':>10} {'verdict':<28}"
)

# ------------------------------------------------------------------------------
# LOGGING SETUP
# ------------------------------------------------------------------------------
# Ensures messages with exact timestamps are printed to the console.
# Useful for long calculations so we know the program hasn't "frozen".
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# HELPER FUNCTION: HANDLING IDENTICAL POINTS
# ------------------------------------------------------------------------------
def find_nonzero_neighbor(tree, point, start_k=3, max_k=100):
    """
    This function solves an extremely rare but critical problem:
    What if two different points in the time series have absolutely identical coordinates?
    The distance to the nearest neighbor would then be 0. However, when calculating a_i(m), 
    we divide by this distance. To avoid division by zero, we must find the next 
    nearest neighbor in order, whose distance is greater than 0.
    """
    # Limit the maximum search depth to the number of actually available points
    max_k = min(max_k, tree.n_samples_fit_)
    k = start_k
    
    # Gradually expand the search radius (k) until we find a non-zero distance
    while k <= max_k:
        # Returns distances and indices for k nearest neighbors
        distances, indices = tree.kneighbors(point, n_neighbors=k, return_distance=True)
        # Zip pairs the distance and index, we iterate through them one by one
        for d, idx in zip(distances[0], indices[0]):
            if d > 0.0:
                return d, idx
        k += 1
        
    # If all points in the dataset are identical (which won't happen with real data),
    # we return the smallest possible positive number the computer knows (epsilon).
    return np.finfo(float).eps, indices[0, -1]

# ------------------------------------------------------------------------------
# MAIN MATHEMATICAL CORE OF CAO'S METHOD FOR A SINGLE DIMENSION (m)
# ------------------------------------------------------------------------------
def calculate_for_m(m, data, tau, d_max):
    """
    Calculates the basic averages E(m) and E*(m) for a specific embedding dimension.
    This function is designed to run independently, allowing us to run multiple 
    instances at once (parallelization).
    """
    N = len(data)
    # Calculate how many points we can actually construct.
    # As dimension m and delay tau increase, we lose points at the end of the series.
    N_valid = N - m * tau
    
    # If we have too little data for such a large dimension, skip the calculation
    if N_valid < 2:
        logger.warning(f"m={m}: Insufficient data (N_valid={N_valid}), skipping.")
        return m, np.nan, np.nan
    
    # ----- STEP 1: TAKENS EMBEDDING (PHASE SPACE RECONSTRUCTION) -----
    # Create matrix X_m, where each row is a point (vector) in m-dimensional space.
    X_m = np.zeros((N_valid, m))
    for k in range(m):
        # Each column is the original time series shifted by k*tau
        X_m[:, k] = data[k * tau : N_valid + k * tau]
    
    # ----- STEP 2: FINDING THE NEAREST NEIGHBOR IN DIMENSION m -----
    # Cao uses the Maximum norm (L_infinity norm) in the paper, so we choose the 'chebyshev' 
    # distance metric. KD_tree is an algorithm that finds neighbors lightning fast.
    # IMPORTANT: this function already runs inside multiprocessing workers.
    # Keep sklearn neighbor search single-threaded here to avoid nested-parallel warnings
    # ("Loky-backed parallel loops cannot be called in a multiprocessing").
    neighbors_model = NearestNeighbors(n_neighbors=2, metric='chebyshev', algorithm='kd_tree', n_jobs=1)
    neighbors_model.fit(X_m)
    
    # Find 2 neighbors for each point. (The first neighbor is the point to itself, we don't care 
    # about that. The second neighbor is the real one we are looking for).
    distances, indices = neighbors_model.kneighbors(X_m, return_distance=True)
    nn_distance = distances[:, 1]       # Distances to the real neighbor (in dimension m)
    nn_index = indices[:, 1]            # Order (index) of this neighbor in the original matrix
    
    # If by chance someone has a zero distance, call our helper function to fix it
    zero_mask = (nn_distance == 0.0)
    if np.any(zero_mask):
        for i in np.where(zero_mask)[0]:
            point = X_m[i:i+1]
            new_dist, new_idx = find_nonzero_neighbor(neighbors_model, point, start_k=3, max_k=100)
            nn_distance[i] = new_dist
            nn_index[i] = new_idx
    
    # ----- STEP 3: SHIFT TO DIMENSION m+1 -----
    # Now we look at where the original point and its neighbor move 
    # when we add one more coordinate to them (i.e., increase the dimension to m+1).
    next_coordinate = data[m * tau : N_valid + m * tau]           # New coordinate of the original point
    nn_next_coordinate = data[nn_index + m * tau]                 # New coordinate of the neighbor
    
    # Calculate the distance between them ONLY in this one new coordinate
    abs_diff_new = np.abs(next_coordinate - nn_next_coordinate)
    
    # Because we use the Chebyshev (Maximum) norm, the total distance in dimension m+1 
    # is simply the maximum of the original distance (in dimension m) and the distance in the new coordinate.
    distance_m_plus_1 = np.maximum(nn_distance, abs_diff_new)
    
    # ----- STEP 4: CALCULATION OF INDICATORS a_i(m) -----
    # This is equation (8.43) or the very core of the deterministic part. 
    # We divide the distance in (m+1) by the distance in (m). If the points move sharply away 
    # from each other, it means they were false neighbors.
    with np.errstate(divide='ignore', invalid='ignore'):
        a_i = distance_m_plus_1 / nn_distance
        a_i = np.where(np.isfinite(a_i), a_i, 1e10) # Safety failsafe against division by zero
    
    # E(m) is the averaging of these ratios over all points.
    E_m = np.mean(a_i)
    # E*(m) is the average of absolute differences only in the new axis (stochastic indicator for E2)
    E_star_m = np.mean(abs_diff_new)
    
    return m, E_m, E_star_m

# ------------------------------------------------------------------------------
# FUNCTION FOR MANAGING PARALLEL PROCESSES
# ------------------------------------------------------------------------------
def cao_method_parallel(data, tau, d_max, num_processes=None):
    """
    This function acts as a "manager". It divides the work (individual dimensions m)
    among available CPU cores. Thanks to this, it calculates many times faster.
    """
    logger.info(f"Starting Cao's method: N={len(data)}, tau={tau}, d_max={d_max}")
    start_total = time.time()
    
    # To calculate indicators E1 and E2 up to d_max, we must know E and E* up to d_max + 1
    m_values = list(range(1, d_max + 2))
    
    # If we don't specify the number of processes, all available CPU cores will be used
    if num_processes is None:
        num_processes = mp.cpu_count()
    logger.info(f"Using {num_processes} parallel processes.")
    
    # The "partial" function here pre-fills the arguments data, tau, and d_max into our calculation function,
    # so the "pool" will only supply the changing number m to the function.
    func = partial(calculate_for_m, data=data, tau=tau, d_max=d_max)
    
    with mp.Pool(processes=num_processes) as pool:
        # imap_unordered scatters the work and returns results as soon as they finish
        results_iter = pool.imap_unordered(func, m_values)
        
        completed = 0
        total = len(m_values)
        results = {}
        
        for m, E_m, E_star_m in results_iter:
            completed += 1
            results[m] = (E_m, E_star_m)
            
            # Simple math to estimate remaining time:
            # (total time so far / number completed) * number remaining
            elapsed_time = time.time() - start_total
            avg_per_dimension = elapsed_time / completed
            remaining = (total - completed) * avg_per_dimension
            logger.info(f"[m={m:3d}] done (E={E_m:.6f}, E*={E_star_m:.6f}) - "
                        f"done {completed}/{total}, approx {remaining:.1f} s remaining")
    
    # Now we align the results, which arrived out of order, into arrays from m=1 to d_max+1
    E = np.full(d_max + 2, np.nan)
    E_star = np.full(d_max + 2, np.nan)
    for m, (val_e, val_es) in results.items():
        if not np.isnan(val_e):
            E[m] = val_e
        if not np.isnan(val_es):
            E_star[m] = val_es
    
    # ----- FINAL CALCULATION OF E1 AND E2 -----
    # E1(m) = E(m+1) / E(m). Stops growing when the dimension is sufficient.
    # E2(m) = E*(m+1) / E*(m). Serves to distinguish deterministic chaos from random noise.
    E1 = np.zeros(d_max + 1)
    E2 = np.zeros(d_max + 1)
    for m in range(1, d_max + 1):
        if E[m] > 0 and E[m+1] > 0:
            E1[m] = E[m+1] / E[m]
        if E_star[m] > 0 and E_star[m+1] > 0:
            E2[m] = E_star[m+1] / E_star[m]
    
    total_time = time.time() - start_total
    logger.info(f"Cao's method completed in {total_time:.2f} seconds.")
    
    # Return arrays sliced from the zeroth index (which is empty, we start from m=1)
    return E1[1:], E2[1:]

# ------------------------------------------------------------------------------
# MAIN SCRIPT (EXECUTION)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    
    # =========================================================================
    # SETTINGS FOR FILES AND THEIR INDIVIDUAL DELAYS (TAU)
    # Here we define the exact filename and delay for each cryptocurrency.
    # We changed a simple list into a list of dictionaries.
    # =========================================================================
    file_settings = [
        {"file": "BTCUSD_BITSTAMP_1h_complete_logreturns.dat", "tau": 5},
        {"file": "ETHUSD_BITSTAMP_1h_complete_logreturns.dat", "tau": 5},
        {"file": "LTCUSD_BITSTAMP_1h_complete_logreturns.dat", "tau": 2},
        {"file": "XRPUSD_BITSTAMP_1h_complete_logreturns.dat", "tau": 2},
        {"file": "LINKUSD_BITSTAMP_1h_complete_logreturns.dat", "tau": 4},
        {"file": "DOGEUSD_BITSTAMP_1h_complete_logreturns.dat", "tau": 6},
        {"file": "ADAUSD_BITSTAMP_1h_complete_logreturns.dat", "tau": 2}
    ]

    config = load_config()
    data_dir = get_data_dir(config)
    output_dir = ensure_dir(os.path.join(get_results_dir(config), "cao"))

    # Reset the aggregated summary so each script run gets a clean table.
    try:
        os.remove(os.path.join(output_dir, SUMMARY_FILE))
    except FileNotFoundError:
        pass

    # Global settings valid for all files
    d_max = 20                    # We examine dimensions m from 1 to 30
    num_processes = None           # Let Python use all CPU cores
    
    # Main loop: Iterates through one file after another from our settings
    for setting in file_settings:
        # Unpack variables from the current dictionary
        filename = setting["file"]
        tau = setting["tau"]
        
        file_path = prefer_liquidity_cut(os.path.join(data_dir, filename))
        
        # Visual separator in the console for clarity
        print("\n" + "="*80)
        logger.info(
            "STARTING PROCESSING: %s (chosen tau = %s) | input file: %s",
            filename,
            tau,
            file_path,
        )
        print("="*80)
        
        # Safety check to see if the file actually exists on the disk
        if not os.path.isfile(file_path):
            logger.error(f"File not found: {file_path}. Skipping to the next...")
            continue
        
        # Load data. We are reading a simple .dat file where each value is on a new line.
        data = np.loadtxt(file_path)
        
        # Starting the calculation core
        start_total = time.time()
        # Here we pass our individual 'tau' extracted from the dictionary into the function
        E1, E2 = cao_method_parallel(data, tau, d_max, num_processes=num_processes)
        total_time = time.time() - start_total
        logger.info(f"Total processing time for {filename}: {total_time:.2f} s")
        
        # =====================================================================
        # PLOTTING AND SAVING THE GRAPH
        # =====================================================================
        # Prepare the X axis (dimensions from 1 to d_max)
        ds = np.arange(1, d_max + 1)
        
        # Setup the "canvas" for the graph with given dimensions
        plt.figure(figsize=(10, 6))
        
        # Plotting curve E1(m) (Blue, solid line) - Deterministic component
        plt.plot(ds, E1, marker='o', linestyle='-', color='blue', label='$E1(m)$ (deterministic)')
        # Plotting curve E2(m) (Red, dashed line) - Stochastic (noise) component
        plt.plot(ds, E2, marker='s', linestyle='--', color='red', label='$E2(m)$ (stochastic)')
        
        # Horizontal reference line at value 1.0 (here E1 typically saturates)
        plt.axhline(y=1.0, color='gray', linestyle=':', label='Reference value 1')
        
        # Remove underscores and ".dat" from the filename and keep only the cryptocurrency name 
        # itself (e.g., "BTCUSD") for the graph title
        coin_name = filename.split('_')[0]
        
        plt.title(f"Cao's Method: {coin_name}, $\\tau = {tau}$")
        plt.xlabel('Embedding dimension $m$')
        plt.ylabel('$E1(m)$ and $E2(m)$')
        
        # Grid settings - so the X axis shows every other tick for better readability
        plt.xticks(np.arange(1, d_max + 1, step=2))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='lower right')
        
        # Dynamically construct the output image name so we don't overwrite old graphs.
        out_name = filename.replace('.dat', f'_tau{tau}_cao_graph.png')
        out_img = os.path.join(output_dir, out_name)
        
        # Save the image in high resolution (300 dpi is standard for printing/theses)
        plt.savefig(out_img, dpi=300)
        logger.info(f"Graph for {coin_name} successfully saved to: {out_img}")
        
        # EXTREMELY IMPORTANT STEP:
        # Close the graph instance in memory. If we used plt.show(),
        # the program would stop, display the image on the monitor, and wait for 
        # the user to manually close it with the cross before moving to the next coin. 
        # Plt.close() ensures smooth background running of the script.
        plt.close()

        # ----- TEXT EXPORT OF RESULTS -----
        # Optimal embedding dimension m*: smallest m where E1 saturates
        # (i.e. E1(m+1)/E1(m) - 1 stays small). We use a 5% threshold.
        E1_arr = np.asarray(E1, dtype=float)
        E2_arr = np.asarray(E2, dtype=float)
        m_optimal = None
        for i in range(len(E1_arr) - 1):
            cur, nxt = E1_arr[i], E1_arr[i + 1]
            if np.isfinite(cur) and np.isfinite(nxt) and cur > 0:
                if abs(nxt - cur) / cur < 0.05 and cur > 0.85:
                    m_optimal = i + 1  # i indexes from 0 -> dimension is i+1
                    break

        e2_finite = E2_arr[np.isfinite(E2_arr)]
        e2_avg = float(np.mean(e2_finite)) if e2_finite.size else float("nan")
        # Cao's noise diagnostic: E2 ~ 1 for all m means stochastic noise,
        # E2 deviating from 1 implies determinism.
        if not e2_finite.size:
            verdict = "no E2 data"
        elif np.all(np.abs(e2_finite - 1.0) < 0.05):
            verdict = "E2~1 -> stochastic / noise-like"
        else:
            verdict = "E2 deviates from 1 -> deterministic"

        rep = Reporter()
        rep.add("=" * 80)
        rep.add(f"Cao's method results - {coin_name} (tau={tau}, d_max={d_max})")
        rep.add(f"Input file : {file_path}")
        rep.add(f"Series len : {len(data)}")
        rep.add(f"Total time : {total_time:.2f} s")
        rep.add("=" * 80)
        rep.add(f"{'m':>3} {'E1(m)':>12} {'E2(m)':>12}")
        rep.add("-" * 32)
        for i, (e1v, e2v) in enumerate(zip(E1_arr, E2_arr), start=1):
            rep.add(f"{i:>3d} {e1v:>12.6f} {e2v:>12.6f}")
        rep.add("-" * 32)
        if m_optimal is not None:
            rep.add(
                f"Optimal m* : {m_optimal} "
                f"(first m where E1 saturates within 5% and E1>0.85)"
            )
        else:
            rep.add("Optimal m* : not found within d_max")
        rep.add(f"Mean E2    : {e2_avg:.6f}")
        rep.add(f"Verdict    : {verdict}")
        rep.add(f"Plot       : {out_img}")
        out_txt = rep.write(
            output_dir, filename.replace(".dat", f"_tau{tau}_cao_results.txt")
        )
        logger.info(f"Saved text report: {out_txt}")

        if m_optimal is not None:
            e1_at = float(E1_arr[m_optimal - 1])
            e1_next = (
                float(E1_arr[m_optimal]) if m_optimal < len(E1_arr) else float("nan")
            )
        else:
            e1_at = float("nan")
            e1_next = float("nan")
        m_opt_str = f"{m_optimal}" if m_optimal is not None else "-"
        summary_row = (
            f"{coin_name:<10} {tau:>4d} {m_opt_str:>10} {e1_at:>10.4f} "
            f"{e1_next:>12.4f} {e2_avg:>10.4f} {verdict:<28}"
        )
        summary_path = append_summary_row(
            output_dir, SUMMARY_FILE, SUMMARY_HEADER, summary_row
        )
        logger.info(f"Appended row to summary: {summary_path}")

    print("\n" + "="*80)
    logger.info("ALL FILES SUCCESSFULLY PROCESSED AND GRAPHS SAVED!")
    print("="*80)
