"""
mutual.py

Implementation of the Fraser & Swinney (1986) algorithm for computing mutual
information I(tau) between a time series s(t) and its delayed version s(t+tau).
The method uses an adaptive partition of the (rank-transformed) plane guided
by chi-square tests for uniformity.

Reference:
Fraser, A. M., & Swinney, H. L. (1986). Independent coordinates for strange
attractors from mutual information. Physical Review A, 33(2), 1134-1140.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import os
from config_loader import (
    load_config,
    get_data_dir,
    get_results_dir,
    ensure_dir,
    prefer_liquidity_cut,
    MUTUAL_SUMMARY_SERIES_COL_W,
    sync_per_coin_bat_tau_from_mutual_summary,
    default_per_coin_settings_bat_path,
)
from report_helper import Reporter, append_summary_row

SUMMARY_FILE = "_mi_summary.txt"
SERIES_COL_W = MUTUAL_SUMMARY_SERIES_COL_W
SUMMARY_HEADER = (
    f"{'series_id':<{SERIES_COL_W}} {'N':>8} {'max_tau':>8} {'first_min_tau':>14} "
    f"{'I(first_min) [bits]':>22} {'I(tau=1) [bits]':>16}"
)

# Mutual information along tau (Fraser & Swinney). Use same max_tau for cut + surrogate runs.
DEFAULT_MAX_TAU = 100

# Increase recursion limit for deeply nested partitions (in case of very fine division)
sys.setrecursionlimit(200000)

# ==============================================================================
# Fraser & Swinney (1986) - Exact implementation (including equations 21 & 22)
# ==============================================================================
def mi_fraser_swinney(s, tau):
    """
    Compute mutual information I(tau) between signal s(t) and s(t+tau)
    using the recursive adaptive partition algorithm of Fraser & Swinney.

    Parameters:
        s   : 1D numpy array of the time series
        tau : time delay (integer)

    Returns:
        I in bits (non-negative; if negative due to numerical noise, returns 0.0)
    """

    # --------------------------------------------------------------------------
    # 1. Data preparation - we need enough points for the given delay tau.
    #    The algorithm requires the number of pairs to be a power of two,
    #    because each recursive step divides the rank intervals exactly in half.
    #    (See Section "Algorithm" in the paper, where they state:
    #     "Our algorithm operates on a pair of sequences of numbers whose
    #      lengths are a power of 2: {[x(t), y(t)]: 0 <= t < 2^n}".)
    # --------------------------------------------------------------------------
    n_total = len(s) - tau          # number of pairs (s(t), s(t+tau))
    if n_total <= 0:
        return 0.0

    # Find the largest power of two not exceeding n_total.
    # Example: n_total=1000 -> nearest lower power is 512.
    n_pow2 = 1 << (int(n_total).bit_length() - 1)
    if n_pow2 < 4:                  # minimum for meaningful chi^2 tests (at least 4 points)
        return 0.0

    # Take the first n_pow2 pairs (x = s(t), y = s(t+tau))
    x = s[:n_pow2]
    y = s[tau:tau + n_pow2]

    # --------------------------------------------------------------------------
    # 2. Rank transformation - crucial step as described in the paper.
    #    Convert x and y to uniform distribution on the interval [0, n_pow2-1].
    #    This ensures that marginal densities are constant: P_s = 1/n_pow2,
    #    P_q = 1/n_pow2. It simplifies the MI calculation because the term
    #    m*log(4) appears in equations (11) and (14). Only the order statistics
    #    are preserved, not the absolute values.
    #    (Paper: "The change of variable goes from floating point (x,y) to
    #     integer (s,q) representation in a fashion that preserves orderings,
    #     with the constraints that ... [s(t)] and [q(t)] are permutations
    #     of the sequence [0 to 2^n - 1].")
    # --------------------------------------------------------------------------
    rx = np.argsort(np.argsort(x)).astype(np.int64)   # ranks of x: 0 .. n_pow2-1
    ry = np.argsort(np.argsort(y)).astype(np.int64)   # ranks of y

    # --------------------------------------------------------------------------
    # 3. Recursive function F(R) according to equations (20a) and (20b).
    #    R is a rectangular cell in the (rank_x, rank_y) plane bounded by
    #    s_low .. s_high on the x-axis and q_low .. q_high on the y-axis.
    #    'indices' are the indices (from the original arrays) of points
    #    that fall into this cell.
    # --------------------------------------------------------------------------
    def compute_F(indices, s_low, s_high, q_low, q_high):
        """
        Recursively compute the contribution F(R) for cell R.
        Returns F(R) according to (20a) or (20b).

        Parameters:
            indices : array of indices (pointing to rx, ry) lying inside R
            s_low, s_high : lower (inclusive) and upper (exclusive) bounds of x-ranks in this cell
            q_low, q_high : lower (inclusive) and upper (exclusive) bounds of y-ranks in this cell
        """
        N = len(indices)
        if N <= 1:
            return 0.0                     # no points or single point -> no contribution

        # If the cell is already so small that it cannot be subdivided further
        # (width <= 1 in either dimension), we treat it as elementary and use
        # equation (20a): F = N * log(N)
        # (Paper: "if there is no substructure in R_m(K_m) ...")
        if s_high - s_low <= 1 or q_high - q_low <= 1:
            return N * np.log(N) if N > 0 else 0.0

        # Midpoints of the interval - we split the cell into 4 equal sub-cells (quadrants)
        s_mid = (s_low + s_high) // 2
        q_mid = (q_low + q_high) // 2

        # Extract the rank values for points in this cell
        s_vals = rx[indices]
        q_vals = ry[indices]

        # ----------------------------------------------------------------------
        # 3a. Create 4 sub-cells (a_i according to equation 21)
        #     ll = low-low, lh = low-high, hl = high-low, hh = high-high
        # ----------------------------------------------------------------------
        mask_ll = (s_vals < s_mid) & (q_vals < q_mid)
        mask_lh = (s_vals < s_mid) & (q_vals >= q_mid)
        mask_hl = (s_vals >= s_mid) & (q_vals < q_mid)
        mask_hh = (s_vals >= s_mid) & (q_vals >= q_mid)

        # Counts of points in each of the 4 sub-cells
        a_counts = np.array([
            np.sum(mask_ll),
            np.sum(mask_lh),
            np.sum(mask_hl),
            np.sum(mask_hh)
        ], dtype=float)

        # ----------------------------------------------------------------------
        # 3b. First chi^2 test - equation (21) from the paper.
        #     Test the null hypothesis that the distribution among the 4 cells
        #     is uniform. The reduced chi^2 statistic for 3 degrees of freedom
        #     (4 cells - 1 constraint = 3) uses a constant (16/5) instead of
        #     the usual (16/3) because it is a reduced chi^2. The threshold 1.547
        #     corresponds to a 20% significance level: if chi^2 < 1.547, we do not
        #     have enough evidence to reject uniformity.
        # ----------------------------------------------------------------------
        chi2_3 = (16.0 / 5.0) * (1.0 / N) * np.sum((a_counts - N / 4.0) ** 2)

        # Indices of points in each sub-cell for potential further subdivision
        idx_ll = indices[mask_ll]
        idx_lh = indices[mask_lh]
        idx_hl = indices[mask_hl]
        idx_hh = indices[mask_hh]

        is_flat = False   # flag indicating whether distribution in this cell is considered uniform

        # ----------------------------------------------------------------------
        # 3c. If the first test suggests uniformity (chi^2 < 1.547), we perform a
        #     stricter test on 16 sub-cells (equation 22). This test requires
        #     that both dimensions have at least 4 units (so we can create a 4x4 grid).
        # ----------------------------------------------------------------------
        if chi2_3 < 1.547:
            # Check if we have sufficient resolution to divide into 16 cells
            if (s_high - s_low >= 4) and (q_high - q_low >= 4):

                # Helper function to obtain 4 sub-counts within a given quadrant
                def get_sub_counts(sub_idx, sl, sh, ql, qh):
                    """
                    Return 4 counts (b_00, b_01, b_10, b_11) for the sub-region
                    defined by bounds sl..sh, ql..qh.
                    """
                    if len(sub_idx) == 0:
                        return np.zeros(4)
                    sm = (sl + sh) // 2
                    qm = (ql + qh) // 2
                    sv = rx[sub_idx]
                    qv = ry[sub_idx]
                    return np.array([
                        np.sum((sv < sm) & (qv < qm)),
                        np.sum((sv < sm) & (qv >= qm)),
                        np.sum((sv >= sm) & (qv < qm)),
                        np.sum((sv >= sm) & (qv >= qm))
                    ], dtype=float)

                # For each of the 4 quadrants, get its 4 sub-counts -> total 16 numbers (b_ij)
                b_ll = get_sub_counts(idx_ll, s_low, s_mid, q_low, q_mid)
                b_lh = get_sub_counts(idx_lh, s_low, s_mid, q_mid, q_high)
                b_hl = get_sub_counts(idx_hl, s_mid, s_high, q_low, q_mid)
                b_hh = get_sub_counts(idx_hh, s_mid, s_high, q_mid, q_high)

                b_counts = np.concatenate([b_ll, b_lh, b_hl, b_hh])   # 16 elements

                # ------------------------------------------------------------------
                # Equation (22) - reduced chi^2 test for 16 cells (15 degrees of freedom)
                # Constant 256/225, threshold 1.287 (again 20% significance level).
                # ------------------------------------------------------------------
                chi2_15 = (256.0 / 225.0) * (1.0 / N) * np.sum((b_counts - N / 16.0) ** 2)

                # If it passes the second test as well, we definitely consider the distribution flat.
                if chi2_15 < 1.287:
                    is_flat = True
            else:
                # Cannot perform the 16-cell test (cell too small) - accept the first test's result
                is_flat = True

        # ----------------------------------------------------------------------
        # 3d. Decision based on tests:
        #     - If flat (is_flat == True) -> use equation (20a): F = N log N
        #     - Otherwise -> equation (20b): recursive division into 4 sub-cells
        #       and sum their F contributions plus an extra term N log 4.
        # ----------------------------------------------------------------------
        if is_flat:
            # Equation (20a): for uniform distribution inside cell R
            return N * np.log(N)
        else:
            # Equation (20b): distribution is not uniform -> split into 4 sub-cells
            F_ll = compute_F(idx_ll, s_low, s_mid, q_low, q_mid)
            F_lh = compute_F(idx_lh, s_low, s_mid, q_mid, q_high)
            F_hl = compute_F(idx_hl, s_mid, s_high, q_low, q_mid)
            F_hh = compute_F(idx_hh, s_mid, s_high, q_mid, q_high)

            # The term N log 4 corresponds to adding log(4) for each point when
            # moving to a finer level (see equations (14) and (16b) in the paper).
            return N * np.log(4) + F_ll + F_lh + F_hl + F_hh

    # --------------------------------------------------------------------------
    # 4. Start recursion on the entire rank space [0, n_pow2) x [0, n_pow2)
    # --------------------------------------------------------------------------
    all_indices = np.arange(n_pow2, dtype=int)
    F_total = compute_F(all_indices, 0, n_pow2, 0, n_pow2)

    # --------------------------------------------------------------------------
    # 5. Compute mutual information I in nats and convert to bits.
    #    Equation (19) from the paper: I = F(R0)/N - log(N)
    #    where N = n_pow2, F(R0) = F_total.
    #    The logarithms used in compute_F are natural logs, so the result is
    #    in nats. Divide by log(2) to obtain bits.
    # --------------------------------------------------------------------------
    I_nats = F_total / n_pow2 - np.log(n_pow2)   # natural logarithms -> nats
    I_bits = I_nats / np.log(2)                  # convert to bits (divide by log(2))

    return max(0.0, I_bits)                     # mutual information cannot be negative


def find_first_minimum(tau_list, mi_list):
    """
    Find the first local minimum of mutual information (excluding tau=1).
    A local minimum is a point that is lower than both its neighbors.

    This implements the criterion proposed by Shaw and cited in the paper:
    choose the delay corresponding to the first minimum of I(tau) for phase
    portrait reconstruction.

    Parameters:
        tau_list : list of tau values
        mi_list  : list of corresponding MI values

    Returns:
        (tau_min, mi_min) or (None, None) if none found.
    """
    for i in range(1, len(mi_list) - 1):
        if mi_list[i] < mi_list[i-1] and mi_list[i] < mi_list[i+1]:
            return tau_list[i], mi_list[i]
    return None, None


# ==============================================================================
# Main part: compute MI for tau = 1..max_tau and visualize
# ==============================================================================
def process_file(file_path, output_dir, max_tau=100):
    basename = os.path.basename(file_path)
    stem = os.path.splitext(basename)[0]
    series_id = stem[:SERIES_COL_W].ljust(SERIES_COL_W)[:SERIES_COL_W]

    r = Reporter()
    r.add("\n" + "=" * 80)
    r.add(f"Processing {stem}: {file_path}")
    r.add("=" * 80)

    s = np.loadtxt(file_path)
    r.add(f"Series length N = {len(s)}")
    r.add(f"Method: Fraser & Swinney (1986) - adaptive partition")
    r.add(f"Tau range: 1..{max_tau}")
    tau_values = list(range(1, max_tau + 1))
    mi_values = []

    r.add("")
    r.add("Computing MI for individual tau values:")
    r.add(f"{'tau':>5} {'I [bits]':>12}")
    r.add("-" * 20)
    for tau in tau_values:
        mi = mi_fraser_swinney(s, tau)
        mi_values.append(mi)
        r.add(f"{tau:>5d} {mi:>12.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(tau_values, mi_values, 'o-', color='navy', linewidth=2, markersize=5)
    plt.xlabel("Time delay tau", fontsize=12)
    plt.ylabel("Mutual information I(tau) [bits]", fontsize=12)
    plt.title(f"Fraser-Swinney mutual information — {stem}", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    first_tau, first_mi = find_first_minimum(tau_values, mi_values)
    if first_tau is not None:
        plt.plot(first_tau, first_mi, 'r*', markersize=15,
                 label=f'First minimum: tau={first_tau}, I={first_mi:.4f} bits')
        plt.legend()
        r.add(f"\n>>> First local minimum: tau = {first_tau}  (I = {first_mi:.4f} bits)")
    else:
        r.add("\n>>> No local minimum found.")

    plt.tight_layout()
    out_plot = os.path.join(output_dir, f"{stem}_mi_plot.png")
    plt.savefig(out_plot, dpi=150)
    plt.close()
    r.add(f"Saved plot: {out_plot}")

    out_txt = r.write(output_dir, f"{stem}_mi_results.txt")
    print(f"Saved text report: {out_txt}")

    first_min_str = f"{first_tau}" if first_tau is not None else "none"
    first_min_val = f"{first_mi:.4f}" if first_tau is not None else "nan"
    summary_row = (
        f"{series_id:<{SERIES_COL_W}} {len(s):>8d} {max_tau:>8d} {first_min_str:>14} "
        f"{first_min_val:>22} {mi_values[0]:>16.4f}"
    )
    summary_path = append_summary_row(
        output_dir, SUMMARY_FILE, SUMMARY_HEADER, summary_row
    )
    print(f"Appended row to summary: {summary_path}")


if __name__ == "__main__":
    config = load_config()
    data_dir = get_data_dir(config)
    output_dir = ensure_dir(os.path.join(get_results_dir(config), "mutual"))
    # Reset the aggregated summary so each script run gets a clean table.
    try:
        os.remove(os.path.join(output_dir, SUMMARY_FILE))
    except FileNotFoundError:
        pass
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
        file_path = prefer_liquidity_cut(os.path.join(data_dir, filename))
        if not os.path.exists(file_path):
            print(f"File not found, skipping: {file_path}")
            continue
        process_file(file_path, output_dir, max_tau=DEFAULT_MAX_TAU)

    status, n_sym = sync_per_coin_bat_tau_from_mutual_summary(config)
    if status == "updated":
        print(f"Updated _per_coin_settings.bat from mutual summary ({n_sym} symbol(s)).")
    elif status == "unchanged":
        print(
            f"Parsed {n_sym} first-minimum tau(s) from _mi_summary.txt; "
            "_per_coin_settings.bat already matched (no rewrite)."
        )
    elif status == "no_bat":
        print(
            f"Parsed {n_sym} tau(s) but _per_coin_settings.bat not found at "
            f"{default_per_coin_settings_bat_path()} — skipped sync."
        )
    else:
        print(
            "No first-minimum tau rows parsed from _mi_summary.txt "
            "(missing file, wrong format, or all 'none'); bat unchanged."
        )
