#!/usr/bin/env python3
r"""Plot TISEAN `lyap_k` S(t) output and optional LLE fit (original series).

This script visualizes the stretching factor S(t) calculated by the Kantz algorithm
(via TISEAN's lyap_k tool) to estimate the Largest Lyapunov Exponent (LLE). 
The LLE is represented by the slope of the linear scaling region of these curves.

Standalone usage (raw curves only):

    py -3 plot_lyap_k_output.py C:\DCh\data\results\lambda_max_test_100\BTCUSD_run2_tau2\BTCUSD_lyap.txt

With the Kantz / hypothesis LLE fit line (median LLE across epsilon blocks; one
representative block's linear window — same rule as `hypothesis.extract_lle_mean_std`):

    py -3 plot_lyap_k_output.py PATH\BASE_lyap.txt --orig-lle-fit --output PATH\BASE_lyap_lle_fit.png

Each `#epsilon= ... dim= ...` block is one S(t) curve:
  column 1 = iteration t (time step)
  column 2 = logarithm of the stretching factor S(t)
  column 3 = number of points with a sufficiently populated neighbourhood
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class LyapBlock:
    """
    Data structure representing a single block of output from lyap_k.
    Each block corresponds to a specific neighborhood size (epsilon) and embedding dimension (dim).
    """
    epsilon: float | None    # The radius of the neighborhood used to find close trajectories
    dim: int | None          # The embedding dimension 'm' used for phase space reconstruction
    rows: np.ndarray         # A 2D array holding the iteration, S(t), and neighbor count data


def parse_lyap_k(path: str) -> list[LyapBlock]:
    """
    Parses a TISEAN lyap_k output text file.
    
    The file format consists of header lines starting with '#' followed by metadata,
    and then columns of numeric data representing the S(t) curve.
    
    Args:
        path (str): The file path to the lyap_k output text file.
        
    Returns:
        list[LyapBlock]: A list of parsed data blocks.
    """
    blocks: list[LyapBlock] = []
    epsilon: float | None = None
    dim: int | None = None
    rows: list[list[float]] = []

    def flush() -> None:
        """
        Helper function to package the currently accumulated rows into a LyapBlock
        and append it to the main blocks list before resetting for the next block.
        """
        nonlocal rows, epsilon, dim
        if rows:
            # Convert the list of lists into a numpy array for easier mathematical slicing later
            blocks.append(LyapBlock(epsilon=epsilon, dim=dim, rows=np.asarray(rows, dtype=float)))
        # Reset the rows accumulator for the next block of data
        rows = []

    # Open the file, ignoring encoding errors to prevent crashes on weird characters
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check if the line is a header containing block metadata
            if line.startswith("#"):
                # Save the previous block's data before starting a new one
                flush()
                
                # Use regex to extract the epsilon (neighborhood size) value
                eps_match = re.search(r"epsilon=\s*([+\-0-9.eE]+)", line)
                # Use regex to extract the dim (embedding dimension) value
                dim_match = re.search(r"dim=\s*(\d+)", line)
                
                # Parse the extracted strings into their respective numeric types if found
                epsilon = float(eps_match.group(1)) if eps_match else None
                dim = int(dim_match.group(1)) if dim_match else None
                continue

            # If it's not a header, it should be a row of data. Split it by whitespace.
            parts = line.split()
            
            # We need at least 2 columns (iteration and S(t)) to form a valid data point
            if len(parts) < 2:
                continue
                
            try:
                # Column 1: Time step / Iteration
                iteration = float(parts[0])
                # Column 2: Logarithm of the average distance between trajectories (Stretching factor)
                stretching_log = float(parts[1])
                # Column 3: Number of reference points that had enough neighbors (if available)
                neighbour_points = float(parts[2]) if len(parts) >= 3 else np.nan
            except ValueError:
                # Skip lines that contain non-numeric data where numbers are expected
                continue
                
            # Store the parsed row
            rows.append([iteration, stretching_log, neighbour_points])

    # Flush any remaining data at the end of the file
    flush()
    return blocks


def block_label(index: int, block: LyapBlock) -> str:
    """
    Constructs a descriptive label for the plot legend based on the block's metadata.
    """
    parts = [f"block {index}"]
    if block.dim is not None:
        parts.append(f"m={block.dim}")
    if block.epsilon is not None:
        parts.append(f"eps={block.epsilon:.3g}")
    # If the block has the 3rd column (neighbor counts), display the maximum number of neighbors found
    if block.rows.size and block.rows.shape[1] >= 3 and np.isfinite(block.rows[:, 2]).any():
        parts.append(f"n~{np.nanmax(block.rows[:, 2]):.0f}")
        
    return ", ".join(parts)


def _block_slopes_for_lle_plot(blocks, min_neighbors):
    """
    Mirrors the logic in `hypothesis.extract_lle_mean_std` to filter blocks
    based on neighborhood density and calculates the linear slope (LLE estimate) for each.
    
    Returns:
        list of tuples: Each tuple contains the block dictionary and its calculated slope.
    """
    # Import internal project dependencies dynamically to avoid circular imports or missing modules in standalone mode
    from hypothesis import MIN_LYAP_NEIGHBORS, _best_linear_slope

    # Determine the strict minimum neighbors threshold to consider a block valid
    mn = int(min_neighbors) if min_neighbors is not None else int(MIN_LYAP_NEIGHBORS)
    pairs = []
    
    # First pass: Try to find valid slopes for blocks that meet the strict min_neighbors criteria
    for blk in blocks:
        if blk["n_neighbors"] < mn:
            continue
        data = blk["data"]
        # Need at least 3 points to reliably fit a line
        if data.shape[0] < 3:
            continue
            
        # Calculate the slope of the linear scaling region (this slope IS the Lyapunov exponent)
        slope = _best_linear_slope(data[:, 0], data[:, 1])
        if np.isfinite(slope):
            pairs.append((blk, float(slope)))

    # Fallback pass: If no blocks met the strict criteria, relax the neighbor constraint and process all blocks
    if not pairs:
        for blk in blocks:
            data = blk["data"]
            if data.shape[0] < 3:
                continue
            slope = _best_linear_slope(data[:, 0], data[:, 1])
            if np.isfinite(slope):
                pairs.append((blk, float(slope)))
                
    return pairs


def plot_orig_lle_fit(lyap_path: str, out_png: str, title: str | None = None) -> None:
    """
    Plots all epsilon S(t) curves and superimposes a bold red line representing 
    the final calculated Largest Lyapunov Exponent (LLE) fit.
    
    This heavily relies on the custom `hypothesis` module to parse and calculate the LLE.
    """
    # Dynamically import required functions from the project's internal modules
    from hypothesis import (
        M_LYAP,
        MIN_LYAP_NEIGHBORS,
        _best_linear_slope_window,
        _parse_lyap_blocks,
        extract_lle_mean_std,
    )
    from hypothesis_config import lyap_min_neighbors

    # Parse the blocks using the hypothesis module's parser, restricted to the target embedding dimension
    blocks = _parse_lyap_blocks(lyap_path, dim=M_LYAP)
    if not blocks:
        print(f"WARNING: No lyap_k blocks parsed from {lyap_path}; skipping LLE fit plot.")
        return

    # Extract the final LLE statistics (median, standard deviation, and number of valid blocks used)
    lle, lle_sd, n_blk = extract_lle_mean_std(lyap_path)
    
    # Get the individual slopes for all valid blocks to find the most representative one
    pairs = _block_slopes_for_lle_plot(blocks, lyap_min_neighbors())

    # Initialize the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot the raw S(t) curves for all parsed blocks
    for i, blk in enumerate(blocks, start=1):
        d = blk["data"]
        if d.shape[0] < 2:
            continue
        lab = f"eps={blk['eps']:.3g}, n_med={blk['n_neighbors']}"
        # Plot with slight transparency. Limit the legend to the first 12 blocks to avoid clutter.
        ax.plot(d[:, 0], d[:, 1], linewidth=1.0, alpha=0.75, label=lab if i <= 12 else None)

    # If we successfully calculated a valid LLE and have slopes to compare against
    if np.isfinite(lle) and pairs:
        # Find the specific block whose local slope is closest to the median overall LLE
        # This gives us a "representative" block to draw the fit line on
        best_blk, best_slope = min(pairs, key=lambda p: abs(p[1] - lle))
        data = best_blk["data"]
        
        # Calculate the exact window (start and end times) where the linear fit was applied
        slope_w, t0, t1, intercept = _best_linear_slope_window(data[:, 0], data[:, 1])
        
        if np.isfinite(slope_w) and np.isfinite(t0) and np.isfinite(t1) and np.isfinite(intercept):
            # Generate points to draw the linear fit line
            tt = np.linspace(t0, t1, max(50, int((t1 - t0) * 4) + 1))
            ss = slope_w * tt + intercept
            
            # Plot the best-fit line in bold crimson so it stands out against the raw curves
            ax.plot(
                tt,
                ss,
                color="crimson",
                linewidth=2.6,
                zorder=6, # Ensure the line is drawn on top of the other curves
                label=(
                    f"LLE median={lle:.5g} (n_blocks={n_blk}); "
                    f"fit window eps={best_blk['eps']:.3g}, local slope={best_slope:.5g}"
                ),
            )

    # Set up the aesthetics of the plot (titles, labels, grid)
    ax.set_title(title or f"lyap_k S(t) + LLE fit: {os.path.basename(lyap_path)}")
    ax.set_xlabel("iteration t")
    ax.set_ylabel("S(t) = logarithm of stretching factor")
    
    # Add a text box in the upper left corner displaying the standard deviation of the LLE
    if np.isfinite(lle_sd):
        ax.text(
            0.02,
            0.98,
            f"LLE std across ε-blocks: {lle_sd:.5g}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
        )
        
    # Final styling adjustments
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    
    # Ensure the output directory exists, then save the figure
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig) # Free up memory
    print(f"Saved plot: {out_png}")


def main() -> None:
    """
    Main entry point for the CLI script. Handles argument parsing and execution flow.
    """
    parser = argparse.ArgumentParser(description="Plot TISEAN lyap_k S(t) blocks.")
    parser.add_argument("input", help="Path to a *_lyap.txt file produced by lyap_k.exe.")
    parser.add_argument(
        "--output",
        help="PNG path to write. Default: next to input as <name>_plot.png or <name>_lle_fit.png.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Plot only blocks with this embedding dimension (raw mode only).",
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=0,
        help="Limit number of plotted blocks after filtering (raw mode only). 0 means all.",
    )
    parser.add_argument(
        "--orig-lle-fit",
        action="store_true",
        help="Use hypothesis LLE logic: plot all epsilon curves and the representative linear fit.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive window after saving.",
    )
    
    # Parse the command-line arguments provided by the user
    args = parser.parse_args()

    # Route 1: The user requested the LLE fit overlay
    if args.orig_lle_fit:
        out = args.output
        if not out:
            # Autogenerate the output filename if not provided
            root, _ext = os.path.splitext(args.input)
            out = root + "_lle_fit.png"
        plot_orig_lle_fit(args.input, out)
        return

    # Route 2: The user just wants to plot the raw S(t) curves natively
    # Parse the raw lyap_k output file
    blocks = parse_lyap_k(args.input)
    
    # Apply user-defined filters
    if args.dim is not None:
        blocks = [block for block in blocks if block.dim == args.dim]
    if args.max_blocks and args.max_blocks > 0:
        blocks = blocks[: args.max_blocks]
        
    if not blocks:
        raise SystemExit("No lyap_k data blocks found for the requested filters.")

    # Autogenerate the output filename if not provided
    output = args.output
    if not output:
        root, _ext = os.path.splitext(args.input)
        output = root + "_plot.png"

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Iterate through all filtered blocks and plot their S(t) curves
    for index, block in enumerate(blocks, start=1):
        rows = block.rows
        # Skip blocks that don't have properly formatted 2D data
        if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 2:
            continue
        # Column 0 is time (x-axis), Column 1 is S(t) (y-axis)
        ax.plot(rows[:, 0], rows[:, 1], linewidth=1.0, label=block_label(index, block))

    # Apply aesthetics and save
    ax.set_title(f"lyap_k S(t) curves: {os.path.basename(args.input)}")
    ax.set_xlabel("iteration t")
    ax.set_ylabel("S(t) = logarithm of stretching factor")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    
    print(f"Saved plot: {output}")
    print(f"Plotted blocks: {len(blocks)}")

    # Display the interactive matplotlib GUI if requested
    if args.show:
        plt.show()

# Standard boilerplate to ensure main() is only called when script is executed directly
if __name__ == "__main__":
    main()