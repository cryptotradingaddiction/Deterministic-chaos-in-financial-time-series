#!/usr/bin/env python3
r"""Plot TISEAN `lyap_k` S(t) output and optional LLE fit (original series).

This script visualizes the stretching factor S(t) calculated by the Kantz algorithm
(via TISEAN's lyap_k tool) to estimate the Largest Lyapunov Exponent (LLE). 
The LLE is represented by the slope of the linear scaling region of these curves.

Standalone usage (raw curves only):

    py -3 plot_lyap_k_output.py data/results/lambda_max_test_100/BTCUSD_run2_tau2/BTCUSD_lyap.txt

With the Kantz / hypothesis LLE fit line (OLS slope of the highest-quality
ε-block; same rule as ``invariants_lyapunov.extract_lle_ols``):

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


def plot_orig_lle_fit(lyap_path: str, out_png: str, title: str | None = None) -> None:
    """
    Plot all epsilon S(t) curves and overlay the OLS linear fit chosen by
    :func:`invariants_lyapunov.extract_lle_ols`.

    The overlay corresponds to the **same** ε-block, window, and slope used by
    the hypothesis test (highest-quality block by ``(t_hi - t_lo) / std_err``),
    so the plot and the TS table cannot disagree.
    """
    from invariants_lyapunov import (
        M_LYAP,
        _parse_lyap_blocks,
        find_best_lle_block,
    )

    # Parse the blocks using the same parser as the hypothesis pipeline.
    blocks = _parse_lyap_blocks(lyap_path, dim=M_LYAP)
    if not blocks:
        print(f"WARNING: No lyap_k blocks parsed from {lyap_path}; skipping LLE fit plot.")
        return

    # Identical selection rule to extract_lle_ols.
    best, candidates = find_best_lle_block(lyap_path)
    n_blocks = len(candidates)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot all raw S(t) curves at m=M_LYAP. Legend is truncated to keep it readable.
    for i, blk in enumerate(blocks, start=1):
        d = blk["data"]
        if d.shape[0] < 2:
            continue
        lab = f"eps={blk['eps']:.3g}, n_med={blk['n_neighbors']}"
        ax.plot(d[:, 0], d[:, 1], linewidth=1.0, alpha=0.75, label=lab if i <= 12 else None)

    # Overlay the OLS fit on the chosen block's window.
    if best is not None:
        (_quality, best_slope, best_std_err, best_eps,
         best_t_lo, best_t_hi, best_intercept, _nn) = best
        tt = np.linspace(best_t_lo, best_t_hi,
                         max(50, int((best_t_hi - best_t_lo) * 4) + 1))
        ss = best_slope * tt + best_intercept
        ax.plot(
            tt, ss,
            color="crimson",
            linewidth=2.6,
            zorder=6,
            label=(
                f"LLE OLS={best_slope:.5g} ± {best_std_err:.3g} "
                f"(eps={best_eps:.3g}, n_blocks={n_blocks})"
            ),
        )
        ax.text(
            0.02, 0.98,
            f"OLS std_err of selected block's slope: {best_std_err:.5g}\n"
            f"window t=[{best_t_lo:.3g}, {best_t_hi:.3g}]",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
        )
    else:
        ax.text(
            0.02, 0.98,
            "extract_lle_ols: no block produced a finite (slope, std_err>0) fit.",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="mistyrose", alpha=0.5),
        )

    ax.set_title(title or f"lyap_k S(t) + LLE OLS fit: {os.path.basename(lyap_path)}")
    ax.set_xlabel("iteration t")
    ax.set_ylabel("S(t) = logarithm of stretching factor")
        
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