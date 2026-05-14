#!/usr/bin/env python3
r"""Plot TISEAN `lyap_k` S(t) output and optional LLE fit (original series).

Standalone usage (raw curves only):

    py -3 plot_lyap_k_output.py C:\DCh\data\results\lambda_max_test_2000\BTCUSD_run2_tau2\BTCUSD_lyap.txt

With the Kantz / hypothesis LLE fit line (median LLE across epsilon blocks; one
representative block's linear window — same rule as `hypothesis.extract_lle_mean_std`):

    py -3 plot_lyap_k_output.py PATH\BASE_lyap.txt --orig-lle-fit --output PATH\BASE_lyap_lle_fit.png

Each `#epsilon= ... dim= ...` block is one S(t) curve:
  column 1 = iteration t
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
    epsilon: float | None
    dim: int | None
    rows: np.ndarray


def parse_lyap_k(path: str) -> list[LyapBlock]:
    blocks: list[LyapBlock] = []
    epsilon: float | None = None
    dim: int | None = None
    rows: list[list[float]] = []

    def flush() -> None:
        nonlocal rows, epsilon, dim
        if rows:
            blocks.append(LyapBlock(epsilon=epsilon, dim=dim, rows=np.asarray(rows, dtype=float)))
        rows = []

    with open(path, encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                flush()
                eps_match = re.search(r"epsilon=\s*([+\-0-9.eE]+)", line)
                dim_match = re.search(r"dim=\s*(\d+)", line)
                epsilon = float(eps_match.group(1)) if eps_match else None
                dim = int(dim_match.group(1)) if dim_match else None
                continue

            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                iteration = float(parts[0])
                stretching_log = float(parts[1])
                neighbour_points = float(parts[2]) if len(parts) >= 3 else np.nan
            except ValueError:
                continue
            rows.append([iteration, stretching_log, neighbour_points])

    flush()
    return blocks


def block_label(index: int, block: LyapBlock) -> str:
    parts = [f"block {index}"]
    if block.dim is not None:
        parts.append(f"m={block.dim}")
    if block.epsilon is not None:
        parts.append(f"eps={block.epsilon:.3g}")
    if block.rows.size and block.rows.shape[1] >= 3 and np.isfinite(block.rows[:, 2]).any():
        parts.append(f"n~{np.nanmax(block.rows[:, 2]):.0f}")
    return ", ".join(parts)


def _block_slopes_for_lle_plot(blocks, min_neighbors):
    """Mirror `extract_lle_mean_std` block filtering; return (block, slope) pairs."""
    from hypothesis import MIN_LYAP_NEIGHBORS, _best_linear_slope

    mn = int(min_neighbors) if min_neighbors is not None else int(MIN_LYAP_NEIGHBORS)
    pairs = []
    for blk in blocks:
        if blk["n_neighbors"] < mn:
            continue
        data = blk["data"]
        if data.shape[0] < 3:
            continue
        slope = _best_linear_slope(data[:, 0], data[:, 1])
        if np.isfinite(slope):
            pairs.append((blk, float(slope)))

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
    """Plot all epsilon S(t) curves plus the linear fit used for the reported LLE (orig series)."""
    from hypothesis import (
        M_LYAP,
        MIN_LYAP_NEIGHBORS,
        _best_linear_slope_window,
        _parse_lyap_blocks,
        extract_lle_mean_std,
    )

    blocks = _parse_lyap_blocks(lyap_path, dim=M_LYAP)
    if not blocks:
        raise SystemExit(f"No lyap_k blocks parsed from {lyap_path}")

    lle, lle_sd, n_blk = extract_lle_mean_std(lyap_path)
    pairs = _block_slopes_for_lle_plot(blocks, MIN_LYAP_NEIGHBORS)

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, blk in enumerate(blocks, start=1):
        d = blk["data"]
        if d.shape[0] < 2:
            continue
        lab = f"eps={blk['eps']:.3g}, n_med={blk['n_neighbors']}"
        ax.plot(d[:, 0], d[:, 1], linewidth=1.0, alpha=0.75, label=lab if i <= 12 else None)

    if np.isfinite(lle) and pairs:
        best_blk, best_slope = min(pairs, key=lambda p: abs(p[1] - lle))
        data = best_blk["data"]
        slope_w, t0, t1, intercept = _best_linear_slope_window(data[:, 0], data[:, 1])
        if np.isfinite(slope_w) and np.isfinite(t0) and np.isfinite(t1) and np.isfinite(intercept):
            tt = np.linspace(t0, t1, max(50, int((t1 - t0) * 4) + 1))
            ss = slope_w * tt + intercept
            ax.plot(
                tt,
                ss,
                color="crimson",
                linewidth=2.6,
                zorder=6,
                label=(
                    f"LLE median={lle:.5g} (n_blocks={n_blk}); "
                    f"fit window eps={best_blk['eps']:.3g}, local slope={best_slope:.5g}"
                ),
            )

    ax.set_title(title or f"lyap_k S(t) + LLE fit: {os.path.basename(lyap_path)}")
    ax.set_xlabel("iteration t")
    ax.set_ylabel("S(t) = logarithm of stretching factor")
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
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_png}")


def main() -> None:
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
    args = parser.parse_args()

    if args.orig_lle_fit:
        out = args.output
        if not out:
            root, _ext = os.path.splitext(args.input)
            out = root + "_lle_fit.png"
        plot_orig_lle_fit(args.input, out)
        return

    blocks = parse_lyap_k(args.input)
    if args.dim is not None:
        blocks = [block for block in blocks if block.dim == args.dim]
    if args.max_blocks and args.max_blocks > 0:
        blocks = blocks[: args.max_blocks]
    if not blocks:
        raise SystemExit("No lyap_k data blocks found for the requested filters.")

    output = args.output
    if not output:
        root, _ext = os.path.splitext(args.input)
        output = root + "_plot.png"

    fig, ax = plt.subplots(figsize=(12, 7))
    for index, block in enumerate(blocks, start=1):
        rows = block.rows
        if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 2:
            continue
        ax.plot(rows[:, 0], rows[:, 1], linewidth=1.0, label=block_label(index, block))

    ax.set_title(f"lyap_k S(t) curves: {os.path.basename(args.input)}")
    ax.set_xlabel("iteration t")
    ax.set_ylabel("S(t) = logarithm of stretching factor")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved plot: {output}")
    print(f"Plotted blocks: {len(blocks)}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
