"""
Lightweight helper to capture analysis output and write it to text files.

Pattern:
    from report_helper import Reporter, append_summary_row
    r = Reporter()
    r.add("=== Header ===")
    r.add(f"value = {x}")
    r.write(out_dir, "btc_results.txt")
"""

from __future__ import annotations

import os
from typing import List


class Reporter:
    """Collects printed lines and writes them to a text file at the end."""

    def __init__(self) -> None:
        self.lines: List[str] = []

    def add(self, line: str = "") -> None:
        print(line)
        self.lines.append(line)

    def add_many(self, lines: List[str]) -> None:
        for line in lines:
            self.add(line)

    def hr(self, char: str = "-", n: int = 70) -> None:
        self.add(char * n)

    def write(self, output_dir: str, filename: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.lines))
            if self.lines and not self.lines[-1].endswith("\n"):
                fh.write("\n")
        return out_path


def append_summary_row(
    output_dir: str,
    filename: str,
    header: str,
    row: str,
    fresh: bool = False,
) -> str:
    """Append `row` to `output_dir/filename`. If the file is empty/new (or
    `fresh=True`), write `header` first."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    if fresh:
        try:
            os.remove(out_path)
        except FileNotFoundError:
            pass
    new = (not os.path.exists(out_path)) or os.path.getsize(out_path) == 0
    with open(out_path, "a", encoding="utf-8") as fh:
        if new:
            fh.write(header.rstrip("\n") + "\n")
        fh.write(row.rstrip("\n") + "\n")
    return out_path
