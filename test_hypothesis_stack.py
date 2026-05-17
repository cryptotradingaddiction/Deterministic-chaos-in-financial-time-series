#!/usr/bin/env python3
"""
Integration smoke tests for the split hypothesis / invariants stack.

This is **not** a self-contained unit-test suite:

- Requires local TISEAN executables under ``Tisean_3.0.0/bin`` (or ``TISEAN_BIN``).
- Requires at least one liquidity-cut ``*_logreturns_cut.dat`` (or set ``DCH_TEST_INPUT``).
- Skips with exit code 0 when prerequisites are missing (use ``--strict`` to fail instead).

Module import / ``py_compile`` checks always run when the script is invoked.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from config_loader import dch_test_point_count, get_data_dir

ROOT = Path(__file__).resolve().parent
TISEAN_BIN = ROOT / "Tisean_3.0.0" / "bin"
TEST_POINTS = dch_test_point_count()
BOOTSTRAP = max(1, int(float(os.environ.get("DCH_TEST_BOOTSTRAP", "3"))))

MODULES = [
    "hypothesis_config",
    "hypothesis_surrogates",
    "hypothesis_ts",
    "tisean_io",
    "invariants_correlation",
    "invariants_lyapunov",
    "invariants_rqa",
    "invariants_compute",
    "hypothesis",
    "rqa_radius",
    "rqa_values",
]

METRIC_RUNS = [
    ("ELLNER", []),
    ("LLE", []),
    (
        "RR,DET",
        [
            "--rqa_radius_mode",
            "fixed",
            "--rqa_radius",
            "0.01",
        ],
    ),
]


def _resolve_test_data() -> Path | None:
    env_path = os.environ.get("DCH_TEST_INPUT", "").strip()
    if env_path:
        candidate = Path(env_path)
        return candidate if candidate.is_file() else None
    default = ROOT / "data" / "BTCUSD_BITSTAMP_1h_complete_logreturns_cut.dat"
    if default.is_file():
        return default
    data_dir = Path(get_data_dir())
    if data_dir.is_dir():
        cuts = sorted(data_dir.glob("*_logreturns_cut.dat"))
        if cuts:
            return cuts[0]
    return None


def _compile_all() -> None:
    for name in MODULES:
        path = ROOT / f"{name}.py"
        if not path.is_file():
            raise FileNotFoundError(path)
        subprocess.check_call([sys.executable, "-m", "py_compile", str(path)])


def _import_smoke() -> None:
    for name in MODULES:
        __import__(name)
    from hypothesis import compute_percentile_radius, format_rqa_radius  # noqa: F401
    from invariants_compute import compute_invariants  # noqa: F401
    from config_loader import dch_test_results_tag

    assert dch_test_point_count() == TEST_POINTS
    assert dch_test_results_tag() == f"test_{TEST_POINTS}"


def _slice_input(src: Path, n: int) -> Path:
    lines = src.read_text(encoding="ascii").splitlines()
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".dat",
        delete=False,
        encoding="ascii",
        newline="\n",
    )
    tmp.write("\n".join(lines[:n]))
    tmp.write("\n")
    tmp.close()
    return Path(tmp.name)


def _run_hypothesis_cli(data_file: Path, metrics: str, extra: list[str], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "hypothesis.py"),
        "--input",
        str(data_file),
        "--base",
        "BTCUSD",
        "--delay",
        "2",
        "--theiler",
        "2",
        "--output_dir",
        str(out_dir),
        "--test_mode",
        "true",
        "--bootstrap_samples",
        str(BOOTSTRAP),
        "--seed",
        "42",
        "--metrics_list",
        metrics,
        *extra,
    ]
    env = os.environ.copy()
    env["PATH"] = str(TISEAN_BIN) + os.pathsep + env.get("PATH", "")
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)
    summary = out_dir / "BTCUSD_surrogate_summary.txt"
    if not summary.is_file():
        raise FileNotFoundError(summary)
    text = summary.read_text(encoding="utf-8")
    for token in metrics.split(","):
        token = token.strip()
        if token and token not in text:
            raise AssertionError(f"metric {token!r} missing from {summary}")
        for line in text.splitlines():
            if line.strip().startswith(token):
                if "insufficient data" in line or " no sd " in line:
                    raise AssertionError(
                        f"metric {token!r} failed: {line.strip()!r}"
                    )
                break
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when data or TISEAN tools are missing (default: skip integration).",
    )
    args = parser.parse_args()

    data_path = _resolve_test_data()
    missing_tisean = [
        exe for exe in ("d2.exe", "c2t.exe", "lyap_k.exe")
        if not (TISEAN_BIN / exe).is_file()
    ]

    print(f"[1/4] py_compile ({len(MODULES)} modules)")
    _compile_all()

    print("[2/4] import smoke")
    _import_smoke()

    if data_path is None:
        msg = "No test input .dat (set DCH_TEST_INPUT or run liquidity.py)"
        if args.strict:
            print(f"FAIL: {msg}", file=sys.stderr)
            return 1
        print(f"SKIP integration: {msg}")
        return 0

    if missing_tisean:
        msg = f"TISEAN tools missing under {TISEAN_BIN}: {', '.join(missing_tisean)}"
        if args.strict:
            print(f"FAIL: {msg}", file=sys.stderr)
            return 1
        print(f"SKIP integration: {msg}")
        return 0

    print(f"[3/4] rqa_radius on {TEST_POINTS} points ({data_path.name})")
    sliced = _slice_input(data_path, TEST_POINTS)
    try:
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "rqa_radius.py"),
                "--input",
                str(sliced),
                "--delay",
                "2",
                "--fallback",
                "0.005",
            ],
            cwd=str(ROOT),
        )

        with tempfile.TemporaryDirectory(prefix="hyp_stack_") as tmp:
            base = Path(tmp)
            for idx, (metrics, extra) in enumerate(METRIC_RUNS, start=1):
                tag = metrics.replace(",", "_")
                out = base / f"run_{idx}_{tag}"
                print(f"[4/4.{idx}] hypothesis.py metrics={metrics}")
                _run_hypothesis_cli(sliced, metrics, extra, out)
    finally:
        sliced.unlink(missing_ok=True)

    print("OK: hypothesis stack integration passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
