#!/usr/bin/env python3
"""
Compact result printer for TISEAN / hypothesis pipeline outputs.

Usage (called from .bat files):
    py -3 print_results.py KIND PATH

KIND is one of:
    file    PATH                      basic file info (size, line count)
    d2      PATH                      multi-block D2 file (*.d2)
    h2      PATH                      multi-block K2 file (*.h2)
    takens  PATH                      Takens estimator output (*_takens.dat)
    lyap    PATH                      lyap_k S(t) blocks (*_lyap.txt)
    rqa     PATH                      RQA metrics text file (rqa_values output)
    boot    PATH                      hypothesis.py surrogate-test summary
    boot_aggregate DIR                aggregate every *_surrogate_summary.txt under DIR
                                       and write _hypothesis_aggregate_summary.txt
    rec     PATH                      recurrence matrix (*_recurr.rec)
    head    PATH [N]                  head of any file (default 12 lines)

Each command prints a small, plain-text summary suitable for live console logs.
The script never raises on missing/empty files; it just prints a [WARN].
"""

import argparse
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _safe_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _line_count(path):
    n = 0
    try:
        with open(path, "rb") as f:
            for _ in f:
                n += 1
    except OSError:
        return 0
    return n


def read_blocks(path):
    """Multi-block TISEAN-style numeric reader.
    A blank line or a comment line ('#'/'!') closes a block."""
    blocks = []
    cur = []
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("!"):
                if cur:
                    blocks.append(np.array(cur, dtype=float))
                    cur = []
                continue
            parts = s.split()
            try:
                cur.append([float(p) for p in parts])
            except ValueError:
                continue
    if cur:
        blocks.append(np.array(cur, dtype=float))
    return [b for b in blocks if b.size > 0]


def _plateau(arr):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    n = a.size
    lo, hi = n // 4, max(n // 4 + 1, 3 * n // 4)
    return float(np.median(a[lo:hi]))


def _saturation(values, last_k=5):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    take = a[-last_k:] if a.size >= last_k else a
    return float(np.median(take))


def _slope_fit(x, y, lo=2, hi=10):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    lo = min(lo, max(0, n - 3))
    hi = min(hi, n)
    if hi - lo < 2:
        return float("nan")
    try:
        s, _ = np.polyfit(x[lo:hi], y[lo:hi], 1)
        return float(s)
    except Exception:
        return float("nan")


def cmd_file(path, _n=None):
    if not os.path.exists(path):
        print(f"  [WARN] Missing file: {path}")
        return
    print(f"  file: {os.path.basename(path)}  size={_safe_size(path):,} B  lines={_line_count(path):,}")


def cmd_head(path, n=12):
    if not os.path.exists(path):
        print(f"  [WARN] Missing file: {path}")
        return
    print(f"  head ({n} lines) of {os.path.basename(path)}:")
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                print(f"    {line.rstrip()}")
    except OSError as e:
        print(f"  [WARN] cannot read: {e}")


def _per_m_table(label, blocks, m_start, value_col=1):
    if not blocks:
        print(f"  [WARN] No data blocks ({label})")
        return
    print(f"  {label} per embedding m:")
    print(f"  {'m':>4}  {label[:10]:>10}  {'pts':>5}")
    plateaus = []
    for i, b in enumerate(blocks, start=m_start):
        if b.ndim < 2 or b.shape[1] <= value_col:
            continue
        v = _plateau(b[:, value_col])
        plateaus.append(v)
        print(f"  {i:>4}  {v:>10.4f}  {b.shape[0]:>5}")
    sat = _saturation(plateaus)
    print(f"  saturation estimate (median of last 5 m): {sat:.4f}")


def cmd_d2(path, _n=None):
    cmd_file(path)
    _per_m_table("D2", read_blocks(path), m_start=1)


def cmd_h2(path, _n=None):
    cmd_file(path)
    _per_m_table("K2", read_blocks(path), m_start=1)


def cmd_takens(path, _n=None):
    cmd_file(path)
    blocks = read_blocks(path)
    if not blocks:
        return
    last = blocks[-1]
    if last.ndim < 2 or last.shape[1] < 2:
        return
    finite = last[np.isfinite(last[:, 1])]
    if finite.size == 0:
        return
    median_dt = float(np.median(finite[:, 1]))
    last_dt = float(finite[-1, 1])
    print(f"  Takens D_T: median={median_dt:.4f}  last={last_dt:.4f}  rows={finite.shape[0]}")


def cmd_lyap(path, _n=None):
    """lyap_k writes one block per (epsilon, m). Group by dim and pick the
    first length scale per m to avoid counting epsilon variants as new m's."""
    import re
    cmd_file(path)
    if not os.path.exists(path):
        return
    by_dim = {}
    cur_dim = None
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.startswith("#"):
                if cur_dim is not None and rows and cur_dim not in by_dim:
                    by_dim[cur_dim] = np.array(rows, dtype=float)
                rows = []
                m = re.search(r"dim=\s*(\d+)", s)
                cur_dim = int(m.group(1)) if m else None
                continue
            if not s:
                continue
            parts = s.split()
            try:
                rows.append([float(parts[0]), float(parts[1])])
            except (ValueError, IndexError):
                continue
    if cur_dim is not None and rows and cur_dim not in by_dim:
        by_dim[cur_dim] = np.array(rows, dtype=float)
    if not by_dim:
        print("  [WARN] No lyap blocks parsed")
        return
    print("  Largest Lyapunov lambda per embedding m (slope on iter 2..10, first epsilon block):")
    print(f"  {'m':>4}  {'lambda':>10}  {'pts':>5}")
    lambdas = []
    for m in sorted(by_dim.keys()):
        b = by_dim[m]
        lam = _slope_fit(b[:, 0], b[:, 1])
        lambdas.append(lam)
        print(f"  {m:>4}  {lam:>10.5f}  {b.shape[0]:>5}")
    sat = _saturation(lambdas)
    print(f"  saturation lambda (median of last 5 m): {sat:.5f}")
    if np.isfinite(sat) and sat > 0:
        T = (1.0 / sat) * np.log(1e-2 / 1e-5)
        print(f"  predictability time T (eps=1e-5, L=1e-2): {T:.2f} h ({T/24:.2f} d)")


def cmd_rec(path, _n=None):
    cmd_file(path)


def cmd_rqa(path, _n=None):
    if not os.path.exists(path):
        print(f"  [WARN] Missing RQA metrics file: {path}")
        return
    print(f"  --- RQA metrics ({os.path.basename(path)}) ---")
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                print(f"    {line.rstrip()}")
    except OSError as e:
        print(f"  [WARN] cannot read: {e}")


def cmd_boot(path, _n=None):
    if not os.path.exists(path):
        print(f"  [WARN] Surrogate-test summary missing: {path}")
        return
    print(f"  --- Surrogate-test summary ({os.path.basename(path)}) ---")
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                print(f"  {line.rstrip()}")
    except OSError as e:
        print(f"  [WARN] cannot read: {e}")
    print("  --- end of summary ---")


def _parse_bootstrap_summary(path):
    """Extract a compact dict of values from one *_surrogate_summary.txt."""
    import re
    info = {
        "symbol": "",
        "tau": "",
        "W": "",
        "B": "",
        "n": "",
        "mode": "",
        "metrics": {},          # orig, std_orig, mean, std, se_surr, z_sigma, pvalue; legacy 'score' = z_sigma
        "conclusion": {},       # name -> "reject H0" / "fail to reject H0" / "insufficient data"
        "T_original": "",
        "T_bootstrap": "",
    }
    metric_names_row = ("D2", "K2", "LLE", "RR", "DET", "LAM", "MAXLINE", "ENTR", "TT")
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None

    m = re.search(r"Permutation surrogate hypothesis test\s+\(([^)]+)\)", text)
    if m:
        info["symbol"] = m.group(1).strip()
    else:
        m = re.search(r"Bootstrap permutation test\s+\(([^)]+)\)", text)
    if m:
        info["symbol"] = m.group(1).strip()
    else:
        m = re.search(r"Bootstrap Permutation Test Results for\s+(\S+)", text)
        if m:
            info["symbol"] = m.group(1).strip()
    m = re.search(r"Parameters:\s*tau=(\d+),\s*W=(\d+),\s*B=(\d+)", text)
    if m:
        info["tau"], info["W"], info["B"] = m.group(1), m.group(2), m.group(3)
    m = re.search(r"Original data length:\s*(\d+)", text)
    if m:
        info["n"] = m.group(1)
    m = re.search(r"Mode:\s*(\S+)", text)
    if m:
        info["mode"] = m.group(1)

    use_std_orig_row = bool(re.search(r"Invariant\s+orig_mean\s+std_orig", text))
    use_se_surr_col = bool(re.search(r"SD\(surr\)\s+SE\(surr\)", text))

    # hypothesis.py v5: ... SD(surr) SE(surr) z_sigma z_SE p-value decision
    row_re_empirical_z_std_se = re.compile(
        r"^(?P<name>\S+)\s+(?P<orig>-?(?:nan|inf|\d+\.\d+))\s+(?P<std_orig>-?(?:nan|inf|\d+\.\d+))\s+"
        r"(?P<mean>-?(?:nan|inf|\d+\.\d+))\s+(?P<std>-?(?:nan|inf|\d+\.\d+))\s+"
        r"(?P<se_surr>-?(?:nan|inf|\d+\.\d+))\s+"
        r"(?P<z_sigma>-?(?:nan|inf|\d+\.\d+))\s+(?P<z_se>-?(?:nan|inf|\d+\.\d+))\s+"
        r"(?P<pvalue>-?(?:nan|inf|\d*\.?\d+(?:[eE][+-]?\d+)?))\s+"
        r"(?P<decision>reject H0|fail to reject H0|insufficient data)\s*$"
    )
    # hypothesis.py v4: orig_mean std_orig Mean(surr) SD(surr) z_sigma z_SE p-value decision
    row_re_empirical_z_std = re.compile(
        r"^(?P<name>\S+)\s+(?P<orig>-?(?:nan|inf|\d+\.\d+))\s+(?P<std_orig>-?(?:nan|inf|\d+\.\d+))\s+"
        r"(?P<mean>-?(?:nan|inf|\d+\.\d+))\s+(?P<std>-?(?:nan|inf|\d+\.\d+))\s+"
        r"(?P<z_sigma>-?(?:nan|inf|\d+\.\d+))\s+(?P<z_se>-?(?:nan|inf|\d+\.\d+))\s+"
        r"(?P<pvalue>-?(?:nan|inf|\d*\.?\d+(?:[eE][+-]?\d+)?))\s+"
        r"(?P<decision>reject H0|fail to reject H0|insufficient data)\s*$"
    )
    # hypothesis.py v3 (no std_orig column)
    row_re_empirical_z = re.compile(
        r"^(?P<name>\S+)\s+(?P<orig>-?(?:nan|inf|\d+\.\d+))\s+(?P<mean>-?(?:nan|inf|\d+\.\d+))\s+"
        r"(?P<std>-?(?:nan|inf|\d+\.\d+))\s+(?P<z_sigma>-?(?:nan|inf|\d+\.\d+))\s+"
        r"(?P<z_se>-?(?:nan|inf|\d+\.\d+))\s+(?P<pvalue>-?(?:nan|inf|\d*\.?\d+(?:[eE][+-]?\d+)?))\s+"
        r"(?P<decision>reject H0|fail to reject H0|insufficient data)\s*$"
    )
    row_re_new_no_sem = re.compile(
        r"^(?P<name>\S+)\s+(?P<orig>-?\d+\.\d+)\s+(?P<mean>-?\d+\.\d+)\s+"
        r"(?P<std>-?\d+\.\d+)\s+(?P<score>-?(?:nan|inf|\d+\.\d+))\s*$"
    )
    row_re_new_no_sem_p = re.compile(
        r"^(?P<name>\S+)\s+(?P<orig>-?\d+\.\d+)\s+(?P<mean>-?\d+\.\d+)\s+"
        r"(?P<std>-?\d+\.\d+)\s+(?P<score>-?(?:nan|inf|\d+\.\d+))\s+"
        r"(?P<pvalue>-?(?:nan|inf|\d*\.?\d+(?:[eE][+-]?\d+)?))\s*$"
    )
    row_re_new_with_sem = re.compile(
        r"^(?P<name>\S+)\s+(?P<orig>-?\d+\.\d+)\s+(?P<mean>-?\d+\.\d+)\s+"
        r"(?P<std>-?\d+\.\d+)\s+(?P<sem>-?(?:nan|inf|\d+\.\d+))\s+(?P<score>-?(?:nan|inf|\d+\.\d+))\s*$"
    )
    row_re_new_with_sem_p = re.compile(
        r"^(?P<name>\S+)\s+(?P<orig>-?\d+\.\d+)\s+(?P<mean>-?\d+\.\d+)\s+"
        r"(?P<std>-?\d+\.\d+)\s+(?P<sem>-?(?:nan|inf|\d+\.\d+))\s+"
        r"(?P<score>-?(?:nan|inf|\d+\.\d+))\s+(?P<pvalue>-?(?:nan|inf|\d*\.?\d+(?:[eE][+-]?\d+)?))\s*$"
    )
    row_re_old = re.compile(
        r"^(?P<name>\S+)\s+(?P<orig>-?\d+\.\d+)\s+(?P<mean>-?\d+\.\d+)\s*\+\-\s*"
        r"(?P<std>-?\d+\.\d+)\s+(?P<score>-?\d+\.\d+)\s*$"
    )
    def _parse_float_tok(s):
        sl = (s or "").strip().lower()
        if sl == "nan":
            return float("nan")
        if sl in ("inf", "+inf"):
            return float("inf")
        if sl == "-inf":
            return float("-inf")
        return float(s)

    def _fill_metric_row(info, name, orig, std_orig, mean, std, se_surr, zs, zse, pvalue, decision):
        info["metrics"][name] = {
            "orig": orig,
            "std_orig": std_orig,
            "mean": mean,
            "std": std,
            "se_surr": se_surr,
            "z_sigma": zs,
            "z_se": zse,
            "score": zs,
            "pvalue": pvalue,
        }
        info["conclusion"][name] = decision

    for line in text.splitlines():
        line_st = line.strip()
        rm = row_re_empirical_z_std_se.match(line_st) if use_se_surr_col else None
        if rm is not None and rm.group("name") in metric_names_row:
            name = rm.group("name")
            zs = _parse_float_tok(rm.group("z_sigma"))
            zse = _parse_float_tok(rm.group("z_se"))
            pv_raw = rm.group("pvalue")
            if pv_raw == "nan":
                pvalue = float("nan")
            elif pv_raw in ("inf", "+inf"):
                pvalue = float("inf")
            elif pv_raw == "-inf":
                pvalue = float("-inf")
            else:
                pvalue = float(pv_raw)
            _fill_metric_row(
                info,
                name,
                _parse_float_tok(rm.group("orig")),
                _parse_float_tok(rm.group("std_orig")),
                _parse_float_tok(rm.group("mean")),
                _parse_float_tok(rm.group("std")),
                _parse_float_tok(rm.group("se_surr")),
                zs,
                zse,
                pvalue,
                rm.group("decision"),
            )
            continue

        rm = row_re_empirical_z_std.match(line_st) if use_std_orig_row else None
        if rm is not None and rm.group("name") in metric_names_row:
            name = rm.group("name")
            zs = _parse_float_tok(rm.group("z_sigma"))
            zse = _parse_float_tok(rm.group("z_se"))
            pv_raw = rm.group("pvalue")
            if pv_raw == "nan":
                pvalue = float("nan")
            elif pv_raw in ("inf", "+inf"):
                pvalue = float("inf")
            elif pv_raw == "-inf":
                pvalue = float("-inf")
            else:
                pvalue = float(pv_raw)
            _fill_metric_row(
                info,
                name,
                _parse_float_tok(rm.group("orig")),
                _parse_float_tok(rm.group("std_orig")),
                _parse_float_tok(rm.group("mean")),
                _parse_float_tok(rm.group("std")),
                float("nan"),
                zs,
                zse,
                pvalue,
                rm.group("decision"),
            )
            continue

        rm = row_re_empirical_z.match(line_st)
        if rm is not None and rm.group("name") in metric_names_row:
            name = rm.group("name")
            zs = _parse_float_tok(rm.group("z_sigma"))
            zse = _parse_float_tok(rm.group("z_se"))
            pv_raw = rm.group("pvalue")
            if pv_raw == "nan":
                pvalue = float("nan")
            elif pv_raw in ("inf", "+inf"):
                pvalue = float("inf")
            elif pv_raw == "-inf":
                pvalue = float("-inf")
            else:
                pvalue = float(pv_raw)
            _fill_metric_row(
                info,
                name,
                _parse_float_tok(rm.group("orig")),
                float("nan"),
                _parse_float_tok(rm.group("mean")),
                _parse_float_tok(rm.group("std")),
                float("nan"),
                zs,
                zse,
                pvalue,
                rm.group("decision"),
            )
            continue

        rm = row_re_new_no_sem.match(line_st)
        if rm is None:
            rm = row_re_new_no_sem_p.match(line_st)
        if rm is None:
            rm = row_re_new_with_sem.match(line_st)
        if rm is None:
            rm = row_re_new_with_sem_p.match(line_st)
        if rm is None or rm.group("name") not in metric_names_row:
            rm = row_re_old.match(line_st)
            if rm is None or rm.group("name") not in metric_names_row:
                continue
            name = rm.group("name")
            score = float(rm.group("score"))
            info["metrics"][name] = {
                "orig": float(rm.group("orig")),
                "std_orig": float("nan"),
                "mean": float(rm.group("mean")),
                "std": float(rm.group("std")),
                "se_surr": float("nan"),
                "z_sigma": score,
                "score": score,
                "pvalue": float("nan"),
            }
        else:
            name = rm.group("name")
            sc_raw = rm.group("score")
            if sc_raw == "nan":
                score = float("nan")
            elif sc_raw == "inf":
                score = float("inf")
            else:
                score = float(sc_raw)
            pv_raw = rm.groupdict().get("pvalue")
            if pv_raw is None or pv_raw == "nan":
                pvalue = float("nan")
            elif pv_raw == "inf":
                pvalue = float("inf")
            else:
                pvalue = float(pv_raw)
            info["metrics"][name] = {
                "orig": float(rm.group("orig")),
                "std_orig": float("nan"),
                "mean": float(rm.group("mean")),
                "std": float(rm.group("std")),
                "se_surr": float("nan"),
                "z_sigma": score,
                "score": score,
                "pvalue": pvalue,
            }

    for name in metric_names_row:
        rm = re.search(
            rf"^\s*{re.escape(name)}\s*:\s*(reject H0|fail to reject H0|insufficient data)",
            text,
            re.MULTILINE,
        )
        if rm:
            info["conclusion"][name] = rm.group(1)

    m = re.search(r"Original T\s*:\s*(.*)", text)
    if m:
        info["T_original"] = m.group(1).strip()
    m = re.search(r"(Bootstrap|Surrogate) T\s*:\s*(.*)", text)
    if m:
        info["T_bootstrap"] = m.group(2).strip()
    return info


def cmd_boot_aggregate(path, _n=None):
    """Walk `path` for *_surrogate_summary.txt files and write an aggregated
    overview as `<path>/_hypothesis_aggregate_summary.txt`."""
    if not os.path.isdir(path):
        print(f"  [WARN] boot_aggregate: not a directory: {path}")
        return

    found = []
    for root, _dirs, files in os.walk(path):
        for fn in files:
            if fn.endswith("_surrogate_summary.txt"):
                found.append(os.path.join(root, fn))
    found.sort()
    if not found:
        print(f"  [WARN] boot_aggregate: no *_surrogate_summary.txt under {path}")
        return

    out_path = os.path.join(path, "_hypothesis_aggregate_summary.txt")
    metric_order = ("D2", "K2", "LLE", "RR", "DET", "LAM", "MAXLINE", "ENTR", "TT")
    # PyRQA scalar metrics: no plateau-window std_orig in hypothesis.py (column is absent / NaN in summaries).
    RQA_METRICS = ("RR", "DET", "LAM", "MAXLINE", "ENTR", "TT")
    DKL_METRICS = ("D2", "K2", "LLE")
    parsed_infos = []
    metrics_present = []
    for fp in found:
        info = _parse_bootstrap_summary(fp)
        if info is None:
            continue
        parsed_infos.append((fp, info))
        metrics_present.extend([k for k in metric_order if k in info.get("metrics", {})])
    metric_names = tuple(dict.fromkeys(metrics_present)) if metrics_present else ("D2", "K2", "LLE")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("Hypothesis pipeline - aggregated summary\n")
        fh.write(f"Scanned root : {path}\n")
        fh.write(f"Files found  : {len(found)}\n")
        fh.write("=" * 110 + "\n\n")

        w_om = 13  # "{metric}_orig_mean" column (e.g. D2_orig_mean)
        w_so = 10  # "{metric}_std_orig"
        header = f"{'Symbol':<8} {'tau':>4} {'W':>3} {'B':>5}"
        for name in metric_names:
            header += (
                f" {f'{name}_orig_mean':>{w_om}} {f'{name}_std_orig':>{w_so}} {f'z_{name}':>7}"
            )
        header += f" {'rej_all':>7}"
        fh.write(header + "\n")
        fh.write("-" * len(header) + "\n")

        for fp, info in parsed_infos:
            if info is None:
                fh.write(f"[WARN] cannot read {fp}\n")
                continue
            row = (
                f"{info['symbol']:<8} {info['tau']:>4} {info['W']:>3} "
                f"{info['B']:>5}"
            )
            for name in metric_names:
                m = info["metrics"].get(name)
                if m is None:
                    row += f" {'nan':>{w_om}} {'nan':>{w_so}} {'nan':>7}"
                else:
                    zv = m.get("z_sigma", m.get("score", float("nan")))
                    so = m.get("std_orig", float("nan"))
                    # RQA metrics have no within-curve std_orig (single PyRQA scalar per series).
                    if name in RQA_METRICS and not np.isfinite(float(so)):
                        so_cell = f"{'—':>{w_so}}"
                    elif np.isfinite(float(so)):
                        so_cell = f"{float(so):>{w_so}.4f}"
                    else:
                        so_cell = f"{'nan':>{w_so}}"
                    row += f" {m['orig']:>{w_om}.4f} {so_cell} {float(zv):>7.4f}"
            has_dkl = all(n in metric_names for n in DKL_METRICS)
            only_rqa = bool(metric_names) and all(n in RQA_METRICS for n in metric_names)
            if has_dkl:
                rej_all = (
                    "YES"
                    if all(info["conclusion"].get(n) == "reject H0" for n in DKL_METRICS)
                    else "NO"
                )
            elif only_rqa:
                rej_all = (
                    "YES"
                    if all(info["conclusion"].get(n) == "reject H0" for n in metric_names)
                    else "NO"
                )
            else:
                rej_all = "—"
            row += f" {rej_all:>7}"
            fh.write(row + "\n")

        fh.write(
            "\nColumn rej_all: YES if every metric in scope rejects H0 at the run alpha (default 0.01). "
            "DKL scope: requires D2, K2, LLE in the summary. RQA-only scope: requires all listed metrics "
            f"from {RQA_METRICS} only. "
            "Mixed or incomplete scopes: — . "
            "std_orig em dash (—): not defined for PyRQA scalars (hypothesis reports NaN).\n\n"
        )

        for fp, info in parsed_infos:
            if info is None:
                continue
            fh.write("=" * 110 + "\n")
            fh.write(f"Source: {fp}\n")
            fh.write(f"Symbol: {info['symbol']}   Mode: {info['mode']}   N: {info['n']}\n")
            fh.write(f"  tau={info['tau']}, W={info['W']}, B={info['B']}\n")
            fh.write(
                f"  {'metric':<6} {'orig_mean':>10} {'std_orig':>10} {'surr_mean':>12} {'surr_std':>12} "
                f"{'SE(surr)':>10} {'z_sigma':>9} {'p-value':>11}  conclusion\n"
            )
            for name in metric_names:
                m = info["metrics"].get(name, {})
                con = info["conclusion"].get(name, "n/a")
                if m:
                    _pv = m.get("pvalue", float("nan"))
                    _pv_s = f"{float(_pv):>11.4f}" if np.isfinite(float(_pv)) else f"{'nan':>11}"
                    zv = m.get("z_sigma", m.get("score", float("nan")))
                    _so = m.get("std_orig", float("nan"))
                    if name in RQA_METRICS and not np.isfinite(float(_so)):
                        _so_s = f"{'—':>10}"
                    elif np.isfinite(float(_so)):
                        _so_s = f"{float(_so):>10.4f}"
                    else:
                        _so_s = f"{'nan':>10}"
                    _se = m.get("se_surr", float("nan"))
                    _se_s = (
                        f"{float(_se):>10.4f}"
                        if np.isfinite(float(_se))
                        else f"{'nan':>10}"
                    )
                    fh.write(
                        f"  {name:<6} {m['orig']:>10.4f} {_so_s} {m['mean']:>12.4f} "
                        f"{m['std']:>12.4f} {_se_s} {float(zv):>9.4f} {_pv_s}  {con}\n"
                    )
                else:
                    fh.write(
                        f"  {name:<6} {'nan':>10} {'nan':>10} {'nan':>12} {'nan':>12} {'nan':>10} {'nan':>9} {'nan':>11}  {con}\n"
                    )
            if info["T_original"]:
                fh.write(f"  Predictability T (original) : {info['T_original']}\n")
            if info["T_bootstrap"]:
                fh.write(f"  Predictability T (surrogates): {info['T_bootstrap']}\n")
            fh.write("\n")

    print(f"  Aggregated hypothesis summary -> {out_path}")
    try:
        with open(out_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                print(f"  {line.rstrip()}")
    except OSError as e:
        print(f"  [WARN] cannot read aggregate: {e}")


HANDLERS = {
    "file": cmd_file,
    "head": cmd_head,
    "d2": cmd_d2,
    "h2": cmd_h2,
    "takens": cmd_takens,
    "lyap": cmd_lyap,
    "rec": cmd_rec,
    "rqa": cmd_rqa,
    "boot": cmd_boot,
    "boot_aggregate": cmd_boot_aggregate,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=sorted(HANDLERS.keys()))
    parser.add_argument("path")
    parser.add_argument("n", nargs="?", type=int, default=None)
    args = parser.parse_args(argv)
    HANDLERS[args.kind](args.path, args.n)


if __name__ == "__main__":
    main()
