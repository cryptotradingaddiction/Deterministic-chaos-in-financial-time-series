#!/usr/bin/env python3
"""
Compact result printer for TISEAN / hypothesis pipeline outputs.

Usage (called from .bat files):
    py -3 print_results.py KIND PATH

KIND is one of:
    file    PATH                      basic file info (size, line count)
    d2      PATH                      diagnostic multi-block local D2 file (*.d2)
    h2      PATH                      legacy multi-block K2 file (*.h2)
    takens  PATH                      Takens estimator output (*_takens.dat)
    ellner_plot_data PATH              gnuplot data for Ellner interval estimates
    takens_value PATH                  CSV row with m=3 Ellner value from the Takens plateau
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


def read_tagged_block(path, dim=3, tag="#dim"):
    rows = []
    current_dim = None
    if not os.path.exists(path):
        return np.empty((0, 2), dtype=float)
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(tag):
                try:
                    current_dim = int(s.split("=")[1].strip())
                except (ValueError, IndexError):
                    current_dim = None
                continue
            if s.startswith("#") or s.startswith("!"):
                continue
            if current_dim != dim:
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return np.array(rows, dtype=float)


def _stable_plateau_values(block, value_col=1, min_points=8):
    """Return `(y_values, r_min, r_max)` for the best plateau window.

    Mirrors `hypothesis.select_plateau_values`. `r_min` / `r_max` are NaN when
    no usable rows are present.
    """
    b = np.asarray(block, dtype=float)
    if b.size == 0 or b.ndim < 2 or b.shape[1] <= value_col:
        return np.array([], dtype=float), float("nan"), float("nan")
    eps = b[:, 0]
    values = b[:, value_col]
    mask = np.isfinite(eps) & np.isfinite(values) & (eps > 0.0) & (values > 0.0)
    eps = eps[mask]
    values = values[mask]
    if values.size == 0:
        return np.array([], dtype=float), float("nan"), float("nan")
    order = np.argsort(np.log(eps))
    eps_sorted = eps[order]
    x = np.log(eps_sorted)
    y = values[order]
    n = y.size
    if n < min_points:
        return y, float(eps_sorted[0]), float(eps_sorted[-1])
    best_score = -np.inf
    best_ij = (0, n)
    for i in range(0, n - min_points + 1):
        for j in range(i + min_points, n + 1):
            xs = x[i:j]
            ys = y[i:j]
            mean_abs = abs(float(np.mean(ys))) + 1e-12
            try:
                slope, _ = np.polyfit(xs, ys, 1)
            except Exception:
                continue
            rel_slope = abs(float(slope)) / mean_abs
            rel_sd = float(np.std(ys, ddof=1)) / mean_abs if ys.size > 1 else np.inf
            length_bonus = (j - i) / n
            score = 0.10 * length_bonus - rel_slope - rel_sd
            if score > best_score:
                best_score = score
                best_ij = (i, j)
    i, j = best_ij
    return y[i:j], float(eps_sorted[i]), float(eps_sorted[j - 1])


def _plateau(block, value_col=1):
    vals, _r_min, _r_max = _stable_plateau_values(block, value_col=value_col)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def _ellner_from_c2(c2_path, r_min, r_max, dim=3):
    """Replicate `hypothesis.compute_ellner_from_c2` to keep cmd_takens_value in sync."""
    if not os.path.exists(c2_path):
        return float("nan")
    rows = read_tagged_block(c2_path, dim=dim, tag="#dim")
    if rows.size == 0:
        return float("nan")
    if not (np.isfinite(r_min) and np.isfinite(r_max)) or r_min <= 0.0 or r_max <= r_min:
        return float("nan")
    r = rows[:, 0]
    c = rows[:, 1]
    finite = np.isfinite(r) & np.isfinite(c) & (r > 0.0) & (c > 0.0)
    r = r[finite]
    c = c[finite]
    if r.size < 2:
        return float("nan")
    order = np.argsort(r)
    r = r[order]
    c = c[order]
    mask = (r >= r_min) & (r <= r_max)
    if int(mask.sum()) < 2:
        return float("nan")
    r_sel = r[mask]
    c_sel = c[mask]
    c_max = float(np.interp(r_max, r_sel, c_sel))
    c_min = float(np.interp(r_min, r_sel, c_sel))
    if not (np.isfinite(c_max) and np.isfinite(c_min)) or c_max <= c_min:
        return float("nan")
    integrand = c_sel / r_sel
    _trapz = getattr(np, "trapezoid", np.trapz)
    integral = float(_trapz(integrand, r_sel))
    if not np.isfinite(integral) or integral <= 0.0:
        return float("nan")
    return float((c_max - c_min) / integral)


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


def _per_m_table(label, blocks, m_start, value_col=1, value_label=None, saturation_label=None):
    if not blocks:
        print(f"  [WARN] No data blocks ({label})")
        return
    print(f"  {label} per embedding m:")
    if value_label is None:
        value_label = label[:10]
    print(f"  {'m':>4}  {value_label[:18]:>18}  {'pts':>5}")
    plateaus = []
    for i, b in enumerate(blocks, start=m_start):
        if b.ndim < 2 or b.shape[1] <= value_col:
            continue
        v = _plateau(b, value_col=value_col)
        plateaus.append(v)
        print(f"  {i:>4}  {v:>18.4f}  {b.shape[0]:>5}")
    sat = _saturation(plateaus)
    if saturation_label is None:
        saturation_label = "saturation estimate (median of last 5 m)"
    print(f"  {saturation_label}: {sat:.4f}")


def cmd_d2(path, _n=None):
    cmd_file(path)
    print("  Diagnostic only: local D2 slopes are not the active hypothesis metric.")
    _per_m_table(
        "Diagnostic local D2 slopes",
        read_blocks(path),
        m_start=1,
        value_label="D2 plateau",
        saturation_label="diagnostic median of last 5 m",
    )


def cmd_h2(path, _n=None):
    cmd_file(path)
    _per_m_table("K2", read_blocks(path), m_start=1)


def cmd_takens(path, _n=None):
    cmd_file(path)
    blocks = read_blocks(path)
    _per_m_table(
        "Takens plateau estimates",
        blocks,
        m_start=1,
        value_label="Takens D_T",
        saturation_label="Takens median of last 5 m",
    )
    if len(blocks) >= 3:
        vals, r_min, r_max = _stable_plateau_values(blocks[2], value_col=1)
        vals = vals[np.isfinite(vals)]
        c2_path = path.replace("_takens.dat", ".c2")
        ellner = _ellner_from_c2(c2_path, r_min, r_max, dim=3)
        if np.isfinite(ellner):
            print(
                "  Ellner extension m=3: "
                f"{ellner:.4f}  (plateau points={int(vals.size)}, "
                f"r_min={r_min:.6g}, r_max={r_max:.6g})"
            )


def cmd_ellner_plot_data(path, _n=None):
    """Emit gnuplot-ready Ellner interval data derived from a Takens file.

    Ellner's estimator is a finite-interval scalar, not a scale-by-scale curve.
    For plotting we therefore draw one horizontal segment per embedding m over
    the exact `[r_min, r_max]` plateau interval selected from the corresponding
    Takens block. This keeps the visual link to the Takens graph while avoiding
    inventing a false pointwise Ellner curve.
    """
    blocks = read_blocks(path)
    c2_path = path.replace("_takens.dat", ".c2")
    for m, block in enumerate(blocks, start=1):
        vals, r_min, r_max = _stable_plateau_values(block, value_col=1)
        vals = vals[np.isfinite(vals)]
        ellner = _ellner_from_c2(c2_path, r_min, r_max, dim=m)
        print(
            f"#m={m} ellner={ellner:.10g} "
            f"r_min={r_min:.10g} r_max={r_max:.10g} plateau_points={int(vals.size)}"
        )
        if np.isfinite(ellner) and np.isfinite(r_min) and np.isfinite(r_max) and r_max > r_min:
            print(f"{r_min:.12g} {ellner:.12g}")
            print(f"{r_max:.12g} {ellner:.12g}")
        print()


def cmd_takens_value(path, _n=None):
    """Emit CSV row: filename, Ellner value, plateau count, and plateau bounds.

    Detects the plateau on the m=3 d_2^(T)(r') curve, then evaluates the Ellner
    estimate on the sibling .c2 file over the auto-detected [r_min, r_max].
    Falls back to NaN when either step is undefined.
    """
    blocks = read_blocks(path)
    value = float("nan")
    points = 0
    r_min = float("nan")
    r_max = float("nan")
    if len(blocks) >= 3:
        vals, r_min, r_max = _stable_plateau_values(blocks[2], value_col=1)
        vals = vals[np.isfinite(vals)]
        points = int(vals.size)
        c2_path = path.replace("_takens.dat", ".c2")
        ellner = _ellner_from_c2(c2_path, r_min, r_max, dim=3)
        if np.isfinite(ellner):
            value = ellner
        elif vals.size:
            value = float(np.mean(vals))
    print(f"{os.path.basename(path)},{value:.10g},{points},{r_min:.10g},{r_max:.10g}")


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
    print("  Diagnostic largest-Lyapunov slopes from lyap_k S(t) (first epsilon block):")
    print(f"  {'m':>4}  {'lambda':>10}  {'pts':>5}")
    lambdas = []
    for m in sorted(by_dim.keys()):
        b = by_dim[m]
        lam = _slope_fit(b[:, 0], b[:, 1])
        lambdas.append(lam)
        print(f"  {m:>4}  {lam:>10.5f}  {b.shape[0]:>5}")
    sat = _saturation(lambdas)
    print(f"  diagnostic lambda summary (median of last 5 m): {sat:.5f}")


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
        "format": "legacy",
        "metrics": {},          # orig, std_orig, mean, std, se_surr, z_sigma, pvalue; legacy 'score' = z_sigma
        "conclusion": {},       # name -> "reject H0" / "fail to reject H0" / "insufficient data"
    }
    metric_names_row = ("D2", "K2", "TAKENS", "ELLNER", "LLE", "RR", "DET", "LAM", "MAXLINE", "ENTR", "TT", "TREND")
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None

    def _parse_float_tok(s):
        sl = (s or "").strip().lower()
        if sl == "nan":
            return float("nan")
        if sl in ("inf", "+inf"):
            return float("inf")
        if sl == "-inf":
            return float("-inf")
        return float(s)

    # Current hypothesis.py format: stationary-bootstrap TS test for TAKENS/ELLNER/LLE.
    m = re.search(r"Stationary-bootstrap hypothesis test\s+\(([^)]+)\)", text)
    if m:
        info["format"] = "stationary_bootstrap_ts"
        info["symbol"] = m.group(1).strip()
        pm = re.search(
            r"Parameters:\s*tau=(\d+),\s*W=(\d+),\s*B=(\d+),\s*stationary_block_mean=([^,]+),\s*TS_threshold=([0-9.]+)",
            text,
        )
        if pm:
            info["tau"], info["W"], info["B"] = pm.group(1), pm.group(2), pm.group(3)
            info["stationary_block_mean"] = pm.group(4)
            info["TS_threshold"] = pm.group(5)
        nm = re.search(r"Original series:\s*n=(\d+)", text)
        if nm:
            info["n"] = nm.group(1)
        info["mode"] = "FULL" if "full" in path.lower() else ("TEST" if "test" in path.lower() else "")

        row_re_boot = re.compile(
            r"^(?P<name>D2|K2|TAKENS|ELLNER|LLE|RR|DET|LAM|MAXLINE|ENTR|TT|TREND)\s+"
            r"(?P<boot_mean>-?(?:nan|inf|\d+\.\d+))\s+"
            r"(?P<boot_sd>-?(?:nan|inf|\d+\.\d+))\s+"
            r"(?P<B>\d+)\s+"
            r"(?P<orig>-?(?:nan|inf|\d+\.\d+))\s+"
            r"(?P<resh>-?(?:nan|inf|\d+\.\d+))\s+"
            r"(?P<normal>-?(?:nan|inf|\d+\.\d+))\s+"
            r"(?P<tref>-?(?:nan|inf|\d+\.\d+))\s+"
            r"(?P<TS>-?(?:nan|inf|\d+\.\d+))\s+"
            r"(?P<absTS>-?(?:nan|inf|\d+\.\d+))\s+"
            r"(?P<decision>reject H0|fail to reject H0|insufficient data|no sd|not bootstrap-tested)\s*$"
        )
        for line in text.splitlines():
            rm = row_re_boot.match(line.strip())
            if not rm:
                continue
            name = rm.group("name")
            info["metrics"][name] = {
                "orig": _parse_float_tok(rm.group("orig")),
                "std_orig": _parse_float_tok(rm.group("boot_sd")),
                "mean": _parse_float_tok(rm.group("resh")),
                "std": float("nan"),
                "normal": _parse_float_tok(rm.group("normal")),
                "t_ref": _parse_float_tok(rm.group("tref")),
                "n": int(rm.group("B")),
                "boot_mean": _parse_float_tok(rm.group("boot_mean")),
                "boot_sd": _parse_float_tok(rm.group("boot_sd")),
                "resh": _parse_float_tok(rm.group("resh")),
                "TS": _parse_float_tok(rm.group("TS")),
                "abs_TS": _parse_float_tok(rm.group("absTS")),
                "F": _parse_float_tok(rm.group("TS")),
                "score": _parse_float_tok(rm.group("TS")),
                "pvalue": float("nan"),
            }
            info["conclusion"][name] = rm.group("decision")
        return info

    # Previous hypothesis.py format: one randperm surrogate + normal/t references.
    m = re.search(r"Single-surrogate hypothesis test\s+\(([^)]+)\)", text)
    if m:
        info["format"] = "single_surrogate"
        info["symbol"] = m.group(1).strip()
        info["B"] = "1"
        pm = re.search(r"Parameters:\s*tau=(\d+),\s*W=(\d+),\s*alpha=([0-9.]+)", text)
        if pm:
            info["tau"], info["W"] = pm.group(1), pm.group(2)
            info["alpha"] = pm.group(3)
        nm = re.search(r"Original series:\s*n=(\d+)", text)
        if nm:
            info["n"] = nm.group(1)
        info["mode"] = "FULL" if "full" in path.lower() else ("TEST" if "test" in path.lower() else "")

        row_re_single = re.compile(
            r"^(?P<name>D2|K2|TAKENS|ELLNER|LLE|RR|DET|LAM|MAXLINE|ENTR|TT|TREND)\s+"
            r"(?P<orig_sd>-?(?:nan|inf|\d+\.\d+))\s+"
            r"(?P<surr_sd>-?(?:nan|inf|\d+\.\d+))\s+"
            r"(?P<n>\d+)\s+"
            r"(?P<orig>-?(?:nan|inf|\d+\.\d+))\s+"
            r"(?P<surr>-?(?:nan|inf|\d+\.\d+))\s+"
            r"(?P<normal>-?(?:nan|inf|\d+\.\d+))\s+"
            r"(?P<tref>-?(?:nan|inf|\d+\.\d+))\s+"
            r"(?P<F>-?(?:nan|inf|\d+\.\d+))\s+"
            r"(?P<pvalue>-?(?:nan|inf|\d*\.?\d+(?:[eE][+-]?\d+)?))\s+"
            r"(?P<decision>reject H0|fail to reject H0|insufficient n|no sd)\s*$"
        )
        for line in text.splitlines():
            rm = row_re_single.match(line.strip())
            if not rm:
                continue
            name = rm.group("name")
            info["metrics"][name] = {
                "orig": _parse_float_tok(rm.group("orig")),
                "std_orig": _parse_float_tok(rm.group("orig_sd")),
                "mean": _parse_float_tok(rm.group("surr")),
                "std": _parse_float_tok(rm.group("surr_sd")),
                "normal": _parse_float_tok(rm.group("normal")),
                "t_ref": _parse_float_tok(rm.group("tref")),
                "n": int(rm.group("n")),
                "F": _parse_float_tok(rm.group("F")),
                "score": _parse_float_tok(rm.group("F")),
                "pvalue": _parse_float_tok(rm.group("pvalue")),
            }
            info["conclusion"][name] = rm.group("decision")
        return info

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
    metric_order = ("D2", "TAKENS", "ELLNER", "K2", "LLE", "RR", "DET", "LAM", "MAXLINE", "ENTR", "TT", "TREND")
    # PyRQA scalar metrics: no plateau-window std_orig in hypothesis.py (column is absent / NaN in summaries).
    RQA_METRICS = ("RR", "DET", "LAM", "MAXLINE", "ENTR", "TT", "TREND")
    DKL_METRICS = ("TAKENS", "ELLNER", "LLE")
    parsed_infos = []
    metrics_present = []
    for fp in found:
        info = _parse_bootstrap_summary(fp)
        if info is None:
            continue
        parsed_infos.append((fp, info))
        metrics_present.extend([k for k in metric_order if k in info.get("metrics", {})])
    metric_names = tuple(dict.fromkeys(metrics_present)) if metrics_present else ("TAKENS", "ELLNER", "LLE")

    if parsed_infos and all(
        info.get("format") in {"single_surrogate", "stationary_bootstrap_ts"}
        for _fp, info in parsed_infos
    ):
        is_stationary = all(info.get("format") == "stationary_bootstrap_ts" for _fp, info in parsed_infos)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("Hypothesis pipeline - aggregated summary\n")
            fh.write(f"Scanned root : {path}\n")
            fh.write(f"Files found  : {len(found)}\n")
            fh.write("=" * 110 + "\n\n")

            w = 11
            header = f"{'Symbol':<8} {'tau':>4} {'W':>3} {'B':>3}"
            for name in metric_names:
                if is_stationary:
                    header += (
                        f" {f'{name}_orig':>{w}} {f'{name}_boot':>{w}}"
                        f" {f'{name}_boot_sd':>{w}} {f'{name}_resh':>{w}}"
                        f" {f'TS_{name}':>{w}} {f'absTS_{name}':>{w}}"
                    )
                else:
                    header += (
                        f" {f'{name}_orig':>{w}} {f'{name}_orig_sd':>{w}}"
                        f" {f'{name}_surr':>{w}} {f'{name}_surr_sd':>{w}}"
                        f" {f'F_{name}':>{w}} {f'p_{name}':>{w}}"
                    )
            header += f" {'rej_all':>7}"
            fh.write(header + "\n")
            fh.write("-" * len(header) + "\n")

            for fp, info in parsed_infos:
                row = f"{info['symbol']:<8} {info['tau']:>4} {info['W']:>3} {info['B']:>3}"
                for name in metric_names:
                    m = info["metrics"].get(name)
                    if not m:
                        row += f" {'nan':>{w}} {'nan':>{w}} {'nan':>{w}} {'nan':>{w}} {'nan':>{w}} {'nan':>{w}}"
                        continue
                    if is_stationary:
                        row += (
                            f" {m['orig']:>{w}.4f} {m.get('boot_mean', float('nan')):>{w}.4f}"
                            f" {m.get('boot_sd', float('nan')):>{w}.4f} {m.get('resh', float('nan')):>{w}.4f}"
                            f" {m.get('TS', float('nan')):>{w}.4f} {m.get('abs_TS', float('nan')):>{w}.4f}"
                        )
                    else:
                        row += (
                            f" {m['orig']:>{w}.4f} {m['std_orig']:>{w}.4f}"
                            f" {m['mean']:>{w}.4f} {m['std']:>{w}.4f}"
                            f" {m['F']:>{w}.4f} {m['pvalue']:>{w}.4f}"
                        )
                rej_all = (
                    "YES"
                    if all(info["conclusion"].get(n) == "reject H0" for n in metric_names)
                    else "NO"
                )
                row += f" {rej_all:>7}"
                fh.write(row + "\n")

            if is_stationary:
                fh.write(
                    "\nCurrent format: stationary-bootstrap TS test. "
                    "For TAKENS/ELLNER/LLE, boot is the mean across B stationary-bootstrap invariant values, "
                    "boot_sd is their sample SD, resh is one fully reshuffled invariant, and "
                    "TS=(boot-resh)/boot_sd. Reject H0 when |TS|>3.\n\n"
                )
            else:
                fh.write(
                    "\nCurrent format: one random-permutation surrogate per series. "
                    "normal and t(3.5) reference values are reported inside each source summary; "
                    "main p-values test H0: Var(T_orig)=Var(T_surr) using a two-sided F-test at alpha=0.01.\n\n"
                )

            for fp, info in parsed_infos:
                fh.write("=" * 110 + "\n")
                fh.write(f"Source: {fp}\n")
                fh.write(f"Symbol: {info['symbol']}   Mode: {info['mode']}   N: {info['n']}\n")
                fh.write(f"  tau={info['tau']}, W={info['W']}, B={info['B']}\n")
                if is_stationary:
                    fh.write(
                        f"  {'metric':<8} {'orig':>10} {'boot':>10} {'boot_sd':>10} {'resh':>10} "
                        f"{'normal':>10} {'t3.5':>10} {'B':>5} {'TS':>10} {'abs_TS':>10}  conclusion\n"
                    )
                else:
                    fh.write(
                        f"  {'metric':<8} {'orig':>10} {'orig_sd':>10} {'surr':>10} {'surr_sd':>10} "
                        f"{'normal':>10} {'t3.5':>10} {'n':>5} {'F':>10} {'p-value':>10}  conclusion\n"
                    )
                for name in metric_names:
                    m = info["metrics"].get(name, {})
                    con = info["conclusion"].get(name, "n/a")
                    if m:
                        if is_stationary:
                            fh.write(
                                f"  {name:<8} {m['orig']:>10.4f} {m.get('boot_mean', float('nan')):>10.4f} "
                                f"{m.get('boot_sd', float('nan')):>10.4f} {m.get('resh', float('nan')):>10.4f} "
                                f"{m['normal']:>10.4f} {m['t_ref']:>10.4f} "
                                f"{m['n']:>5} {m.get('TS', float('nan')):>10.4f} {m.get('abs_TS', float('nan')):>10.4f}  {con}\n"
                            )
                        else:
                            fh.write(
                                f"  {name:<8} {m['orig']:>10.4f} {m['std_orig']:>10.4f} "
                                f"{m['mean']:>10.4f} {m['std']:>10.4f} "
                                f"{m['normal']:>10.4f} {m['t_ref']:>10.4f} "
                                f"{m['n']:>5} {m['F']:>10.4f} {m['pvalue']:>10.4f}  {con}\n"
                            )
                    else:
                        fh.write(
                            f"  {name:<8} {'nan':>10} {'nan':>10} {'nan':>10} {'nan':>10} "
                            f"{'nan':>10} {'nan':>10} {'nan':>5} {'nan':>10} {'nan':>10}  {con}\n"
                        )
                fh.write("\n")

        print(f"  Aggregated hypothesis summary -> {out_path}")
        try:
            with open(out_path, encoding="utf-8", errors="replace") as fh:
                print(fh.read())
        except OSError:
            pass
        return

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
            "\nColumn rej_all: YES if every metric in scope rejects H0 under the TS threshold. "
            "DKL scope: requires TAKENS, ELLNER, LLE in the summary. RQA-only scope: requires all listed metrics "
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
    "ellner_plot_data": cmd_ellner_plot_data,
    "takens_value": cmd_takens_value,
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
