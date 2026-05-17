#!/usr/bin/env python3
"""
Central configuration and path helpers for the DCh pipeline.

This module is the single Python entry point for:

- **config.yaml** — merged with :data:`DEFAULT_CONFIG` when PyYAML is available.
- **Environment overrides** — ``DCH_TEST_MODE``, ``DCH_TEST_POINTS``, etc. (used by
  ``.bat`` files via ``_dch_test_env.bat`` and by :func:`desktop_app`).
- **Per-coin TISEAN parameters** — parsed from ``Tisean_3.0.0/bin/_per_coin_settings.bat``.
- **Derived summaries** — embedding delay τ from ``mutual/_mi_summary.txt`` and
  Theiler window W (equals τ: ``W_D2_<sym> := TAU_D2_<sym>`` after ``theilers_w.bat``).

Scripts should prefer these helpers over hard-coded ``C:\\DCh\\data`` paths so test
mode, liquidity cuts, and per-coin τ/W stay consistent across Python and batch steps.
"""

import os
import re
from datetime import datetime
from pathlib import Path

try:
    import yaml
except Exception:
    # PyYAML is optional at import time; without it only DEFAULT_CONFIG applies.
    yaml = None


# ---------------------------------------------------------------------------
# Built-in defaults (used when config.yaml is missing or omits keys)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # All time series and TISEAN result trees live under these roots.
    "paths": {"data_dir": r"C:\DCh\data", "results_dir": r"C:\DCh\data\results"},
    # Bitstamp download window; null "to" → today's UTC date in get_download_range().
    "download": {"from": None, "to": None},
    "liquidity": {
        # "liquidity" = first hour where rolling zero-return % drops below tolerance;
        # "fixed" / "fixed_date" (alias) = keep the last fixed_tail_points rows (e.g. 17520 h).
        "mode": "fixed",
        "window_size": 720,
        "tolerance": 1.0,
        # Optional end of analysis window in liquidity mode (ISO-like string).
        # null = use the last timestamp in each file (no artificial cutoff).
        "analysis_end": None,
        # Used only when mode is "fixed" or "fixed_date": number of trailing samples to keep.
        "fixed_tail_points": 1000,
        # When True, liquidity.py writes *_logreturns_cut.* siblings used by the active pipeline.
        "create_cut_files": True,
        "create_backup_before_cut": True,
    },
    # Filename suffixes for discover_data_files() and data_file() helpers.
    "files": {
        "raw_csv_suffix": "_BITSTAMP_1h_complete.csv",
        "logreturns_dat_suffix": "_BITSTAMP_1h_complete_logreturns.dat",
        "logreturns_csv_suffix": "_BITSTAMP_1h_complete_logreturns.csv",
    },
}


def _deep_merge(base, override):
    """Recursively merge *override* into *base* (dict values only; leaves replace)."""
    result = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path=None):
    """
    Load merged configuration: :data:`DEFAULT_CONFIG` + optional ``config.yaml``.

    If *config_path* is omitted, looks for ``config.yaml`` next to this file.
    When PyYAML is not installed or the file is absent, returns defaults only.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    cfg = dict(DEFAULT_CONFIG)
    if yaml is not None and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            parsed = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, parsed)
    return cfg


def get_data_dir(config=None):
    """Absolute normalized path to the data directory (OHLC, log-returns, temp trims)."""
    cfg = config or load_config()
    data_dir = cfg.get("paths", {}).get("data_dir", DEFAULT_CONFIG["paths"]["data_dir"])
    return os.path.normpath(data_dir)


def get_results_dir(config=None):
    """Absolute normalized path to the results root (plots, summaries, hypothesis output)."""
    cfg = config or load_config()
    default_results = DEFAULT_CONFIG["paths"]["results_dir"]
    results_dir = cfg.get("paths", {}).get("results_dir", default_results)
    return os.path.normpath(results_dir)


def ensure_dir(path):
    """Create *path* (and parents) if missing; return *path* unchanged."""
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Test-mode environment (mirrors Tisean_3.0.0/bin/_dch_test_env.bat)
# ---------------------------------------------------------------------------

# Default row count when DCH_TEST_MODE=true and DCH_TEST_POINTS is unset.
DEFAULT_DCH_TEST_POINTS = 100


def dch_test_point_count() -> int:
    """
    Number of samples used in test mode (first N rows of each series).

    Override with environment variable ``DCH_TEST_POINTS`` (integer ≥ 1).
    Batch files set ``TEST_POINT_COUNT`` from the same value via ``_dch_test_env.bat``.
    """
    raw = os.environ.get("DCH_TEST_POINTS", "").strip()
    if raw:
        try:
            return max(1, int(float(raw)))
        except ValueError:
            pass
    return DEFAULT_DCH_TEST_POINTS


def dch_test_results_tag() -> str:
    """
    Suffix for test-mode result folders, e.g. ``test_100``.

    Matches directory names like ``correlation_dimension_test_100`` and
    ``data/results_test_100`` used by invariant ``.bat`` scripts.
    """
    return f"test_{dch_test_point_count()}"


def dch_test_mode_from_env() -> bool:
    """
    True when the pipeline should use short test series and ``*_test_<N>`` folders.

    Reads ``DCH_TEST_MODE`` (truthy: ``1``, ``true``, ``yes``, ``y``, ``on``).
    Desktop GUI and ``_dch_test_env.bat`` set this for smoke runs.
    """
    v = os.environ.get("DCH_TEST_MODE", "false").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _hypothesis_results_branch(config, dirname_full: str, dirname_test: str) -> Path:
    """
    Pick FULL vs test results subdirectory for one invariant family.

    Each invariant ``.bat`` writes ``_hypothesis_aggregate_summary.txt`` under either
    ``<results_dir>/<dirname_full>/`` or ``<results_dir>/<dirname_test>/``. This helper
    chooses the directory that actually contains that aggregate file, preferring test
    paths when ``DCH_TEST_MODE`` is on but falling back to FULL if only FULL exists
    (e.g. user ran full pipeline earlier).
    """
    rd = Path(get_results_dir(config or load_config()))
    agg = "_hypothesis_aggregate_summary.txt"
    p_full = rd / dirname_full / agg
    p_test = rd / dirname_test / agg
    want_test = dch_test_mode_from_env()
    if want_test:
        if p_test.is_file():
            return rd / dirname_test
        if p_full.is_file():
            return rd / dirname_full
        return rd / dirname_test
    if p_full.is_file():
        return rd / dirname_full
    if p_test.is_file():
        return rd / dirname_test
    return rd / dirname_full


def hypothesis_correlation_dimension_dir(config=None) -> Path:
    """Directory for D2 / Takens / Ellner hypothesis outputs (FULL or test_<N>)."""
    return _hypothesis_results_branch(
        config,
        "correlation_dimension_full",
        f"correlation_dimension_{dch_test_results_tag()}",
    )


def hypothesis_lambda_max_dir(config=None) -> Path:
    """Directory for LLE (lyap_k) hypothesis outputs (FULL or test_<N>)."""
    return _hypothesis_results_branch(
        config,
        "lambda_max_full",
        f"lambda_max_{dch_test_results_tag()}",
    )


def hypothesis_rqa_dir(config=None) -> Path:
    """Directory for RQA hypothesis outputs (FULL or test_<N>)."""
    return _hypothesis_results_branch(
        config,
        "rqa_full",
        f"rqa_{dch_test_results_tag()}",
    )


# ---------------------------------------------------------------------------
# Download and liquidity window
# ---------------------------------------------------------------------------


def get_download_range(config=None):
    """
    Return ``(start, end)`` date strings for ``crypto_data_all.py``.

    *start* may be ``None`` (downloader uses per-asset defaults).
    *end* defaults to today's UTC date when omitted in config.
    """
    cfg = config or load_config()
    start = cfg.get("download", {}).get("from")
    end = cfg.get("download", {}).get("to")
    if not end:
        end = datetime.utcnow().strftime("%Y-%m-%d")
    return start, end


def get_liquidity_settings(config=None):
    """
    Return merged and validated liquidity / cut-window settings from config.yaml.

    Normalizes ``mode`` to ``"liquidity"`` or ``"fixed"`` (``fixed_date`` is an alias
    for ``fixed``). Coerces numeric fields and boolean flags so downstream scripts
    do not need to repeat validation logic.
    """
    cfg = config or load_config()
    defaults = DEFAULT_CONFIG.get("liquidity", {})
    user = cfg.get("liquidity") or {}
    out = _deep_merge(dict(defaults), user)
    mode = str(out.get("mode") or "liquidity").strip().lower()
    if mode == "fixed_date":
        mode = "fixed"
    if mode not in ("liquidity", "fixed"):
        mode = "liquidity"
    out["mode"] = mode
    try:
        out["window_size"] = int(out["window_size"])
    except (TypeError, ValueError):
        out["window_size"] = int(defaults["window_size"])
    try:
        out["tolerance"] = float(out["tolerance"])
    except (TypeError, ValueError):
        out["tolerance"] = float(defaults["tolerance"])
    try:
        out["fixed_tail_points"] = int(out.get("fixed_tail_points", defaults["fixed_tail_points"]))
    except (TypeError, ValueError):
        out["fixed_tail_points"] = int(defaults["fixed_tail_points"])
    out["create_cut_files"] = bool(out.get("create_cut_files", True))
    out["create_backup_before_cut"] = bool(out.get("create_backup_before_cut", True))
    return out


def data_file(symbol_no_slash, suffix, config=None):
    """
    Build ``<data_dir>/<symbol><suffix>`` for a coin symbol without slashes.

    Example: ``data_file("BTCUSD", "_BITSTAMP_1h_complete_logreturns.dat")``.
    """
    return os.path.join(get_data_dir(config), f"{symbol_no_slash}{suffix}")


# ---------------------------------------------------------------------------
# Per-coin batch settings (_per_coin_settings.bat)
# ---------------------------------------------------------------------------


def default_per_coin_settings_bat_path():
    """Absolute path to ``Tisean_3.0.0/bin/_per_coin_settings.bat`` beside this repo."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Tisean_3.0.0",
        "bin",
        "_per_coin_settings.bat",
    )


def parse_per_coin_settings_bat(path=None):
    """
    Parse ``set NAME=value`` assignments from the shared batch settings file.

    Keys are normalized to UPPER CASE for case-insensitive lookup. Text after ``REM``
    on the same line is stripped (batch-style comments). Quoted values have outer
    quotes removed.

    Returns an empty dict if the file does not exist (callers use numeric fallbacks).
    """
    path = path or default_per_coin_settings_bat_path()
    out = {}
    if not os.path.isfile(path):
        return out
    set_re = re.compile(r"^\s*set\s+([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$", re.I)
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.split("REM", 1)[0].strip()
            if not line:
                continue
            m = set_re.match(line)
            if not m:
                continue
            key, val = m.group(1), m.group(2).strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in {'"', "'"}:
                val = val[1:-1]
            out[key.upper()] = val
    return out


def rqa_params_for_symbol(symbol: str, settings_flat=None):
    """
    Per-coin RQA parameters from ``_per_coin_settings.bat``.

    Variable names: ``TAU_RQA_<sym>``, ``RAD_RQA_<sym>``, ``W_D2_<sym>``.

    ``W_D2_<sym>`` is the shared Theiler window written by ``theilers_w.bat`` and
    consumed by every downstream stage (``d2.exe`` / ``lyap_k.exe`` via ``-t``,
    PyRQA ``theiler_corrector`` in ``rqa_values.py`` and ``hypothesis.py``).

    Returns ``(tau, radius, theiler_w)``. Defaults match ``RQA.bat`` fallbacks when
    a variable is missing.
    """
    s = settings_flat if settings_flat is not None else parse_per_coin_settings_bat()
    sk = {k.upper(): v for k, v in s.items()}

    def _get(name, default):
        return sk.get(name.upper(), default)

    tau = int(float(_get(f"TAU_RQA_{symbol}", "3")))
    rad = float(_get(f"RAD_RQA_{symbol}", "0.005"))
    theiler_w = int(float(_get(f"W_D2_{symbol}", "0")))
    return tau, rad, theiler_w


def prefer_liquidity_cut(file_path):
    """
    Prefer liquidity-cut log-return files when they exist.

    The active pipeline uses ``liquidity.py`` outputs so all downstream programs
    analyse the same liquid time window. Callers pass the canonical
    ``*_logreturns.csv`` or ``*_logreturns.dat`` path; this helper redirects to the
    sibling ``*_logreturns_cut.*`` file when available.

    Raises FileNotFoundError if a ``_logreturns.`` path was given but the ``_cut``
    sibling is missing (run ``liquidity.py`` first).
    """
    normalized = os.path.normpath(file_path)
    cut_path = normalized
    if "_logreturns." in normalized:
        cut_path = normalized.replace("_logreturns.", "_logreturns_cut.")
    if os.path.exists(cut_path):
        return os.path.normpath(cut_path)
    if cut_path != normalized:
        raise FileNotFoundError(
            f"Required liquidity-cut data file is missing: {cut_path}. "
            "Run C:\\DCh\\liquidity.py before this pipeline."
        )
    return normalized


# ---------------------------------------------------------------------------
# Mutual-information summary → embedding tau (Fraser & Swinney first minimum)
# ---------------------------------------------------------------------------

MUTUAL_SUMMARY_FILENAME = "_mi_summary.txt"
# Must match ``SERIES_COL_W`` in ``mutual.py`` (fixed-width summary rows).
MUTUAL_SUMMARY_SERIES_COL_W = 46

# Legacy static defaults when ``mutual/_mi_summary.txt`` is missing or has no minimum.
TAU_FALLBACK_BY_SYMBOL = {
    "BTCUSD": 3,
    "ETHUSD": 3,
    "LTCUSD": 2,
    "XRPUSD": 3,
    "LINKUSD": 4,
    "DOGEUSD": 6,
    "ADAUSD": 4,
}


def mutual_summary_path(config=None):
    """``<results_dir>/mutual/_mi_summary.txt`` — written by ``mutual.py``."""
    cfg = config if config is not None else load_config()
    return os.path.join(get_results_dir(cfg), "mutual", MUTUAL_SUMMARY_FILENAME)


def parse_mutual_first_min_tau_map(config=None) -> dict[str, int]:
    """
    Parse ``mutual/_mi_summary.txt`` → ``{ 'BTCUSD': tau, ... }``.

    Reads the ``first_min_tau`` column via regex (robust to padding drift).
    Uses ``utf-8-sig`` so a BOM on the first line does not break header detection.
    Rows with ``none`` / ``nan`` are skipped. τ is clamped to ≥ 1.
    """
    path = mutual_summary_path(config)
    out: dict[str, int] = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        text = fh.read()

    # One row: <SYMBOL>_BITSTAMP_...<spaces>N max_tau first_min ...
    row_re = re.compile(
        r"^((?:BTCUSD|ETHUSD|LTCUSD|XRPUSD|LINKUSD|DOGEUSD|ADAUSD)_[^\s]+)\s+"
        r"(\d+)\s+(\d+)\s+(\S+)",
        re.MULTILINE | re.IGNORECASE,
    )
    for m in row_re.finditer(text):
        stem, _n, _mt, fm = m.group(1), m.group(2), m.group(3), m.group(4)
        sym = stem.split("_")[0].upper()
        tok = fm.strip().lower()
        if tok in ("none", "nan"):
            continue
        try:
            tau = int(round(float(fm)))
        except (TypeError, ValueError):
            continue
        if tau < 1:
            tau = 1
        out[sym] = tau
    return out


def tau_for_symbol_from_mutual(symbol: str, config=None) -> int:
    """
    Embedding delay τ for *symbol*.

    Prefer first MI minimum from ``mutual.py`` summary; otherwise
    :data:`TAU_FALLBACK_BY_SYMBOL`, else 3.
    """
    sym = str(symbol).upper().replace("/", "")
    m = parse_mutual_first_min_tau_map(config)
    if sym in m:
        return m[sym]
    return int(TAU_FALLBACK_BY_SYMBOL.get(sym, 3))


def sync_per_coin_bat_tau_from_mutual_summary(config=None, bat_path=None) -> tuple[str, int]:
    """
    Write τ into ``_per_coin_settings.bat`` from the mutual-information summary.

    Updates ``TAU_D2_*``, ``TAU_LLE_*``, and ``TAU_RQA_*`` for each symbol found in
    the summary so TISEAN ``.bat`` files and ``hypothesis.py`` share one delay per coin.

    Returns ``(status, n_symbols)`` where *status* is one of:

    - ``"updated"`` — bat file was modified,
    - ``"unchanged"`` — values already matched,
    - ``"empty_taus"`` — summary had no usable τ,
    - ``"no_bat"`` — settings file missing.
    """
    bat_path = bat_path or default_per_coin_settings_bat_path()
    taus = parse_mutual_first_min_tau_map(config)
    if not taus:
        return "empty_taus", 0
    if not os.path.isfile(bat_path):
        return "no_bat", len(taus)
    with open(bat_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    orig = text
    for sym, tau_val in taus.items():
        for prefix in ("TAU_D2_", "TAU_LLE_", "TAU_RQA_"):
            var = f"{prefix}{sym}"
            pat = re.compile(rf"(^set\s+{re.escape(var)}=)(\d+)\s*$", re.I | re.M)
            text = pat.sub(lambda m, tv=tau_val: m.group(1) + str(int(tv)), text)
    if text != orig:
        with open(bat_path, "w", encoding="utf-8", newline="\r\n") as f:
            f.write(text)
        return "updated", len(taus)
    return "unchanged", len(taus)


# ---------------------------------------------------------------------------
# Theiler-window summary → W_D2_<sym> (TISEAN corr + stp pipeline)
# ---------------------------------------------------------------------------

THEILER_SUMMARY_FILENAME = "_theiler_summary.txt"


def theiler_summary_path(config=None) -> str:
    """
    Path to ``_theiler_summary.txt`` produced by ``theilers_w.bat``.

    Resolution order:

    1. If ``DCH_TEST_MODE`` and ``theiler_w_test_<N>/_theiler_summary.txt`` exists → use it.
    2. Else if ``theiler_w/_theiler_summary.txt`` exists → use FULL run.
    3. Else fall back to test path if present, else FULL path (may not exist yet).
    """
    cfg = config if config is not None else load_config()
    rd = Path(get_results_dir(cfg))
    if dch_test_mode_from_env():
        candidate_test = rd / f"theiler_w_{dch_test_results_tag()}" / THEILER_SUMMARY_FILENAME
        if candidate_test.is_file():
            return str(candidate_test)
    candidate_full = rd / "theiler_w" / THEILER_SUMMARY_FILENAME
    if candidate_full.is_file():
        return str(candidate_full)
    candidate_test = rd / f"theiler_w_{dch_test_results_tag()}" / THEILER_SUMMARY_FILENAME
    if candidate_test.is_file():
        return str(candidate_test)
    return str(candidate_full)


def parse_theiler_w_map(config=None) -> dict[str, int]:
    """
    Parse ``_theiler_summary.txt`` → ``{ 'BTCUSD': W_final, ... }``.

    Expected columns (from ``theilers_w.bat`` / ``detect_theiler.py``):

        SYMBOL  N  tau_d2  tau_a  W_formula  W_stp  W_final

    Legacy 6-column rows (``acf_zero`` instead of ``tau_a`` / ``W_formula``) are also
    accepted; the **last** integer on each data line is always taken as ``W_final``.
    Comment lines starting with ``#`` are ignored by the regex anchor on symbol names.
    """
    path = theiler_summary_path(config)
    out: dict[str, int] = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        text = fh.read()
    row_re = re.compile(
        r"^\s*((?:BTCUSD|ETHUSD|LTCUSD|XRPUSD|LINKUSD|DOGEUSD|ADAUSD))"
        r"((?:\s+-?\d+){5,6})\s*$",
        re.MULTILINE | re.IGNORECASE,
    )
    for m in row_re.finditer(text):
        sym = m.group(1).upper()
        nums = m.group(2).split()
        try:
            w_final = int(nums[-1])
        except (TypeError, ValueError):
            continue
        if w_final < 0:
            w_final = 0
        out[sym] = w_final
    return out


def sync_per_coin_bat_w_d2_from_theiler_summary(
    config=None, bat_path=None
) -> tuple[str, int]:
    """
    Set ``W_D2_<sym> = TAU_D2_<sym>`` in ``_per_coin_settings.bat``.

    **Project rule:** Theiler window **W equals embedding delay τ** (not ``W_stp`` or
    ``tau_a`` alone). Called at the end of ``theilers_w.bat`` after ACF/STP diagnostics.

    Returns ``(status, n_symbols)`` with *status* ∈
    ``{"updated", "unchanged", "empty_taus", "no_bat"}``.
    """
    bat_path = bat_path or default_per_coin_settings_bat_path()
    per_coin = parse_per_coin_settings_bat(bat_path)
    if not per_coin:
        return "empty_taus", 0
    sk = {k.upper(): v for k, v in per_coin.items()}
    taus: dict[str, int] = {}
    for key, val in sk.items():
        if not key.startswith("TAU_D2_"):
            continue
        sym = key[7:]
        try:
            taus[sym] = int(float(val))
        except (TypeError, ValueError):
            continue
    if not taus:
        return "empty_taus", 0
    if not os.path.isfile(bat_path):
        return "no_bat", len(taus)
    with open(bat_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    orig = text
    for sym, tau_val in taus.items():
        var = f"W_D2_{sym}"
        pat = re.compile(rf"(^set\s+{re.escape(var)}=)(\d+)\s*$", re.I | re.M)
        text = pat.sub(lambda m, tv=tau_val: m.group(1) + str(int(tv)), text)
    if text != orig:
        with open(bat_path, "w", encoding="utf-8", newline="\r\n") as f:
            f.write(text)
        return "updated", len(taus)
    return "unchanged", len(taus)
