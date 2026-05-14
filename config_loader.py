#!/usr/bin/env python3
import os
import re
from datetime import datetime

try:
    import yaml
except Exception:
    yaml = None


DEFAULT_CONFIG = {
    "paths": {"data_dir": r"C:\DCh\data", "results_dir": r"C:\DCh\data\results"},
    "download": {"from": None, "to": None},
    "files": {
        "raw_csv_suffix": "_BITSTAMP_1h_complete.csv",
        "logreturns_dat_suffix": "_BITSTAMP_1h_complete_logreturns.dat",
        "logreturns_csv_suffix": "_BITSTAMP_1h_complete_logreturns.csv",
    },
}


def _deep_merge(base, override):
    result = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    cfg = dict(DEFAULT_CONFIG)
    if yaml is not None and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            parsed = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, parsed)
    return cfg


def get_data_dir(config=None):
    cfg = config or load_config()
    data_dir = cfg.get("paths", {}).get("data_dir", DEFAULT_CONFIG["paths"]["data_dir"])
    return os.path.normpath(data_dir)


def get_results_dir(config=None):
    cfg = config or load_config()
    default_results = DEFAULT_CONFIG["paths"]["results_dir"]
    results_dir = cfg.get("paths", {}).get("results_dir", default_results)
    return os.path.normpath(results_dir)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def get_download_range(config=None):
    cfg = config or load_config()
    start = cfg.get("download", {}).get("from")
    end = cfg.get("download", {}).get("to")
    if not end:
        end = datetime.utcnow().strftime("%Y-%m-%d")
    return start, end


def data_file(symbol_no_slash, suffix, config=None):
    return os.path.join(get_data_dir(config), f"{symbol_no_slash}{suffix}")


def default_per_coin_settings_bat_path():
    """Path to `Tisean_3.0.0/bin/_per_coin_settings.bat` next to this package."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Tisean_3.0.0",
        "bin",
        "_per_coin_settings.bat",
    )


def parse_per_coin_settings_bat(path=None):
    """
    Parse `set NAME=value` assignments from the shared batch settings file.
    Keys are normalized to UPPER CASE for lookup.
    Lines after REM on the same line are ignored (batch-style).
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
    Per-coin RQA parameters from `_per_coin_settings.bat`:
      TAU_RQA_<sym>, RAD_RQA_<sym>, W_D2_<sym> (Theiler; same as RQA.bat / hypothesis.py).

    Returns (tau, radius, theiler_w). Defaults match RQA.bat fallbacks.
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

    The active pipeline uses `liquidity.py` outputs so all downstream programs
    analyse the same liquid time window. Callers pass the canonical
    `*_logreturns.csv` or `*_logreturns.dat` path; this helper redirects to the
    sibling `*_logreturns_cut.*` file when available.
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
