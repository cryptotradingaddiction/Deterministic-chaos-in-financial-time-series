#!/usr/bin/env python3
import os
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


def prefer_liquidity_cut(file_path):
    """
    Prefer liquidity-cut file variants when they exist.

    Supported naming:
    - *_logreturns.dat  -> *_logreturns_cut.dat
    - *_logreturns.csv  -> *_logreturns_cut.csv
    """
    normalized = os.path.normpath(file_path)
    cut_candidate = None
    if normalized.endswith("_logreturns.dat"):
        cut_candidate = normalized.replace("_logreturns.dat", "_logreturns_cut.dat")
    elif normalized.endswith("_logreturns.csv"):
        cut_candidate = normalized.replace("_logreturns.csv", "_logreturns_cut.csv")

    if cut_candidate and os.path.exists(cut_candidate):
        return cut_candidate
    return normalized
