#!/usr/bin/env python3
"""Build thesis-style summary table from original FULL results.

Output:
    C:\\Users\\Teodor\\Documents\\RQA_surrogate_test.docx
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd

try:
    from docx import Document
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: python-docx. Install with: py -3 -m pip install python-docx"
    ) from exc


RESULTS = Path(r"C:\DCh\data\results")
DATA_DIR = Path(r"C:\DCh\data")
OUT_DOCX = Path(r"C:\Users\Teodor\Documents\RQA_surrogate_test.docx")
SYMS = ["BTCUSD", "ETHUSD", "XRPUSD", "LTCUSD", "LINKUSD", "DOGEUSD", "ADAUSD"]
HEADERS = ["Aktivum / USD", "BTC", "ETH", "XRP", "LTC", "LINK", "DOGE", "ADA"]


@dataclass
class TableData:
    sample_start: dict[str, str]
    sample_end: dict[str, str]
    n_vals: dict[str, str]
    tau: dict[str, str]
    tau_w: dict[str, str]
    m_vals: dict[str, str]
    dc: dict[str, str]
    d2: dict[str, str]
    k2: dict[str, str]
    lle: dict[str, str]
    rr: dict[str, str]
    det: dict[str, str]
    lam: dict[str, str]
    maxline: dict[str, str]
    entr: dict[str, str]
    tt: dict[str, str]


def _parse_agg_col(path: Path, col_index: int, decimals: int = 4, as_int: bool = False) -> dict[str, str]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, str] = {}
    for line in txt.splitlines():
        m = re.match(r"^(ADAUSD|BTCUSD|DOGEUSD|ETHUSD|LINKUSD|LTCUSD|XRPUSD)\s+", line.strip())
        if not m:
            continue
        parts = line.split()
        sym = parts[0]
        val = float(parts[col_index])
        if as_int:
            out[sym] = str(int(round(val)))
        else:
            out[sym] = f"{val:.{decimals}f}"
    return out


def _parse_tau_map() -> dict[str, str]:
    path = RESULTS / "correlation_dimension_full" / "_hypothesis_aggregate_summary.txt"
    txt = path.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, str] = {}
    for line in txt.splitlines():
        m = re.match(r"^(ADAUSD|BTCUSD|DOGEUSD|ETHUSD|LINKUSD|LTCUSD|XRPUSD)\s+(\d+)\s+", line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _parse_tauw_map() -> dict[str, str]:
    df = pd.read_fwf(RESULTS / "tau_w" / "_tau_w_summary.txt")
    out: dict[str, str] = {}
    for _, r in df.iterrows():
        sym = str(r["Symbol"]).strip()
        if sym in SYMS:
            out[sym] = f"{float(r['final_tau_w']):.4f}"
    return out


def _parse_dc_map() -> dict[str, str]:
    df = pd.read_fwf(RESULTS / "2dc" / "_2dc_summary.txt")
    out: dict[str, str] = {}
    for _, r in df.iterrows():
        sid = str(r["series_id"]).strip()
        sym = sid.split("_")[0]
        if sym in SYMS:
            out[sym] = f"{float(r['d_c']):.4f}"
    return out


def _parse_start_end_n() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    starts: dict[str, str] = {}
    ends: dict[str, str] = {}
    nvals: dict[str, str] = {}
    for sym in SYMS:
        cut = DATA_DIR / f"{sym}_BITSTAMP_1h_complete_logreturns_cut.csv"
        raw = DATA_DIR / f"{sym}_BITSTAMP_1h_complete_logreturns.csv"
        path = cut if cut.exists() else raw
        df = pd.read_csv(path)
        d = pd.to_datetime(df["datetime_str"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        d = d.dropna()
        starts[sym] = d.iloc[0].strftime("%d.%m.%Y, %H:%M")
        ends[sym] = d.iloc[-1].strftime("%d.%m.%Y, %H:%M")
        nvals[sym] = str(len(df))
    return starts, ends, nvals


def _parse_d2() -> dict[str, str]:
    return _parse_agg_col(
        RESULTS / "correlation_dimension_full" / "_hypothesis_aggregate_summary.txt",
        col_index=4,
        decimals=4,
    )


def _parse_k2() -> dict[str, str]:
    return _parse_agg_col(
        RESULTS / "correlation_entropy_full" / "_hypothesis_aggregate_summary.txt",
        col_index=4,
        decimals=4,
    )


def _parse_lle() -> dict[str, str]:
    return _parse_agg_col(
        RESULTS / "lambda_max_full" / "_hypothesis_aggregate_summary.txt",
        col_index=4,
        decimals=4,
    )


def _parse_rqa_row(metric_col: str, decimals: int = 4) -> dict[str, str]:
    col_ix = {
        "RR_orig_mean": 4,
        "DET_orig_mean": 7,
        "LAM_orig_mean": 10,
        "MAXLINE_orig_mean": 13,
        "ENTR_orig_mean": 16,
        "TT_orig_mean": 19,
    }[metric_col]
    return _parse_agg_col(
        RESULTS / "rqa_full" / "_hypothesis_aggregate_summary.txt",
        col_index=col_ix,
        decimals=decimals,
        as_int=(metric_col == "MAXLINE_orig_mean"),
    )


def collect() -> TableData:
    starts, ends, nvals = _parse_start_end_n()
    return TableData(
        sample_start=starts,
        sample_end=ends,
        n_vals=nvals,
        tau=_parse_tau_map(),
        tau_w=_parse_tauw_map(),
        m_vals={s: "3" for s in SYMS},
        dc=_parse_dc_map(),
        d2=_parse_d2(),
        k2=_parse_k2(),
        lle=_parse_lle(),
        rr=_parse_rqa_row("RR_orig_mean"),
        det=_parse_rqa_row("DET_orig_mean"),
        lam=_parse_rqa_row("LAM_orig_mean"),
        maxline=_parse_rqa_row("MAXLINE_orig_mean"),
        entr=_parse_rqa_row("ENTR_orig_mean"),
        tt=_parse_rqa_row("TT_orig_mean"),
    )


def row_values(label: str, mapping: dict[str, str]) -> list[str]:
    return [label] + [mapping.get(sym, "") for sym in SYMS]


def write_doc(data: TableData) -> None:
    doc = Document()
    doc.add_heading("Tabulka invariantu (originální FULL běhy z /results)", level=1)

    rows = [
        row_values("Sample start", data.sample_start),
        row_values("Sample end", data.sample_end),
        row_values("N", data.n_vals),
        row_values(r"\mathbit{\tau}", data.tau),
        row_values(r"\mathbit{\tau}w", data.tau_w),
        row_values(r"\mathbit{m}", data.m_vals),
        row_values(r"\mathbit{d}c", data.dc),
        row_values(r"\mathbit{d}2", data.d2),
        row_values(r"\mathbit{K}2", data.k2),
        row_values(r"\mathbit{\lambda}_{\mathbit{max}}", data.lle),
        row_values("RR", data.rr),
        row_values("DET", data.det),
        row_values("LAM", data.lam),
        row_values("MAXLINE", data.maxline),
        row_values("ENTR", data.entr),
        row_values("TT", data.tt),
    ]

    table = doc.add_table(rows=1, cols=len(HEADERS))
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, h in enumerate(HEADERS):
        hdr[i].text = h

    for rv in rows:
        c = table.add_row().cells
        for i, v in enumerate(rv):
            c[i].text = v

    doc.add_paragraph("")
    doc.add_paragraph(
        "Zdroj: C:\\DCh\\data\\results\\{correlation_dimension_full, correlation_entropy_full, lambda_max_full, rqa_full}, "
        "plus C:\\DCh\\data\\results\\{tau_w,2dc,mutual} a *_logreturns_cut.csv."
    )
    OUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT_DOCX))


def main() -> None:
    write_doc(collect())
    print(f"Saved: {OUT_DOCX}")


if __name__ == "__main__":
    main()

