#!/usr/bin/env python3
"""Unit tests for documents.py canonical aggregate row selection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from documents import (
    SYMS,
    _parse_aggregate_rows,
    _parse_agg_named_col,
    _select_canonical_aggregate_rows,
)


class CanonicalAggregateSelectionTests(unittest.TestCase):
    def test_select_canonical_prefers_matching_tau(self):
        rows_by_sym = {
            "BTCUSD": [
                {"tau": "3", "ELLNER_orig": "2.6980"},
                {"tau": "5", "ELLNER_orig": "2.7649"},
            ],
            "XRPUSD": [
                {"tau": "2", "ELLNER_orig": "3.2901"},
                {"tau": "3", "ELLNER_orig": "3.3222"},
            ],
        }
        tau_map = {"BTCUSD": 3, "XRPUSD": 2}
        picked = _select_canonical_aggregate_rows(rows_by_sym, tau_map)
        self.assertEqual(picked["BTCUSD"]["ELLNER_orig"], "2.6980")
        self.assertEqual(picked["XRPUSD"]["ELLNER_orig"], "3.2901")

    def test_parse_aggregate_rows_from_temp_file(self):
        text = """Symbol    tau   W   B ELLNER_orig
BTCUSD      3   3 100      2.6980
BTCUSD      5   5 100      2.7649
ETHUSD      3   3 100      2.6959
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "_hypothesis_aggregate_summary.txt"
            path.write_text(text, encoding="utf-8")
            _header, rows = _parse_aggregate_rows(path, tau_map={"BTCUSD": 3, "ETHUSD": 3})
            self.assertEqual(rows["BTCUSD"]["ELLNER_orig"], "2.6980")
            out = _parse_agg_named_col(path, "ELLNER_orig", tau_map={"BTCUSD": 3, "ETHUSD": 3})
            self.assertEqual(out["BTCUSD"], "2.6980")
            self.assertEqual(out["ETHUSD"], "2.6959")


if __name__ == "__main__":
    unittest.main()
