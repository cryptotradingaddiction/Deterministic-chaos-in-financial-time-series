#!/usr/bin/env python3
"""Interactive recurrence plot (RQA) explorer for the DCh desktop app."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from attractor_core import M_MAX, M_MIN, TAU_MAX, coord_label, max_feasible_tau
from config_loader import PIPELINE_SYMBOLS, rqa_params_for_symbol, tau_for_symbol_from_mutual
from hypothesis_config import RQA_RADIUS_PERCENTILE_DEFAULT
from recurrence_worker import RecurrenceComputeResult, run_recurrence_compute, set_worker_thread_count


class RecurrencePlotWidget(QWidget):
    """PySide6 panel: RQA recurrence matrix, diagonal profile, phase-space slice."""

    def __init__(
        self,
        config=None,
        *,
        test_mode_provider=None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        from config_loader import load_config

        self.config = config or load_config()
        self._test_mode_provider = test_mode_provider
        self._request_id = 0
        self._cached: RecurrenceComputeResult | None = None
        self._rp_cmap = ListedColormap(["#0f172a", "#f8fafc", "#475569"])
        set_worker_thread_count(4)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(200)
        self._debounce.timeout.connect(self._start_compute)
        self._build_ui()
        self._on_symbol_changed()
        self.refresh(force=True)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        toolbar = QFrame()
        toolbar.setObjectName("settingsFrame")
        tb = QVBoxLayout(toolbar)
        tb.setContentsMargins(10, 8, 10, 8)
        tb.setSpacing(6)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Asset:"))
        self.cmb_symbol = QComboBox()
        for sym in PIPELINE_SYMBOLS:
            self.cmb_symbol.addItem(sym.replace("USD", ""), sym)
        row1.addWidget(self.cmb_symbol, stretch=1)
        row1.addWidget(QLabel("tau:"))
        self.spin_tau = QSpinBox()
        self.spin_tau.setRange(1, TAU_MAX)
        row1.addWidget(self.spin_tau)
        self.btn_tau_rqa = QPushButton("RQA tau")
        self.btn_tau_rqa.setObjectName("btnGhost")
        row1.addWidget(self.btn_tau_rqa)
        row1.addWidget(QLabel("m:"))
        self.spin_m = QSpinBox()
        self.spin_m.setRange(M_MIN, min(M_MAX, 12))
        self.spin_m.setValue(3)
        row1.addWidget(self.spin_m)
        row1.addWidget(QLabel("W:"))
        self.spin_theiler = QSpinBox()
        self.spin_theiler.setRange(0, 500)
        row1.addWidget(self.spin_theiler)
        self.lbl_feasible = QLabel("")
        self.lbl_feasible.setObjectName("previewPath")
        row1.addWidget(self.lbl_feasible, stretch=1)
        tb.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Radius:"))
        self.cmb_radius_mode = QComboBox()
        self.cmb_radius_mode.addItem(f"Percentile ({RQA_RADIUS_PERCENTILE_DEFAULT:g}%)", "percentile")
        self.cmb_radius_mode.addItem("Manual", "manual")
        self.cmb_radius_mode.addItem("BAT RAD_RQA", "bat")
        row2.addWidget(self.cmb_radius_mode)
        self.spin_percentile = QDoubleSpinBox()
        self.spin_percentile.setRange(0.5, 50.0)
        self.spin_percentile.setValue(RQA_RADIUS_PERCENTILE_DEFAULT)
        self.spin_percentile.setDecimals(1)
        row2.addWidget(self.spin_percentile)
        self.spin_radius = QDoubleSpinBox()
        self.spin_radius.setRange(1e-8, 1.0)
        self.spin_radius.setDecimals(8)
        self.spin_radius.setSingleStep(0.0001)
        row2.addWidget(self.spin_radius)
        row2.addWidget(QLabel("Display cap:"))
        self.spin_display_cap = QSpinBox()
        self.spin_display_cap.setRange(400, 6000)
        self.spin_display_cap.setValue(2500)
        row2.addWidget(self.spin_display_cap)
        self.chk_test_limit = QCheckBox("TEST row limit")
        row2.addWidget(self.chk_test_limit)
        tb.addLayout(row2)

        row_win = QHBoxLayout()
        row_win.addWidget(QLabel("Time window:"))
        self.slider_win_start = QSlider(Qt.Orientation.Horizontal)
        self.slider_win_start.setRange(0, 99)
        self.slider_win_end = QSlider(Qt.Orientation.Horizontal)
        self.slider_win_end.setRange(1, 100)
        self.slider_win_end.setValue(100)
        self.lbl_win = QLabel("0% – 100%")
        row_win.addWidget(self.slider_win_start, stretch=1)
        row_win.addWidget(self.slider_win_end, stretch=1)
        row_win.addWidget(self.lbl_win)
        tb.addLayout(row_win)

        row3 = QHBoxLayout()
        for label, slot in (
            ("Refresh", lambda: self.refresh(force=True)),
            ("Save PNG", self._save_png),
        ):
            btn = QPushButton(label)
            btn.setObjectName("btnPrimary" if label == "Refresh" else "btnGhost")
            btn.clicked.connect(slot)
            row3.addWidget(btn)
        row3.addStretch(1)
        tb.addLayout(row3)
        root.addWidget(toolbar)

        self.meta = QLabel("Recurrence plot: black = recurrent pairs, grey band = Theiler exclusion.")
        self.meta.setObjectName("previewMeta")
        self.meta.setWordWrap(True)
        root.addWidget(self.meta)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        left = QFrame()
        left.setObjectName("settingsFrame")
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("RQA statistics"))
        self.stats_panel = QTextEdit()
        self.stats_panel.setObjectName("textPreview")
        self.stats_panel.setReadOnly(True)
        left_layout.addWidget(self.stats_panel, stretch=1)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(8.5, 7.0), dpi=100)
        self.fig.patch.set_facecolor("#0f172a")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setMinimumHeight(480)
        right_layout.addWidget(self.canvas)
        splitter.addWidget(right)
        splitter.setSizes([320, 900])

        self.cmb_symbol.currentIndexChanged.connect(self._on_symbol_changed)
        self.btn_tau_rqa.clicked.connect(self._apply_rqa_params)
        for w in (self.spin_tau, self.spin_m, self.spin_theiler, self.spin_display_cap):
            w.valueChanged.connect(self.refresh)
        self.cmb_radius_mode.currentIndexChanged.connect(self._on_radius_mode_changed)
        self.spin_percentile.valueChanged.connect(self.refresh)
        self.spin_radius.valueChanged.connect(self.refresh)
        self.slider_win_start.valueChanged.connect(self._on_window_changed)
        self.slider_win_end.valueChanged.connect(self._on_window_changed)
        self._on_radius_mode_changed()

    def current_symbol(self) -> str:
        return str(self.cmb_symbol.currentData())

    def _window_fracs(self) -> tuple[float, float]:
        s = self.slider_win_start.value() / 100.0
        e = self.slider_win_end.value() / 100.0
        if e <= s:
            e = min(1.0, s + 0.01)
        return s, e

    def _test_point_cap(self) -> int | None:
        if not self.chk_test_limit.isChecked():
            return None
        if self._test_mode_provider is not None and not self._test_mode_provider():
            return None
        from config_loader import dch_test_point_count
        return int(dch_test_point_count())

    def _radius_mode(self) -> str:
        mode = str(self.cmb_radius_mode.currentData())
        if mode == "bat":
            return "manual"
        return mode

    def _manual_radius_value(self) -> float:
        if str(self.cmb_radius_mode.currentData()) == "bat":
            _, rad, _ = rqa_params_for_symbol(self.current_symbol())
            return float(rad)
        return float(self.spin_radius.value())

    def _on_radius_mode_changed(self) -> None:
        mode = str(self.cmb_radius_mode.currentData())
        self.spin_percentile.setEnabled(mode == "percentile")
        self.spin_radius.setEnabled(mode == "manual")
        self.refresh()

    def _apply_rqa_params(self) -> None:
        tau, rad, w = rqa_params_for_symbol(self.current_symbol())
        self.spin_tau.setValue(int(tau))
        self.spin_theiler.setValue(int(w))
        self.spin_radius.setValue(float(rad))
        self._update_feasibility_hint()
        self.refresh()

    def _on_symbol_changed(self) -> None:
        self._apply_rqa_params()
        self.refresh()

    def _on_window_changed(self) -> None:
        s, e = self._window_fracs()
        self.lbl_win.setText(f"{int(s * 100)}% – {int(e * 100)}%")
        self.refresh()

    def _update_feasibility_hint(self) -> None:
        try:
            from attractor_core import load_logreturns
            s, e = self._window_fracs()
            series = load_logreturns(
                self.current_symbol(), self.config,
                max_points=self._test_point_cap(), start_frac=s, end_frac=e,
            )
            n = len(series)
        except (FileNotFoundError, OSError):
            self.lbl_feasible.setText("")
            return
        m, tau = int(self.spin_m.value()), int(self.spin_tau.value())
        en = max(0, n - (m - 1) * tau)
        self.lbl_feasible.setText(
            f"N={n} | embedded~{en} | max tau={max_feasible_tau(n, m)}"
        )

    def refresh(self, *, force: bool = False) -> None:
        self._debounce.stop()
        if force:
            self._start_compute()
        else:
            self._debounce.start()

    def _start_compute(self) -> None:
        self._request_id += 1
        rid = self._request_id
        s, e = self._window_fracs()
        sym = self.current_symbol()
        self.meta.setText(f"Computing recurrence plot for {sym}…")
        self._update_feasibility_hint()
        run_recurrence_compute(
            rid,
            {
                "symbol": sym,
                "config": self.config,
                "tau": int(self.spin_tau.value()),
                "m": int(self.spin_m.value()),
                "theiler_w": int(self.spin_theiler.value()),
                "radius_mode": self._radius_mode(),
                "manual_radius": self._manual_radius_value(),
                "percentile": float(self.spin_percentile.value()),
                "start_frac": s,
                "end_frac": e,
                "test_cap": self._test_point_cap(),
                "max_display": int(self.spin_display_cap.value()),
            },
            self._on_compute_done,
            self._on_compute_fail,
        )

    def _on_compute_fail(self, request_id: int, message: str) -> None:
        if request_id != self._request_id:
            return
        self.meta.setText(message)
        self.stats_panel.setPlainText(message)
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", color="#e2e8f0", transform=ax.transAxes)
        self.canvas.draw_idle()

    def _on_compute_done(self, result: RecurrenceComputeResult) -> None:
        if result.request_id != self._request_id:
            return
        self._cached = result
        self.stats_panel.setPlainText(result.stats_text)
        self._draw_plots(result)
        b = result.bundle
        metrics = b["metrics"]
        self.meta.setText(
            f"{result.symbol}: RP {b['n_display']}x{b['n_display']} "
            f"(from {b['n_embedded']} embedded). "
            f"RR={metrics.get('RR', float('nan')):.4f} DET={metrics.get('DET', float('nan')):.4f}"
        )

    def _draw_plots(self, result: RecurrenceComputeResult) -> None:
        b = result.bundle
        matrix = b["matrix"]
        tau, m = result.tau, result.m

        self.fig.clf()
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.2, 1.0], hspace=0.32, wspace=0.28)

        ax_rp = self.fig.add_subplot(gs[0, 0])
        ax_rp.set_facecolor("#0f172a")
        if matrix.size:
            ax_rp.imshow(
                matrix, origin="lower", aspect="auto", cmap=self._rp_cmap,
                vmin=0, vmax=2, interpolation="nearest",
            )
        ax_rp.set_title(
            f"Recurrence plot (eps={b['radius']:.4g})",
            color="#e2e8f0", fontsize=10,
        )
        ax_rp.set_xlabel("time index (display subsample)", color="#cbd5e1", fontsize=8)
        ax_rp.set_ylabel("time index (display subsample)", color="#cbd5e1", fontsize=8)
        ax_rp.tick_params(colors="#94a3b8", labelsize=7)

        ax_diag = self.fig.add_subplot(gs[0, 1])
        ax_diag.set_facecolor("#0f172a")
        dk, dd = b["diag_k"], b["diag_density"]
        if dk.size:
            ax_diag.plot(dk, dd, "o-", color="#38bdf8", markersize=3, linewidth=1.2)
            trend = result.bundle["metrics"].get("TREND", float("nan"))
            if np.isfinite(trend):
                ax_diag.set_title(f"Diagonal density (TREND={trend:.4g})", color="#e2e8f0", fontsize=10)
            else:
                ax_diag.set_title("Diagonal recurrence density", color="#e2e8f0", fontsize=10)
        ax_diag.set_xlabel("diagonal lag k", color="#cbd5e1", fontsize=8)
        ax_diag.set_ylabel("recurrence density", color="#cbd5e1", fontsize=8)
        ax_diag.tick_params(colors="#94a3b8", labelsize=7)
        ax_diag.grid(True, color="#334155", alpha=0.35)

        ax_phase = self.fig.add_subplot(gs[1, :])
        ax_phase.set_facecolor("#0f172a")
        pts = b["phase2d"]
        if pts.shape[0]:
            step = max(1, pts.shape[0] // 4000)
            sl = pts[::step]
            ax_phase.plot(sl[:, 0], sl[:, 1], color="#60a5fa", linewidth=0.5, alpha=0.75)
        ax_phase.set_xlabel(coord_label(0, tau), color="#cbd5e1", fontsize=8)
        ax_phase.set_ylabel(coord_label(min(1, m - 1), tau), color="#cbd5e1", fontsize=8)
        ax_phase.set_title(f"Phase-space slice (m={m}, tau={tau})", color="#e2e8f0", fontsize=10)
        ax_phase.tick_params(colors="#94a3b8", labelsize=7)
        ax_phase.grid(True, color="#334155", alpha=0.35)

        self.fig.suptitle(
            f"{result.symbol} — recurrence / RQA  |  Theiler W={result.theiler_w}",
            color="#e2e8f0", fontsize=11,
        )
        self.canvas.draw_idle()

    def _save_png(self) -> None:
        from config_loader import ensure_dir, get_results_dir

        if self._cached is None:
            QMessageBox.information(self, "Save PNG", "Nothing to save — run Refresh first.")
            return
        out_dir = Path(ensure_dir(str(Path(get_results_dir(self.config)) / "recurrence_viewer")))
        out_path = out_dir / (
            f"{self.current_symbol()}_rp_tau{self.spin_tau.value()}_m{self.spin_m.value()}.png"
        )
        try:
            self.fig.savefig(out_path, dpi=150, facecolor=self.fig.get_facecolor())
            self.meta.setText(f"Saved: {out_path}")
        except OSError as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
