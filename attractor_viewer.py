#!/usr/bin/env python3
"""
Interactive Takens attractor explorer for the DCh desktop app.

Features: tau-m heatmap, PCA / manual / animated dim triples, time window,
recurrence colouring (2D), surrogate overlay, matplotlib + OpenGL renderers,
auto LOD, local expansion proxy in statistics.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from attractor_cache import get_embedding, get_surrogate, series_key
from attractor_core import (
    M_MAX,
    M_MIN,
    TAU_MAX,
    axis_limits_with_margin,
    coord_label,
    dim_triple_sequence,
    load_logreturns,
    max_feasible_tau,
    project_embedding,
    recurrence_density_2d,
    select_lod_indices,
    subsample_indices,
)
from attractor_gl import GlAttractorCanvas, gl_available
from attractor_worker import (
    AttractorComputeResult,
    CaoFnnVizResult,
    HeatmapResult,
    run_attractor_compute,
    run_cao_fnn_viz,
    run_heatmap_compute,
    set_worker_thread_count,
)
from config_loader import PIPELINE_SYMBOLS, tau_for_symbol_from_mutual

# Re-export for backward compatibility
__all__ = ["Attractor3DWidget", "delay_embedding", "load_logreturns"]


class Attractor3DWidget(QWidget):
    """PySide6 panel: controls, heatmap, statistics, matplotlib / OpenGL views."""

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
        self._canvas: FigureCanvasQTAgg | None = None
        self._gl: GlAttractorCanvas | None = None
        self._ax = None
        self._view_centers = None
        self._view_half_spans = None
        self._zoom_level = 1.0
        self._default_elev = 24.0
        self._default_azim = -58.0
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(1800)
        self._anim_timer.timeout.connect(self._anim_step)
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(180)
        self._debounce_timer.timeout.connect(self._start_debounced_compute)
        self._anim_idx = 0
        self._heatmap_m_vals: np.ndarray | None = None
        self._heatmap_tau_vals: np.ndarray | None = None
        self._suppress_anim_refresh = False
        self._request_id = 0
        self._heatmap_request_id = 0
        self._cached: AttractorComputeResult | None = None
        self._cao_cached: CaoFnnVizResult | None = None
        self._cao_request_id = 0
        set_worker_thread_count(4)
        self._build_ui()
        if gl_available():
            self.cmb_renderer.setCurrentIndex(self.cmb_renderer.findData("gl"))
            self.plot_stack.setCurrentIndex(1)
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
        row1.addWidget(QLabel("τ:"))
        self.spin_tau = QSpinBox()
        self.spin_tau.setRange(1, TAU_MAX)
        self.spin_tau.setValue(3)
        row1.addWidget(self.spin_tau)
        self.btn_tau_mi = QPushButton("MI τ")
        self.btn_tau_mi.setObjectName("btnGhost")
        row1.addWidget(self.btn_tau_mi)
        row1.addWidget(QLabel("m:"))
        self.spin_m = QSpinBox()
        self.spin_m.setRange(M_MIN, M_MAX)
        self.spin_m.setValue(3)
        row1.addWidget(self.spin_m)
        self.lbl_feasible = QLabel("")
        self.lbl_feasible.setObjectName("previewPath")
        row1.addWidget(self.lbl_feasible, stretch=1)
        tb.addLayout(row1)

        row1b = QHBoxLayout()
        row1b.addWidget(QLabel("Projection:"))
        self.cmb_projection = QComboBox()
        self.cmb_projection.addItem("Manual axes", "manual")
        self.cmb_projection.addItem("PCA (3D)", "pca")
        self.cmb_projection.addItem("Animate dim triples", "animate")
        self.cmb_projection.addItem("Cao FNN (dual view)", "cao_fnn")
        row1b.addWidget(self.cmb_projection, stretch=1)
        row1b.addWidget(QLabel("Renderer:"))
        self.cmb_renderer = QComboBox()
        self.cmb_renderer.addItem("Matplotlib", "mpl")
        gl_label = "OpenGL (pyqtgraph)" if gl_available() else "OpenGL (unavailable)"
        self.cmb_renderer.addItem(gl_label, "gl")
        if not gl_available():
            self.cmb_renderer.model().item(1).setEnabled(False)
        row1b.addWidget(self.cmb_renderer)
        tb.addLayout(row1b)

        row_axes = QHBoxLayout()
        row_axes.addWidget(QLabel("View axes:"))
        row_axes.addWidget(QLabel("X"))
        self.cmb_dim_x = QComboBox()
        row_axes.addWidget(self.cmb_dim_x)
        row_axes.addWidget(QLabel("Y"))
        self.cmb_dim_y = QComboBox()
        row_axes.addWidget(self.cmb_dim_y)
        row_axes.addWidget(QLabel("Z"))
        self.cmb_dim_z = QComboBox()
        row_axes.addWidget(self.cmb_dim_z)
        self.btn_axes_default = QPushButton("0,1,2")
        self.btn_axes_default.setObjectName("btnGhost")
        row_axes.addWidget(self.btn_axes_default)
        row_axes.addStretch(1)
        tb.addLayout(row_axes)

        row_win = QHBoxLayout()
        row_win.addWidget(QLabel("Time window:"))
        self.slider_win_start = QSlider(Qt.Orientation.Horizontal)
        self.slider_win_start.setRange(0, 99)
        self.slider_win_start.setValue(0)
        self.slider_win_end = QSlider(Qt.Orientation.Horizontal)
        self.slider_win_end.setRange(1, 100)
        self.slider_win_end.setValue(100)
        self.lbl_win = QLabel("0% – 100%")
        self.lbl_win.setMinimumWidth(90)
        row_win.addWidget(self.slider_win_start, stretch=1)
        row_win.addWidget(self.slider_win_end, stretch=1)
        row_win.addWidget(self.lbl_win)
        tb.addLayout(row_win)

        row2 = QHBoxLayout()
        self.chk_color_time = QCheckBox("Colour by time")
        self.chk_color_time.setChecked(True)
        row2.addWidget(self.chk_color_time)
        self.chk_scatter = QCheckBox("Scatter")
        row2.addWidget(self.chk_scatter)
        self.chk_recurrence = QCheckBox("Recurrence colour (2D)")
        row2.addWidget(self.chk_recurrence)
        self.chk_surrogate = QCheckBox("Surrogate overlay")
        self.chk_surrogate.setToolTip("Random permutation of the series (reshuffle null).")
        row2.addWidget(self.chk_surrogate)
        self.chk_auto_lod = QCheckBox("Auto LOD")
        self.chk_auto_lod.setChecked(True)
        row2.addWidget(self.chk_auto_lod)
        self.chk_cap_display = QCheckBox("Manual cap")
        row2.addWidget(self.chk_cap_display)
        self.spin_cap = QSpinBox()
        self.spin_cap.setRange(1000, 300000)
        self.spin_cap.setValue(50000)
        self.spin_cap.setEnabled(False)
        row2.addWidget(self.spin_cap)
        self.chk_test_limit = QCheckBox("TEST row limit")
        row2.addWidget(self.chk_test_limit)
        self.chk_rosenstein = QCheckBox("Rosenstein 1-step (not LLE)")
        self.chk_rosenstein.setChecked(False)
        self.chk_rosenstein.setToolTip(
            "Median one-step nearest-neighbour expansion rate (Rosenstein-style). "
            "Exploratory only — not the Kant/lyap_k LLE used in hypothesis testing."
        )
        row2.addWidget(self.chk_rosenstein)
        row2.addWidget(QLabel("Line w:"))
        self.spin_linewidth = QDoubleSpinBox()
        self.spin_linewidth.setRange(0.1, 3.0)
        self.spin_linewidth.setSingleStep(0.1)
        self.spin_linewidth.setValue(0.4)
        row2.addWidget(self.spin_linewidth)
        tb.addLayout(row2)

        row3 = QHBoxLayout()
        for btn, slot in (
            (QPushButton("Refresh"), lambda: self.refresh(force=True)),
            (QPushButton("Reset view"), self._reset_view),
            (QPushButton("Zoom +"), lambda: self._zoom_by(0.8)),
            (QPushButton("Zoom -"), lambda: self._zoom_by(1.25)),
            (QPushButton("Save PNG"), self._save_png),
        ):
            btn.setObjectName("btnPrimary" if btn.text() == "Refresh" else "btnGhost")
            btn.clicked.connect(slot)
            row3.addWidget(btn)
        self.lbl_zoom = QLabel("100%")
        row3.addWidget(self.lbl_zoom)
        row3.addStretch(1)
        tb.addLayout(row3)

        root.addWidget(toolbar)
        self.meta = QLabel("Heatmap: click a cell to set τ and m.")
        self.meta.setObjectName("previewMeta")
        self.meta.setWordWrap(True)
        root.addWidget(self.meta)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        left = QFrame()
        left.setObjectName("settingsFrame")
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Phase-space statistics"))
        self.stats_panel = QTextEdit()
        self.stats_panel.setObjectName("textPreview")
        self.stats_panel.setReadOnly(True)
        left_layout.addWidget(self.stats_panel, stretch=2)

        hm_row = QHBoxLayout()
        hm_row.addWidget(QLabel("τ–m map:"))
        self.cmb_heatmap_metric = QComboBox()
        self.cmb_heatmap_metric.addItem("embedded N", "embedded_n")
        self.cmb_heatmap_metric.addItem("log10(N)", "log_embedded_n")
        self.cmb_heatmap_metric.addItem("mean |corr|", "mean_corr")
        self.cmb_heatmap_metric.addItem("path length", "path_length")
        hm_row.addWidget(self.cmb_heatmap_metric)
        self.btn_heatmap = QPushButton("Update")
        self.btn_heatmap.setObjectName("btnGhost")
        hm_row.addWidget(self.btn_heatmap)
        left_layout.addLayout(hm_row)

        self.heatmap_fig = Figure(figsize=(3.4, 3.0), dpi=96)
        self.heatmap_fig.patch.set_facecolor("#0f172a")
        self.heatmap_canvas = FigureCanvasQTAgg(self.heatmap_fig)
        self.heatmap_canvas.mpl_connect("button_press_event", self._on_heatmap_click)
        self.heatmap_canvas.setMinimumHeight(220)
        left_layout.addWidget(self.heatmap_canvas, stretch=1)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_stack = QStackedWidget()
        fig = Figure(figsize=(7.5, 6.5), dpi=100)
        fig.patch.set_facecolor("#0f172a")
        self._canvas = FigureCanvasQTAgg(fig)
        self._canvas.setMinimumHeight(420)
        self._canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
        self.plot_stack.addWidget(self._canvas)
        self._gl = GlAttractorCanvas()
        self.plot_stack.addWidget(self._gl)
        self._cao_fig = Figure(figsize=(7.5, 6.5), dpi=100)
        self._cao_fig.patch.set_facecolor("#0f172a")
        self._cao_canvas = FigureCanvasQTAgg(self._cao_fig)
        self._cao_canvas.setMinimumHeight(420)
        self.plot_stack.addWidget(self._cao_canvas)
        right_layout.addWidget(self.plot_stack)
        splitter.addWidget(right)
        splitter.setSizes([360, 900])

        self.cmb_symbol.currentIndexChanged.connect(self._on_symbol_changed)
        self.btn_tau_mi.clicked.connect(self._apply_mi_tau)
        self.spin_m.valueChanged.connect(self._on_m_changed)
        self.spin_tau.valueChanged.connect(self._on_tau_changed)
        self.cmb_projection.currentIndexChanged.connect(self._on_projection_changed)
        self.cmb_renderer.currentIndexChanged.connect(self._on_renderer_changed)
        self.btn_axes_default.clicked.connect(self._set_default_view_axes)
        self.cmb_dim_x.currentIndexChanged.connect(self._on_view_axes_changed)
        self.cmb_dim_y.currentIndexChanged.connect(self._on_view_axes_changed)
        self.cmb_dim_z.currentIndexChanged.connect(self._on_view_axes_changed)
        self.chk_cap_display.toggled.connect(lambda c: self.spin_cap.setEnabled(c))
        self.slider_win_start.valueChanged.connect(self._on_window_changed)
        self.slider_win_end.valueChanged.connect(self._on_window_changed)
        self.btn_heatmap.clicked.connect(self._update_heatmap)
        self.cmb_heatmap_metric.currentIndexChanged.connect(self._update_heatmap)
        self.spin_linewidth.valueChanged.connect(self._on_style_changed)
        for w in (self.chk_scatter, self.chk_color_time, self.chk_auto_lod, self.chk_cap_display):
            w.toggled.connect(self._on_style_changed)
        self.chk_recurrence.toggled.connect(self.refresh)
        self.chk_surrogate.toggled.connect(self.refresh)
        self.chk_rosenstein.toggled.connect(self.refresh)

        self._sync_dim_combos()
        self._on_symbol_changed()

    def current_symbol(self) -> str:
        return str(self.cmb_symbol.currentData())

    def _projection_mode(self) -> str:
        return str(self.cmb_projection.currentData())

    def _renderer_mode(self) -> str:
        return str(self.cmb_renderer.currentData())

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

    def _on_symbol_changed(self) -> None:
        self._apply_mi_tau()
        self._update_heatmap()
        self.refresh()

    def _apply_mi_tau(self) -> None:
        self.spin_tau.setValue(int(tau_for_symbol_from_mutual(self.current_symbol(), self.config)))
        self._update_feasibility_hint()

    def _on_m_changed(self) -> None:
        self._sync_dim_combos()
        self._update_feasibility_hint()
        if self._projection_mode() == "animate":
            self._anim_idx = 0
        self.refresh()

    def _on_tau_changed(self) -> None:
        self._sync_dim_combos()
        self._update_feasibility_hint()
        self.refresh()

    def _on_window_changed(self) -> None:
        s, e = self._window_fracs()
        self.lbl_win.setText(f"{int(s*100)}% – {int(e*100)}%")
        self.refresh()

    def _on_projection_changed(self) -> None:
        mode = self._projection_mode()
        manual = mode == "manual"
        cao = mode == "cao_fnn"
        self.cmb_dim_x.setEnabled(manual or cao)
        self.cmb_dim_y.setEnabled(manual or cao)
        self.cmb_dim_z.setEnabled(manual or cao)
        self.btn_axes_default.setEnabled(manual or cao)
        self.cmb_renderer.setEnabled(not cao)
        if cao:
            self._anim_timer.stop()
            self.plot_stack.setCurrentIndex(2)
            self.meta.setText("Cao FNN: filled = true neighbors, hollow = false neighbors (a_i(m) ratio).")
        elif mode == "animate":
            self._anim_idx = 0
            self._anim_timer.start()
        else:
            self._anim_timer.stop()
        self.refresh()

    def _on_renderer_changed(self) -> None:
        if self._projection_mode() == "cao_fnn":
            self.plot_stack.setCurrentIndex(2)
            return
        self.plot_stack.setCurrentIndex(1 if self._renderer_mode() == "gl" else 0)
        self._repaint_from_cache()

    def _anim_step(self) -> None:
        if self._projection_mode() != "animate":
            self._anim_timer.stop()
            return
        m = int(self.spin_m.value())
        seq = dim_triple_sequence(m)
        if not seq:
            return
        self._anim_idx = (self._anim_idx + 1) % len(seq)
        triple = seq[self._anim_idx]
        self._suppress_anim_refresh = True
        for combo, pick in zip((self.cmb_dim_x, self.cmb_dim_y, self.cmb_dim_z), triple):
            idx = combo.findData(pick)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        self._suppress_anim_refresh = False
        if self._cached is not None:
            self._fast_reproject_manual()
        else:
            self.refresh()

    def _set_default_view_axes(self) -> None:
        m = int(self.spin_m.value())
        for combo, pick in zip((self.cmb_dim_x, self.cmb_dim_y, self.cmb_dim_z), (0, 1, min(2, m - 1))):
            idx = combo.findData(pick)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def _sync_dim_combos(self) -> None:
        m = int(self.spin_m.value())
        tau = int(self.spin_tau.value())
        prev = (self.cmb_dim_x.currentData(), self.cmb_dim_y.currentData(), self.cmb_dim_z.currentData())
        for combo in (self.cmb_dim_x, self.cmb_dim_y, self.cmb_dim_z):
            combo.blockSignals(True)
            combo.clear()
            for d in range(m):
                combo.addItem(f"dim {d} ({coord_label(d, tau)})", d)
            combo.blockSignals(False)
        self.cmb_dim_z.setEnabled(m >= 3)
        for combo, default, old in zip((self.cmb_dim_x, self.cmb_dim_y, self.cmb_dim_z), (0, 1, min(2, m - 1)), prev):
            want = int(old) if old is not None and 0 <= int(old) < m else default
            idx = combo.findData(want)
            combo.setCurrentIndex(max(0, idx))

    def _view_dims(self) -> tuple[int, int, int]:
        m = int(self.spin_m.value())
        return (
            int(self.cmb_dim_x.currentData()),
            int(self.cmb_dim_y.currentData()),
            int(self.cmb_dim_z.currentData()) if m >= 3 else 0,
        )

    def _on_view_axes_changed(self) -> None:
        if self._suppress_anim_refresh:
            return
        if self._projection_mode() == "animate":
            self._anim_timer.stop()
            self.cmb_projection.setCurrentIndex(self.cmb_projection.findData("manual"))
        mode = self._projection_mode()
        if mode == "cao_fnn":
            self.refresh()
        elif self._cached is not None and mode == "manual":
            self._fast_reproject_manual()
        else:
            self.refresh()

    def _update_feasibility_hint(self) -> None:
        try:
            s, e = self._window_fracs()
            series = load_logreturns(
                self.current_symbol(), self.config, max_points=self._test_point_cap(),
                start_frac=s, end_frac=e,
            )
            n = len(series)
        except (FileNotFoundError, OSError):
            self.lbl_feasible.setText("")
            return
        m, tau = int(self.spin_m.value()), int(self.spin_tau.value())
        en = max(0, n - (m - 1) * tau)
        self.lbl_feasible.setText(
            f"N={n} | max τ={max_feasible_tau(n, m)} | embedded≈{en}"
            + ("" if en > 0 else " | INVALID")
        )

    def _update_heatmap(self) -> None:
        self._heatmap_request_id += 1
        rid = self._heatmap_request_id
        s, e = self._window_fracs()
        m_max_hm = min(30, M_MAX)
        m_values = np.arange(M_MIN, m_max_hm + 1)
        try:
            series = load_logreturns(
                self.current_symbol(), self.config, max_points=self._test_point_cap(),
                start_frac=s, end_frac=e,
            )
            n = len(series)
            tau_cap = min(200, max_feasible_tau(n, M_MIN))
            tau_values = np.unique(np.linspace(1, max(1, tau_cap), num=min(40, tau_cap), dtype=int))
        except (FileNotFoundError, OSError) as exc:
            self._draw_heatmap_error(str(exc))
            return

        self._heatmap_m_vals = m_values
        self._heatmap_tau_vals = tau_values
        metric = str(self.cmb_heatmap_metric.currentData())
        run_heatmap_compute(
            rid,
            {
                "symbol": self.current_symbol(),
                "config": self.config,
                "start_frac": s,
                "end_frac": e,
                "test_cap": self._test_point_cap(),
                "m_values": m_values,
                "tau_values": tau_values,
                "metric": metric,
            },
            self._on_heatmap_done,
            self._on_heatmap_fail,
        )

    def _draw_heatmap_error(self, message: str) -> None:
        self.heatmap_fig.clf()
        ax = self.heatmap_fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", color="#e2e8f0", transform=ax.transAxes)
        self.heatmap_canvas.draw_idle()

    def _on_heatmap_done(self, result: HeatmapResult) -> None:
        if result.request_id != self._heatmap_request_id:
            return
        self._heatmap_m_vals = result.m_values
        self._heatmap_tau_vals = result.tau_values
        self.heatmap_fig.clf()
        ax = self.heatmap_fig.add_subplot(111)
        ax.set_facecolor("#0f172a")
        im = ax.imshow(
            result.grid, aspect="auto", origin="lower", cmap="viridis",
            extent=[
                result.tau_values[0] - 0.5, result.tau_values[-1] + 0.5,
                result.m_values[0] - 0.5, result.m_values[-1] + 0.5,
            ],
        )
        ax.plot(self.spin_tau.value(), self.spin_m.value(), "r*", markersize=10)
        ax.set_xlabel("tau", color="#cbd5e1")
        ax.set_ylabel("m", color="#cbd5e1")
        ax.set_title(f"tau-m ({result.metric})", color="#e2e8f0", fontsize=9)
        ax.tick_params(colors="#94a3b8", labelsize=7)
        self.heatmap_fig.colorbar(im, ax=ax, fraction=0.046)
        self.heatmap_fig.tight_layout()
        self.heatmap_canvas.draw_idle()

    def _on_heatmap_fail(self, request_id: int, message: str) -> None:
        if request_id != self._heatmap_request_id:
            return
        self._draw_heatmap_error(message)

    def _on_heatmap_click(self, event) -> None:
        if event.inaxes is None or self._heatmap_m_vals is None or self._heatmap_tau_vals is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        tau = int(round(event.xdata))
        m = int(round(event.ydata))
        tau = int(np.clip(tau, 1, TAU_MAX))
        m = int(np.clip(m, M_MIN, M_MAX))
        self.spin_tau.setValue(tau)
        self.spin_m.setValue(m)
        self.refresh()

    def _view_mask_from_limits(self, pts: np.ndarray) -> np.ndarray | None:
        if self._ax is None or self._zoom_level >= 0.95:
            return None
        try:
            xlo, xhi = self._ax.get_xlim()
            ylo, yhi = self._ax.get_ylim()
            mask = (
                (pts[:, 0] >= xlo) & (pts[:, 0] <= xhi)
                & (pts[:, 1] >= ylo) & (pts[:, 1] <= yhi)
            )
            if pts.shape[1] >= 3 and hasattr(self._ax, "get_zlim"):
                zlo, zhi = self._ax.get_zlim()
                mask &= (pts[:, 2] >= zlo) & (pts[:, 2] <= zhi)
            return mask
        except Exception:
            return None

    def _store_view_limits(self, pts: np.ndarray) -> None:
        self._view_centers = tuple(float(np.mean(pts[:, i])) for i in range(3))
        self._view_half_spans = tuple(
            (axis_limits_with_margin(pts[:, i])[1] - axis_limits_with_margin(pts[:, i])[0]) / 2.0
            for i in range(3)
        )
        self._zoom_level = 1.0
        self.lbl_zoom.setText("100%")

    def _apply_zoom_limits(self, *, redraw: bool = True) -> None:
        if self._ax is None or self._view_centers is None:
            return
        cx, cy, cz = self._view_centers
        hx, hy, hz = self._view_half_spans
        z = self._zoom_level
        self._ax.set_xlim(cx - hx * z, cx + hx * z)
        self._ax.set_ylim(cy - hy * z, cy + hy * z)
        if getattr(self, "_last_projection", "") == "3d" and hasattr(self._ax, "set_zlim"):
            self._ax.set_zlim(cz - hz * z, cz + hz * z)
        self.lbl_zoom.setText(f"{int(round(100 / max(z, 1e-6)))}%")
        if redraw and self._canvas is not None:
            self._canvas.draw_idle()

    def _zoom_by(self, factor: float) -> None:
        self._zoom_level = float(np.clip(self._zoom_level * factor, 0.02, 50.0))
        if self._renderer_mode() == "mpl":
            self._apply_zoom_limits()
            if self.chk_auto_lod.isChecked():
                self._repaint_from_cache()
        else:
            self.lbl_zoom.setText(f"{int(round(100 / max(self._zoom_level, 1e-6)))}%")
            self._repaint_from_cache()

    def _on_scroll_zoom(self, event) -> None:
        if self._ax is None or event.inaxes != self._ax:
            return
        self._zoom_by(0.9 if event.button == "up" else 1.1)

    def _reset_view(self) -> None:
        self._zoom_level = 1.0
        if self._ax is not None:
            self._ax.view_init(elev=self._default_elev, azim=self._default_azim)
        if self._gl is not None:
            self._gl.reset_view()
        self._apply_zoom_limits()
        self._repaint_from_cache()

    def _redraw_style_only(self) -> None:
        if getattr(self, "_last_plot_pts", None) is None:
            return
        self._draw_matplotlib(
            self._last_plot_pts, self._last_t_norm, self._last_title,
            self._last_axis_labels, self._last_projection, self._last_colors,
            self._last_overlay,
        )
        self._apply_zoom_limits()

    def _draw_matplotlib(
        self, pts, t_norm, title, axis_labels, projection, colors, overlay,
    ) -> None:
        fig = self._canvas.figure
        fig.clf()
        lw = float(self.spin_linewidth.value())
        fig.patch.set_facecolor("#0f172a")

        if projection == "2d":
            self._ax = fig.add_subplot(111)
            ax = self._ax
            ax.set_facecolor("#0f172a")
            c = colors if colors is not None else (t_norm if self.chk_color_time.isChecked() else None)
            if self.chk_scatter.isChecked():
                if c is not None:
                    sc = ax.scatter(pts[:, 0], pts[:, 1], c=c, cmap="viridis", s=max(1.0, lw * 6), alpha=0.75)
                    fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02, label="time" if colors is None else "rec")
                else:
                    ax.scatter(pts[:, 0], pts[:, 1], c="#60a5fa", s=max(1.0, lw * 6), alpha=0.75)
            elif c is not None and len(pts) > 1:
                import matplotlib.cm as cm
                from matplotlib.collections import LineCollection
                segments = np.stack([pts[:-1], pts[1:]], axis=1)
                cmap = cm.get_cmap("viridis" if colors is None else "hot")
                lc = LineCollection(segments, colors=cmap(c[:-1] if colors is None else c), linewidths=lw, alpha=0.9)
                ax.add_collection(lc)
                ax.autoscale_view()
            else:
                ax.plot(pts[:, 0], pts[:, 1], color="#60a5fa", linewidth=lw, alpha=0.85)
            if overlay is not None:
                ax.plot(overlay[:, 0], overlay[:, 1], color="#f87171", linewidth=lw * 0.8, alpha=0.45)
            ax.set_xlabel(axis_labels[0], color="#cbd5e1")
            ax.set_ylabel(axis_labels[1], color="#cbd5e1")
            ax.tick_params(colors="#94a3b8", labelsize=8)
            ax.grid(True, color="#334155", alpha=0.35)
        else:
            self._ax = fig.add_subplot(111, projection="3d")
            ax = self._ax
            ax.set_facecolor("#0f172a")
            if self.chk_scatter.isChecked():
                c = t_norm if self.chk_color_time.isChecked() else None
                if c is not None:
                    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=c, cmap="viridis", s=max(1.0, lw * 6), alpha=0.7)
                    fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02, label="time")
                else:
                    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="#60a5fa", s=max(1.0, lw * 6), alpha=0.7)
            elif self.chk_color_time.isChecked() and len(pts) > 1:
                import matplotlib.cm as cm
                from mpl_toolkits.mplot3d.art3d import Line3DCollection
                segments = np.stack([pts[:-1], pts[1:]], axis=1)
                lc = Line3DCollection(segments, colors=cm.get_cmap("viridis")(t_norm[:-1]), linewidths=lw, alpha=0.9)
                ax.add_collection3d(lc)
                ax.auto_scale_xyz(pts[:, 0], pts[:, 1], pts[:, 2])
            else:
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#60a5fa", linewidth=lw, alpha=0.85)
            if overlay is not None:
                ax.plot(overlay[:, 0], overlay[:, 1], overlay[:, 2], color="#f87171", linewidth=lw * 0.8, alpha=0.4)
            ax.set_xlabel(axis_labels[0], color="#cbd5e1", labelpad=8)
            ax.set_ylabel(axis_labels[1], color="#cbd5e1", labelpad=8)
            ax.set_zlabel(axis_labels[2], color="#cbd5e1", labelpad=8)
            ax.tick_params(colors="#94a3b8", labelsize=8)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.fill = False
                axis.pane.set_edgecolor("#334155")
            ax.grid(True, color="#334155", alpha=0.35)
            ax.view_init(elev=self._default_elev, azim=self._default_azim)

        ax.set_title(title, color="#e2e8f0", fontsize=11)
        fig.tight_layout()
        self._canvas.draw_idle()

    def _save_png(self) -> None:
        from config_loader import ensure_dir, get_results_dir

        out_dir = Path(ensure_dir(str(Path(get_results_dir(self.config)) / "phase_3d")))
        out_path = out_dir / (
            f"{self.current_symbol()}_phase3d_tau{self.spin_tau.value()}_m{self.spin_m.value()}.png"
        )
        try:
            if self._projection_mode() == "cao_fnn":
                fig = self._cao_fig
            elif self._renderer_mode() == "gl" and self._gl is not None:
                QMessageBox.information(self, "Save PNG", "Switch to Matplotlib renderer to save PNG from the canvas.")
                return
            else:
                fig = self._canvas.figure
            fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
            self.meta.setText(f"Saved: {out_path}")
        except OSError as exc:
            QMessageBox.warning(self, "Save failed", str(exc))

    def refresh(self, *, force: bool = False) -> None:
        self._debounce_timer.stop()
        if force:
            self._start_debounced_compute()
        else:
            self._debounce_timer.start()

    def _start_debounced_compute(self) -> None:
        if self._projection_mode() == "cao_fnn":
            self._start_cao_fnn_compute()
        else:
            self._start_background_compute()

    def _cao_projection_pairs(self) -> tuple[tuple[int, int], tuple[int, int]]:
        m = int(self.spin_m.value())
        dims = self._view_dims()
        z = dims[2] if m >= 3 else 1
        return (dims[0], z), (dims[1], z)

    def _start_cao_fnn_compute(self) -> None:
        self._cao_request_id += 1
        rid = self._cao_request_id
        s, e = self._window_fracs()
        sym = self.current_symbol()
        tau, m = int(self.spin_tau.value()), int(self.spin_m.value())
        proj_a, proj_b = self._cao_projection_pairs()
        self.plot_stack.setCurrentIndex(2)
        self.meta.setText(f"Cao FNN: computing {sym} m={m} tau={tau}…")
        run_cao_fnn_viz(
            rid,
            {
                "symbol": sym,
                "config": self.config,
                "tau": tau,
                "m": m,
                "start_frac": s,
                "end_frac": e,
                "test_cap": self._test_point_cap(),
                "proj_a": proj_a,
                "proj_b": proj_b,
            },
            self._on_cao_fnn_done,
            self._on_cao_fnn_fail,
        )

    def _on_cao_fnn_fail(self, request_id: int, message: str) -> None:
        if request_id != self._cao_request_id:
            return
        self.meta.setText(message)
        self.stats_panel.setPlainText(message)
        self._cao_fig.clf()
        ax = self._cao_fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", color="#e2e8f0", transform=ax.transAxes)
        self._cao_canvas.draw_idle()

    def _on_cao_fnn_done(self, result: CaoFnnVizResult) -> None:
        if result.request_id != self._cao_request_id:
            return
        self._cao_cached = result
        self.lbl_feasible.setText(result.feasible_text)
        self.stats_panel.setPlainText(result.stats_text)
        self._draw_cao_fnn(result)
        ex = result.exemplars
        self.meta.setText(
            f"Cao FNN {result.symbol}: E({ex.m})={ex.E_m:.3f}, "
            f"true a_i={ex.a_i_true:.2f}, false a_i={ex.a_i_false:.2f}"
        )

    def _draw_cao_fnn(self, result: CaoFnnVizResult) -> None:
        from attractor_cao_viz import axis_pair_labels

        ex = result.exemplars
        emb = ex.embedded_m
        tau = ex.tau
        lw = float(self.spin_linewidth.value())
        idx = select_lod_indices(
            emb.shape[0], zoom_level=1.0, auto_lod=self.chk_auto_lod.isChecked(),
            manual_cap=int(self.spin_cap.value()) if self.chk_cap_display.isChecked() else None,
            gl_mode=False,
        )

        self._cao_fig.clf()
        ax_l, ax_r = self._cao_fig.subplots(1, 2)
        for ax in (ax_l, ax_r):
            ax.set_facecolor("#0f172a")
            ax.tick_params(colors="#94a3b8", labelsize=8)
            ax.grid(True, color="#334155", alpha=0.35)

        def draw_panel(ax, dim_h: int, dim_v: int, subtitle: str) -> None:
            ax.plot(
                emb[idx, dim_h], emb[idx, dim_v],
                color="#475569", linewidth=lw * 0.7, alpha=0.55,
            )
            for i, j, filled, color in (
                (ex.true_i, ex.true_j, True, "#38bdf8"),
                (ex.false_i, ex.false_j, False, "#f87171"),
            ):
                xi, yi = emb[i, dim_h], emb[i, dim_v]
                xj, yj = emb[j, dim_h], emb[j, dim_v]
                face = color if filled else "none"
                ax.scatter([xi, xj], [yi, yj], s=72, facecolors=[face, face],
                           edgecolors=color, linewidths=1.8, zorder=5)
                ax.plot([xi, xj], [yi, yj], color=color, linewidth=1.0, alpha=0.6, zorder=4)
            xlab, ylab = axis_pair_labels(dim_h, dim_v, tau)
            ax.set_xlabel(xlab, color="#cbd5e1")
            ax.set_ylabel(ylab, color="#cbd5e1")
            ax.set_title(subtitle, color="#e2e8f0", fontsize=9)

        pa, pb = ex.proj_a, ex.proj_b
        draw_panel(ax_l, pa[0], pa[1], f"dim {pa[0]} vs {pa[1]}  |  E({ex.m})={ex.E_m:.3f}")
        draw_panel(ax_r, pb[0], pb[1], f"dim {pb[0]} vs {pb[1]}")
        self._cao_fig.suptitle(
            f"Cao false-nearest-neighbors  m={ex.m}  tau={tau}  "
            f"(filled=true, hollow=false)",
            color="#e2e8f0", fontsize=11,
        )
        self._cao_fig.tight_layout()
        self._cao_canvas.draw_idle()

    def _start_background_compute(self) -> None:
        self._request_id += 1
        rid = self._request_id
        s, e = self._window_fracs()
        sym = self.current_symbol()
        tau, m = int(self.spin_tau.value()), int(self.spin_m.value())
        self.meta.setText(f"Computing {sym} τ={tau} m={m}…")
        run_attractor_compute(
            rid,
            {
                "symbol": sym,
                "config": self.config,
                "tau": tau,
                "m": m,
                "start_frac": s,
                "end_frac": e,
                "test_cap": self._test_point_cap(),
                "view_dims": self._view_dims(),
                "proj_mode": (
                    "manual"
                    if self._projection_mode() in ("animate", "cao_fnn")
                    else self._projection_mode()
                ),
                "window_note": f"{int(s*100)}% – {int(e*100)}% of series",
                "want_surrogate": self.chk_surrogate.isChecked(),
                "want_recurrence": self.chk_recurrence.isChecked(),
                "want_rosenstein_rate": self.chk_rosenstein.isChecked(),
            },
            self._on_compute_done,
            self._on_compute_fail,
        )

    def _on_compute_fail(self, request_id: int, message: str) -> None:
        if request_id != self._request_id:
            return
        self.meta.setText(message)
        self.stats_panel.setPlainText(message)
        if self._canvas is not None:
            self._canvas.figure.clf()
            ax = self._canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, message, ha="center", va="center", color="#e2e8f0", transform=ax.transAxes)
            self._canvas.draw_idle()

    def _on_compute_done(self, result: AttractorComputeResult) -> None:
        if result.request_id != self._request_id:
            return
        self._cached = result
        self._last_embedded = result.embedded
        self._last_series = result.series
        self.lbl_feasible.setText(result.feasible_text)
        self.stats_panel.setPlainText(result.stats_text)
        self._repaint_from_cache()

    def _fast_reproject_manual(self) -> None:
        if self._cached is None:
            self.refresh()
            return
        c = self._cached
        view_dims = self._view_dims()
        view_pts, axis_labels, projection = project_embedding(
            c.embedded, "manual", view_dims, c.tau,
        )
        recurrence_full = c.recurrence_full
        if self.chk_recurrence.isChecked() and projection == "2d":
            recurrence_full = recurrence_density_2d(view_pts)

        overlay_view = None
        if self.chk_surrogate.isChecked():
            ws, we = self._window_fracs()
            sk = series_key(c.symbol, self.config, ws, we, self._test_point_cap())
            surr = get_surrogate(c.series, sk, seed=0)
            emb_s = get_embedding(surr, c.tau, c.m, (sk, "surr"))
            overlay_view, _, _ = project_embedding(emb_s, "manual", view_dims, c.tau)

        self._cached = replace(
            c,
            view_pts=view_pts,
            axis_labels=axis_labels,
            projection=projection,
            view_dims=view_dims,
            recurrence_full=recurrence_full,
            overlay_view=overlay_view,
        )
        self._repaint_from_cache()

    def _on_style_changed(self) -> None:
        if self._projection_mode() == "cao_fnn":
            if self._cao_cached is not None:
                self._draw_cao_fnn(self._cao_cached)
            return
        self._repaint_from_cache()

    def _repaint_from_cache(self) -> None:
        if self._projection_mode() == "cao_fnn":
            if self._cao_cached is not None:
                self._draw_cao_fnn(self._cao_cached)
            return
        if self._cached is None:
            self.refresh(force=True)
            return
        r = self._cached
        manual_cap = int(self.spin_cap.value()) if self.chk_cap_display.isChecked() else None
        view_mask = self._view_mask_from_limits(r.view_pts)
        gl_mode = self._renderer_mode() == "gl"
        idx_local = select_lod_indices(
            r.view_pts.shape[0],
            zoom_level=self._zoom_level,
            auto_lod=self.chk_auto_lod.isChecked(),
            manual_cap=manual_cap,
            view_mask=view_mask,
            gl_mode=gl_mode,
        )
        subsampled = len(idx_local) < r.view_pts.shape[0]
        pts = r.view_pts[idx_local]
        t_norm = idx_local.astype(np.float64) / max(1, r.view_pts.shape[0] - 1)

        colors = None
        if self.chk_recurrence.isChecked() and r.projection == "2d" and r.recurrence_full is not None:
            colors = r.recurrence_full[idx_local]
            colors = colors / max(float(np.max(colors)), 1.0)

        overlay = None
        if r.overlay_view is not None and self.chk_surrogate.isChecked():
            ov = r.overlay_view
            overlay = ov[idx_local] if ov.shape[0] == r.view_pts.shape[0] else ov[subsample_indices(ov.shape[0], len(idx_local))]

        cap_note = f" | window {r.window_note}"
        if subsampled:
            cap_note += f" | drawn {len(idx_local)}/{r.view_pts.shape[0]}"
        title = f"{r.symbol} m={r.m} tau={r.tau} N={r.embedded.shape[0]}{cap_note}"

        self._last_plot_pts = pts
        self._last_t_norm = t_norm
        self._last_axis_labels = r.axis_labels
        self._last_projection = r.projection
        self._last_colors = colors
        self._last_overlay = overlay
        self._last_title = title

        rs = r.rosenstein_rate
        rs_str = f"{rs:.4f}" if not np.isnan(rs) else "off"
        self.meta.setText(
            f"{r.symbol}: {len(r.series)} pts -> {r.embedded.shape[0]} embedded. "
            f"Rosenstein 1-step={rs_str}. Click heatmap or animate triples."
        )

        if gl_mode and self._gl is not None:
            self.plot_stack.setCurrentIndex(1)
            color_data = t_norm if self.chk_color_time.isChecked() and colors is None else colors
            self._gl.plot(
                pts, colors=color_data, scatter=self.chk_scatter.isChecked(), overlay_pts=overlay,
            )
        else:
            self.plot_stack.setCurrentIndex(0)
            self._draw_matplotlib(pts, t_norm, title, r.axis_labels, r.projection, colors, overlay)
            pts_for_zoom = np.column_stack([pts, np.zeros(len(pts))]) if r.projection == "2d" else pts
            self._store_view_limits(pts_for_zoom)
            self._apply_zoom_limits()
