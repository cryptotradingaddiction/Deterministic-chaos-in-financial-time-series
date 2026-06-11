#!/usr/bin/env python3
"""High-performance PyQtGraph OpenGL canvas for the attractor viewer."""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

_GL_AVAILABLE = False
_GL_IMPORT_ERROR = ""

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl

    _GL_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    _GL_IMPORT_ERROR = str(exc)

# Above this count, draw scatter/line-strip on a decimated subset for orbit FPS.
_GL_LINE_BUDGET = 14000
_GL_SCATTER_BUDGET = 25000


def gl_available() -> bool:
    return _GL_AVAILABLE


def gl_import_error() -> str:
    return _GL_IMPORT_ERROR


class GlAttractorCanvas(QWidget):
    """GPU-backed trajectory view (default interactive renderer)."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._fallback = QLabel(
            "OpenGL renderer unavailable.\nInstall: py -3 -m pip install pyqtgraph PyOpenGL"
        )
        self._fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._view = None
        self._scatter = None
        self._overlay_scatter = None
        self._line = None
        if _GL_AVAILABLE:
            self._build_gl()
        else:
            self._layout.addWidget(self._fallback)

    def _build_gl(self) -> None:
        self._view = gl.GLViewWidget()
        self._view.setBackgroundColor((15, 23, 42))
        self._view.opts["distance"] = 40
        self._scatter = gl.GLScatterPlotItem(size=2.0, pxMode=True)
        self._overlay_scatter = gl.GLScatterPlotItem(size=2.0, pxMode=True)
        self._line = gl.GLLinePlotItem(mode="line_strip", width=1.2, antialias=False)
        self._view.addItem(self._line)
        self._view.addItem(self._scatter)
        self._view.addItem(self._overlay_scatter)
        self._layout.addWidget(self._view)

    def clear(self) -> None:
        if not _GL_AVAILABLE or self._scatter is None:
            return
        empty = np.zeros((0, 3), dtype=np.float32)
        self._scatter.setData(pos=empty)
        self._overlay_scatter.setData(pos=empty)
        self._line.setData(pos=empty)
    @staticmethod
    def _to_f32_xyz(pts: np.ndarray) -> np.ndarray:
        if pts.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if pts.shape[1] == 2:
            out = np.empty((pts.shape[0], 3), dtype=np.float32)
            out[:, 0:2] = pts.astype(np.float32, copy=False)
            out[:, 2] = 0.0
            return out
        return np.ascontiguousarray(pts[:, :3], dtype=np.float32)

    @staticmethod
    def _decimate(pts3: np.ndarray, colors: np.ndarray | None, budget: int):
        n = pts3.shape[0]
        if n <= budget:
            return pts3, colors
        step = int(np.ceil(n / budget))
        sl = slice(None, None, step)
        c = colors[sl] if colors is not None else None
        return pts3[sl], c

    def plot(
        self,
        pts: np.ndarray,
        *,
        colors: np.ndarray | None = None,
        scatter: bool = False,
        overlay_pts: np.ndarray | None = None,
    ) -> None:
        if not _GL_AVAILABLE or self._view is None:
            return
        if pts.size == 0:
            self.clear()
            return

        pts3 = self._to_f32_xyz(pts)
        budget = _GL_SCATTER_BUDGET if scatter else _GL_LINE_BUDGET
        draw_pts, draw_colors = self._decimate(pts3, colors, budget)

        if scatter:
            self._line.setVisible(False)
            self._scatter.setVisible(True)
            if draw_colors is not None and draw_colors.ndim == 1:
                rgba = pg.colormap.get("viridis").map(draw_colors, mode="byte")
            elif draw_colors is not None:
                rgba = draw_colors
            else:
                rgba = None
            self._scatter.setData(pos=draw_pts, color=rgba, size=3.0)
        else:
            self._line.setVisible(True)
            self._scatter.setVisible(draw_colors is not None)
            self._line.setData(pos=draw_pts, color=(0.38, 0.65, 0.98, 0.88))
            if draw_colors is not None and draw_colors.ndim == 1:
                rgba = pg.colormap.get("viridis").map(draw_colors, mode="byte")
                self._scatter.setData(pos=draw_pts, color=rgba, size=2.0)
            else:
                self._scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))

        if overlay_pts is not None and overlay_pts.size:
            ov = self._to_f32_xyz(overlay_pts)
            ov, _ = self._decimate(ov, None, _GL_SCATTER_BUDGET)
            red = np.tile(np.array([248, 113, 113, 120], dtype=np.ubyte), (ov.shape[0], 1))
            self._overlay_scatter.setVisible(True)
            self._overlay_scatter.setData(pos=ov, color=red, size=2.2)
        else:
            self._overlay_scatter.setVisible(False)
            self._overlay_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))

    def reset_view(self) -> None:
        if self._view is not None:
            self._view.setCameraPosition(distance=40, elevation=24, azimuth=-58)
