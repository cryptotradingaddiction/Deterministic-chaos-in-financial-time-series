#!/usr/bin/env python3
"""Background workers for attractor compute and heatmap generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from attractor_cache import get_embedding, get_pca_view, get_series, get_surrogate, series_key
from attractor_cao_viz import CaoFnnExemplars, build_cao_fnn_exemplars
from attractor_core import (
    build_tau_m_heatmap,
    compute_phase_space_stats,
    coord_label,
    local_expansion_proxy,
    max_feasible_tau,
    project_embedding,
    recurrence_density_2d,
)


@dataclass
class AttractorComputeResult:
    request_id: int
    symbol: str
    tau: int
    m: int
    window_note: str
    series: np.ndarray
    embedded: np.ndarray
    view_pts: np.ndarray
    axis_labels: list[str]
    projection: str
    view_dims: tuple[int, int, int]
    stats_text: str
    rosenstein_rate: float
    overlay_view: np.ndarray | None
    recurrence_full: np.ndarray | None
    feasible_text: str


class AttractorComputeTask(QRunnable):
    def __init__(self, request_id: int, params: dict):
        super().__init__()
        self.request_id = request_id
        self.params = params
        self.signals = _WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            p = self.params
            sk = series_key(
                p["symbol"], p["config"], p["start_frac"], p["end_frac"], p["test_cap"],
            )
            series = get_series(
                p["symbol"], p["config"],
                start_frac=p["start_frac"], end_frac=p["end_frac"], test_cap=p["test_cap"],
            )
            tau, m = int(p["tau"]), int(p["m"])
            embedded = get_embedding(series, tau, m, sk)
            view_dims = tuple(p["view_dims"])
            proj_mode = p["proj_mode"]

            if proj_mode == "pca":
                if embedded.shape[1] >= 2:
                    from sklearn.decomposition import PCA

                    n_comp = 3 if embedded.shape[1] >= 3 else 2
                    view_pts = get_pca_view(embedded, (sk, tau, m))
                    axis_labels = [f"PC{i + 1}" for i in range(n_comp)]
                    projection = "3d" if n_comp == 3 else "2d"
                else:
                    view_pts, axis_labels, projection = project_embedding(
                        embedded, "manual", view_dims, tau,
                    )
            else:
                view_pts, axis_labels, projection = project_embedding(
                    embedded, "manual", view_dims, tau,
                )

            overlay_view = None
            if p["want_surrogate"]:
                surr = get_surrogate(series, sk, seed=0)
                emb_s = get_embedding(surr, tau, m, (sk, "surr"))
                if proj_mode == "pca":
                    overlay_view = get_pca_view(emb_s, (sk, tau, m, "surr"))
                else:
                    overlay_view, _, _ = project_embedding(emb_s, "manual", view_dims, tau)

            recurrence_full = None
            if p["want_recurrence"] and projection == "2d":
                recurrence_full = recurrence_density_2d(view_pts)

            lam = float("nan")
            if p["want_rosenstein_rate"]:
                lam = float(local_expansion_proxy(embedded, tau)["rosenstein_rate"])

            stats = compute_phase_space_stats(
                p["symbol"], series, embedded, tau, m,
                display_n=view_pts.shape[0], subsampled=False,
                view_dims=view_dims, view_pts=view_pts,
                window_note=p["window_note"], coord_labels=axis_labels,
            )
            stats.rosenstein_rate = lam

            n = len(series)
            en = embedded.shape[0]
            feasible = (
                f"N={n} | max τ={max_feasible_tau(n, m)} | embedded≈{en}"
                + ("" if en > 0 else " | INVALID")
            )

            self.signals.finished.emit(
                AttractorComputeResult(
                    request_id=self.request_id,
                    symbol=p["symbol"],
                    tau=tau,
                    m=m,
                    window_note=p["window_note"],
                    series=series,
                    embedded=embedded,
                    view_pts=view_pts,
                    axis_labels=axis_labels,
                    projection=projection,
                    view_dims=view_dims,
                    stats_text=stats.format_text(),
                    rosenstein_rate=lam,
                    overlay_view=overlay_view,
                    recurrence_full=recurrence_full,
                    feasible_text=feasible,
                )
            )
        except Exception as exc:
            self.signals.failed.emit(self.request_id, str(exc))


@dataclass
class CaoFnnVizResult:
    request_id: int
    symbol: str
    exemplars: CaoFnnExemplars
    stats_text: str
    feasible_text: str


class CaoFnnVizTask(QRunnable):
    def __init__(self, request_id: int, params: dict):
        super().__init__()
        self.request_id = request_id
        self.params = params
        self.signals = _WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            p = self.params
            series = get_series(
                p["symbol"], p["config"],
                start_frac=p["start_frac"], end_frac=p["end_frac"], test_cap=p["test_cap"],
            )
            m, tau = int(p["m"]), int(p["tau"])
            exemplars = build_cao_fnn_exemplars(
                series, m, tau,
                proj_a=tuple(p["proj_a"]),
                proj_b=tuple(p["proj_b"]),
            )
            n = len(series)
            en = exemplars.embedded_m.shape[0]
            feasible = (
                f"N={n} | Cao embedded≈{en} (N-m*tau)"
                + ("" if en > 0 else " | INVALID")
            )
            self.signals.finished.emit(
                CaoFnnVizResult(
                    request_id=self.request_id,
                    symbol=p["symbol"],
                    exemplars=exemplars,
                    stats_text=exemplars.format_stats(p["symbol"]),
                    feasible_text=feasible,
                )
            )
        except Exception as exc:
            self.signals.failed.emit(self.request_id, str(exc))


@dataclass
class HeatmapResult:
    request_id: int
    grid: np.ndarray
    m_values: np.ndarray
    tau_values: np.ndarray
    metric: str


class HeatmapComputeTask(QRunnable):
    def __init__(self, request_id: int, params: dict):
        super().__init__()
        self.request_id = request_id
        self.params = params
        self.signals = _WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            p = self.params
            series = get_series(
                p["symbol"], p["config"],
                start_frac=p["start_frac"], end_frac=p["end_frac"], test_cap=p["test_cap"],
            )
            n = len(series)
            grid = build_tau_m_heatmap(
                n, p["m_values"], p["tau_values"],
                metric=p["metric"], series=series,
            )
            self.signals.finished.emit(
                HeatmapResult(
                    request_id=self.request_id,
                    grid=grid,
                    m_values=p["m_values"],
                    tau_values=p["tau_values"],
                    metric=p["metric"],
                )
            )
        except Exception as exc:
            self.signals.failed.emit(self.request_id, str(exc))


class _WorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(int, str)


_pool = QThreadPool.globalInstance()
_active_tasks: set[QRunnable] = set()


def _retain_task(task: QRunnable) -> None:
    _active_tasks.add(task)


def _release_task(task: QRunnable) -> None:
    _active_tasks.discard(task)


def run_attractor_compute(request_id: int, params: dict, on_ok, on_fail) -> None:
    task = AttractorComputeTask(request_id, params)
    _retain_task(task)

    def _done(result: AttractorComputeResult) -> None:
        _release_task(task)
        on_ok(result)

    def _fail(rid: int, message: str) -> None:
        _release_task(task)
        on_fail(rid, message)

    task.signals.finished.connect(_done)
    task.signals.failed.connect(_fail)
    _pool.start(task)


def run_cao_fnn_viz(request_id: int, params: dict, on_ok, on_fail) -> None:
    task = CaoFnnVizTask(request_id, params)
    _retain_task(task)

    def _done(result: CaoFnnVizResult) -> None:
        _release_task(task)
        on_ok(result)

    def _fail(rid: int, message: str) -> None:
        _release_task(task)
        on_fail(rid, message)

    task.signals.finished.connect(_done)
    task.signals.failed.connect(_fail)
    _pool.start(task)


def run_heatmap_compute(request_id: int, params: dict, on_ok, on_fail) -> None:
    task = HeatmapComputeTask(request_id, params)
    _retain_task(task)

    def _done(result: HeatmapResult) -> None:
        _release_task(task)
        on_ok(result)

    def _fail(rid: int, message: str) -> None:
        _release_task(task)
        on_fail(rid, message)

    task.signals.finished.connect(_done)
    task.signals.failed.connect(_fail)
    _pool.start(task)


def set_worker_thread_count(n: int) -> None:
    _pool.setMaxThreadCount(max(2, n))
