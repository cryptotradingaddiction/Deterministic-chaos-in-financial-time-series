#!/usr/bin/env python3
"""Background worker for recurrence plot computation."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from attractor_cache import get_series, series_key
from recurrence_core import compute_recurrence_bundle, format_recurrence_stats, resolve_radius


@dataclass
class RecurrenceComputeResult:
    request_id: int
    symbol: str
    tau: int
    m: int
    theiler_w: int
    bundle: dict
    stats_text: str


class RecurrenceComputeTask(QRunnable):
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
                start_frac=p.get("start_frac", 0.0),
                end_frac=p.get("end_frac", 1.0),
                test_cap=p.get("test_cap"),
            )
            tau, m = int(p["tau"]), int(p["m"])
            theiler_w = int(p["theiler_w"])
            radius = resolve_radius(
                series,
                delay=tau,
                m=m,
                mode=p["radius_mode"],
                manual_radius=float(p["manual_radius"]),
                percentile=float(p["percentile"]),
            )
            bundle = compute_recurrence_bundle(
                series,
                tau=tau,
                m=m,
                radius=radius,
                theiler_w=theiler_w,
                max_display=int(p.get("max_display", 2500)),
            )
            stats = format_recurrence_stats(
                p["symbol"],
                tau=tau,
                m=m,
                radius=radius,
                theiler_w=theiler_w,
                n_series=len(series),
                n_embedded=bundle["n_embedded"],
                n_display=bundle["n_display"],
                metrics=bundle["metrics"],
                theiler_corrector=bundle["theiler_corrector"],
            )
            self.signals.finished.emit(
                RecurrenceComputeResult(
                    request_id=self.request_id,
                    symbol=p["symbol"],
                    tau=tau,
                    m=m,
                    theiler_w=theiler_w,
                    bundle=bundle,
                    stats_text=stats,
                )
            )
        except Exception as exc:
            self.signals.failed.emit(self.request_id, str(exc))


class _WorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(int, str)


_pool = QThreadPool.globalInstance()
_active_tasks: set[QRunnable] = set()


def run_recurrence_compute(request_id: int, params: dict, on_ok, on_fail) -> None:
    task = RecurrenceComputeTask(request_id, params)
    _active_tasks.add(task)

    def _done(result: RecurrenceComputeResult) -> None:
        _active_tasks.discard(task)
        on_ok(result)

    def _fail(rid: int, message: str) -> None:
        _active_tasks.discard(task)
        on_fail(rid, message)

    task.signals.finished.connect(_done)
    task.signals.failed.connect(_fail)
    _pool.start(task)


def set_worker_thread_count(n: int) -> None:
    _pool.setMaxThreadCount(max(2, n))
