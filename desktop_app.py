#!/usr/bin/env python3
import os
import sys
from html import escape
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config_loader import load_config, get_data_dir, get_results_dir


ROOT = Path(__file__).resolve().parent
PY = "py"
PY_ARGS = ["-3"]


@dataclass
class CommandSpec:
    name: str
    program: str
    args: list[str]
    cwd: Path


class PipelineApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DCh Pipeline Desktop")
        self.resize(1400, 900)

        self.config = load_config()
        self.data_dir = Path(get_data_dir(self.config))
        self.results_dir = Path(get_results_dir(self.config))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.current_processes: dict[str, QProcess] = {}
        self.pipeline_queue: list[object] = []
        self.pipeline_running = False
        self.stop_requested = False
        self._all_artifacts: list[Path] = []
        self._preview_pixmap: QPixmap | None = None
        self._pipeline_total_steps = 0
        self._pipeline_completed_steps = 0
        self._current_step_name = ""

        self._build_ui()
        self._apply_theme()
        self._load_artifacts()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        splitter.addWidget(left)

        self.steps = QListWidget()
        for s in self._step_names():
            QListWidgetItem(s, self.steps)
        self.steps.setCurrentRow(0)
        left_layout.addWidget(QLabel("Pipeline Steps"))
        left_layout.addWidget(self.steps)

        btn_row = QHBoxLayout()
        self.btn_run_full = QPushButton("Run Full Pipeline")
        self.btn_run_selected = QPushButton("Run Selected Step")
        self.btn_stop = QPushButton("Stop")
        btn_row.addWidget(self.btn_run_full)
        btn_row.addWidget(self.btn_run_selected)
        btn_row.addWidget(self.btn_stop)
        left_layout.addLayout(btn_row)

        self.btn_refresh = QPushButton("Refresh Artifacts")
        left_layout.addWidget(self.btn_refresh)
        self.chk_test_mode = QCheckBox("Global TEST_MODE (2000 rows)")
        self.chk_test_mode.setChecked(False)
        left_layout.addWidget(self.chk_test_mode)

        self.progress_label = QLabel("Progress: idle")
        left_layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        splitter.addWidget(right)

        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs)

        logs_tab = QWidget()
        logs_layout = QVBoxLayout(logs_tab)
        logs_layout.addWidget(QLabel("Live Logs"))
        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        logs_layout.addWidget(self.logs)
        self.tabs.addTab(logs_tab, "Logs")

        artifacts_tab = QWidget()
        artifacts_layout = QVBoxLayout(artifacts_tab)
        artifacts_layout.addWidget(QLabel("Artifacts (PNG/TXT)"))
        filter_row = QHBoxLayout()
        self.artifact_search = QLineEdit()
        self.artifact_search.setPlaceholderText("Search artifacts...")
        self.artifact_type = QComboBox()
        self.artifact_type.addItems(["All", "Images", "Text"])
        filter_row.addWidget(self.artifact_search)
        filter_row.addWidget(self.artifact_type)
        artifacts_layout.addLayout(filter_row)

        self.artifact_stats = QLabel("Artifacts: 0 shown")
        artifacts_layout.addWidget(self.artifact_stats)

        self.artifacts = QListWidget()
        artifacts_layout.addWidget(self.artifacts)

        self.btn_open_file = QPushButton("Open Selected File")
        artifacts_layout.addWidget(self.btn_open_file)
        self.tabs.addTab(artifacts_tab, "Artifacts")

        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        self.preview_label = QLabel("Image preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(260)
        self.preview_label.setStyleSheet("border: 1px solid #555;")
        preview_layout.addWidget(self.preview_label)

        self.text_preview = QTextEdit()
        self.text_preview.setReadOnly(True)
        preview_layout.addWidget(self.text_preview)
        self.tabs.addTab(preview_tab, "Preview")

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 8)

        self.btn_run_full.clicked.connect(self.run_full_pipeline)
        self.btn_run_selected.clicked.connect(self.run_selected_step)
        self.btn_stop.clicked.connect(self.stop_all)
        self.btn_refresh.clicked.connect(self._load_artifacts)
        self.artifacts.itemSelectionChanged.connect(self._preview_selected_artifact)
        self.btn_open_file.clicked.connect(self._open_selected_file)
        self.artifact_search.textChanged.connect(self._apply_artifact_filters)
        self.artifact_type.currentIndexChanged.connect(self._apply_artifact_filters)

    def _step_names(self):
        return [
            "1. crypto_data_all.py",
            "2. compute_logreturns.py",
            "3. mutual.py",
            "4. tau_w.py",
            "5. phase_2D.py",
            "6. phase_3D.py",
            "7. cao_.py",
            "8. 2dc.py",
            "9. hypothesis.bat (distributed wrapper: Takens/Ellner/LLE TS tests + RQA summaries)",
        ]

    def _python_cmd(self, script_name: str):
        return CommandSpec(
            name=script_name,
            program=PY,
            args=PY_ARGS + [str(ROOT / script_name)],
            cwd=ROOT,
        )

    def _bat_cmd(self, bat_name: str):
        return CommandSpec(
            name=bat_name,
            program="cmd",
            args=["/c", bat_name],
            cwd=ROOT / "Tisean_3.0.0" / "bin",
        )

    def _full_pipeline(self):
        return [
            self._python_cmd("crypto_data_all.py"),
            self._python_cmd("compute_logreturns.py"),
            self._python_cmd("mutual.py"),
            self._python_cmd("tau_w.py"),
            self._python_cmd("phase_2D.py"),
            self._python_cmd("phase_3D.py"),
            self._python_cmd("cao_.py"),
            self._python_cmd("2dc.py"),
            self._bat_cmd("hypothesis.bat"),
        ]

    def _append_log(self, msg: str):
        text = msg.rstrip()
        if not text:
            return
        safe = escape(text).replace("\n", "<br>")
        self.logs.append(f"<span style='color:#cfd6df'>{safe}</span>")
        self.logs.append("")
        self.logs.verticalScrollBar().setValue(self.logs.verticalScrollBar().maximum())

    def _append_status(self, badge: str, msg: str):
        colors = {
            "RUN": "#2b6cb0",
            "OK": "#2f855a",
            "FAIL": "#c53030",
            "INFO": "#6b7280",
        }
        color = colors.get(badge, "#6b7280")
        safe = escape(msg.rstrip()).replace("\n", "<br>")
        html = (
            f"<span style='display:inline-block;background:{color};color:white;"
            f"padding:2px 6px;border-radius:8px;font-weight:600;'>"
            f"{escape(badge)}</span> "
            f"<span style='color:#e2e8f0'>{safe}</span>"
        )
        self.logs.append(html)
        self.logs.append("")
        self.logs.verticalScrollBar().setValue(self.logs.verticalScrollBar().maximum())

    def _apply_theme(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #16181d; color: #e6e6e6; }
            QPushButton { background: #2b6cb0; border: none; padding: 6px 10px; border-radius: 6px; color: white; }
            QPushButton:hover { background: #3182ce; }
            QListWidget, QTextEdit, QLineEdit, QComboBox { background: #0f1115; border: 1px solid #30343b; border-radius: 6px; }
            QLabel { color: #d8d8d8; }
            """
        )

    def _run_command(self, spec: CommandSpec, on_done):
        proc = QProcess(self)
        proc.setProgram(spec.program)
        proc.setArguments(spec.args)
        proc.setWorkingDirectory(str(spec.cwd))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("DCH_TEST_MODE", "true" if self.chk_test_mode.isChecked() else "false")
        proc.setProcessEnvironment(env)

        self.current_processes[spec.name] = proc
        self._current_step_name = spec.name
        self._append_status("RUN", f"{spec.name} | cwd={spec.cwd} | test_mode={'true' if self.chk_test_mode.isChecked() else 'false'}")
        self._append_log(f"[CMD] {spec.program} {' '.join(spec.args)}")
        self._refresh_progress_ui()

        def ready_out():
            out = bytes(proc.readAllStandardOutput()).decode(errors="replace")
            if out:
                self._append_log(f"[{spec.name}] {out}")

        def ready_err():
            err = bytes(proc.readAllStandardError()).decode(errors="replace")
            if err:
                self._append_log(f"[{spec.name}][ERR] {err}")

        def finished(exit_code, exit_status):
            self.current_processes.pop(spec.name, None)
            ok = (exit_code == 0 and exit_status == QProcess.NormalExit)
            if ok:
                self._append_status("OK", f"{spec.name} finished | exit_code={exit_code}")
            else:
                self._append_status("FAIL", f"{spec.name} failed | exit_code={exit_code}")
            on_done(ok)

        proc.readyReadStandardOutput.connect(ready_out)
        proc.readyReadStandardError.connect(ready_err)
        proc.finished.connect(finished)
        proc.start()

    def _run_next_pipeline_item(self):
        if self.stop_requested:
            self.pipeline_running = False
            self._append_status("INFO", "Pipeline stopped by user.")
            self._refresh_progress_ui()
            return
        if not self.pipeline_queue:
            self.pipeline_running = False
            self._append_status("OK", "Pipeline finished.")
            self._load_artifacts()
            self._refresh_progress_ui()
            return

        item = self.pipeline_queue.pop(0)
        if isinstance(item, list):
            pending = {"count": len(item), "ok": True}
            self._append_log(f"[GROUP] Starting concurrent group ({len(item)} commands)")

            def done_one(ok):
                pending["count"] -= 1
                pending["ok"] = pending["ok"] and ok
                if pending["count"] == 0:
                    if not pending["ok"]:
                        self.pipeline_running = False
                        self._append_status("FAIL", "Pipeline stopped due to failure in concurrent group.")
                        self._refresh_progress_ui()
                        return
                    self._pipeline_completed_steps += 1
                    self._append_status("OK", "Concurrent group completed.")
                    self._refresh_progress_ui()
                    QTimer.singleShot(10, self._run_next_pipeline_item)

            for spec in item:
                self._run_command(spec, done_one)
        else:
            def done_single(ok):
                if not ok:
                    self.pipeline_running = False
                    self._append_status("FAIL", "Pipeline stopped due to step failure.")
                    self._refresh_progress_ui()
                    return
                self._pipeline_completed_steps += 1
                self._refresh_progress_ui()
                QTimer.singleShot(10, self._run_next_pipeline_item)

            self._run_command(item, done_single)

    def run_full_pipeline(self):
        if self.pipeline_running:
            QMessageBox.warning(self, "Pipeline running", "Pipeline is already running.")
            return
        self.stop_requested = False
        self.pipeline_running = True
        self.pipeline_queue = self._full_pipeline()
        self._pipeline_total_steps = len(self.pipeline_queue)
        self._pipeline_completed_steps = 0
        self._current_step_name = ""
        self._append_status("INFO", "Starting full pipeline...")
        self._refresh_progress_ui()
        self._run_next_pipeline_item()

    def run_selected_step(self):
        if self.pipeline_running:
            QMessageBox.warning(self, "Pipeline running", "Stop current run first.")
            return
        idx = self.steps.currentRow()
        if idx < 0:
            return
        mapping = self._full_pipeline()
        selected = mapping[idx]
        self.stop_requested = False
        self.pipeline_running = True
        self.pipeline_queue = [selected]
        self._pipeline_total_steps = 1
        self._pipeline_completed_steps = 0
        self._current_step_name = ""
        self._append_status("INFO", f"Starting selected step: {self.steps.currentItem().text()}")
        self._refresh_progress_ui()
        self._run_next_pipeline_item()

    def stop_all(self):
        self.stop_requested = True
        for name, proc in list(self.current_processes.items()):
            self._append_status("INFO", f"Terminating {name}")
            proc.kill()
        self.current_processes.clear()
        self.pipeline_running = False
        self._refresh_progress_ui()

    def _refresh_progress_ui(self):
        total = self._pipeline_total_steps
        done = self._pipeline_completed_steps
        if total <= 0:
            self.progress_label.setText("Progress: idle")
            self.progress_bar.setValue(0)
            return
        pct = int((done / total) * 100)
        step_text = f"Step {min(done + 1, total)}/{total}"
        if not self.pipeline_running and done >= total:
            self.progress_label.setText(f"Progress: Completed {total}/{total}")
            self.progress_bar.setValue(100)
            return
        current = f" | Running: {self._current_step_name}" if self._current_step_name else ""
        self.progress_label.setText(f"Progress: {step_text} ({done}/{total} done){current}")
        self.progress_bar.setValue(pct)

    def _artifact_roots(self):
        roots = [
            ROOT,
            self.data_dir,
            self.results_dir,
            self.data_dir / "results",
            self.data_dir / "results_test_2000",
            self.data_dir / "results_full",
            self.data_dir / "results_info_dim_test_2000",
            self.data_dir / "results_info_dim_full",
        ]
        return [p for p in roots if p.exists()]

    def _load_artifacts(self):
        exts = {".png", ".jpg", ".jpeg", ".txt"}
        items = []
        for base in self._artifact_roots():
            for path in base.rglob("*"):
                if path.is_file() and path.suffix.lower() in exts:
                    items.append(path)
        self._all_artifacts = sorted(set(items), key=lambda p: p.stat().st_mtime, reverse=True)[:800]
        self._apply_artifact_filters()
        self._append_log(f"[ARTIFACTS] Loaded {len(self._all_artifacts)} files.")

    def _apply_artifact_filters(self):
        search = (self.artifact_search.text() or "").strip().lower()
        mode = self.artifact_type.currentText()
        self.artifacts.clear()

        shown = 0
        for p in self._all_artifacts:
            suffix = p.suffix.lower()
            is_image = suffix in {".png", ".jpg", ".jpeg"}
            is_text = suffix == ".txt"

            if mode == "Images" and not is_image:
                continue
            if mode == "Text" and not is_text:
                continue

            haystack = str(p).lower()
            if search and search not in haystack:
                continue

            try:
                rel = p.relative_to(ROOT)
            except ValueError:
                rel = p
            display = f"{p.name}  |  {rel}"
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, str(p))
            self.artifacts.addItem(item)
            shown += 1

        self.artifact_stats.setText(f"Artifacts: {shown} shown / {len(self._all_artifacts)} total")

    def _preview_selected_artifact(self):
        item = self.artifacts.currentItem()
        if not item:
            return
        raw_path = item.data(Qt.UserRole)
        if not raw_path:
            return
        path = Path(raw_path)
        if not path.exists():
            return
        self.text_preview.clear()
        self.preview_label.clear()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            pix = QPixmap(str(path))
            if not pix.isNull():
                self._preview_pixmap = pix
                self._refresh_preview_pixmap()
            self.text_preview.setPlainText(str(path))
        else:
            self._preview_pixmap = None
            try:
                txt = path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                txt = f"Unable to read file: {e}"
            self.preview_label.setText("Text file selected")
            self.text_preview.setPlainText(txt[:200000])

    def _open_selected_file(self):
        item = self.artifacts.currentItem()
        if not item:
            return
        raw_path = item.data(Qt.UserRole)
        if not raw_path:
            return
        path = str(raw_path)
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                QMessageBox.information(self, "Open file", f"Selected file:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Open failed", f"Could not open file:\n{e}")

    def _refresh_preview_pixmap(self):
        if self._preview_pixmap is None:
            return
        self.preview_label.setPixmap(
            self._preview_pixmap.scaled(
                self.preview_label.width(),
                self.preview_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_preview_pixmap()


def main():
    app = QApplication(sys.argv)
    win = PipelineApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
