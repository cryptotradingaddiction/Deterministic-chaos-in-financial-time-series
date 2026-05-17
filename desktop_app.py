#!/usr/bin/env python3
"""
PySide6 desktop runner for the DCh cryptocurrency chaos-analysis pipeline.

Responsibilities:

- Run the **11-step** chain (download → log-returns → liquidity → diagnostics →
  ``hypothesis.bat``) or a single selected step.
- Stream **stdout/stderr** from child processes into a coloured HTML log.
- Forward **environment variables** consumed by ``.bat`` files and
  ``hypothesis.py`` (test mode, bootstrap count, dimension metrics, etc.).
- Browse **artifacts** (PNG/TXT under ``data/`` and ``results/``) and preview
  images with fit / 100% / custom zoom.

Child processes are started via :class:`QProcess` (not a shell), with working
directories set per step: Python scripts run from the repo root; TISEAN
``.bat`` files run from ``Tisean_3.0.0/bin``.
"""

import os
import sys
from datetime import datetime
from html import escape
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QEvent, QProcess, QProcessEnvironment, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Shared paths and test-mode row count (must match _dch_test_env.bat).
from config_loader import (
    load_config,
    dch_test_point_count,
    dch_test_results_tag,
    get_data_dir,
    get_results_dir,
)
from hypothesis_config import DEFAULT_BOOTSTRAP_SAMPLES


# Repo root (``C:\DCh``) — cwd for all ``py -3 <script>.py`` invocations.
ROOT = Path(__file__).resolve().parent
# Windows Python launcher; use ``py -3`` so the same interpreter as CLI docs.
PY = "py"
PY_ARGS = ["-3"]

# ---------------------------------------------------------------------------
# Pipeline step status (sidebar list icons and colours)
# ---------------------------------------------------------------------------

STEP_PENDING = "pending"
STEP_RUNNING = "running"
STEP_OK = "ok"
STEP_FAIL = "fail"

# Unicode prefixes shown before each step name in the sidebar.
STEP_ICON = {
    STEP_PENDING: "○",
    STEP_RUNNING: "●",
    STEP_OK: "✓",
    STEP_FAIL: "✗",
}

# Foreground colours for sidebar step list items.
STEP_COLOR = {
    STEP_PENDING: QColor("#94a3b8"),
    STEP_RUNNING: QColor("#60a5fa"),
    STEP_OK: QColor("#4ade80"),
    STEP_FAIL: QColor("#f87171"),
}

# Filter presets for the Artifacts tab (substring match on full path, lowercased).
ARTIFACT_PRESETS: dict[str, list[str] | None] = {
    "All": None,
    "STP / ACF (theiler)": ["stp", "acf", "theiler"],
    "Lyapunov (lyap_k)": ["lyap", "lambda_max"],
    "RQA / recurrence": ["rqa", "recurr", "recurrence"],
    "Dimension (d2 / takens)": ["correlation_dimension", "takens", "ellner", ".c2", ".d2"],
    "Hypothesis summaries": ["surrogate_summary", "hypothesis"],
}

# After these steps succeed, auto-switch to Artifacts and select the newest PNG.
PLOT_STEP_KEYWORDS = (
    "theilers_w",
    "hypothesis",
    "lambda_max",
    "correlation_dimension",
    "rqa",
    "phase_",
    "cao_",
    "2dc",
)


@dataclass
class CommandSpec:
    """One subprocess to launch (Python script or ``cmd /c`` batch)."""

    name: str  # Display name and key in ``current_processes`` (e.g. ``mutual.py``)
    program: str  # Executable: ``py`` or ``cmd``
    args: list[str]  # argv tail (e.g. ``["-3", "C:\\DCh\\mutual.py"]`` or ``["/c", "hypothesis.bat"]``)
    cwd: Path  # Working directory for the child process


@dataclass
class StepDef:
    """Sidebar row: short label shown in the list + hover tooltip."""

    short: str
    tooltip: str


class PipelineApp(QMainWindow):
    """
    Main window: sidebar controls, tabbed logs / artifacts / preview, pipeline runner.

    Pipeline state:

    - ``pipeline_queue`` — FIFO of :class:`CommandSpec` or ``list[CommandSpec]`` for
      concurrent groups (reserved; current queue is sequential only).
    - ``current_processes`` — active ``QProcess`` instances keyed by ``spec.name``.
    - ``_step_status`` — parallel array to ``_step_defs`` with STEP_* constants.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DCh Pipeline Desktop")
        self.resize(1500, 960)

        # Paths from config.yaml (same as config_loader helpers used by CLI scripts).
        self.config = load_config()
        self.data_dir = Path(get_data_dir(self.config))
        self.results_dir = Path(get_results_dir(self.config))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # --- Subprocess / pipeline runner state ---
        self.current_processes: dict[str, QProcess] = {}
        self.pipeline_queue: list[object] = []  # CommandSpec | list[CommandSpec]
        self.pipeline_running = False
        self.stop_requested = False  # User pressed Stop; drain queue without starting new steps
        self._pipeline_total_steps = 0
        self._pipeline_completed_steps = 0
        self._current_step_name = ""  # Shown in progress label while a step runs
        self._pipeline_failed = False  # Any step returned non-zero → progress bar red

        # --- Sidebar step list (filled in _build_ui from _pipeline_step_defs) ---
        self._step_defs: list[StepDef] = []
        self._step_status: list[str] = []

        # --- Artifacts + preview ---
        self._all_artifacts: list[Path] = []  # Newest-first cache (cap 1200 files)
        self._preview_pixmap: QPixmap | None = None  # Full-resolution image for zoom
        self._preview_path: Path | None = None
        self._preview_zoom_mode = "fit"  # "fit" | "100" | "custom"
        self._preview_zoom_factor = 1.0  # Used when mode == "custom"
        self._artifacts_tab: QWidget | None = None  # For setCurrentIndex after plot steps

        self._build_ui()
        self._reset_step_statuses()
        self._apply_theme()
        self._apply_monospace_fonts()
        self._load_artifacts()

    @staticmethod
    def _mono_font() -> QFont:
        """Prefer Cascadia Mono / Consolas for log and text preview panes."""
        for family in ("Cascadia Mono", "Consolas", "Courier New"):
            font = QFont(family, 10)
            if font.exactMatch() or family == "Courier New":
                font.setStyleHint(QFont.StyleHint.Monospace)
                return font
        font = QFont()
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(10)
        return font

    def _apply_monospace_fonts(self):
        """Apply :meth:`_mono_font` to log console and text preview."""
        mono = self._mono_font()
        self.logs.setFont(mono)
        self.text_preview.setFont(mono)

    def _section_label(self, text: str) -> QLabel:
        """Muted subsection heading (styled via ``#sectionLabel`` in QSS)."""
        label = QLabel(text)
        label.setObjectName("sectionLabel")
        return label

    def _build_ui(self):
        """
        Construct the full window: left sidebar + right tabs.

        Layout: ``QSplitter`` — sidebar (steps, settings, buttons, test mode) |
        progress card + ``QTabWidget`` (Logs, Artifacts, Preview).
        """
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(8)
        layout.addWidget(splitter)

        # ----- Left sidebar -----
        sidebar_card = QFrame()
        sidebar_card.setObjectName("sidebarCard")
        left_layout = QVBoxLayout(sidebar_card)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(10)
        splitter.addWidget(sidebar_card)

        title = QLabel("DCh Pipeline")
        title.setObjectName("appTitle")
        left_layout.addWidget(title)

        # Ordered list matching _full_pipeline() indices for status icons.
        self._step_defs = self._pipeline_step_defs()
        self.steps = QListWidget()
        self.steps.setObjectName("pipelineSteps")
        for defn in self._step_defs:
            item = QListWidgetItem(defn.short)
            item.setToolTip(defn.tooltip)
            self.steps.addItem(item)
        self.steps.setCurrentRow(0)
        left_layout.addWidget(self._section_label("Pipeline steps"))
        left_layout.addWidget(self.steps, stretch=1)

        # Env vars forwarded in _run_command → DCH_RUN_HYPOTHESIS, DCH_DIMENSION_METRICS,
        # DCH_BOOTSTRAP_SAMPLES (read by correlation_dimension.bat / hypothesis.py).
        settings_frame = QFrame()
        settings_frame.setObjectName("settingsFrame")
        settings_layout = QVBoxLayout(settings_frame)
        settings_layout.setContentsMargins(10, 8, 10, 8)
        settings_layout.setSpacing(6)
        settings_layout.addWidget(self._section_label("Hypothesis / TISEAN"))
        self.chk_run_hypothesis = QCheckBox("Run hypothesis in .bat steps")
        self.chk_run_hypothesis.setChecked(True)
        settings_layout.addWidget(self.chk_run_hypothesis)
        dim_row = QHBoxLayout()
        dim_row.addWidget(QLabel("D2 metrics:"))
        self.cmb_dimension_metrics = QComboBox()
        self.cmb_dimension_metrics.addItems(["ELLNER", "TAKENS", "TAKENS,ELLNER"])
        dim_row.addWidget(self.cmb_dimension_metrics, stretch=1)
        settings_layout.addLayout(dim_row)
        boot_row = QHBoxLayout()
        boot_row.addWidget(QLabel("Bootstrap B:"))
        self.spin_bootstrap = QSpinBox()
        self.spin_bootstrap.setRange(1, 500)
        self.spin_bootstrap.setValue(DEFAULT_BOOTSTRAP_SAMPLES)
        boot_row.addWidget(self.spin_bootstrap)
        settings_layout.addLayout(boot_row)
        left_layout.addWidget(settings_frame)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.btn_run_selected = QPushButton("Run selected")
        self.btn_run_selected.setObjectName("btnPrimary")
        self.btn_run_full = QPushButton("Run full")
        self.btn_run_full.setObjectName("btnSecondary")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("btnDanger")
        btn_row.addWidget(self.btn_run_selected, stretch=2)
        btn_row.addWidget(self.btn_run_full, stretch=2)
        btn_row.addWidget(self.btn_stop, stretch=1)
        left_layout.addLayout(btn_row)

        self.btn_refresh = QPushButton("Refresh artifacts")
        self.btn_refresh.setObjectName("btnSecondary")
        left_layout.addWidget(self.btn_refresh)

        test_frame = QFrame()
        test_frame.setObjectName("testModeFrame")
        test_layout = QVBoxLayout(test_frame)
        test_layout.setContentsMargins(10, 8, 10, 8)
        # Sets DCH_TEST_MODE=true and DCH_TEST_POINTS (default 100 via config_loader).
        n_test = dch_test_point_count()
        self.chk_test_mode = QCheckBox(
            f"TEST_MODE — {n_test} rows (DCH_TEST_POINTS)"
        )
        self.chk_test_mode.setChecked(False)
        test_layout.addWidget(self.chk_test_mode)
        left_layout.addWidget(test_frame)

        # ----- Right pane: progress + tabs -----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        splitter.addWidget(right)

        progress_card = QFrame()
        progress_card.setObjectName("progressCard")
        progress_layout = QVBoxLayout(progress_card)
        progress_layout.setContentsMargins(14, 12, 14, 12)
        progress_layout.setSpacing(6)
        self.progress_label = QLabel("Idle — no pipeline running")
        self.progress_label.setObjectName("progressLabel")
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("pipelineProgress")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(12)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        right_layout.addWidget(progress_card)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainTabs")
        right_layout.addWidget(self.tabs, stretch=1)

        # Tab 1: HTML log (stdout/stderr + status badges).
        logs_tab = QWidget()
        logs_layout = QVBoxLayout(logs_tab)
        logs_layout.setContentsMargins(8, 8, 8, 8)
        logs_layout.setSpacing(8)
        log_toolbar = QHBoxLayout()
        log_toolbar.addWidget(self._section_label("Live logs"))
        log_toolbar.addStretch(1)
        self.chk_follow_log = QCheckBox("Follow tail")
        self.chk_follow_log.setChecked(True)
        self.btn_clear_log = QPushButton("Clear")
        self.btn_clear_log.setObjectName("btnGhost")
        self.btn_clear_log.setFixedWidth(72)
        log_toolbar.addWidget(self.chk_follow_log)
        log_toolbar.addWidget(self.btn_clear_log)
        logs_layout.addLayout(log_toolbar)
        self.logs = QTextEdit()
        self.logs.setObjectName("logConsole")
        self.logs.setReadOnly(True)
        logs_layout.addWidget(self.logs)
        self.tabs.addTab(logs_tab, "Logs")

        # Tab 2: tree of PNG/TXT under data/ and results/ (see _artifact_roots).
        self._artifacts_tab = QWidget()
        artifacts_layout = QVBoxLayout(self._artifacts_tab)
        artifacts_layout.setContentsMargins(8, 8, 8, 8)
        artifacts_layout.setSpacing(8)
        artifacts_layout.addWidget(self._section_label("Artifacts (PNG / TXT)"))
        filter_row = QHBoxLayout()
        self.artifact_search = QLineEdit()
        self.artifact_search.setPlaceholderText("Search path or filename...")
        self.artifact_preset = QComboBox()
        for label in ARTIFACT_PRESETS:
            self.artifact_preset.addItem(label)
        self.artifact_type = QComboBox()
        self.artifact_type.addItems(["All", "Images", "Text"])
        filter_row.addWidget(self.artifact_search, stretch=2)
        filter_row.addWidget(self.artifact_preset, stretch=1)
        filter_row.addWidget(self.artifact_type)
        artifacts_layout.addLayout(filter_row)

        self.artifact_stats = QLabel("Artifacts: 0 shown")
        artifacts_layout.addWidget(self.artifact_stats)

        self.artifact_tree = QTreeWidget()
        self.artifact_tree.setObjectName("artifactTree")
        self.artifact_tree.setHeaderLabels(["Name", "Folder / path"])
        self.artifact_tree.setColumnWidth(0, 220)
        self.artifact_tree.setAlternatingRowColors(True)
        self.artifact_tree.setRootIsDecorated(True)
        artifacts_layout.addWidget(self.artifact_tree)

        self.btn_open_file = QPushButton("Open selected file")
        self.btn_open_file.setObjectName("btnSecondary")
        artifacts_layout.addWidget(self.btn_open_file)
        self.tabs.addTab(self._artifacts_tab, "Artifacts")

        # Tab 3: image (scroll + zoom) or plain-text preview for .txt summaries.
        self.preview_tab = QWidget()
        preview_layout = QVBoxLayout(self.preview_tab)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.setSpacing(8)

        preview_toolbar = QHBoxLayout()
        preview_toolbar.addWidget(self._section_label("Preview"))
        preview_toolbar.addStretch(1)
        self.btn_preview_open = QPushButton("Open file")
        self.btn_preview_open.setObjectName("btnGhost")
        self.btn_preview_copy_path = QPushButton("Copy path")
        self.btn_preview_copy_path.setObjectName("btnGhost")
        preview_toolbar.addWidget(self.btn_preview_open)
        preview_toolbar.addWidget(self.btn_preview_copy_path)
        self.btn_preview_fit = QPushButton("Fit")
        self.btn_preview_fit.setObjectName("btnGhost")
        self.btn_preview_100 = QPushButton("100%")
        self.btn_preview_100.setObjectName("btnGhost")
        self.btn_preview_zoom_in = QPushButton("+")
        self.btn_preview_zoom_in.setObjectName("btnGhost")
        self.btn_preview_zoom_in.setFixedWidth(36)
        self.btn_preview_zoom_out = QPushButton("−")
        self.btn_preview_zoom_out.setObjectName("btnGhost")
        self.btn_preview_zoom_out.setFixedWidth(36)
        preview_toolbar.addWidget(self.btn_preview_fit)
        preview_toolbar.addWidget(self.btn_preview_100)
        preview_toolbar.addWidget(self.btn_preview_zoom_out)
        preview_toolbar.addWidget(self.btn_preview_zoom_in)
        preview_layout.addLayout(preview_toolbar)

        self.preview_meta = QLabel("Select an artifact from the list")
        self.preview_meta.setObjectName("previewMeta")
        self.preview_meta.setWordWrap(True)
        preview_layout.addWidget(self.preview_meta)

        self.preview_path_label = QLabel("")
        self.preview_path_label.setObjectName("previewPath")
        self.preview_path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.preview_path_label.setWordWrap(True)
        preview_layout.addWidget(self.preview_path_label)

        self.preview_stack = QStackedWidget()
        self.preview_stack.setObjectName("previewStack")

        image_page = QWidget()
        image_page_layout = QVBoxLayout(image_page)
        image_page_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setObjectName("previewScroll")
        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.preview_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.preview_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.preview_image = QLabel("Select a PNG/JPEG artifact to preview")
        self.preview_image.setObjectName("imagePreview")
        self.preview_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self.preview_scroll.setWidget(self.preview_image)
        # Refit image when the scroll viewport is resized (see eventFilter).
        self.preview_scroll.viewport().installEventFilter(self)
        image_page_layout.addWidget(self.preview_scroll)
        self.preview_stack.addWidget(image_page)

        self.text_preview = QTextEdit()
        self.text_preview.setObjectName("textPreview")
        self.text_preview.setReadOnly(True)
        self.preview_stack.addWidget(self.text_preview)

        self.preview_stack.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        preview_layout.addWidget(self.preview_stack, stretch=1)

        self.tabs.addTab(self.preview_tab, "Preview")

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 1040])

        # Wire signals (pipeline actions, artifact filters, preview zoom).
        self.btn_run_full.clicked.connect(self.run_full_pipeline)
        self.btn_run_selected.clicked.connect(self.run_selected_step)
        self.btn_stop.clicked.connect(self.stop_all)
        self.btn_clear_log.clicked.connect(self.logs.clear)
        self.btn_refresh.clicked.connect(self._load_artifacts)
        self.btn_open_file.clicked.connect(self._open_selected_file)
        self.btn_preview_open.clicked.connect(self._open_selected_file)
        self.btn_preview_copy_path.clicked.connect(self._copy_preview_path)
        self.btn_preview_fit.clicked.connect(lambda: self._set_preview_zoom("fit"))
        self.btn_preview_100.clicked.connect(lambda: self._set_preview_zoom("100"))
        self.btn_preview_zoom_in.clicked.connect(self._preview_zoom_in)
        self.btn_preview_zoom_out.clicked.connect(self._preview_zoom_out)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.artifact_search.textChanged.connect(self._apply_artifact_filters)
        self.artifact_preset.currentIndexChanged.connect(self._apply_artifact_filters)
        self.artifact_type.currentIndexChanged.connect(self._apply_artifact_filters)
        self.artifact_tree.itemSelectionChanged.connect(self._preview_selected_artifact)

    def _pipeline_step_defs(self) -> list[StepDef]:
        """Human-readable labels for the 11 pipeline steps (order = _full_pipeline)."""
        return [
            StepDef("1. crypto_data_all", "Download Bitstamp OHLC → CSV"),
            StepDef("2. logreturns", "compute_logreturns.py"),
            StepDef("3. liquidity", "liquidity.py — cut series"),
            StepDef("4. mutual", "mutual.py — MI minima → tau"),
            StepDef("5. tau_w", "tau_w.py"),
            StepDef("6. theilers_w", "theilers_w.bat — ACF/STP, W_D2_<sym>, PNG plots"),
            StepDef("7. phase_2D", "phase_2D.py"),
            StepDef("8. phase_3D", "phase_3D.py"),
            StepDef("9. cao_", "cao_.py — embedding dimension"),
            StepDef("10. 2dc", "2dc.py"),
            StepDef("11. hypothesis", "hypothesis.bat — D2 + LLE + RQA + results.docx"),
        ]

    def _reset_step_statuses(self):
        """Mark every sidebar step pending (called at pipeline start)."""
        self._step_status = [STEP_PENDING] * len(self._step_defs)
        for idx in range(self.steps.count()):
            self._refresh_step_list_item(idx)

    def _refresh_step_list_item(self, idx: int):
        """Update one QListWidget row: icon prefix + foreground colour by status."""
        if idx < 0 or idx >= self.steps.count():
            return
        item = self.steps.item(idx)
        status = self._step_status[idx] if idx < len(self._step_status) else STEP_PENDING
        short = self._step_defs[idx].short if idx < len(self._step_defs) else f"Step {idx + 1}"
        item.setText(f"{STEP_ICON.get(status, '○')}  {short}")
        item.setForeground(QBrush(STEP_COLOR.get(status, STEP_COLOR[STEP_PENDING])))
        if status == STEP_RUNNING:
            self.steps.setCurrentRow(idx)

    def _step_index_for_spec(self, spec: CommandSpec) -> int:
        """Map a running :class:`CommandSpec` back to sidebar index (-1 if unknown)."""
        pipeline = self._full_pipeline()
        for idx, cmd in enumerate(pipeline):
            if cmd.name == spec.name:
                return idx
        return -1

    def _step_produces_plots(self, spec: CommandSpec) -> bool:
        """True if this step may write PNGs we should auto-select after success."""
        name = spec.name.lower()
        return any(key in name for key in PLOT_STEP_KEYWORDS)

    def _python_cmd(self, script_name: str):
        """Build ``py -3 <ROOT>/<script_name>`` running from repo root."""
        return CommandSpec(
            name=script_name,
            program=PY,
            args=PY_ARGS + [str(ROOT / script_name)],
            cwd=ROOT,
        )

    def _bat_cmd(self, bat_name: str):
        """Build ``cmd /c <bat_name>`` with cwd ``Tisean_3.0.0/bin``."""
        return CommandSpec(
            name=bat_name,
            program="cmd",
            args=["/c", bat_name],
            cwd=ROOT / "Tisean_3.0.0" / "bin",
        )

    def _full_pipeline(self):
        """Ordered commands for **Run full** (must match _pipeline_step_defs)."""
        return [
            self._python_cmd("crypto_data_all.py"),
            self._python_cmd("compute_logreturns.py"),
            self._python_cmd("liquidity.py"),
            self._python_cmd("mutual.py"),
            self._python_cmd("tau_w.py"),
            self._bat_cmd("theilers_w.bat"),
            self._python_cmd("phase_2D.py"),
            self._python_cmd("phase_3D.py"),
            self._python_cmd("cao_.py"),
            self._python_cmd("2dc.py"),
            self._bat_cmd("hypothesis.bat"),
        ]

    def _scroll_log_to_end(self):
        """Scroll log to bottom when Follow tail is enabled."""
        if self.chk_follow_log.isChecked():
            self.logs.verticalScrollBar().setValue(self.logs.verticalScrollBar().maximum())

    def _append_log(self, msg: str):
        """Append a plain process line with colour by ERR/WARN keywords."""
        text = msg.rstrip()
        if not text:
            return
        safe = escape(text).replace("\n", "<br>")
        upper = text.upper()
        if "[ERR]" in upper or "ERROR" in upper or "TRACEBACK" in upper:
            color = "#fca5a5"
        elif "[WARN]" in upper or "WARNING" in upper:
            color = "#fcd34d"
        else:
            color = "#cbd5e1"
        self.logs.append(f"<span style='color:{color}'>{safe}</span>")
        self.logs.append("")
        self._scroll_log_to_end()

    def _append_status(self, badge: str, msg: str):
        """Append a timestamped badge row (RUN / OK / FAIL / INFO)."""
        colors = {
            "RUN": "#2563eb",
            "OK": "#16a34a",
            "FAIL": "#dc2626",
            "INFO": "#64748b",
        }
        color = colors.get(badge, "#64748b")
        ts = datetime.now().strftime("%H:%M:%S")
        safe = escape(msg.rstrip()).replace("\n", "<br>")
        html = (
            f"<span style='color:#64748b;font-size:11px;'>{ts}</span> "
            f"<span style='display:inline-block;background:{color};color:white;"
            f"padding:2px 8px;border-radius:6px;font-weight:600;font-size:11px;'>"
            f"{escape(badge)}</span> "
            f"<span style='color:#e2e8f0'>{safe}</span>"
        )
        self.logs.append(html)
        self.logs.append("")
        self._scroll_log_to_end()

    def _set_progress_state(self, state: str):
        """Drive progress bar colour via QSS property: idle | running | ok | fail."""
        self.progress_bar.setProperty("state", state)
        style = self.progress_bar.style()
        style.unpolish(self.progress_bar)
        style.polish(self.progress_bar)

    def _set_pipeline_controls_busy(self, busy: bool):
        """Disable Run buttons while pipeline active; enable Stop only when busy."""
        self.btn_run_full.setEnabled(not busy)
        self.btn_run_selected.setEnabled(not busy)
        self.btn_stop.setEnabled(busy)

    def _apply_theme(self):
        """Apply dark Fusion-style QSS (object names match selectors below)."""
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #12141a;
                color: #e8eaed;
                font-family: "Segoe UI", "Segoe UI Variable", sans-serif;
                font-size: 11pt;
            }
            QLabel { color: #d1d5db; }
            QLabel#appTitle {
                font-size: 15pt;
                font-weight: 700;
                color: #f8fafc;
                padding-bottom: 4px;
            }
            QLabel#sectionLabel {
                font-size: 10pt;
                font-weight: 600;
                color: #94a3b8;
                text-transform: none;
            }
            QLabel#progressLabel {
                font-size: 10pt;
                color: #cbd5e1;
            }
            QFrame#sidebarCard, QFrame#progressCard {
                background: #1a1d26;
                border: 1px solid #2d3340;
                border-radius: 10px;
            }
            QFrame#testModeFrame, QFrame#settingsFrame {
                background: #151922;
                border: 1px solid #334155;
                border-radius: 8px;
            }
            QTreeWidget#artifactTree {
                background: #0c0e13;
                border: 1px solid #2d3340;
                border-radius: 8px;
            }
            QTreeWidget#artifactTree::item {
                padding: 4px 2px;
            }
            QTreeWidget#artifactTree::item:selected {
                background: #1e3a5f;
            }
            QHeaderView::section {
                background: #1a1d26;
                color: #94a3b8;
                padding: 6px;
                border: none;
            }
            QSpinBox {
                background: #0c0e13;
                border: 1px solid #2d3340;
                border-radius: 6px;
                padding: 4px;
                color: #e2e8f0;
            }
            QPushButton#btnPrimary {
                background: #2563eb;
                color: #ffffff;
                border: none;
                padding: 8px 14px;
                border-radius: 8px;
                font-weight: 600;
            }
            QPushButton#btnPrimary:hover { background: #3b82f6; }
            QPushButton#btnPrimary:pressed { background: #1d4ed8; }
            QPushButton#btnPrimary:disabled {
                background: #374151;
                color: #9ca3af;
            }
            QPushButton#btnSecondary {
                background: #252a35;
                color: #e2e8f0;
                border: 1px solid #3d4654;
                padding: 8px 14px;
                border-radius: 8px;
            }
            QPushButton#btnSecondary:hover {
                background: #2f3642;
                border-color: #4b5563;
            }
            QPushButton#btnDanger {
                background: #7f1d1d;
                color: #fef2f2;
                border: none;
                padding: 8px 12px;
                border-radius: 8px;
                font-weight: 600;
            }
            QPushButton#btnDanger:hover { background: #b91c1c; }
            QPushButton#btnDanger:pressed { background: #991b1b; }
            QPushButton#btnGhost {
                background: transparent;
                color: #94a3b8;
                border: 1px solid #3d4654;
                padding: 4px 10px;
                border-radius: 6px;
                font-size: 10pt;
            }
            QPushButton#btnGhost:hover {
                color: #e2e8f0;
                border-color: #64748b;
            }
            QListWidget#pipelineSteps, QListWidget, QTextEdit#logConsole, QTextEdit,
            QLineEdit, QComboBox {
                background: #0c0e13;
                border: 1px solid #2d3340;
                border-radius: 8px;
                padding: 4px;
                selection-background-color: #2563eb;
                selection-color: #ffffff;
            }
            QListWidget#pipelineSteps::item {
                padding: 8px 6px;
                border-radius: 4px;
            }
            QListWidget#pipelineSteps::item:selected {
                background: #1e3a5f;
                color: #f1f5f9;
            }
            QListWidget#pipelineSteps::item:hover:!selected {
                background: #1f2430;
            }
            QTabWidget#mainTabs::pane {
                border: 1px solid #2d3340;
                border-radius: 8px;
                top: -1px;
                background: #14171e;
            }
            QTabBar::tab {
                background: #1a1d26;
                color: #94a3b8;
                padding: 8px 16px;
                margin-right: 4px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background: #14171e;
                color: #f1f5f9;
                font-weight: 600;
            }
            QTabBar::tab:hover:!selected { color: #cbd5e1; }
            QProgressBar#pipelineProgress {
                background: #0c0e13;
                border: 1px solid #2d3340;
                border-radius: 6px;
                text-align: center;
                color: #94a3b8;
                font-size: 9pt;
            }
            QProgressBar#pipelineProgress::chunk {
                background: #2563eb;
                border-radius: 5px;
            }
            QProgressBar#pipelineProgress[state="ok"]::chunk { background: #16a34a; }
            QProgressBar#pipelineProgress[state="fail"]::chunk { background: #dc2626; }
            QProgressBar#pipelineProgress[state="idle"]::chunk { background: #374151; }
            QScrollArea#previewScroll {
                background: #0a0c10;
                border: 1px solid #2d3340;
                border-radius: 8px;
            }
            QLabel#imagePreview {
                background: #0c0e13;
                padding: 4px;
                color: #64748b;
            }
            QLabel#previewMeta {
                color: #94a3b8;
                font-size: 10pt;
            }
            QLabel#previewPath {
                color: #64748b;
                font-size: 9pt;
                font-family: "Cascadia Mono", "Consolas", monospace;
            }
            QTextEdit#textPreview {
                min-height: 120px;
            }
            QStackedWidget#previewStack {
                min-height: 420px;
            }
            QCheckBox { color: #cbd5e1; spacing: 8px; }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid #4b5563;
                background: #0c0e13;
            }
            QCheckBox::indicator:checked {
                background: #2563eb;
                border-color: #2563eb;
            }
            QSplitter::handle { background: #2d3340; }
            QSplitter::handle:horizontal { width: 8px; margin: 0 2px; }
            """
        )

    def _run_command(self, spec: CommandSpec, on_done):
        """
        Start one child process and stream output until it exits.

        Injects environment variables for test mode and hypothesis settings, then
        calls ``on_done(ok)`` once with *ok* = (exit code 0 and normal exit).
        """
        proc = QProcess(self)
        proc.setProgram(spec.program)
        proc.setArguments(spec.args)
        proc.setWorkingDirectory(str(spec.cwd))
        # Mirror _dch_test_env.bat + desktop settings so .bat and Python agree.
        env = QProcessEnvironment.systemEnvironment()
        test_on = self.chk_test_mode.isChecked()
        env.insert("DCH_TEST_MODE", "true" if test_on else "false")
        if test_on:
            from hypothesis_config import (
                DEFAULT_DCH_LYAP_MIN_NEIGHBORS_TEST,
                DEFAULT_DCH_LYAP_STEPS_TEST,
            )

            env.insert("DCH_TEST_POINTS", str(dch_test_point_count()))
            env.insert("DCH_LYAP_STEPS", str(DEFAULT_DCH_LYAP_STEPS_TEST))
            env.insert(
                "DCH_LYAP_MIN_NEIGHBORS", str(DEFAULT_DCH_LYAP_MIN_NEIGHBORS_TEST)
            )
        else:
            env.insert("DCH_TEST_MODE", "false")
        env.insert(
            "DCH_RUN_HYPOTHESIS",
            "true" if self.chk_run_hypothesis.isChecked() else "false",
        )
        env.insert("DCH_DIMENSION_METRICS", self.cmb_dimension_metrics.currentText())
        env.insert("DCH_BOOTSTRAP_SAMPLES", str(self.spin_bootstrap.value()))
        proc.setProcessEnvironment(env)

        self.current_processes[spec.name] = proc
        self._current_step_name = spec.name
        step_idx = self._step_index_for_spec(spec)
        if step_idx >= 0:
            self._step_status[step_idx] = STEP_RUNNING
            self._refresh_step_list_item(step_idx)
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
                for line in err.splitlines():
                    if line.strip():
                        self._append_log(f"[{spec.name}][ERR] {line}")

        def finished(exit_code, exit_status):
            # QProcess.finished(int, QProcess.ExitStatus) — always disconnects implicitly.
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

    # -------------------------------------------------------------------------
    # Pipeline queue (sequential steps; optional concurrent groups)
    # -------------------------------------------------------------------------

    def _run_next_pipeline_item(self):
        """
        Pop and run the next queue item (recursive via QTimer after each step).

        Queue entries are either a single :class:`CommandSpec` or a list of specs
        run concurrently (failure in any member stops the pipeline).
        """
        if self.stop_requested:
            self.pipeline_running = False
            self._set_pipeline_controls_busy(False)
            self._append_status("INFO", "Pipeline stopped by user.")
            self._refresh_progress_ui()
            return
        if not self.pipeline_queue:
            self.pipeline_running = False
            self._set_pipeline_controls_busy(False)
            self._append_status("OK", "Pipeline finished.")
            self._load_artifacts(log=True)
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
                        self._pipeline_failed = True
                        self._set_pipeline_controls_busy(False)
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
                self._on_step_completed(item, ok)
                if not ok:
                    self.pipeline_running = False
                    self._pipeline_failed = True
                    self._set_pipeline_controls_busy(False)
                    self._append_status("FAIL", "Pipeline stopped due to step failure.")
                    self._refresh_progress_ui()
                    return
                self._pipeline_completed_steps += 1
                self._refresh_progress_ui()
                QTimer.singleShot(10, self._run_next_pipeline_item)

            self._run_command(item, done_single)

    def run_full_pipeline(self):
        """Enqueue all steps from :meth:`_full_pipeline` and start the runner."""
        if self.pipeline_running:
            QMessageBox.warning(self, "Pipeline running", "Pipeline is already running.")
            return
        self.stop_requested = False
        self._pipeline_failed = False
        self._reset_step_statuses()
        self.pipeline_running = True
        self._set_pipeline_controls_busy(True)
        self.pipeline_queue = self._full_pipeline()
        self._pipeline_total_steps = len(self.pipeline_queue)
        self._pipeline_completed_steps = 0
        self._current_step_name = ""
        self._append_status("INFO", "Starting full pipeline...")
        self._refresh_progress_ui()
        self._run_next_pipeline_item()

    def run_selected_step(self):
        """Run only the step at the current sidebar selection index."""
        if self.pipeline_running:
            QMessageBox.warning(self, "Pipeline running", "Stop current run first.")
            return
        idx = self.steps.currentRow()
        if idx < 0:
            return
        mapping = self._full_pipeline()
        selected = mapping[idx]
        self.stop_requested = False
        self._pipeline_failed = False
        self._reset_step_statuses()
        step_idx = self.steps.currentRow()
        if step_idx >= 0:
            self._step_status[step_idx] = STEP_PENDING
            self._refresh_step_list_item(step_idx)
        self.pipeline_running = True
        self._set_pipeline_controls_busy(True)
        self.pipeline_queue = [selected]
        self._pipeline_total_steps = 1
        self._pipeline_completed_steps = 0
        self._current_step_name = ""
        self._append_status("INFO", f"Starting selected step: {self.steps.currentItem().text()}")
        self._refresh_progress_ui()
        self._run_next_pipeline_item()

    def stop_all(self):
        """Set stop flag, kill all running QProcess children, reset UI."""
        self.stop_requested = True
        for name, proc in list(self.current_processes.items()):
            self._append_status("INFO", f"Terminating {name}")
            proc.kill()
        self.current_processes.clear()
        self.pipeline_running = False
        self._set_pipeline_controls_busy(False)
        self._refresh_progress_ui()

    def _refresh_progress_ui(self):
        """Update progress label, bar percent, and bar colour from pipeline state."""
        total = self._pipeline_total_steps
        done = self._pipeline_completed_steps
        if total <= 0:
            self.progress_label.setText("Idle — no pipeline running")
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("")
            self._set_progress_state("idle")
            return
        pct = int((done / total) * 100) if total else 0
        if not self.pipeline_running and done >= total:
            if self._pipeline_failed:
                self.progress_label.setText(f"Finished with errors — {done}/{total} steps")
                self._set_progress_state("fail")
            else:
                self.progress_label.setText(f"Completed — all {total} steps OK")
                self._set_progress_state("ok")
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("100%")
            return
        if self._pipeline_failed and not self.pipeline_running:
            self.progress_label.setText(f"Stopped — {done}/{total} steps completed")
            self.progress_bar.setValue(pct)
            self.progress_bar.setFormat(f"{pct}%")
            self._set_progress_state("fail")
            return
        step_no = min(done + 1, total)
        current = self._current_step_name or "…"
        self.progress_label.setText(f"Step {step_no}/{total} · {current}")
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"{pct}%")
        self._set_progress_state("running")

    # -------------------------------------------------------------------------
    # Artifacts browser (PNG / TXT under data/ and results/)
    # -------------------------------------------------------------------------

    def _artifact_roots(self):
        """
        Directories scanned recursively for PNG/JPG/TXT artifacts.

        Includes ``results_<test_tag>`` / ``results_full`` temp trim folders when present;
        only existing paths are returned.
        """
        test_tag = dch_test_results_tag()
        roots = [
            ROOT,
            self.data_dir,
            self.results_dir,
            self.data_dir / "results",
            self.data_dir / f"results_{test_tag}",
            self.data_dir / "results_full",
        ]
        return [p for p in roots if p.exists()]

    def _load_artifacts(self, log: bool = True):
        """Rescan artifact roots, sort by mtime (newest first), cap at 1200 files."""
        exts = {".png", ".jpg", ".jpeg", ".txt"}
        items = []
        for base in self._artifact_roots():
            for path in base.rglob("*"):
                if path.is_file() and path.suffix.lower() in exts:
                    items.append(path)
        self._all_artifacts = sorted(set(items), key=lambda p: p.stat().st_mtime, reverse=True)[:1200]
        self._apply_artifact_filters()
        if log:
            self._append_log(f"[ARTIFACTS] Loaded {len(self._all_artifacts)} files.")

    def _artifact_matches_preset(self, path: Path, preset_label: str) -> bool:
        """True if *path* matches any keyword for the Artifacts preset (or preset is All)."""
        keys = ARTIFACT_PRESETS.get(preset_label)
        if not keys:
            return True
        hay = str(path).lower()
        return any(k in hay for k in keys)

    def _apply_artifact_filters(self):
        """Rebuild artifact tree from ``_all_artifacts`` using search / type / preset."""
        search = (self.artifact_search.text() or "").strip().lower()
        mode = self.artifact_type.currentText()
        preset = self.artifact_preset.currentText()
        self.artifact_tree.clear()

        filtered: list[Path] = []
        for p in self._all_artifacts:
            suffix = p.suffix.lower()
            is_image = suffix in {".png", ".jpg", ".jpeg"}
            is_text = suffix == ".txt"
            if mode == "Images" and not is_image:
                continue
            if mode == "Text" and not is_text:
                continue
            if not self._artifact_matches_preset(p, preset):
                continue
            haystack = str(p).lower()
            if search and search not in haystack:
                continue
            filtered.append(p)

        groups: dict[str, list[Path]] = {}
        for p in filtered:
            try:
                rel = p.relative_to(self.data_dir)
                folder = str(rel.parent).replace("\\", "/")
                if folder in (".", ""):
                    folder = "(data root)"
            except ValueError:
                try:
                    rel = p.relative_to(ROOT)
                    folder = str(rel.parent).replace("\\", "/") or "(project)"
                except ValueError:
                    folder = str(p.parent)
            groups.setdefault(folder, []).append(p)

        shown = 0
        for folder in sorted(groups.keys(), key=str.lower):
            folder_item = QTreeWidgetItem([folder, ""])
            folder_item.setData(0, Qt.ItemDataRole.UserRole, "")
            folder_item.setFirstColumnSpanned(True)
            for p in sorted(groups[folder], key=lambda x: x.name.lower()):
                try:
                    rel = p.relative_to(self.data_dir)
                except ValueError:
                    rel = p
                file_item = QTreeWidgetItem([p.name, str(rel)])
                file_item.setData(0, Qt.ItemDataRole.UserRole, str(p))
                folder_item.addChild(file_item)
                shown += 1
            self.artifact_tree.addTopLevelItem(folder_item)
            folder_item.setExpanded(True)

        self.artifact_tree.resizeColumnToContents(0)
        self.artifact_stats.setText(f"Artifacts: {shown} files in {len(groups)} folders")

    # -------------------------------------------------------------------------
    # Preview tab (image zoom + text summaries)
    # -------------------------------------------------------------------------

    def _on_step_completed(self, spec: CommandSpec, ok: bool):
        """
        Sidebar status update after one sequential step; optional artifact UX.

        On success: refresh artifact cache; if the step produces plots, switch to
        the Artifacts tab and highlight the newest PNG.
        """
        idx = self._step_index_for_spec(spec)
        if idx >= 0:
            self._step_status[idx] = STEP_OK if ok else STEP_FAIL
            self._refresh_step_list_item(idx)
        if ok:
            self._load_artifacts(log=False)
            if self._step_produces_plots(spec):
                if self._artifacts_tab is not None:
                    art_idx = self.tabs.indexOf(self._artifacts_tab)
                    if art_idx >= 0:
                        self.tabs.setCurrentIndex(art_idx)
                QTimer.singleShot(150, self._select_newest_png)

    def _select_newest_png(self):
        """Select the most recently modified PNG in the artifact cache."""
        pngs = [p for p in self._all_artifacts if p.suffix.lower() == ".png"]
        if not pngs:
            return
        newest = max(pngs, key=lambda p: p.stat().st_mtime)
        self._select_artifact_in_tree(newest)

    def _select_artifact_in_tree(self, path: Path):
        """Find tree item whose UserRole holds *path* and set as current selection."""
        target = str(path)
        for i in range(self.artifact_tree.topLevelItemCount()):
            folder = self.artifact_tree.topLevelItem(i)
            for j in range(folder.childCount()):
                child = folder.child(j)
                if child.data(0, Qt.ItemDataRole.UserRole) == target:
                    self.artifact_tree.setCurrentItem(child)
                    return

    def _selected_artifact_path(self) -> Path | None:
        """Absolute path from the selected leaf row (folder rows have empty UserRole)."""
        items = self.artifact_tree.selectedItems()
        if not items:
            return None
        raw = items[0].data(0, Qt.ItemDataRole.UserRole)
        if not raw:
            return None
        return Path(raw)

    def _format_preview_meta(self, path: Path) -> str:
        """One-line summary: filename and file size for the preview header."""
        try:
            size_kb = path.stat().st_size / 1024.0
            size_s = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.2f} MB"
        except OSError:
            size_s = "?"
        return f"{path.name}  ·  {size_s}"

    def _show_preview_path(self, path: Path | None):
        """Update path label and meta line; clear when no file selected."""
        self._preview_path = path
        if path is None:
            self.preview_path_label.setText("")
            self.preview_meta.setText("Select an artifact from the list")
            return
        self.preview_path_label.setText(str(path))
        self.preview_meta.setText(self._format_preview_meta(path))

    def _set_preview_zoom(self, mode: str):
        """Set zoom mode (fit / 100 / custom) and refresh scaled pixmap."""
        self._preview_zoom_mode = mode
        if mode == "100":
            self._preview_zoom_factor = 1.0
        QTimer.singleShot(0, self._refresh_preview_pixmap)

    def _preview_zoom_in(self):
        """Increase custom zoom factor (max 400%), switching out of fit mode."""
        if self._preview_pixmap is None or self._preview_pixmap.isNull():
            return
        if self._preview_zoom_mode == "fit":
            self._preview_zoom_mode = "custom"
            self._preview_zoom_factor = 1.0
        self._preview_zoom_factor = min(4.0, self._preview_zoom_factor * 1.25)
        self._preview_zoom_mode = "custom"
        self._refresh_preview_pixmap()

    def _preview_zoom_out(self):
        """Decrease custom zoom factor (min 25%)."""
        if self._preview_pixmap is None or self._preview_pixmap.isNull():
            return
        if self._preview_zoom_mode == "fit":
            self._preview_zoom_mode = "custom"
            self._preview_zoom_factor = 1.0
        self._preview_zoom_factor = max(0.25, self._preview_zoom_factor / 1.25)
        self._preview_zoom_mode = "custom"
        self._refresh_preview_pixmap()

    def _preview_selected_artifact(self):
        """Load selected tree file into image or text preview stack."""
        path = self._selected_artifact_path()
        if path is None:
            self._preview_pixmap = None
            self._show_preview_path(None)
            self.preview_stack.setCurrentIndex(0)
            self.preview_image.setText("Select a PNG/JPEG artifact to preview")
            self.preview_image.setPixmap(QPixmap())
            return
        if not path.exists():
            self._append_status("INFO", f"Artifact missing: {path}")
            return
        self._show_preview_path(path)
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            pix = QPixmap(str(path))
            if pix.isNull():
                self._preview_pixmap = None
                self.preview_stack.setCurrentIndex(0)
                self.preview_image.setText("Could not load image")
                self.preview_image.setPixmap(QPixmap())
                return
            self._preview_pixmap = pix
            self._preview_zoom_mode = "fit"
            self.preview_stack.setCurrentIndex(0)
            zoom_note = ""
            self.preview_meta.setText(
                f"{self._format_preview_meta(path)}  ·  {pix.width()}×{pix.height()} px{zoom_note}"
            )
            QTimer.singleShot(0, self._refresh_preview_pixmap)
            preview_idx = self.tabs.indexOf(self.preview_tab)
            if preview_idx >= 0:
                self.tabs.setCurrentIndex(preview_idx)
        else:
            self._preview_pixmap = None
            self.preview_stack.setCurrentIndex(1)
            try:
                txt = path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                txt = f"Unable to read file: {e}"
            self.text_preview.setPlainText(txt[:200000])

    def _copy_preview_path(self):
        """Copy absolute path of current preview file to the system clipboard."""
        if not self._preview_path:
            return
        QApplication.clipboard().setText(str(self._preview_path))

    def _on_tab_changed(self, index: int):
        """Refit image when user switches to Preview (viewport may have changed)."""
        if self.tabs.widget(index) is self.preview_tab:
            QTimer.singleShot(0, self._refresh_preview_pixmap)

    def _open_selected_file(self):
        """Open selected artifact with the OS default app (``os.startfile`` on Windows)."""
        path_obj = self._selected_artifact_path()
        if path_obj is None:
            return
        path = str(path_obj)
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                QMessageBox.information(self, "Open file", f"Selected file:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Open failed", f"Could not open file:\n{e}")

    def _preview_viewport_size(self):
        """Usable width/height inside preview scroll area (small margin)."""
        vp = self.preview_scroll.viewport()
        return max(80, vp.width() - 12), max(80, vp.height() - 12)

    def _refresh_preview_pixmap(self):
        """Scale ``_preview_pixmap`` per zoom mode and assign to ``preview_image`` QLabel."""
        if self._preview_pixmap is None or self._preview_pixmap.isNull():
            return
        if self.preview_stack.currentIndex() != 0:
            return
        src = self._preview_pixmap
        if self._preview_zoom_mode == "fit":
            max_w, max_h = self._preview_viewport_size()
            scaled = src.scaled(
                max_w,
                max_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        elif self._preview_zoom_mode == "100":
            scaled = src
        else:
            tw = max(1, int(src.width() * self._preview_zoom_factor))
            th = max(1, int(src.height() * self._preview_zoom_factor))
            scaled = src.scaled(
                tw,
                th,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self.preview_image.setPixmap(scaled)
        self.preview_image.resize(scaled.size())
        mode_lbl = self._preview_zoom_mode
        if mode_lbl == "custom":
            mode_lbl = f"{self._preview_zoom_factor:.0%}"
        self.preview_image.setToolTip(
            f"{src.width()}×{src.height()} px native · shown {scaled.width()}×{scaled.height()} ({mode_lbl})"
        )

    def resizeEvent(self, event):
        """Keep fit-to-viewport scaling correct when the main window is resized."""
        super().resizeEvent(event)
        self._refresh_preview_pixmap()

    def eventFilter(self, obj, event):
        """Refit preview on scroll-area viewport resize (installed on viewport)."""
        if (
            hasattr(self, "preview_scroll")
            and obj is self.preview_scroll.viewport()
            and event.type() == QEvent.Type.Resize
        ):
            QTimer.singleShot(0, self._refresh_preview_pixmap)
        return super().eventFilter(obj, event)


def main():
    """Application entry: Fusion style + :class:`PipelineApp` main window."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = PipelineApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
