"""Efficient desktop GUI for the Python event-camera simulator using PySide6."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import cv2 as cv
import numpy as np
from PySide6 import QtCore, QtGui
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import SimulatorConfig
from .demo import write_synthetic_video
from .events import EventStream
from .pipeline import SimulationResult
from .sources import FrameSource, inspect_source, open_source
from .visualization import EventVideoRenderer


class SimulationWorker(QThread):
    """Background worker thread to run simulation without blocking the UI."""

    progress = Signal(int, int)  # current_frame, total_frames
    finished = Signal(dict, object)  # stats, result
    error = Signal(str)

    def __init__(
        self,
        source: FrameSource,
        config: SimulatorConfig,
        renderer: EventVideoRenderer | None = None,
        max_frames: int | None = None,
    ):
        super().__init__()
        self.source = source
        self.config = config
        self.renderer = renderer
        self.max_frames = max_frames
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        try:
            total_frames = self.source.total_frames or self.max_frames or 100
            last_timestamp_us = 0
            frames_processed = 0
            started = time.perf_counter()

            from .simulator import EventSimulator

            sim = EventSimulator(self.config)
            events_list: list[np.ndarray] = []
            contrast_events = 0
            background_events = 0
            last_frame: np.ndarray | None = None

            for frame in self.source:
                if self._is_cancelled:
                    break
                if self.max_frames and frames_processed >= self.max_frames:
                    break

                last_frame = frame.image.copy()

                if not sim.initialized:
                    sim.initialize(frame.image, frame.timestamp_us)
                    if self.renderer is not None:
                        self.renderer.add(np.empty(0), frame.image, frame.timestamp_us)
                    frames_processed += 1
                    last_timestamp_us = frame.timestamp_us
                    self.progress.emit(frames_processed, total_frames)
                    continue

                pair_res = sim.process(frame.image, frame.timestamp_us)
                if pair_res.events.size:
                    events_list.append(pair_res.events)
                contrast_events += pair_res.contrast_events
                background_events += pair_res.background_events

                if self.renderer is not None:
                    self.renderer.add(pair_res.events, frame.image, frame.timestamp_us)
                    self.renderer.maybe_write(frame.timestamp_us)

                frames_processed += 1
                last_timestamp_us = frame.timestamp_us
                self.progress.emit(frames_processed, total_frames)

            if self.renderer is not None:
                self.renderer.close()

            elapsed = time.perf_counter() - started
            stream = EventStream.concatenate(events_list)
            stats = stream.summary(input_duration_us=last_timestamp_us)
            stats["frames_processed"] = frames_processed
            stats["contrast_events"] = contrast_events
            stats["background_events"] = background_events
            stats["cancelled"] = self._is_cancelled
            stats["elapsed_seconds"] = elapsed
            stats["width"] = self.source.width
            stats["height"] = self.source.height

            preview_panel = None
            if last_frame is not None:
                preview_renderer = EventVideoRenderer(
                    self.source.width, self.source.height, self.config.visualization
                )
                all_events = stream.events
                if all_events.size > 0:
                    t_end = int(all_events["timestamp_us"][-1])
                    t_start = max(
                        0, t_end - self.config.visualization.accumulation_time_us
                    )
                    recent_events = all_events[all_events["timestamp_us"] >= t_start]
                    valid = (recent_events["x"] < self.source.width) & (
                        recent_events["y"] < self.source.height
                    )
                    if np.any(valid):
                        preview_renderer.add(recent_events[valid], last_frame, t_end)
                preview_panel = preview_renderer.render_combined_panel(last_frame)

            sim_res = SimulationResult(
                event_stream=stream,
                statistics=stats,
                frames_processed=frames_processed,
                elapsed_seconds=elapsed,
                video_path=self.config.visualization.output_video,
                preview_panel=preview_panel,
            )
            self.finished.emit(stats, sim_res)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
        finally:
            self.source.close()


class MainWindow(QMainWindow):
    """Main window for the evsim PySide6 desktop GUI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Event Camera Simulator (evsim)")
        self.resize(1180, 780)

        self.worker: SimulationWorker | None = None
        self.last_result: SimulationResult | None = None
        self.temp_demo_path: Path | None = None

        self._init_ui()
        self._apply_preset("Realistic")

    def _init_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left control panel (Scrollable)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(440)
        left_scroll.setMaximumWidth(520)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(12)

        # 1. Source Group
        source_group = QGroupBox("1. Input Source")
        source_layout = QVBoxLayout(source_group)

        src_type_layout = QHBoxLayout()
        src_type_layout.addWidget(QLabel("Mode:"))
        self.combo_source_type = QComboBox()
        self.combo_source_type.addItems(
            ["Synthetic 960 FPS Demo", "Video File", "Image Sequence"]
        )
        self.combo_source_type.currentIndexChanged.connect(self._on_source_type_changed)
        src_type_layout.addWidget(self.combo_source_type)
        source_layout.addLayout(src_type_layout)

        path_layout = QHBoxLayout()
        self.edit_source_path = QLineEdit()
        self.edit_source_path.setPlaceholderText(
            "Path to video file or image sequence directory"
        )
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._browse_source)
        path_layout.addWidget(self.edit_source_path)
        path_layout.addWidget(self.btn_browse)
        source_layout.addLayout(path_layout)

        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Fallback FPS:"))
        self.spin_fallback_fps = QDoubleSpinBox()
        self.spin_fallback_fps.setRange(1.0, 10000.0)
        self.spin_fallback_fps.setValue(960.0)
        fps_layout.addWidget(self.spin_fallback_fps)
        source_layout.addLayout(fps_layout)

        self.lbl_source_info = QLabel("Source: Built-in synthetic 960 FPS disk scene")
        self.lbl_source_info.setStyleSheet("color: #4CAF50; font-weight: bold;")
        source_layout.addWidget(self.lbl_source_info)

        left_layout.addWidget(source_group)

        # 2. Parameters Group
        param_group = QGroupBox("2. Sensor && Simulation Parameters")
        param_form = QFormLayout(param_group)

        preset_layout = QHBoxLayout()
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(["Realistic", "Ideal"])
        self.combo_preset.currentTextChanged.connect(self._apply_preset)
        preset_layout.addWidget(self.combo_preset)
        param_form.addRow("Preset:", preset_layout)

        self.spin_pos_th = QDoubleSpinBox()
        self.spin_pos_th.setRange(0.01, 2.0)
        self.spin_pos_th.setSingleStep(0.02)
        self.spin_pos_th.setValue(0.20)
        param_form.addRow("C+ (ON Threshold):", self.spin_pos_th)

        self.spin_neg_th = QDoubleSpinBox()
        self.spin_neg_th.setRange(0.01, 2.0)
        self.spin_neg_th.setSingleStep(0.02)
        self.spin_neg_th.setValue(0.15)
        param_form.addRow("C- (OFF Threshold):", self.spin_neg_th)

        self.spin_ts_res = QSpinBox()
        self.spin_ts_res.setRange(1, 10000)
        self.spin_ts_res.setValue(1)
        self.spin_ts_res.setSuffix(" μs")
        param_form.addRow("Timestamp Resolution:", self.spin_ts_res)

        self.spin_refractory = QSpinBox()
        self.spin_refractory.setRange(0, 100000)
        self.spin_refractory.setValue(100)
        self.spin_refractory.setSuffix(" μs")
        param_form.addRow("Refractory Period:", self.spin_refractory)

        self.spin_accum = QSpinBox()
        self.spin_accum.setRange(100, 1000000)
        self.spin_accum.setValue(5000)
        self.spin_accum.setSingleStep(1000)
        self.spin_accum.setSuffix(" μs")
        param_form.addRow("Accumulation Time:", self.spin_accum)

        self.combo_backend = QComboBox()
        self.combo_backend.addItems(["vectorized", "loop"])
        param_form.addRow("Backend:", self.combo_backend)

        self.chk_threshold_var = QCheckBox("Pixel Threshold Mismatch (σ=0.02)")
        self.chk_threshold_var.setChecked(True)
        param_form.addRow(self.chk_threshold_var)

        self.chk_bg_noise = QCheckBox("Background Activity (Leak noise)")
        self.chk_bg_noise.setChecked(True)
        param_form.addRow(self.chk_bg_noise)

        self.chk_linearization = QCheckBox("Gamma Linearization (sRGB to Linear)")
        self.chk_linearization.setChecked(False)
        param_form.addRow(self.chk_linearization)

        self.spin_max_frames = QSpinBox()
        self.spin_max_frames.setRange(0, 100000)
        self.spin_max_frames.setValue(0)
        self.spin_max_frames.setSpecialValueText("Unlimited")
        param_form.addRow("Max Frames:", self.spin_max_frames)

        left_layout.addWidget(param_group)

        # 3. Output Settings Group
        out_group = QGroupBox("3. Output Settings")
        out_form = QFormLayout(out_group)

        self.chk_save_csv = QCheckBox("Save CSV")
        self.edit_csv_path = QLineEdit("output/gui_events.csv")
        out_form.addRow(self.chk_save_csv, self.edit_csv_path)

        self.chk_save_npz = QCheckBox("Save NPZ")
        self.chk_save_npz.setChecked(True)
        self.edit_npz_path = QLineEdit("output/gui_events.npz")
        out_form.addRow(self.chk_save_npz, self.edit_npz_path)

        self.chk_save_video = QCheckBox("Generate Video")
        self.chk_save_video.setChecked(True)
        self.edit_video_path = QLineEdit("output/gui_video.avi")
        out_form.addRow(self.chk_save_video, self.edit_video_path)

        left_layout.addWidget(out_group)

        # 4. Action buttons and progress
        action_group = QGroupBox("Execution")
        action_layout = QVBoxLayout(action_group)

        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("Start Simulation")
        self.btn_run.setFixedHeight(40)
        self.btn_run.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; font-size: 14px;"
        )
        self.btn_run.clicked.connect(self._start_simulation)
        btn_layout.addWidget(self.btn_run)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setFixedHeight(40)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_simulation)
        btn_layout.addWidget(self.btn_cancel)
        action_layout.addLayout(btn_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        action_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Ready")
        action_layout.addWidget(self.lbl_status)

        left_layout.addWidget(action_group)
        left_layout.addStretch()

        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)

        # Right display panel (Tabs: Statistics, Visual Preview, JSON Log)
        self.tabs = QTabWidget()

        # Tab 1: Statistics Table
        tab_stats = QWidget()
        stats_layout = QVBoxLayout(tab_stats)
        self.table_stats = QTableWidget(0, 2)
        self.table_stats.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table_stats.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        stats_layout.addWidget(self.table_stats)
        self.tabs.addTab(tab_stats, "Statistics")

        # Tab 2: Visual Preview
        tab_preview = QWidget()
        preview_layout = QVBoxLayout(tab_preview)
        self.lbl_preview_image = QLabel(
            "Visual preview will appear here after simulation."
        )
        self.lbl_preview_image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview_image.setStyleSheet(
            "background-color: #1e1e1e; color: #888; border-radius: 4px;"
        )
        preview_scroll = QScrollArea()
        preview_scroll.setWidgetResizable(True)
        preview_scroll.setWidget(self.lbl_preview_image)
        preview_layout.addWidget(preview_scroll)
        self.tabs.addTab(tab_preview, "Visual Preview (Input | Events | Overlay)")

        # Tab 3: Raw JSON
        tab_log = QWidget()
        log_layout = QVBoxLayout(tab_log)
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        log_layout.addWidget(self.text_log)
        self.tabs.addTab(tab_log, "JSON Summary")

        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)

    def _on_source_type_changed(self, index: int) -> None:
        if index == 0:  # Built-in Demo
            self.edit_source_path.setEnabled(False)
            self.btn_browse.setEnabled(False)
            self.lbl_source_info.setText(
                "Source: Built-in synthetic 960 FPS disk scene (320x240, 1.0s)"
            )
        else:
            self.edit_source_path.setEnabled(True)
            self.btn_browse.setEnabled(True)
            self.lbl_source_info.setText("Please select an input source...")

    def _browse_source(self) -> None:
        if self.combo_source_type.currentIndex() == 1:  # Video
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Video File",
                "",
                "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)",
            )
        else:  # Image Sequence
            path = QFileDialog.getExistingDirectory(
                self, "Select Image Sequence Directory"
            )
        if path:
            self.edit_source_path.setText(path)
            self._inspect_selected_source(path)

    def _inspect_selected_source(self, path: str) -> None:
        try:
            meta = inspect_source(path, fallback_fps=self.spin_fallback_fps.value())
            frames_str = (
                f"{meta['frames']} frames" if meta.get("frames") else "unknown frames"
            )
            dur_str = f", {meta['duration_s']:.2f}s" if meta.get("duration_s") else ""
            self.lbl_source_info.setText(
                f"Source: {meta.get('width', '?')}x{meta.get('height', '?')} @ {meta.get('fps', self.spin_fallback_fps.value()):.1f} FPS ({frames_str}{dur_str})"
            )
        except Exception as exc:  # noqa: BLE001
            self.lbl_source_info.setText(f"Error inspecting source: {exc}")

    def _apply_preset(self, preset_name: str) -> None:
        if preset_name == "Realistic":
            self.spin_pos_th.setValue(0.20)
            self.spin_neg_th.setValue(0.15)
            self.spin_refractory.setValue(100)
            self.chk_threshold_var.setChecked(True)
            self.chk_bg_noise.setChecked(True)
            self.chk_linearization.setChecked(False)
        else:  # Ideal
            self.spin_pos_th.setValue(0.20)
            self.spin_neg_th.setValue(0.20)
            self.spin_refractory.setValue(0)
            self.chk_threshold_var.setChecked(False)
            self.chk_bg_noise.setChecked(False)
            self.chk_linearization.setChecked(False)

    def _build_config(self) -> SimulatorConfig:
        config = SimulatorConfig()
        config.sensor.positive_threshold = self.spin_pos_th.value()
        config.sensor.negative_threshold = self.spin_neg_th.value()
        config.sensor.timestamp_resolution_us = self.spin_ts_res.value()
        config.sensor.refractory_period_us = self.spin_refractory.value()
        config.simulation.backend = self.combo_backend.currentText()
        config.visualization.accumulation_time_us = self.spin_accum.value()

        config.noise.enable_threshold_variation = self.chk_threshold_var.isChecked()
        config.noise.threshold_sigma = (
            0.02 if self.chk_threshold_var.isChecked() else 0.0
        )
        config.noise.background_rate_hz = 0.02 if self.chk_bg_noise.isChecked() else 0.0

        config.input.linearize_gamma = self.chk_linearization.isChecked()
        config.input.fallback_fps = self.spin_fallback_fps.value()

        max_f = self.spin_max_frames.value()
        config.runtime.max_frames = max(0, max_f)

        if self.chk_save_csv.isChecked():
            config.output.csv_path = self.edit_csv_path.text().strip()
        if self.chk_save_npz.isChecked():
            config.output.npz_path = self.edit_npz_path.text().strip()
        if self.chk_save_video.isChecked():
            config.visualization.output_video = self.edit_video_path.text().strip()

        config.validate()
        return config

    def _start_simulation(self) -> None:
        try:
            config = self._build_config()
            src_mode = self.combo_source_type.currentIndex()

            if src_mode == 0:  # Built-in synthetic demo
                temp_dir = Path(tempfile.gettempdir()) / "evsim_demo"
                temp_dir.mkdir(parents=True, exist_ok=True)
                self.temp_demo_path = temp_dir / "input_demo.avi"
                write_synthetic_video(
                    self.temp_demo_path, fps=960.0, seconds=1.0, width=320, height=240
                )
                source = open_source(self.temp_demo_path, fallback_fps=960.0)
            else:
                path_str = self.edit_source_path.text().strip()
                if not path_str:
                    QMessageBox.warning(
                        self, "Missing Input", "Please specify an input path."
                    )
                    return
                source = open_source(path_str, fallback_fps=config.input.fallback_fps)

            renderer = None
            if config.visualization.output_video:
                renderer = EventVideoRenderer(
                    source.width, source.height, config.visualization
                )
                renderer.open(config.visualization.output_video)

            self.btn_run.setEnabled(False)
            self.btn_cancel.setEnabled(True)
            self.progress_bar.setValue(0)
            self.lbl_status.setText("Simulation running...")

            self.worker = SimulationWorker(
                source=source,
                config=config,
                renderer=renderer,
                max_frames=config.runtime.max_frames or None,
            )
            self.worker.progress.connect(self._on_progress)
            self.worker.finished.connect(self._on_finished)
            self.worker.error.connect(self._on_error)
            self.worker.start()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error Starting Simulation", str(exc))
            self.btn_run.setEnabled(True)
            self.btn_cancel.setEnabled(False)

    def _cancel_simulation(self) -> None:
        if self.worker and self.worker.isRunning():
            self.lbl_status.setText("Cancelling simulation...")
            self.worker.cancel()

    @Slot(int, int)
    def _on_progress(self, current: int, total: int) -> None:
        self.progress_bar.setMaximum(max(total, 1))
        self.progress_bar.setValue(current)
        self.lbl_status.setText(f"Processing frame {current} of {total}...")

    @Slot(dict, object)
    def _on_finished(self, stats: dict, result: SimulationResult) -> None:
        self.last_result = result
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText(
            "Simulation complete!"
            if not stats.get("cancelled")
            else "Simulation cancelled."
        )

        # Save files if requested
        if result.event_stream.count:
            if self.chk_save_csv.isChecked():
                csv_p = Path(self.edit_csv_path.text().strip())
                csv_p.parent.mkdir(parents=True, exist_ok=True)
                result.event_stream.save_csv(csv_p)
            if self.chk_save_npz.isChecked():
                npz_p = Path(self.edit_npz_path.text().strip())
                npz_p.parent.mkdir(parents=True, exist_ok=True)
                result.event_stream.save_npz(npz_p)

        # Update stats table
        self._populate_stats_table(stats)
        self.text_log.setText(json.dumps(stats, indent=2))

        # Generate and display preview
        self._render_preview(result)

    @Slot(str)
    def _on_error(self, err_msg: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText("Error occurred.")
        QMessageBox.critical(self, "Simulation Error", err_msg)

    def _populate_stats_table(self, stats: dict) -> None:
        display_keys = [
            ("Frames Processed", stats.get("frames_processed")),
            ("Total Events", stats.get("event_count")),
            ("ON Events (+1)", stats.get("on_events")),
            ("OFF Events (-1)", stats.get("off_events")),
            (
                "ON / OFF Ratio",
                f"{stats.get('on_off_ratio', 0.0):.4f}"
                if stats.get("on_off_ratio")
                else "N/A",
            ),
            ("Input Duration", f"{stats.get('input_duration_s', 0.0):.4f} s"),
            (
                "Active Event Duration",
                f"{stats.get('active_event_duration_s', 0.0):.4f} s",
            ),
            (
                "Event Rate (over input)",
                f"{stats.get('event_rate_over_input_hz', 0.0):.1f} Hz",
            ),
            (
                "Event Rate (over active)",
                f"{stats.get('event_rate_over_active_hz', 0.0):.1f} Hz",
            ),
            (
                "Monotonic Timestamps",
                "True" if stats.get("monotonic_timestamps") else "False",
            ),
        ]
        self.table_stats.setRowCount(len(display_keys))
        for row, (metric, val) in enumerate(display_keys):
            self.table_stats.setItem(row, 0, QTableWidgetItem(metric))
            self.table_stats.setItem(row, 1, QTableWidgetItem(str(val)))

    def _render_preview(self, result: SimulationResult) -> None:
        try:
            if result.preview_panel is not None:
                panel = result.preview_panel
            else:
                width = int(result.statistics.get("width") or 320)
                height = int(result.statistics.get("height") or 240)
                frame = np.full((height, width), 128, dtype=np.uint8)
                renderer = EventVideoRenderer(
                    width, height, SimulatorConfig().visualization
                )
                if result.events.size:
                    t_end = int(result.events["timestamp_us"][-1])
                    t_start = max(0, t_end - self.spin_accum.value())
                    recent_events = result.events[
                        result.events["timestamp_us"] >= t_start
                    ]
                    valid = (recent_events["x"] < width) & (recent_events["y"] < height)
                    if np.any(valid):
                        renderer.add(recent_events[valid], frame, t_end)
                panel = renderer.render_combined_panel(frame)

            # Convert BGR OpenCV image to QPixmap
            rgb_panel = cv.cvtColor(panel, cv.COLOR_BGR2RGB)
            h, w, ch = rgb_panel.shape
            bytes_per_line = ch * w
            qimg = QtGui.QImage(
                rgb_panel.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888
            )
            pixmap = QtGui.QPixmap.fromImage(qimg)
            self.lbl_preview_image.setPixmap(
                pixmap.scaled(
                    self.lbl_preview_image.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
            )
            self.tabs.setCurrentIndex(1)  # switch to preview tab
        except Exception as exc:  # noqa: BLE001
            self.lbl_preview_image.setText(f"Preview rendering error: {exc}")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if "-h" in args or "--help" in args:
        print(
            "Usage: evsim-gui\n\nLaunch the PySide6 desktop GUI for the Python Event Camera Simulator."
        )
        return 0
    app = QApplication.instance()
    if app is None:
        app = QApplication([sys.argv[0]] + list(args))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
