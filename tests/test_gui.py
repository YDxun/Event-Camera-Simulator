import os
from pathlib import Path

import pytest

# Ensure PySide6 runs headlessly in testing/CI
os.environ["QT_QPA_PLATFORM"] = "offscreen"

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from evsim.gui import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_gui_window_initialization(qapp):
    window = MainWindow()
    assert window.windowTitle() == "Event Camera Simulator (evsim)"
    assert window.combo_preset.currentText() == "Realistic"
    assert window.spin_pos_th.value() == 0.20
    assert window.spin_neg_th.value() == 0.15


def test_gui_preset_switching(qapp):
    window = MainWindow()
    window.combo_preset.setCurrentText("Ideal")
    assert window.spin_pos_th.value() == 0.20
    assert window.spin_neg_th.value() == 0.20
    assert window.spin_refractory.value() == 0
    assert not window.chk_threshold_var.isChecked()

    window.combo_preset.setCurrentText("Realistic")
    assert window.spin_pos_th.value() == 0.20
    assert window.spin_neg_th.value() == 0.15
    assert window.spin_refractory.value() == 100
    assert window.chk_threshold_var.isChecked()


def test_gui_build_config(qapp):
    window = MainWindow()
    window.spin_pos_th.setValue(0.25)
    window.spin_neg_th.setValue(0.18)
    window.spin_ts_res.setValue(5)
    window.spin_refractory.setValue(200)
    window.spin_max_frames.setValue(10)

    cfg = window._build_config()
    assert cfg.sensor.positive_threshold == 0.25
    assert cfg.sensor.negative_threshold == 0.18
    assert cfg.sensor.timestamp_resolution_us == 5
    assert cfg.sensor.refractory_period_us == 200
    assert cfg.runtime.max_frames == 10


def test_gui_stats_table_population(qapp):
    window = MainWindow()
    mock_stats = {
        "frames_processed": 50,
        "event_count": 1234,
        "on_events": 600,
        "off_events": 634,
        "on_off_ratio": 600 / 634,
        "input_duration_s": 0.05,
        "active_event_duration_s": 0.048,
        "event_rate_over_input_hz": 24680.0,
        "event_rate_over_active_hz": 25708.3,
        "monotonic_timestamps": True,
    }
    window._populate_stats_table(mock_stats)
    assert window.table_stats.rowCount() > 0
    # Check that event_count is in table
    found_events = False
    for row in range(window.table_stats.rowCount()):
        if window.table_stats.item(row, 0).text() == "Total Events":
            assert window.table_stats.item(row, 1).text() == "1234"
            found_events = True
    assert found_events
