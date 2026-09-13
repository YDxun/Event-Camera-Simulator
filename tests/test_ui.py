from pathlib import Path
from zipfile import ZipFile

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

import app


def test_ui_config_builds_expected_non_ideal_parameters():
    config = app._build_config(
        preset="Enhanced",
        positive_threshold=0.25,
        negative_threshold=0.15,
        timestamp_resolution_us=10,
        refractory_period_us=250,
        threshold_variation=True,
        background_activity=True,
        linearization=False,
        accumulation_time_us=5_000,
        max_frames=12,
    )
    assert config.sensor.positive_threshold == 0.25
    assert config.sensor.negative_threshold == 0.15
    assert config.sensor.timestamp_resolution_us == 10
    assert config.sensor.refractory_period_us == 250
    assert config.noise.enable_threshold_variation
    assert config.noise.background_rate_hz == 0.05
    assert config.runtime.max_frames == 12


def test_ui_zip_extraction_rejects_path_traversal(tmp_path: Path):
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError):
        app._safe_extract_zip(archive, tmp_path / "extract")


def test_ui_small_simulation_runs_end_to_end():
    test_app = AppTest.from_file("app.py").run(timeout=30)
    test_app.number_input[3].set_value(5)
    test_app.button[0].click()
    test_app.run(timeout=60)
    assert not test_app.exception
    assert "Result Preview" in [item.value for item in test_app.subheader]
    assert any(metric.label == "Events" for metric in test_app.metric)
