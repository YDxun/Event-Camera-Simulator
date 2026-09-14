from pathlib import Path
from zipfile import ZipFile

import cv2 as cv
import numpy as np
import pytest

pytest.importorskip("streamlit")
import app
from streamlit.testing.v1 import AppTest

from evsim.sources import VideoSource


class _ZeroSizeMetadataCapture:
    def __init__(self, capture):
        self._capture = capture

    def __getattr__(self, name):
        return getattr(self._capture, name)

    def get(self, prop):
        if prop in (cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT):
            return 0
        return self._capture.get(prop)


def test_video_source_falls_back_to_first_decoded_frame(tmp_path: Path, monkeypatch):
    video_path = tmp_path / "input.avi"
    writer = cv.VideoWriter(str(video_path), cv.VideoWriter_fourcc(*"MJPG"), 30.0, (32, 24), True)
    assert writer.isOpened()
    writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()

    real_capture = cv.VideoCapture
    monkeypatch.setattr(
        cv,
        "VideoCapture",
        lambda path: _ZeroSizeMetadataCapture(real_capture(path)),
    )
    source = VideoSource(video_path)
    assert source.width == 32
    assert source.height == 24
    source.close()


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
        fallback_fps=480.0,
    )
    assert config.sensor.positive_threshold == 0.25
    assert config.sensor.negative_threshold == 0.15
    assert config.sensor.timestamp_resolution_us == 10
    assert config.sensor.refractory_period_us == 250
    assert config.noise.enable_threshold_variation
    assert config.noise.background_rate_hz == 0.05
    assert config.runtime.max_frames == 12
    assert config.input.fallback_fps == 480.0


def test_ui_zip_extraction_rejects_path_traversal(tmp_path: Path):
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError):
        app._safe_extract_zip(archive, tmp_path / "extract")


def test_ui_small_simulation_runs_end_to_end():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    test_app = AppTest.from_file(str(app_path)).run(timeout=30)
    test_app.number_input[3].set_value(5)
    test_app.button[0].click()
    test_app.run(timeout=120)
    assert not test_app.exception
    assert "Result Preview" in [item.value for item in test_app.subheader]
    assert any(metric.label == "Events" for metric in test_app.metric)
