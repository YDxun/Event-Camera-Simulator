from pathlib import Path

import numpy as np

import cv2 as cv

from evsim.config import SimulatorConfig
from evsim.events import EVENT_DTYPE, EventStream
from evsim.sources import ImageSequenceSource, VideoSource, write_image
from evsim.visualization import EventVideoRenderer


def test_event_stream_npz_and_csv_roundtrip(tmp_path: Path):
    events = np.array(
        [(0, 1, 2, 1), (1000, 3, 4, -1), (2500, 5, 6, 1)],
        dtype=EVENT_DTYPE,
    )
    stream = EventStream(events)
    npz = tmp_path / "events.npz"
    csv = tmp_path / "events.csv"
    stream.save_npz(npz)
    stream.save_csv(csv)
    assert np.array_equal(EventStream.load_npz(npz).events, events)
    assert np.array_equal(EventStream.load_csv(csv).events, events)


def test_image_sequence_natural_order_and_timestamps(tmp_path: Path):
    root = tmp_path / "seq"
    frames_dir = root / "frame"
    frames_dir.mkdir(parents=True)
    for name, value in (("img10.png", 10), ("img2.png", 20), ("img1.png", 30)):
        write_image(frames_dir / name, np.full((6, 7), value, dtype=np.uint8))
    (root / "ts_frame.txt").write_text("100\n200\n300\n", encoding="utf-8")
    source = ImageSequenceSource(root)
    loaded = list(source)
    assert [frame.timestamp_us for frame in loaded] == [100, 200, 300]
    assert loaded[0].image.shape == (6, 7)
    assert source.has_timestamps


def test_video_source_and_renderer(tmp_path: Path):
    video_path = tmp_path / "input.avi"
    writer = cv.VideoWriter(
        str(video_path), cv.VideoWriter_fourcc(*"MJPG"), 30.0, (32, 24), True
    )
    assert writer.isOpened()
    for value in (0, 60, 120):
        gray = np.full((24, 32), value, dtype=np.uint8)
        writer.write(cv.cvtColor(gray, cv.COLOR_GRAY2BGR))
    writer.release()

    source = VideoSource(video_path)
    frames = list(source)
    source.close()
    assert len(frames) == 3
    assert frames[0].image.shape == (24, 32)

    config = SimulatorConfig()
    config.visualization.accumulation_time_us = 1000
    renderer = EventVideoRenderer(32, 24, config.visualization)
    output = renderer.open(tmp_path / "events.avi")
    events = np.array([(500, 10, 10, 1)], dtype=EVENT_DTYPE)
    renderer.add(events, frames[0].image, frames[0].timestamp_us)
    assert renderer.render_count_frame().shape == (24, 32)
    assert renderer.render_representation_panel(frames[0].image).shape == (
        24,
        32 * 4,
        3,
    )
    renderer.maybe_write(frames[1].timestamp_us)
    renderer.close()
    assert output.exists()
    capture = cv.VideoCapture(str(output))
    assert capture.isOpened()
    assert int(capture.get(cv.CAP_PROP_FRAME_COUNT)) >= 1
    capture.release()


def test_summary_reports_active_and_input_event_rates():
    events = np.array([(400_000, 1, 1, 1), (600_000, 1, 1, -1)], dtype=EVENT_DTYPE)
    summary = EventStream(events).summary(input_duration_us=1_000_000)
    assert summary["active_event_duration_s"] == 0.2
    assert summary["input_duration_s"] == 1.0
    assert summary["event_rate_over_active_hz"] == 10.0
    assert summary["event_rate_over_input_hz"] == 2.0
    assert summary["event_rate_hz"] == summary["event_rate_over_active_hz"]
