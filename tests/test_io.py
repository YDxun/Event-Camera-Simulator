import stat
import tempfile
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import app
import av
import cv2 as cv
import numpy as np
import pytest

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


def test_invalid_timestamp_file_is_strict_unless_fallback_enabled(tmp_path: Path):
    root = tmp_path / "seq"
    root.mkdir()
    for index in range(3):
        write_image(root / f"{index}.png", np.zeros((2, 2), dtype=np.uint8))
    (root / "ts_frame.txt").write_text("100\n90\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid timestamp file"):
        ImageSequenceSource(root)
    source = ImageSequenceSource(root, allow_timestamp_fallback=True)
    assert source.timestamp_source == "fps"
    assert "FPS fallback" in source.timestamp_warning


def test_video_source_uses_container_pts(tmp_path: Path):
    path = tmp_path / "vfr.mkv"
    container = av.open(str(path), "w")
    stream = container.add_stream("ffv1", rate=1000)
    stream.width = 8
    stream.height = 8
    stream.pix_fmt = "gray"
    for pts in (0, 10, 35):
        frame = av.VideoFrame.from_ndarray(
            np.full((8, 8), pts, dtype=np.uint8), format="gray"
        )
        frame.pts = pts
        frame.time_base = Fraction(1, 1000)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    source = VideoSource(path)
    timestamps = [frame.timestamp_us for frame in source]
    source.close()
    assert timestamps == [0, 10_000, 35_000]
    assert source.timestamp_source == "pts"


def test_video_source_falls_back_when_frame_pts_is_missing():
    class Container:
        def __init__(self):
            self.frames = [
                av.VideoFrame.from_ndarray(np.zeros((2, 2), dtype=np.uint8), "gray")
                for _ in range(2)
            ]

        def seek(self, *_args, **_kwargs):
            return None

        def decode(self, _stream):
            return iter(self.frames)

    source = VideoSource.__new__(VideoSource)
    source.container = Container()
    source.stream = object()
    source.width = 2
    source.height = 2
    source.fps = 100.0
    source.total_frames = 2
    source.timestamp_source = "pts"
    source.timestamp_warning = None
    source._index = 0
    timestamps = [frame.timestamp_us for frame in source]
    assert timestamps == [0, 10_000]
    assert source.timestamp_source == "pts_with_fps_fallback"
    assert "no PTS" in source.timestamp_warning


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


def test_zip_rejects_symbolic_links_and_expansion_limit(tmp_path: Path, monkeypatch):
    link_archive = tmp_path / "link.zip"
    info = ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(link_archive, "w") as archive:
        archive.writestr(info, "target")
    with pytest.raises(ValueError, match="symbolic links"):
        app._safe_extract_zip(link_archive, tmp_path / "links")

    large_archive = tmp_path / "large.zip"
    with ZipFile(large_archive, "w") as archive:
        archive.writestr("large.bin", b"12345")
    monkeypatch.setattr(app, "MAX_ZIP_TOTAL_BYTES", 4)
    with pytest.raises(ValueError, match="2 GiB safety limit"):
        app._safe_extract_zip(large_archive, tmp_path / "large")


def test_replacing_streamlit_workdir_cleans_previous_directory(tmp_path: Path):
    temporary = tempfile.TemporaryDirectory(dir=tmp_path)
    path = Path(temporary.name)
    state = {"_workdir": temporary}
    app._clear_previous_workdir(state)
    assert "_workdir" not in state
    assert not path.exists()
