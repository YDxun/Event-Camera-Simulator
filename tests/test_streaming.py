from pathlib import Path

import numpy as np
import pytest

from evsim.config import SimulatorConfig
from evsim.events import DiskEventSink, EventStream
from evsim.pipeline import simulate_source
from evsim.simulator import EventSimulator
from evsim.sources import Frame, FrameSource


class ArraySource(FrameSource):
    def __init__(self, frames: list[np.ndarray]):
        self.frames = frames
        self.width = frames[0].shape[1]
        self.height = frames[0].shape[0]
        self.total_frames = len(frames)
        self.fps = 1000.0
        self.timestamp_source = "test"
        self.timestamp_warning = None

    def __iter__(self):
        for index, image in enumerate(self.frames):
            yield Frame(image, index * 1000, index)

    def reset(self):
        return None


def test_streaming_outputs_match_retained_mode(tmp_path: Path):
    frames = [np.full((4, 5), value, dtype=np.uint8) for value in (20, 80, 160, 40)]
    config = SimulatorConfig()
    retained = simulate_source(ArraySource(frames), config, progress=False)
    csv_path = tmp_path / "events.csv"
    npz_path = tmp_path / "events.npz"
    streamed = simulate_source(
        ArraySource(frames),
        config,
        progress=False,
        retain_events=False,
        event_sink=DiskEventSink(csv_path, npz_path),
    )

    assert streamed.events.size == 0
    assert np.array_equal(EventStream.load_csv(csv_path).events, retained.events)
    assert np.array_equal(EventStream.load_npz(npz_path).events, retained.events)
    for key in retained.statistics.keys() - {
        "processing_fps",
        "pixel_frames_per_second",
    }:
        assert streamed.statistics[key] == retained.statistics[key]


def test_candidate_event_limit_fails_before_expansion():
    config = SimulatorConfig()
    config.sensor.positive_threshold = 0.01
    config.runtime.max_candidate_events_per_frame = 1
    simulator = EventSimulator(config)
    simulator.initialize(np.zeros((1, 1), dtype=np.uint8), 0)
    with pytest.raises(RuntimeError, match="max_candidate_events_per_frame"):
        simulator.process(np.full((1, 1), 255, dtype=np.uint8), 1000)


def test_streaming_zero_events_writes_valid_npz(tmp_path: Path):
    path = tmp_path / "empty.npz"
    frames = [np.zeros((2, 2), dtype=np.uint8)] * 2
    result = simulate_source(
        ArraySource(frames),
        SimulatorConfig(),
        progress=False,
        retain_events=False,
        event_sink=DiskEventSink(npz_path=path),
    )
    assert result.statistics["event_count"] == 0
    assert EventStream.load_npz(path).count == 0


def test_renderer_is_closed_when_simulation_fails():
    class FailingRenderer:
        last_frame = None
        actual_output_path = None

        def __init__(self):
            self.closed = False

        def add(self, *_args):
            raise RuntimeError("render failed")

        def close(self):
            self.closed = True

    renderer = FailingRenderer()
    frames = [np.zeros((2, 2), dtype=np.uint8)]
    with pytest.raises(RuntimeError, match="render failed"):
        simulate_source(ArraySource(frames), SimulatorConfig(), renderer=renderer)
    assert renderer.closed
