"""End-to-end simulation pipeline."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import SimulatorConfig
from .events import EventStream, empty_events
from .simulator import EventSimulator
from .sources import FrameSource, inspect_source, open_source
from .visualization import EventVideoRenderer


@dataclass
class SimulationResult:
    event_stream: EventStream
    statistics: dict[str, Any]
    frames_processed: int
    elapsed_seconds: float
    video_path: str | None = None

    @property
    def events(self) -> np.ndarray:
        return self.event_stream.events


def simulate_source(
    source: FrameSource,
    config: SimulatorConfig,
    max_frames: int | None = None,
    progress: bool = True,
    renderer: EventVideoRenderer | None = None,
) -> SimulationResult:
    simulator = EventSimulator(config)
    parts: list[np.ndarray] = []
    frames_processed = 0
    started = time.perf_counter()

    iterator = iter(source)
    first = next(iterator, None)
    if first is None:
        raise RuntimeError("Input contains no frames")
    simulator.initialize(first.image, first.timestamp_us)
    frames_processed = 1
    first_timestamp_us = first.timestamp_us
    last_timestamp_us = first.timestamp_us
    if renderer is not None:
        renderer.add(empty_events(), first.image, first.timestamp_us)

    limit = max_frames if max_frames and max_frames > 0 else None
    total = source.total_frames if source.total_frames else None
    bar = None
    if progress:
        try:
            from tqdm import tqdm

            bar = tqdm(total=limit or total, desc="Simulating", unit="frame")
            bar.update(1)
        except ImportError:
            bar = None

    for frame in iterator:
        if limit is not None and frames_processed >= limit:
            break
        pair = simulator.process(frame.image, frame.timestamp_us)
        last_timestamp_us = frame.timestamp_us
        parts.append(pair.events)
        if renderer is not None:
            renderer.add(pair.events, frame.image, frame.timestamp_us)
            renderer.maybe_write(frame.timestamp_us)
        frames_processed += 1
        if bar is not None:
            bar.update(1)

    if bar is not None:
        bar.close()
    if renderer is not None:
        renderer.close()

    events = np.concatenate(parts) if parts else empty_events()
    stream = EventStream(events)
    elapsed = time.perf_counter() - started
    stats = stream.summary(input_duration_us=last_timestamp_us - first_timestamp_us)
    stats["frames_processed"] = frames_processed
    stats["width"] = source.width
    stats["height"] = source.height
    stats["processing_fps"] = frames_processed / elapsed if elapsed > 0 else 0.0
    stats["pixel_frames_per_second"] = (
        frames_processed * source.width * source.height / elapsed if elapsed > 0 else 0.0
    )
    result = SimulationResult(stream, stats, frames_processed, elapsed)
    if renderer is not None and renderer.actual_output_path is not None:
        result.video_path = str(renderer.actual_output_path)
    return result


def simulate_path(path: str | Path, config: SimulatorConfig) -> SimulationResult:
    source = open_source(
        path,
        fallback_fps=config.input.fallback_fps,
        timestamp_scale_us=config.input.timestamp_scale_us,
    )
    try:
        return simulate_source(
            source,
            config,
            max_frames=config.runtime.max_frames or None,
            progress=config.runtime.progress,
        )
    finally:
        source.close()


def source_metadata(path: str | Path, config: SimulatorConfig) -> dict[str, Any]:
    return inspect_source(
        path,
        fallback_fps=config.input.fallback_fps,
        timestamp_scale_us=config.input.timestamp_scale_us,
    )
