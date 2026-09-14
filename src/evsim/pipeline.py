"""End-to-end simulation pipeline."""

from __future__ import annotations

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
    """Outcome and diagnostics of a simulation run.

    Attributes:
        event_stream: Sorted EventStream containing all generated events.
        statistics: Dictionary of event count, rate, and timing metrics.
        frames_processed: Number of video frames processed.
        elapsed_seconds: Wall-clock duration of the simulation in seconds.
        video_path: Path to the generated visualization video file, if enabled.
    """

    event_stream: EventStream
    statistics: dict[str, Any]
    frames_processed: int
    elapsed_seconds: float
    video_path: str | None = None
    preview_panel: np.ndarray | None = None

    @property
    def events(self) -> np.ndarray:
        """Convenience property accessing raw structured NumPy event array."""
        return self.event_stream.events


def simulate_source(
    source: FrameSource,
    config: SimulatorConfig,
    max_frames: int | None = None,
    progress: bool = True,
    renderer: EventVideoRenderer | None = None,
) -> SimulationResult:
    """Execute the event simulation pipeline over an open FrameSource.

    Args:
        source: Active FrameSource instance (video or image sequence).
        config: Simulation parameters and sensor model settings.
        max_frames: Optional frame processing limit.
        progress: Whether to display a tqdm progress bar in terminal.
        renderer: Optional EventVideoRenderer to produce accumulation video.

    Returns:
        SimulationResult containing events, summary statistics, and timing.
    """
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
    preview_panel = None
    if renderer is not None and renderer.last_frame is not None:
        preview_panel = renderer.render_combined_panel(renderer.last_frame)
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
        frames_processed * source.width * source.height / elapsed
        if elapsed > 0
        else 0.0
    )
    result = SimulationResult(
        stream,
        stats,
        frames_processed,
        elapsed,
        preview_panel=preview_panel,
    )
    if renderer is not None and renderer.actual_output_path is not None:
        result.video_path = str(renderer.actual_output_path)
    return result


def simulate_path(path: str | Path, config: SimulatorConfig) -> SimulationResult:
    """Convenience wrapper to open a path and run the simulation pipeline.

    Args:
        path: Path to video file or image sequence directory.
        config: SimulatorConfig controlling sensor, noise, and outputs.

    Returns:
        SimulationResult containing events and statistics.
    """
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
    """Inspect resolution, frame count, FPS, and timestamps for a given path."""
    return inspect_source(
        path,
        fallback_fps=config.input.fallback_fps,
        timestamp_scale_us=config.input.timestamp_scale_us,
    )
