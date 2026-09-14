"""End-to-end simulation pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import SimulatorConfig
from .events import EventSink, EventStream, empty_events
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
    retain_events: bool = True,
    event_sink: EventSink | None = None,
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

    bar = None
    completed = False
    first_event_us: int | None = None
    last_event_us: int | None = None
    event_count = 0
    on_events = 0
    off_events = 0
    x_min: int | None = None
    x_max: int | None = None
    y_min: int | None = None
    y_max: int | None = None
    preview_panel = None
    sink_finalized = False
    try:
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
        if progress:
            from tqdm import tqdm

            bar = tqdm(total=limit or total, desc="Simulating", unit="frame")
            bar.update(1)

        for frame in iterator:
            if limit is not None and frames_processed >= limit:
                break
            pair = simulator.process(frame.image, frame.timestamp_us)
            events = pair.events
            last_timestamp_us = frame.timestamp_us
            if events.size:
                event_count += int(events.size)
                on_events += int(np.count_nonzero(events["polarity"] > 0))
                off_events += int(np.count_nonzero(events["polarity"] < 0))
                chunk_x_min = int(events["x"].min())
                chunk_x_max = int(events["x"].max())
                chunk_y_min = int(events["y"].min())
                chunk_y_max = int(events["y"].max())
                x_min = chunk_x_min if x_min is None else min(x_min, chunk_x_min)
                x_max = chunk_x_max if x_max is None else max(x_max, chunk_x_max)
                y_min = chunk_y_min if y_min is None else min(y_min, chunk_y_min)
                y_max = chunk_y_max if y_max is None else max(y_max, chunk_y_max)
                first_event_us = (
                    int(events["timestamp_us"][0])
                    if first_event_us is None
                    else min(first_event_us, int(events["timestamp_us"][0]))
                )
                last_event_us = (
                    int(events["timestamp_us"][-1])
                    if last_event_us is None
                    else max(last_event_us, int(events["timestamp_us"][-1]))
                )
            if retain_events:
                parts.append(events)
            if event_sink is not None:
                event_sink.write(events)
            if renderer is not None:
                renderer.add(events, frame.image, frame.timestamp_us)
                renderer.maybe_write(frame.timestamp_us)
            frames_processed += 1
            if bar is not None:
                bar.update(1)
        if renderer is not None and renderer.last_frame is not None:
            preview_panel = renderer.render_combined_panel(renderer.last_frame)
        if event_sink is not None:
            event_sink.close()
            sink_finalized = True
        completed = True
    except Exception:
        if event_sink is not None and not sink_finalized:
            event_sink.abort()
        raise
    finally:
        if bar is not None:
            bar.close()
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                if completed:
                    raise

    events = np.concatenate(parts) if parts else empty_events()
    stream = EventStream(events)
    elapsed = time.perf_counter() - started
    input_duration_us = last_timestamp_us - first_timestamp_us
    if retain_events:
        stats = stream.summary(input_duration_us=input_duration_us)
    elif event_count == 0:
        stats = EventStream(empty_events()).summary(input_duration_us=input_duration_us)
    else:
        active_duration_s = (last_event_us - first_event_us) / 1_000_000.0
        input_duration_s = input_duration_us / 1_000_000.0
        stats = {
            "event_count": event_count,
            "on_events": on_events,
            "off_events": off_events,
            "on_fraction": on_events / event_count,
            "off_fraction": off_events / event_count,
            "timestamp_min_us": first_event_us,
            "timestamp_max_us": last_event_us,
            "active_event_duration_s": active_duration_s,
            "input_duration_s": input_duration_s,
            "event_rate_over_active_hz": (
                event_count / active_duration_s if active_duration_s > 0 else 0.0
            ),
            "event_rate_over_input_hz": (
                event_count / input_duration_s if input_duration_s > 0 else 0.0
            ),
            "event_rate_hz": event_count / active_duration_s
            if active_duration_s > 0
            else 0.0,
            "event_rate_definition": "active_duration_alias",
            "monotonic_timestamps": True,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        }
    stats["frames_processed"] = frames_processed
    stats["width"] = source.width
    stats["height"] = source.height
    stats["timestamp_source"] = getattr(source, "timestamp_source", "unknown")
    stats["timestamp_warning"] = getattr(source, "timestamp_warning", None)
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
        allow_timestamp_fallback=config.input.allow_timestamp_fallback,
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
        allow_timestamp_fallback=config.input.allow_timestamp_fallback,
    )
