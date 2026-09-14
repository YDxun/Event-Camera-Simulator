"""Self-contained synthetic high-FPS demo and video generation."""

from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np


def demo_frame(width: int, height: int, index: int, total: int) -> np.ndarray:
    """Render a synthetic grayscale frame with moving bright/dark disks and background textures.

    The scene combines:
    1. A linear horizontal illumination gradient.
    2. A vertical spatial sinusoid texture.
    3. A bright disk translating across the frame with sinusoidal vertical oscillation.
    4. A dark disk translating in the opposite direction.
    5. A global periodic intensity modulation (flashing illumination).

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        index: Current frame index [0, total - 1].
        total: Total number of frames in the sequence.

    Returns:
        2D uint8 image array of shape (height, width).
    """
    x = np.arange(width, dtype=np.float32)[None, :]
    y = np.arange(height, dtype=np.float32)[:, None]
    phase = index / max(total - 1, 1)
    frame = 45.0 + 55.0 * x / max(width - 1, 1)
    frame = np.broadcast_to(frame, (height, width)).astype(np.float32).copy()
    frame += 20.0 * np.sin(2.0 * np.pi * y / max(height, 1))
    radius = max(12.0, min(width, height) * 0.16)
    cx = radius + phase * (width - 2.0 * radius)
    cy = height * (0.50 + 0.18 * np.sin(2.0 * np.pi * phase))
    disk = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
    frame[disk] = 235.0
    cx2 = width - cx
    cy2 = height * (0.50 - 0.18 * np.sin(2.0 * np.pi * phase))
    dark = (x - cx2) ** 2 + (y - cy2) ** 2 <= (radius * 0.75) ** 2
    frame[dark] = 25.0
    ramp = 0.92 + 0.08 * np.sin(2.0 * np.pi * 3.0 * phase)
    frame *= ramp
    return np.clip(frame, 0.0, 255.0).astype(np.uint8)


def write_synthetic_video(
    path: str | Path,
    fps: float,
    seconds: float,
    width: int,
    height: int,
) -> Path:
    """Generate and write a synthetic high-FPS video file to disk.

    Args:
        path: Target file path (.avi).
        fps: Frame rate in Hertz (e.g. 960.0).
        seconds: Duration of the video in seconds.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Path to the saved video file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    total = round(fps * seconds)
    writer = cv.VideoWriter(
        str(path),
        cv.VideoWriter_fourcc(*"MJPG"),
        float(fps),
        (width, height),
        isColor=True,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create demo video: {path}")
    for index in range(total):
        gray = demo_frame(width, height, index, total)
        writer.write(cv.cvtColor(gray, cv.COLOR_GRAY2BGR))
    writer.release()
    return path


def run_demo(
    output_dir: str | Path,
    fps: float = 960.0,
    seconds: float = 1.0,
    width: int = 320,
    height: int = 240,
) -> dict[str, object]:
    """Execute end-to-end self-contained 960 FPS demo generating video, events, and metrics.

    Args:
        output_dir: Destination directory for all output artifacts.
        fps: Frame rate of synthetic input (default: 960.0).
        seconds: Duration in seconds (default: 1.0).
        width: Video width (default: 320).
        height: Video height (default: 240).

    Returns:
        Dictionary of summary statistics and artifact locations.
    """
    from .config import SimulatorConfig
    from .pipeline import simulate_source
    from .sources import open_source
    from .visualization import EventVideoRenderer

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_video = write_synthetic_video(
        output / "input_960fps.avi", fps, seconds, width, height
    )

    config = SimulatorConfig()
    config.input.fallback_fps = fps
    config.sensor.positive_threshold = 0.20
    config.sensor.negative_threshold = 0.15
    config.sensor.refractory_period_us = 100
    config.noise.enable_threshold_variation = True
    config.noise.threshold_sigma = 0.02
    config.noise.background_rate_hz = 0.02
    config.visualization.accumulation_time_us = 5_000
    config.visualization.output_video = str(output / "event_video.avi")
    config.output.csv_path = str(output / "events.csv")
    config.output.npz_path = str(output / "events.npz")
    config.output.statistics_path = str(output / "statistics.json")
    config.runtime.progress = True
    config.save(output / "config_used.json")

    source = open_source(input_video, fallback_fps=fps)
    try:
        renderer = EventVideoRenderer(width, height, config.visualization)
        renderer.open(config.visualization.output_video)
        result = simulate_source(source, config, progress=True, renderer=renderer)
    finally:
        source.close()

    result.event_stream.save_csv(config.output.csv_path)
    result.event_stream.save_npz(config.output.npz_path)
    stats = dict(result.statistics)
    stats["input_video"] = str(input_video)
    stats["event_video"] = result.video_path or config.visualization.output_video
    stats["config"] = config.to_dict()
    Path(config.output.statistics_path).write_text(
        __import__("json").dumps(stats, indent=2) + "\n", encoding="utf-8"
    )

    total = max(int(fps * seconds), 1)
    final_frame = demo_frame(width, height, total - 1, total)
    preview = EventVideoRenderer(width, height, config.visualization)
    preview.add(result.events, final_frame, total * 1_000_000)
    cv.imwrite(
        str(output / "event_preview.png"), preview.render_combined_panel(final_frame)
    )
    return stats
