"""Accumulated event visualization and illustrative video output.

Visualization Model:
--------------------
Unlike traditional cameras where frames represent integrated radiance over exposure,
event cameras produce sparse, asynchronous event streams. To visualize events:
1. Events are accumulated over a frame-aligned temporal window of duration Delta T_acc
   (default: 10,000 us = 10 ms).
2. Color encoding conventions in BGR (OpenCV format):
   - Background (no events): Dark Gray (25, 25, 25)
   - ON events (polarity = +1): Red (20, 80, 255)
   - OFF events (polarity = -1): Blue (255, 80, 20)
   - Simultaneous ON & OFF within window: Magenta (255, 0, 255)
3. Accumulation affects only visualization displays and video output; the underlying
   (x, y, t, p) event stream maintains exact microsecond timestamps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import cv2 as cv
import numpy as np

from .config import VisualizationConfig


class EventVideoRenderer:
    """Accumulate events and render input/event/overlay panels into video or previews."""

    def __init__(self, width: int, height: int, config: VisualizationConfig):
        self.width = width
        self.height = height
        self.config = config
        self.on_count = np.zeros((height, width), dtype=np.int32)
        self.off_count = np.zeros((height, width), dtype=np.int32)
        self.window_start_us: int | None = None
        self.window_end_us: int | None = None
        self.last_frame: np.ndarray | None = None
        self.writer: cv.VideoWriter | None = None
        self.actual_output_path: Path | None = None
        self.frames_written = 0

    def add(self, events: np.ndarray, frame: np.ndarray, timestamp_us: int) -> None:
        """Accumulate events from a newly processed frame into the spatial histogram."""
        if events.size:
            x = events["x"].astype(np.intp)
            y = events["y"].astype(np.intp)
            linear_index = np.ravel_multi_index((y, x), (self.height, self.width))
            positive = events["polarity"] > 0
            np.add.at(self.on_count.reshape(-1), linear_index[positive], 1)
            np.add.at(self.off_count.reshape(-1), linear_index[~positive], 1)
        if self.window_start_us is None:
            self.window_start_us = int(timestamp_us)
        self.window_end_us = int(timestamp_us)
        self.last_frame = frame

    def maybe_write(self, timestamp_us: int) -> None:
        """Write a video panel frame if the accumulation window duration has elapsed."""
        if self.window_start_us is None:
            return
        elapsed = int(timestamp_us) - self.window_start_us
        if elapsed >= self.config.accumulation_time_us:
            self.flush(timestamp_us)

    def flush(self, timestamp_us: int | None = None) -> None:
        """Flush the accumulated events as a rendered video frame and reset accumulation."""
        if self.last_frame is None or self.writer is None:
            self.clear()
            return
        panel = self.render_combined_panel(self.last_frame)
        self.writer.write(panel)
        self.frames_written += 1
        self.clear()
        if timestamp_us is not None:
            self.window_start_us = int(timestamp_us)
            self.window_end_us = int(timestamp_us)

    def clear(self) -> None:
        """Zero out the event accumulation buffers."""
        self.on_count.fill(0)
        self.off_count.fill(0)
        self.window_start_us = None
        self.window_end_us = None

    def render_event_frame(self) -> np.ndarray:
        """Render a color-coded 2D event image (ON=red, OFF=blue, both=magenta) in BGR."""
        image = np.full((self.height, self.width, 3), 25, dtype=np.uint8)
        image[self.off_count > 0] = (255, 80, 20)
        image[self.on_count > 0] = (20, 80, 255)
        image[(self.on_count > 0) & (self.off_count > 0)] = (255, 0, 255)
        return image

    def render_overlay(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            image = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        elif frame.shape[2] == 3:
            image = frame.copy()
        else:
            image = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
        event_frame = self.render_event_frame()
        mask = (self.on_count > 0) | (self.off_count > 0)
        opacity = float(self.config.overlay_opacity)
        blended = cv.addWeighted(image, 1.0 - opacity, event_frame, opacity, 0.0)
        image[mask] = blended[mask]
        return image

    def render_polarity_frame(self) -> np.ndarray:
        """Render a polarity-coded event frame (ON red, OFF blue)."""
        return self.render_event_frame()

    def render_count_frame(self) -> np.ndarray:
        counts = self.on_count + self.off_count
        return np.clip(counts * 32, 0, 255).astype(np.uint8)

    def render_representation_panel(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            original = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        else:
            original = frame.copy()
        polarity = self.render_polarity_frame()
        count = cv.cvtColor(self.render_count_frame(), cv.COLOR_GRAY2BGR)
        overlay = self.render_overlay(frame)
        panel = np.hstack((original, polarity, count, overlay))
        labels = ("Input", "Polarity events", "Event density", "Overlay")
        for index, label in enumerate(labels):
            cv.putText(
                panel,
                label,
                (index * self.width + 8, 22),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
                cv.LINE_AA,
            )
        return panel

    def render_combined_panel(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            original = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        else:
            original = frame.copy()
        events = self.render_event_frame()
        overlay = self.render_overlay(frame)
        panel = np.hstack((original, events, overlay))
        cv.putText(
            panel,
            "Input",
            (8, 22),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv.LINE_AA,
        )
        cv.putText(
            panel,
            "Events (ON red / OFF blue)",
            (self.width + 8, 22),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv.LINE_AA,
        )
        cv.putText(
            panel,
            "Overlay",
            (2 * self.width + 8, 22),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv.LINE_AA,
        )
        return panel

    def open(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        candidates: list[tuple[Path, int]] = []
        if path.suffix.lower() == ".mp4":
            candidates.append((path, cv.VideoWriter_fourcc(*"mp4v")))
            candidates.append((path, cv.VideoWriter_fourcc(*"avc1")))
        candidates.append((path.with_suffix(".avi"), cv.VideoWriter_fourcc(*"MJPG")))
        panel_size = (self.width * 3, self.height)
        for candidate, fourcc in candidates:
            writer = cv.VideoWriter(
                str(candidate),
                fourcc,
                float(self.config.playback_fps),
                panel_size,
                isColor=True,
            )
            if writer.isOpened():
                self.writer = writer
                self.actual_output_path = candidate
                return candidate
            writer.release()
        raise RuntimeError(f"Could not open a video writer for {path}")

    def close(self) -> None:
        if self.last_frame is not None and self.writer is not None:
            self.flush()
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
