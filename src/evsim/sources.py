"""High-FPS video and timestamp-aware image-sequence readers."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import cv2 as cv
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_IMAGE_DECODE_ERRORS = (OSError, ValueError)


@dataclass(frozen=True, slots=True)
class Frame:
    """A single frame from a video or image sequence.

    Attributes:
        image: Grayscale image array of shape (H, W) with dtype uint8.
        timestamp_us: Integer timestamp in microseconds from sequence start.
        index: Zero-based frame sequence number.
    """

    image: np.ndarray
    timestamp_us: int
    index: int


class FrameSource:
    """Abstract base protocol for streaming high-FPS frames into the simulator.

    Subclasses must implement `__iter__` and `reset()`. Context manager
    semantics (`with source:`) ensure resources like file handles or OpenCV
    VideoCapture streams are deterministically released.
    """

    width: int
    height: int
    fps: float | None
    total_frames: int | None

    def __iter__(self) -> Iterator[Frame]:
        """Iterate over frames sequentially in chronological order."""
        raise NotImplementedError

    def reset(self) -> None:
        """Rewind the stream back to the initial frame."""
        raise NotImplementedError

    def close(self) -> None:
        """Release underlying system resources (e.g. video capture handles)."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _natural_key(path: Path) -> list[object]:
    """Extract natural alphanumeric sort key so 'frame_2.png' precedes 'frame_10.png'."""
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", path.stem)
    ]


def read_image(path: str | Path, flags: int = cv.IMREAD_UNCHANGED) -> np.ndarray | None:
    """Read an image file across platforms, supporting paths with non-ASCII characters.

    Standard cv.imread fails on Windows when paths contain unicode characters.
    Using np.fromfile followed by cv.imdecode bypasses the OS path limitations.

    Args:
        path: Path to the image file.
        flags: OpenCV image decoding flags (default: cv.IMREAD_UNCHANGED).

    Returns:
        The decoded image as a NumPy ndarray, or None if reading/decoding fails.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv.imdecode(data, flags)
    except _IMAGE_DECODE_ERRORS:
        return None


def write_image(path: str | Path, image: np.ndarray) -> bool:
    """Write an image file across platforms, supporting paths with non-ASCII characters.

    Args:
        path: Target file path.
        image: Image array to write.

    Returns:
        True if the image was successfully encoded and written, False otherwise.
    """
    path = Path(path)
    ok, encoded = cv.imencode(path.suffix, image)
    if not ok:
        return False
    try:
        encoded.tofile(path)
        return True
    except OSError:
        return False


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert an arbitrary input image (grayscale, BGR, BGRA) to 2D uint8 grayscale.

    Args:
        image: Input image array of shape (H, W), (H, W, 1), (H, W, 3) or (H, W, 4).

    Returns:
        2D single-channel grayscale array of shape (H, W).
    """
    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    channels = image.shape[2]
    if channels == 1:
        return image[:, :, 0]
    if channels == 3:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if channels == 4:
        return cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported channel count: {channels}")


class VideoSource(FrameSource):
    """Reads frames, FPS and resolution from a video file via OpenCV VideoCapture.

    Timestamps are synthesized from frame index and frame rate:
    `timestamp_us = round((index / fps) * 1_000_000.0)`.

    Args:
        path: Path to video file (.mp4, .avi, etc.).
        fallback_fps: FPS to assume if OpenCV reports 0 or invalid FPS.
    """

    def __init__(self, path: str | Path, fallback_fps: float = 960.0):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Video not found: {self.path}")
        self.capture = cv.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")
        self.width = int(self.capture.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        if self.width <= 0 or self.height <= 0:
            ok, first_frame = self.capture.read()
            if ok and first_frame is not None and first_frame.size > 0:
                self.height, self.width = first_frame.shape[:2]
            self.capture.set(cv.CAP_PROP_POS_FRAMES, 0)
        if self.width <= 0 or self.height <= 0:
            raise RuntimeError(f"Cannot determine video resolution: {self.path}")
        fps = float(self.capture.get(cv.CAP_PROP_FPS))
        self.fps = fps if fps > 0 else float(fallback_fps)
        declared = int(self.capture.get(cv.CAP_PROP_FRAME_COUNT))
        self.total_frames = declared if declared > 0 else None
        self._index = 0

    def __iter__(self) -> Iterator[Frame]:
        self.reset()
        while True:
            ok, image = self.capture.read()
            if not ok:
                break
            if image is None or image.size == 0:
                continue
            timestamp_us = round((self._index / self.fps) * 1_000_000.0)
            yield Frame(_to_gray(image), timestamp_us, self._index)
            self._index += 1

    def reset(self) -> None:
        self._index = 0
        self.capture.set(cv.CAP_PROP_POS_FRAMES, 0)

    def close(self) -> None:
        self.capture.release()


class ImageSequenceSource(FrameSource):
    """Reads a folder of images, optionally using THU-HSEVI-style ts_frame.txt timestamps.

    If `ts_frame.txt` is found in the sequence directory or root, real timestamps
    are parsed and converted to microseconds. Otherwise, uniform timestamps
    are synthesized from `fallback_fps`.

    Args:
        path: Path to directory containing images or sequence root.
        fallback_fps: Fallback frame rate if no timestamp file exists.
        timestamp_scale_us: Multiplier to convert raw timestamp units to microseconds.
    """

    def __init__(
        self,
        path: str | Path,
        fallback_fps: float = 960.0,
        timestamp_scale_us: float = 1.0,
    ):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Image sequence not found: {self.path}")
        self.root = self.path
        if self.path.is_dir() and (self.path / "frame").is_dir():
            self.frame_dir = self.path / "frame"
        elif self.path.is_dir():
            self.frame_dir = self.path
        else:
            raise NotADirectoryError(f"Image sequence path is not a directory: {path}")

        self.paths = sorted(
            [
                p
                for p in self.frame_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=_natural_key,
        )
        if not self.paths:
            raise RuntimeError(f"No image files found in: {self.frame_dir}")

        first = read_image(self.paths[0], cv.IMREAD_UNCHANGED)
        if first is None:
            raise RuntimeError(f"Failed to read first image: {self.paths[0]}")
        self.height, self.width = first.shape[:2]
        self.fps = float(fallback_fps)
        self.total_frames = len(self.paths)

        self.timestamps_us = self._load_timestamps(
            fallback_fps=fallback_fps, scale=timestamp_scale_us
        )
        self.has_timestamps = self._timestamp_file is not None
        self._index = 0

    def _find_timestamp_file(self) -> Path | None:
        candidates = [
            self.root / "ts_frame.txt",
            self.frame_dir / "ts_frame.txt",
            self.root.parent / "ts_frame.txt",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def _load_timestamps(self, fallback_fps: float, scale: float) -> np.ndarray:
        self._timestamp_file = self._find_timestamp_file()
        values: list[float] = []
        if self._timestamp_file is not None:
            text = self._timestamp_file.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                if match:
                    values.append(float(match.group(0)))
        if len(values) >= len(self.paths):
            timestamps = np.rint(np.asarray(values[: len(self.paths)]) * scale).astype(
                np.int64
            )
            if np.all(np.diff(timestamps) > 0):
                return timestamps
        self._timestamp_file = None
        if fallback_fps <= 0:
            raise ValueError("fallback_fps must be positive when timestamps are absent")
        index = np.arange(len(self.paths), dtype=np.float64)
        return np.rint(index * (1_000_000.0 / fallback_fps)).astype(np.int64)

    def __iter__(self) -> Iterator[Frame]:
        self.reset()
        for index, path in enumerate(self.paths):
            image = read_image(path, cv.IMREAD_UNCHANGED)
            if image is None:
                raise RuntimeError(f"Failed to read image: {path}")
            yield Frame(_to_gray(image), int(self.timestamps_us[index]), index)

    def reset(self) -> None:
        self._index = 0

    def inspect(self) -> dict[str, object]:
        return {
            "kind": "image_sequence",
            "path": str(self.path),
            "frame_dir": str(self.frame_dir),
            "width": self.width,
            "height": self.height,
            "frames": self.total_frames,
            "fallback_fps": self.fps,
            "has_timestamp_file": self.has_timestamps,
            "timestamp_file": str(self._timestamp_file)
            if self._timestamp_file
            else None,
            "timestamp_start_us": int(self.timestamps_us[0]),
            "timestamp_end_us": int(self.timestamps_us[-1]),
            "duration_s": float(
                (self.timestamps_us[-1] - self.timestamps_us[0]) / 1_000_000.0
            ),
        }


def open_source(
    path: str | Path,
    fallback_fps: float = 960.0,
    timestamp_scale_us: float = 1.0,
) -> FrameSource:
    """Factory function returning the appropriate FrameSource for a file or directory.

    Args:
        path: Path to a video file (.mp4, .avi, etc.) or an image directory.
        fallback_fps: Default frame rate if source does not provide one.
        timestamp_scale_us: Timestamp scaling factor for image sequence timestamps.

    Returns:
        A FrameSource instance (VideoSource or ImageSequenceSource).

    Raises:
        FileNotFoundError: If path does not exist.
    """
    source_path = Path(path)
    if source_path.is_dir():
        return ImageSequenceSource(
            source_path,
            fallback_fps=fallback_fps,
            timestamp_scale_us=timestamp_scale_us,
        )
    if source_path.is_file():
        return VideoSource(source_path, fallback_fps=fallback_fps)
    raise FileNotFoundError(f"Input path not found: {path}")


def inspect_source(
    path: str | Path,
    fallback_fps: float = 960.0,
    timestamp_scale_us: float = 1.0,
) -> dict[str, object]:
    """Inspect input metadata (resolution, frame count, FPS, duration) without full decode.

    Args:
        path: Path to video file or image sequence directory.
        fallback_fps: Fallback FPS if absent.
        timestamp_scale_us: Multiplier to convert raw timestamp units to microseconds.

    Returns:
        Dictionary containing metadata summary.
    """
    source = open_source(path, fallback_fps, timestamp_scale_us)
    try:
        if isinstance(source, ImageSequenceSource):
            return source.inspect()
        return {
            "kind": "video",
            "path": str(Path(path)),
            "width": source.width,
            "height": source.height,
            "frames": source.total_frames,
            "fps": source.fps,
            "duration_s": (
                source.total_frames / source.fps
                if source.total_frames and source.fps
                else None
            ),
        }
    finally:
        source.close()
