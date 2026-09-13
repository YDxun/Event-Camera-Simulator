"""High-FPS video and timestamp-aware image-sequence readers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class Frame:
    image: np.ndarray
    timestamp_us: int
    index: int


class FrameSource:
    """Common frame-source interface."""

    width: int
    height: int
    fps: float | None
    total_frames: int | None

    def __iter__(self) -> Iterator[Frame]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _natural_key(path: Path) -> list[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", path.stem)
    ]


def read_image(path: str | Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    """Read images on Windows even when the path contains non-ASCII characters."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, flags)
    except (OSError, ValueError):
        return None


def write_image(path: str | Path, image: np.ndarray) -> bool:
    """Write images on Windows even when the path contains non-ASCII characters."""
    path = Path(path)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        return False
    try:
        encoded.tofile(path)
        return True
    except OSError:
        return False


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    channels = image.shape[2]
    if channels == 1:
        return image[:, :, 0]
    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported channel count: {channels}")

class VideoSource(FrameSource):
    """Reads frames, FPS and resolution from a video via OpenCV."""

    def __init__(self, path: str | Path, fallback_fps: float = 960.0):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Video not found: {self.path}")
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.width <= 0 or self.height <= 0:
            ok, first_frame = self.capture.read()
            if ok and first_frame is not None and first_frame.size > 0:
                self.height, self.width = first_frame.shape[:2]
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if self.width <= 0 or self.height <= 0:
            raise RuntimeError(f"Cannot determine video resolution: {self.path}")
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.fps = fps if fps > 0 else float(fallback_fps)
        declared = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
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
            timestamp_us = int(round((self._index / self.fps) * 1_000_000.0))
            yield Frame(_to_gray(image), timestamp_us, self._index)
            self._index += 1

    def reset(self) -> None:
        self._index = 0
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def close(self) -> None:
        self.capture.release()

class ImageSequenceSource(FrameSource):
    """Reads an image folder, optionally using THU-HSEVI-style ts_frame.txt."""

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

        first = read_image(self.paths[0], cv2.IMREAD_UNCHANGED)
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
            image = read_image(path, cv2.IMREAD_UNCHANGED)
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
            "timestamp_file": str(self._timestamp_file) if self._timestamp_file else None,
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
