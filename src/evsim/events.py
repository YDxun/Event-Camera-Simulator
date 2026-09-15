"""Event representation, memory layout, and event-stream IO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol, Self

import numpy as np

# Structured NumPy array definition for neuromorphic events (x, y, t, p).
# - timestamp_us: 64-bit integer timestamp in microseconds (overflow safe for millennia).
# - x: 16-bit unsigned pixel x-coordinate [0, width - 1].
# - y: 16-bit unsigned pixel y-coordinate [0, height - 1].
# - polarity: 8-bit signed integer (+1 for ON brightness increase, -1 for OFF decrease).
EVENT_DTYPE = np.dtype(
    [
        ("timestamp_us", np.int64),
        ("x", np.uint16),
        ("y", np.uint16),
        ("polarity", np.int8),
    ]
)


class EventSink(Protocol):
    """Incremental event destination used by memory-bounded simulations."""

    def write(self, events: np.ndarray) -> None: ...

    def close(self) -> None: ...

    def abort(self) -> None: ...


class DiskEventSink:
    """Stream CSV output and assemble NPZ output through disk-backed chunks."""

    def __init__(
        self, csv_path: str | Path | None = None, npz_path: str | Path | None = None
    ) -> None:
        if not csv_path and not npz_path:
            raise ValueError("DiskEventSink requires a CSV or NPZ output path")
        self.csv_path = Path(csv_path) if csv_path else None
        self.npz_path = Path(npz_path) if npz_path else None
        self._csv = None
        self._temporary = TemporaryDirectory(prefix="evsim_events_")
        self._chunk_paths: list[Path] = []
        self._count = 0
        if self.csv_path:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv = self.csv_path.open("w", encoding="utf-8", newline="")
            self._csv.write("timestamp_s,x,y,polarity\n")

    def write(self, events: np.ndarray) -> None:
        if not events.size:
            return
        ordered = sort_events(events)
        if self._csv is not None:
            self._csv.writelines(
                f"{int(row['timestamp_us']) / 1_000_000.0:.9f},"
                f"{int(row['x'])},{int(row['y'])},{int(row['polarity'])}\n"
                for row in ordered
            )
        if self.npz_path:
            path = (
                Path(self._temporary.name) / f"chunk_{len(self._chunk_paths):08d}.npy"
            )
            np.save(path, ordered, allow_pickle=False)
            self._chunk_paths.append(path)
        self._count += int(ordered.size)

    def close(self) -> None:
        try:
            if self._csv is not None:
                self._csv.close()
                self._csv = None
            if self.npz_path:
                self.npz_path.parent.mkdir(parents=True, exist_ok=True)
                if self._count == 0:
                    np.savez_compressed(self.npz_path, events=empty_events())
                    return
                # NPZ is finalized from a memmap, so long runs stay disk-backed.
                # 最终格式不变，但合并过程不会把全部事件重新塞回内存。
                array_path = Path(self._temporary.name) / "events.npy"
                combined = np.lib.format.open_memmap(
                    array_path, mode="w+", dtype=EVENT_DTYPE, shape=(self._count,)
                )
                offset = 0
                for path in self._chunk_paths:
                    chunk = np.load(path, mmap_mode="r", allow_pickle=False)
                    combined[offset : offset + chunk.size] = chunk
                    offset += int(chunk.size)
                combined.flush()
                np.savez_compressed(self.npz_path, events=combined)
                del combined
        finally:
            self._temporary.cleanup()

    def abort(self) -> None:
        if self._csv is not None:
            self._csv.close()
            self._csv = None
        self._temporary.cleanup()
        for path in (self.csv_path, self.npz_path):
            if path is not None:
                path.unlink(missing_ok=True)


def empty_events() -> np.ndarray:
    """Return an empty 1D NumPy structured array with `EVENT_DTYPE`."""
    return np.empty(0, dtype=EVENT_DTYPE)


def sort_events(events: np.ndarray) -> np.ndarray:
    """Deterministically sort events chronologically and spatially.

    Sort order: `timestamp_us` -> `y` -> `x` -> `polarity`.
    NumPy's `np.lexsort` evaluates keys in reverse order (last argument is primary).
    """
    if events.size == 0:
        return events.astype(EVENT_DTYPE, copy=False)
    order = np.lexsort(
        (
            events["polarity"],
            events["x"],
            events["y"],
            events["timestamp_us"],
        )
    )
    return events[order]


@dataclass
class EventStream:
    """A chronologically sortable stream of `(x, y, t, p)` events.

    Provides high-level property access, serialization to compressed NPZ
    and standard CSV, as well as summary rate and polarity statistics.
    """

    events: np.ndarray

    def __post_init__(self) -> None:
        if self.events.dtype != EVENT_DTYPE:
            self.events = self.events.astype(EVENT_DTYPE, copy=False)
        self.events = sort_events(self.events)

    def __len__(self) -> int:
        return int(self.events.size)

    @property
    def count(self) -> int:
        """Total number of events in the stream."""
        return len(self)

    @property
    def timestamp_us(self) -> np.ndarray:
        """1D array of event timestamps in microseconds."""
        return self.events["timestamp_us"]

    @property
    def x(self) -> np.ndarray:
        """1D array of pixel x-coordinates."""
        return self.events["x"]

    @property
    def y(self) -> np.ndarray:
        """1D array of pixel y-coordinates."""
        return self.events["y"]

    @property
    def polarity(self) -> np.ndarray:
        """1D array of event polarities (+1 for ON, -1 for OFF)."""
        return self.events["polarity"]

    @property
    def duration_us(self) -> int:
        """Temporal span from first to last event in microseconds."""
        if self.events.size < 2:
            return 0
        return int(self.timestamp_us[-1] - self.timestamp_us[0])

    def is_monotonic(self) -> bool:
        """Check whether event timestamps are strictly non-decreasing."""
        return bool(np.all(np.diff(self.timestamp_us) >= 0))

    @classmethod
    def concatenate(cls, parts: list[np.ndarray]) -> Self:
        """Combine multiple event structured arrays into a single sorted EventStream."""
        if not parts:
            return cls(empty_events())
        non_empty = [p for p in parts if p.size > 0]
        if not non_empty:
            return cls(empty_events())
        return cls(np.concatenate(non_empty))

    def save_csv(self, path: str | Path) -> None:
        """Save the event stream to CSV formatted as 'timestamp_s,x,y,polarity'.

        Uses buffered streaming chunks to write large streams efficiently
        without allocating massive 2D floating-point intermediate arrays.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("timestamp_s,x,y,polarity\n")
            n = len(self)
            if n == 0:
                return
            chunk_size = 65536
            ts_s = self.timestamp_us / 1_000_000.0
            xs = self.x
            ys = self.y
            pols = self.polarity
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                lines = [
                    f"{t:.9f},{x},{y},{p}\n"
                    for t, x, y, p in zip(
                        ts_s[start:end],
                        xs[start:end],
                        ys[start:end],
                        pols[start:end],
                        strict=True,
                    )
                ]
                f.writelines(lines)

    def save_npz(self, path: str | Path) -> None:
        """Save the event stream to compressed NPZ format with key 'events'."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, events=self.events)

    @classmethod
    def load_npz(cls, path: str | Path) -> Self:
        """Load an EventStream from a compressed NPZ file containing an 'events' array."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            if "events" not in data:
                raise ValueError(f"{path} does not contain an 'events' array")
            return cls(data["events"].astype(EVENT_DTYPE, copy=False))

    @classmethod
    def load_csv(cls, path: str | Path) -> Self:
        """Load an EventStream from a CSV file with 'timestamp_s,x,y,polarity' header."""
        path = Path(path)
        table = np.genfromtxt(
            path, delimiter=",", names=True, dtype=None, encoding="utf-8"
        )
        events = np.empty(np.size(table), dtype=EVENT_DTYPE)
        events["timestamp_us"] = np.rint(table["timestamp_s"] * 1_000_000.0).astype(
            np.int64
        )
        events["x"] = table["x"].astype(np.uint16)
        events["y"] = table["y"].astype(np.uint16)
        events["polarity"] = table["polarity"].astype(np.int8)
        return cls(events)

    def summary(self, input_duration_us: int | None = None) -> dict[str, Any]:
        """Return summary statistics.

        ``event_rate_hz`` is retained as an alias of the active-duration rate.
        New code should use the explicit ``event_rate_over_active_hz`` and,
        when input duration is available, ``event_rate_over_input_hz``.
        """

        count = len(self)
        if count == 0:
            return {
                "event_count": 0,
                "on_events": 0,
                "off_events": 0,
                "active_event_duration_s": 0.0,
                "input_duration_s": (
                    input_duration_us / 1_000_000.0
                    if input_duration_us is not None
                    else None
                ),
                "event_rate_over_active_hz": 0.0,
                "event_rate_over_input_hz": 0.0,
                "event_rate_hz": 0.0,
                "event_rate_definition": "active_duration_alias",
                "monotonic_timestamps": True,
            }
        ts = self.timestamp_us
        active_duration_us = int(ts[-1] - ts[0])
        active_duration_s = active_duration_us / 1_000_000.0
        input_duration_s = (
            input_duration_us / 1_000_000.0
            if input_duration_us is not None
            else active_duration_s
        )
        on_count = int(np.count_nonzero(self.polarity > 0))
        off_count = int(np.count_nonzero(self.polarity < 0))
        active_rate = count / active_duration_s if active_duration_s > 0 else 0.0
        input_rate = count / input_duration_s if input_duration_s > 0 else 0.0
        return {
            "event_count": count,
            "on_events": on_count,
            "off_events": off_count,
            "on_fraction": on_count / count,
            "off_fraction": off_count / count,
            "timestamp_min_us": int(ts[0]),
            "timestamp_max_us": int(ts[-1]),
            "active_event_duration_s": active_duration_s,
            "input_duration_s": input_duration_s,
            "event_rate_over_active_hz": active_rate,
            "event_rate_over_input_hz": input_rate,
            "event_rate_hz": active_rate,
            "event_rate_definition": "active_duration_alias",
            "monotonic_timestamps": self.is_monotonic(),
            "x_min": int(self.x.min()),
            "x_max": int(self.x.max()),
            "y_min": int(self.y.min()),
            "y_max": int(self.y.max()),
        }
