"""Event representation and event-stream IO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EVENT_DTYPE = np.dtype(
    [
        ("timestamp_us", np.int64),
        ("x", np.uint16),
        ("y", np.uint16),
        ("polarity", np.int8),
    ]
)


def empty_events() -> np.ndarray:
    return np.empty(0, dtype=EVENT_DTYPE)


def sort_events(events: np.ndarray) -> np.ndarray:
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
    """A chronologically sortable stream of ``(x, y, t, p)`` events."""

    events: np.ndarray

    def __post_init__(self) -> None:
        if self.events.dtype != EVENT_DTYPE:
            self.events = self.events.astype(EVENT_DTYPE, copy=False)
        self.events = sort_events(self.events)

    def __len__(self) -> int:
        return int(self.events.size)

    @property
    def timestamp_us(self) -> np.ndarray:
        return self.events["timestamp_us"]

    @property
    def x(self) -> np.ndarray:
        return self.events["x"]

    @property
    def y(self) -> np.ndarray:
        return self.events["y"]

    @property
    def polarity(self) -> np.ndarray:
        return self.events["polarity"]

    @property
    def duration_us(self) -> int:
        if self.events.size < 2:
            return 0
        return int(self.timestamp_us[-1] - self.timestamp_us[0])

    def is_monotonic(self) -> bool:
        return bool(np.all(np.diff(self.timestamp_us) >= 0))

    def save_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = np.column_stack(
            (
                self.timestamp_us.astype(np.float64) / 1_000_000.0,
                self.x.astype(np.int64),
                self.y.astype(np.int64),
                self.polarity.astype(np.int64),
            )
        )
        np.savetxt(
            path,
            columns,
            delimiter=",",
            header="timestamp_s,x,y,polarity",
            comments="",
            fmt=["%.9f", "%d", "%d", "%d"],
        )

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, events=self.events)

    @classmethod
    def load_npz(cls, path: str | Path) -> "EventStream":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            if "events" not in data:
                raise ValueError(f"{path} does not contain an 'events' array")
            return cls(data["events"].astype(EVENT_DTYPE, copy=False))

    @classmethod
    def load_csv(cls, path: str | Path) -> "EventStream":
        path = Path(path)
        table = np.genfromtxt(
            path, delimiter=",", names=True, dtype=None, encoding="utf-8"
        )
        events = np.empty(np.size(table), dtype=EVENT_DTYPE)
        events["timestamp_us"] = np.rint(
            table["timestamp_s"] * 1_000_000.0
        ).astype(np.int64)
        events["x"] = table["x"].astype(np.uint16)
        events["y"] = table["y"].astype(np.uint16)
        events["polarity"] = table["polarity"].astype(np.int8)
        return cls(events)

    def summary(self) -> dict[str, Any]:
        count = len(self)
        if count == 0:
            return {
                "event_count": 0,
                "on_events": 0,
                "off_events": 0,
                "event_rate_hz": 0.0,
                "duration_s": 0.0,
                "monotonic_timestamps": True,
            }
        ts = self.timestamp_us
        duration_us = int(ts[-1] - ts[0])
        duration_s = duration_us / 1_000_000.0
        on_count = int(np.count_nonzero(self.polarity > 0))
        off_count = int(np.count_nonzero(self.polarity < 0))
        return {
            "event_count": count,
            "on_events": on_count,
            "off_events": off_count,
            "on_fraction": on_count / count,
            "off_fraction": off_count / count,
            "timestamp_min_us": int(ts[0]),
            "timestamp_max_us": int(ts[-1]),
            "duration_s": duration_s,
            "event_rate_hz": count / duration_s if duration_s > 0 else 0.0,
            "monotonic_timestamps": self.is_monotonic(),
            "x_min": int(self.x.min()),
            "x_max": int(self.x.max()),
            "y_min": int(self.y.min()),
            "y_max": int(self.y.max()),
        }
