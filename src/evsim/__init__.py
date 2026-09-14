"""Python event-camera simulator package."""

from __future__ import annotations

from .config import SimulatorConfig
from .events import EventStream
from .simulator import EventSimulator

__all__ = ["EventSimulator", "EventStream", "SimulatorConfig"]
__version__ = "1.0.0"
