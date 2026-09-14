"""Python event-camera simulator package."""

from .config import SimulatorConfig
from .events import EventStream
from .simulator import EventSimulator

__all__ = ["EventSimulator", "EventStream", "SimulatorConfig"]
__version__ = "1.0.0"
