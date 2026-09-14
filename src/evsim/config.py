"""Configuration model and validation for the event-camera simulator."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class InputConfig:
    """Settings for input video or image-sequence loading and preprocessing.

    Attributes:
        fallback_fps: Frame rate assumed if not specified in video/image metadata.
        linearize: Whether to invert gamma encoding (approximate radiometric linear radiance).
        gamma: Exponent for gamma linearization (typically 2.2 for sRGB).
        bit_depth: Sensor bit depth (8 or 16), determines normalization scale (2^N - 1).
        timestamp_scale_us: Scaling factor to convert raw image timestamps into microseconds.
    """

    fallback_fps: float = 960.0
    linearize: bool = False
    gamma: float = 2.2
    bit_depth: int = 8
    timestamp_scale_us: float = 1.0


@dataclass
class SensorConfig:
    """Neuromorphic sensor analog pixel circuit parameters.

    Attributes:
        positive_threshold: Log-intensity change threshold C+ to fire an ON event.
        negative_threshold: Log-intensity change threshold C- to fire an OFF event.
        log_epsilon: Small constant added before log to avoid log(0) and model dark current.
        timestamp_resolution_us: Temporal quantization bin size in microseconds (>= 1).
        refractory_period_us: Dead time in microseconds after an event where no new event can fire.
    """

    positive_threshold: float = 0.20
    negative_threshold: float = 0.20
    log_epsilon: float = 1e-3
    timestamp_resolution_us: int = 1
    refractory_period_us: int = 0


@dataclass
class SimulationConfig:
    """Algorithm execution and interpolation parameters.

    Attributes:
        interpolation: 'linear' for sub-frame event timestamp interpolation, or 'none'.
        backend: 'vectorized' (NumPy broadcasted, fast) or 'loop' (pixel-by-pixel reference).
    """

    interpolation: str = "linear"
    backend: str = "vectorized"


@dataclass
class NoiseConfig:
    """Stochastic noise parameters simulating physical sensor non-idealities.

    Attributes:
        enable_threshold_variation: Whether to apply per-pixel Gaussian threshold mismatch.
        threshold_sigma: Standard deviation of pixel threshold variation (Gaussian).
        background_rate_hz: Mean Poisson background event rate per pixel in Hertz.
        random_seed: Seed for reproducible noise generation.
    """

    enable_threshold_variation: bool = False
    threshold_sigma: float = 0.03
    background_rate_hz: float = 0.0
    random_seed: int = 42


@dataclass
class VisualizationConfig:
    """Accumulation and rendering options for video and preview panels.

    Attributes:
        accumulation_time_us: Sliding temporal window duration in microseconds for event frames.
        playback_fps: Playback frame rate of rendered output video.
        overlay_opacity: Alpha blending factor [0.0, 1.0] for overlaying events onto grayscale.
        output_video: Path to save rendered video (.mp4 or .avi).
        display: Whether to display interactive UI window.
    """

    accumulation_time_us: int = 10_000
    playback_fps: float = 30.0
    overlay_opacity: float = 0.65
    output_video: str = ""
    display: bool = False


@dataclass
class OutputConfig:
    """Paths for saving simulation outputs and event streams.

    Attributes:
        csv_path: Path to write text CSV event stream ('timestamp_s,x,y,polarity').
        npz_path: Path to write compressed NumPy NPZ event archive.
        statistics_path: Path to write JSON summary statistics.
    """

    csv_path: str = ""
    npz_path: str = ""
    statistics_path: str = ""


@dataclass
class RuntimeConfig:
    """Runtime execution limits and diagnostics.

    Attributes:
        max_frames: Stop processing after this many frames (0 = process all frames).
        progress: Whether to show a progress bar in terminal.
    """

    max_frames: int = 0
    progress: bool = True


@dataclass
class SimulatorConfig:
    """Root configuration tree combining all simulation sub-configurations."""

    input: InputConfig | None = None
    sensor: SensorConfig | None = None
    simulation: SimulationConfig | None = None
    noise: NoiseConfig | None = None
    visualization: VisualizationConfig | None = None
    output: OutputConfig | None = None
    runtime: RuntimeConfig | None = None

    def __post_init__(self) -> None:
        self.input = self.input or InputConfig()
        self.sensor = self.sensor or SensorConfig()
        self.simulation = self.simulation or SimulationConfig()
        self.noise = self.noise or NoiseConfig()
        self.visualization = self.visualization or VisualizationConfig()
        self.output = self.output or OutputConfig()
        self.runtime = self.runtime or RuntimeConfig()
        self.validate()

    def validate(self) -> None:
        if self.input.fallback_fps <= 0:
            raise ValueError("input.fallback_fps must be positive")
        if self.input.gamma <= 0:
            raise ValueError("input.gamma must be positive")
        if self.input.bit_depth not in (8, 16):
            raise ValueError("input.bit_depth must be 8 or 16")
        if self.input.timestamp_scale_us <= 0:
            raise ValueError("input.timestamp_scale_us must be positive")
        if self.sensor.positive_threshold <= 0:
            raise ValueError("sensor.positive_threshold must be positive")
        if self.sensor.negative_threshold <= 0:
            raise ValueError("sensor.negative_threshold must be positive")
        if self.sensor.log_epsilon <= 0:
            raise ValueError("sensor.log_epsilon must be positive")
        if self.sensor.timestamp_resolution_us < 1:
            raise ValueError("sensor.timestamp_resolution_us must be >= 1")
        if self.sensor.refractory_period_us < 0:
            raise ValueError("sensor.refractory_period_us must be >= 0")
        if self.simulation.interpolation not in {"linear", "none"}:
            raise ValueError("simulation.interpolation must be 'linear' or 'none'")
        if self.simulation.backend not in {"vectorized", "loop"}:
            raise ValueError("simulation.backend must be 'vectorized' or 'loop'")
        if self.noise.threshold_sigma < 0:
            raise ValueError("noise.threshold_sigma must be >= 0")
        if self.noise.background_rate_hz < 0:
            raise ValueError("noise.background_rate_hz must be >= 0")
        if self.visualization.accumulation_time_us <= 0:
            raise ValueError("visualization.accumulation_time_us must be positive")
        if self.visualization.playback_fps <= 0:
            raise ValueError("visualization.playback_fps must be positive")
        if not 0.0 <= self.visualization.overlay_opacity <= 1.0:
            raise ValueError("visualization.overlay_opacity must be in [0, 1]")
        if self.runtime.max_frames < 0:
            raise ValueError("runtime.max_frames must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> SimulatorConfig:
        raw = raw or {}
        allowed_sections = {f.name for f in fields(cls)}
        unknown = set(raw) - allowed_sections
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")

        def build(section_cls: type[T], values: Any) -> T:
            values = values or {}
            if not isinstance(values, dict):
                raise TypeError(f"{section_cls.__name__} must be a JSON object")
            section_allowed = {f.name for f in fields(section_cls)}
            extra = set(values) - section_allowed
            if extra:
                raise ValueError(
                    f"Unknown keys in {section_cls.__name__}: {sorted(extra)}"
                )
            return section_cls(**values)

        return cls(
            input=build(InputConfig, raw.get("input")),
            sensor=build(SensorConfig, raw.get("sensor")),
            simulation=build(SimulationConfig, raw.get("simulation")),
            noise=build(NoiseConfig, raw.get("noise")),
            visualization=build(VisualizationConfig, raw.get("visualization")),
            output=build(OutputConfig, raw.get("output")),
            runtime=build(RuntimeConfig, raw.get("runtime")),
        )

    @classmethod
    def load(cls, path: str | Path | None) -> SimulatorConfig:
        if path is None or str(path) == "":
            return cls()
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            raise TypeError("Configuration root must be a JSON object")
        return cls.from_dict(raw)
