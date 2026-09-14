"""Command-line interface for the Python event-camera simulator using Tyro."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import tyro

from .config import SimulatorConfig
from .demo import run_demo
from .events import EventStream
from .pipeline import simulate_source
from .sources import inspect_source, open_source
from .validation import benchmark_backends, run_core_validation
from .visualization import EventVideoRenderer


def _apply_overrides(config: SimulatorConfig, cmd: SimulateCmd) -> None:
    if cmd.backend:
        config.simulation.backend = cmd.backend
    if cmd.positive_threshold is not None:
        config.sensor.positive_threshold = cmd.positive_threshold
    if cmd.negative_threshold is not None:
        config.sensor.negative_threshold = cmd.negative_threshold
    if cmd.timestamp_resolution is not None:
        config.sensor.timestamp_resolution_us = cmd.timestamp_resolution
    if cmd.refractory_period is not None:
        config.sensor.refractory_period_us = cmd.refractory_period
    if cmd.accumulation_time is not None:
        config.visualization.accumulation_time_us = cmd.accumulation_time
    if cmd.max_frames is not None:
        config.runtime.max_frames = cmd.max_frames
    if cmd.no_progress:
        config.runtime.progress = False
    config.validate()


def _write_result(result_stats: dict, stream: EventStream, config: SimulatorConfig) -> None:
    if config.output.csv_path:
        stream.save_csv(config.output.csv_path)
    if config.output.npz_path:
        stream.save_npz(config.output.npz_path)
    if config.output.statistics_path:
        path = Path(config.output.statistics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result_stats, indent=2) + "\n", encoding="utf-8")


@dataclass
class InspectCmd:
    """Inspect video or image-sequence metadata."""

    input: Annotated[str, tyro.conf.arg(aliases=["-i"])]
    """Path to input video file or image sequence directory."""
    config: Annotated[str | None, tyro.conf.arg(aliases=["-c"])] = None
    """Optional configuration JSON file."""
    fallback_fps: float | None = None
    """Fallback FPS if not present in source metadata."""

    def run(self) -> int:
        cfg = SimulatorConfig.load(self.config) if self.config else SimulatorConfig()
        fps = self.fallback_fps or cfg.input.fallback_fps
        metadata = inspect_source(
            self.input,
            fallback_fps=fps,
            timestamp_scale_us=cfg.input.timestamp_scale_us,
        )
        print(json.dumps(metadata, indent=2))
        return 0


@dataclass
class SimulateCmd:
    """Generate events and optional video from high-FPS frames."""

    input: Annotated[str, tyro.conf.arg(aliases=["-i"])]
    """Path to input video file or image sequence directory."""
    config: Annotated[str | None, tyro.conf.arg(aliases=["-c"])] = None
    """Optional configuration JSON file."""
    output_csv: Annotated[str | None, tyro.conf.arg(aliases=["-o"])] = None
    """Path to save event stream in CSV format."""
    output_npz: str | None = None
    """Path to save event stream in compressed NPZ format."""
    video: Annotated[str | None, tyro.conf.arg(aliases=["-v"])] = None
    """Path to save illustrative event accumulation video."""
    backend: Annotated[Literal["vectorized", "loop"] | None, tyro.conf.arg(aliases=["-b"])] = None
    """Simulation backend: 'vectorized' (default, fast) or 'loop' (reference)."""
    positive_threshold: float | None = None
    """Positive contrast threshold C+."""
    negative_threshold: float | None = None
    """Negative contrast threshold C-."""
    timestamp_resolution: int | None = None
    """Timestamp quantization resolution in microseconds."""
    refractory_period: int | None = None
    """Pixel refractory period in microseconds."""
    accumulation_time: int | None = None
    """Event accumulation window in microseconds for video visualization."""
    max_frames: int | None = None
    """Maximum number of frames to process."""
    no_progress: bool = False
    """Disable progress bar."""

    def run(self) -> int:
        config = SimulatorConfig.load(self.config) if self.config else SimulatorConfig()
        _apply_overrides(config, self)
        if self.output_csv:
            config.output.csv_path = self.output_csv
        if self.output_npz:
            config.output.npz_path = self.output_npz
        if self.video:
            config.visualization.output_video = self.video

        metadata = inspect_source(
            self.input,
            fallback_fps=config.input.fallback_fps,
            timestamp_scale_us=config.input.timestamp_scale_us,
        )
        source = open_source(
            self.input,
            fallback_fps=config.input.fallback_fps,
            timestamp_scale_us=config.input.timestamp_scale_us,
        )
        renderer = None
        if config.visualization.output_video:
            renderer = EventVideoRenderer(source.width, source.height, config.visualization)
            renderer.open(config.visualization.output_video)
        try:
            result = simulate_source(
                source,
                config,
                max_frames=config.runtime.max_frames or None,
                progress=config.runtime.progress,
                renderer=renderer,
            )
        finally:
            source.close()

        stats = dict(result.statistics)
        stats["input_metadata"] = metadata
        stats["video_path"] = result.video_path
        _write_result(stats, result.event_stream, config)
        print(json.dumps(stats, indent=2))
        return 0


@dataclass
class ValidateCmd:
    """Run analytic and backend equivalence checks."""

    seed: int = 7
    """Random seed for noise generation in validation checks."""
    output_json: str | None = None
    """Path to save validation results as JSON."""

    def run(self) -> int:
        result = run_core_validation(seed=self.seed)
        if self.output_json:
            path = Path(self.output_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1


@dataclass
class BenchmarkCmd:
    """Compare vectorized and pixel-loop backends."""

    width: int = 320
    """Frame width for benchmark."""
    height: int = 240
    """Frame height for benchmark."""
    frames: int = 60
    """Number of frames to benchmark."""
    seed: int = 42
    """Random seed for synthetic benchmark frames."""
    output_json: str | None = None
    """Path to save benchmark results as JSON."""

    def run(self) -> int:
        result = benchmark_backends(self.width, self.height, self.frames, self.seed)
        if self.output_json:
            path = Path(self.output_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0


@dataclass
class DemoCmd:
    """Generate and simulate a self-contained 960 FPS demo."""

    output_dir: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "results_python/demo"
    """Output directory for generated demo artifacts."""
    fps: float = 960.0
    """Frame rate of synthetic video."""
    seconds: float = 1.0
    """Duration of synthetic video in seconds."""
    width: int = 320
    """Frame width."""
    height: int = 240
    """Frame height."""

    def run(self) -> int:
        result = run_demo(
            self.output_dir,
            fps=self.fps,
            seconds=self.seconds,
            width=self.width,
            height=self.height,
        )
        print(json.dumps(result, indent=2))
        return 0


@dataclass
class GuiCmd:
    """Launch the PySide6 desktop GUI."""

    def run(self) -> int:
        from .gui import main as gui_main

        return gui_main()


EvsimCommands = (
    Annotated[InspectCmd, tyro.conf.subcommand(name="inspect")]
    | Annotated[SimulateCmd, tyro.conf.subcommand(name="simulate")]
    | Annotated[ValidateCmd, tyro.conf.subcommand(name="validate")]
    | Annotated[BenchmarkCmd, tyro.conf.subcommand(name="benchmark")]
    | Annotated[DemoCmd, tyro.conf.subcommand(name="demo")]
    | Annotated[GuiCmd, tyro.conf.subcommand(name="gui")]
)


def main(argv: list[str] | None = None) -> int:
    try:
        cmd = tyro.cli(
            EvsimCommands,
            args=argv,
            description="Python event-camera simulator: high-FPS frames to events.",
        )
        return int(cmd.run())
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 2
    except (FileNotFoundError, RuntimeError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
