"""Command-line interface for the Python event-camera simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import SimulatorConfig
from .demo import run_demo
from .events import EventStream
from .pipeline import simulate_source
from .sources import inspect_source, open_source
from .validation import benchmark_backends, run_core_validation
from .visualization import EventVideoRenderer


def _apply_overrides(config: SimulatorConfig, args: argparse.Namespace) -> None:
    if args.backend:
        config.simulation.backend = args.backend
    if args.positive_threshold is not None:
        config.sensor.positive_threshold = args.positive_threshold
    if args.negative_threshold is not None:
        config.sensor.negative_threshold = args.negative_threshold
    if args.timestamp_resolution is not None:
        config.sensor.timestamp_resolution_us = args.timestamp_resolution
    if args.refractory_period is not None:
        config.sensor.refractory_period_us = args.refractory_period
    if args.accumulation_time is not None:
        config.visualization.accumulation_time_us = args.accumulation_time
    if args.max_frames is not None:
        config.runtime.max_frames = args.max_frames
    if args.no_progress:
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

def cmd_inspect(args: argparse.Namespace) -> int:
    config = SimulatorConfig.load(args.config)
    metadata = inspect_source(
        args.input,
        fallback_fps=args.fallback_fps or config.input.fallback_fps,
        timestamp_scale_us=config.input.timestamp_scale_us,
    )
    print(json.dumps(metadata, indent=2))
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    config = SimulatorConfig.load(args.config)
    _apply_overrides(config, args)
    if args.output_csv:
        config.output.csv_path = args.output_csv
    if args.output_npz:
        config.output.npz_path = args.output_npz
    if args.video:
        config.visualization.output_video = args.video

    metadata = inspect_source(
        args.input,
        fallback_fps=config.input.fallback_fps,
        timestamp_scale_us=config.input.timestamp_scale_us,
    )
    source = open_source(
        args.input,
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

def cmd_validate(args: argparse.Namespace) -> int:
    result = run_core_validation(seed=args.seed)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    result = benchmark_backends(args.width, args.height, args.frames, args.seed)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    result = run_demo(
        args.output_dir,
        fps=args.fps,
        seconds=args.seconds,
        width=args.width,
        height=args.height,
    )
    print(json.dumps(result, indent=2))
    return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evsim",
        description="Python event-camera simulator: high-FPS frames to events.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_p = sub.add_parser("inspect", help="Inspect video or image-sequence metadata")
    inspect_p.add_argument("-i", "--input", required=True)
    inspect_p.add_argument("-c", "--config")
    inspect_p.add_argument("--fallback-fps", type=float)
    inspect_p.set_defaults(func=cmd_inspect)

    sim_p = sub.add_parser("simulate", help="Generate events and optional video")
    sim_p.add_argument("-i", "--input", required=True)
    sim_p.add_argument("-c", "--config")
    sim_p.add_argument("-o", "--output-csv")
    sim_p.add_argument("--output-npz")
    sim_p.add_argument("-v", "--video")
    sim_p.add_argument("-b", "--backend", choices=["vectorized", "loop"])
    sim_p.add_argument("--positive-threshold", type=float)
    sim_p.add_argument("--negative-threshold", type=float)
    sim_p.add_argument("--timestamp-resolution", type=int)
    sim_p.add_argument("--refractory-period", type=int)
    sim_p.add_argument("--accumulation-time", type=int)
    sim_p.add_argument("--max-frames", type=int)
    sim_p.add_argument("--no-progress", action="store_true")
    sim_p.set_defaults(func=cmd_simulate)

    validate_p = sub.add_parser("validate", help="Run analytic and backend equivalence checks")
    validate_p.add_argument("--seed", type=int, default=7)
    validate_p.add_argument("--output-json")
    validate_p.set_defaults(func=cmd_validate)

    bench_p = sub.add_parser("benchmark", help="Compare vectorized and pixel-loop backends")
    bench_p.add_argument("--width", type=int, default=320)
    bench_p.add_argument("--height", type=int, default=240)
    bench_p.add_argument("--frames", type=int, default=60)
    bench_p.add_argument("--seed", type=int, default=42)
    bench_p.add_argument("--output-json")
    bench_p.set_defaults(func=cmd_benchmark)

    demo_p = sub.add_parser("demo", help="Generate and simulate a self-contained 960 FPS demo")
    demo_p.add_argument("-o", "--output-dir", default="results_python/demo")
    demo_p.add_argument("--fps", type=float, default=960.0)
    demo_p.add_argument("--seconds", type=float, default=1.0)
    demo_p.add_argument("--width", type=int, default=320)
    demo_p.add_argument("--height", type=int, default=240)
    demo_p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
