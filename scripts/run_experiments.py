"""Reproducible CA experiments: threshold, FPS interpolation and noise ablation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evsim.config import SimulatorConfig
from evsim.demo import demo_frame
from evsim.events import EventStream
from evsim.simulator import EventSimulator


def simulate_frames(config: SimulatorConfig, images: list[np.ndarray]) -> EventStream:
    simulator = EventSimulator(config)
    simulator.initialize(images[0], 0)
    parts = [
        simulator.process(image, index * 1000).events
        for index, image in enumerate(images[1:], start=1)
    ]
    return EventStream(np.concatenate(parts) if parts else np.empty(0))


def ideal_config() -> SimulatorConfig:
    config = SimulatorConfig.load("configs/ideal.json")
    config.runtime.progress = False
    return config

def threshold_sweep(output_dir: Path) -> list[dict[str, float]]:
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    frames = 160
    width, height = 160, 120
    images = [demo_frame(width, height, index, frames) for index in range(frames)]
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        config = ideal_config()
        config.sensor.positive_threshold = threshold
        config.sensor.negative_threshold = threshold
        stream = simulate_frames(config, images)
        rows.append(
            {
                "threshold": threshold,
                "event_count": len(stream),
                "event_rate_hz": len(stream) / ((frames - 1) / 960.0),
            }
        )
    np.savetxt(
        output_dir / "threshold_sweep.csv",
        np.array([[row["threshold"], row["event_count"], row["event_rate_hz"]] for row in rows]),
        delimiter=",",
        header="threshold,event_count,event_rate_hz",
        comments="",
        fmt=["%.4f", "%d", "%.3f"],
    )
    values = np.array([row["threshold"] for row in rows])
    counts = np.array([row["event_count"] for row in rows])
    plt.figure(figsize=(6, 4))
    plt.plot(values, counts, marker="o")
    plt.xlabel("Contrast threshold C")
    plt.ylabel("Event count")
    plt.title("Threshold sweep")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_sweep.png", dpi=160)
    plt.close()
    return rows

def noise_ablation(output_dir: Path) -> list[dict[str, float]]:
    frames = 160
    width, height = 120, 96
    images = [demo_frame(width, height, index, frames) for index in range(frames)]
    variants = [
        ("ideal", False, 0.0),
        ("threshold_variation", True, 0.0),
        ("background", False, 0.05),
        ("combined", True, 0.05),
    ]
    rows: list[dict[str, float]] = []
    for name, vary, background in variants:
        config = ideal_config()
        config.noise.enable_threshold_variation = vary
        config.noise.threshold_sigma = 0.03 if vary else 0.0
        config.noise.background_rate_hz = background
        stream = simulate_frames(config, images)
        rows.append(
            {
                "variant": name,
                "event_count": len(stream),
                "on_events": int(np.count_nonzero(stream.polarity > 0)),
                "off_events": int(np.count_nonzero(stream.polarity < 0)),
            }
        )
    with (output_dir / "noise_ablation.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")
    names = [row["variant"] for row in rows]
    counts = [row["event_count"] for row in rows]
    plt.figure(figsize=(7, 4))
    plt.bar(names, counts)
    plt.ylabel("Event count")
    plt.title("Noise ablation")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "noise_ablation.png", dpi=160)
    plt.close()
    return rows

def fps_sweep(output_dir: Path) -> list[dict[str, float]]:
    duration_s = 1.5
    fps_values = [120, 240, 480, 960, 1920, 3840]
    epsilon = 1e-3

    def continuous_log(t: np.ndarray) -> np.ndarray:
        return -1.5 + 0.4 * t + 0.08 * np.sin(2.0 * np.pi * 7.0 * t)

    def frame_from_log(value: float) -> np.ndarray:
        intensity = np.clip(np.exp(value) - epsilon, 0.0, 1.0)
        dn = int(np.rint(intensity * 255.0))
        return np.array([[dn]], dtype=np.uint8)

    def run(fps: int) -> np.ndarray:
        indices = np.arange(int(round(duration_s * fps)), dtype=np.float64)
        times_s = indices / fps
        images = [frame_from_log(float(value)) for value in continuous_log(times_s)]
        timestamps_us = np.rint(times_s * 1_000_000.0).astype(np.int64)
        config = ideal_config()
        config.sensor.positive_threshold = 0.06
        config.sensor.negative_threshold = 0.06
        simulator = EventSimulator(config)
        simulator.initialize(images[0], int(timestamps_us[0]))
        parts = [
            simulator.process(image, int(timestamp)).events
            for image, timestamp in zip(images[1:], timestamps_us[1:])
        ]
        return np.concatenate(parts) if parts else np.empty(0)

    reference = run(fps_values[-1])
    rows: list[dict[str, float]] = []
    for fps in fps_values:
        events = run(fps)
        count_error = len(events) - len(reference)
        overlap = min(len(events), len(reference))
        if overlap:
            timing_rmse_us = float(
                np.sqrt(np.mean((events["timestamp_us"][:overlap] - reference["timestamp_us"][:overlap]) ** 2))
            )
        else:
            timing_rmse_us = float("nan")
        rows.append(
            {
                "fps": fps,
                "event_count": len(events),
                "count_error_vs_3840": count_error,
                "timing_rmse_us_vs_3840": timing_rmse_us,
            }
        )
    np.savetxt(
        output_dir / "fps_sweep.csv",
        np.array([[row["fps"], row["event_count"], row["count_error_vs_3840"], row["timing_rmse_us_vs_3840"]] for row in rows]),
        delimiter=",",
        header="fps,event_count,count_error_vs_3840,timing_rmse_us_vs_3840",
        comments="",
        fmt=["%d", "%d", "%d", "%.6f"],
    )
    fps = np.array([row["fps"] for row in rows])
    rmse = np.array([row["timing_rmse_us_vs_3840"] for row in rows])
    count_error = np.array([row["count_error_vs_3840"] for row in rows])
    plt.figure(figsize=(7, 4))
    plt.plot(fps, rmse, marker="o", label="Timestamp RMSE")
    plt.plot(fps, np.abs(count_error), marker="s", label="Absolute count error")
    plt.xscale("log", base=2)
    plt.xlabel("Input FPS")
    plt.ylabel("Error vs 3840 FPS reference")
    plt.title("Temporal interpolation convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fps_sweep.png", dpi=160)
    plt.close()
    return rows

def main() -> int:
    from evsim.validation import benchmark_backends, run_core_validation

    output_dir = ROOT / "results_python" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = run_core_validation()
    benchmark = benchmark_backends(width=96, height=72, frames=30)
    threshold = threshold_sweep(output_dir)
    noise = noise_ablation(output_dir)
    fps = fps_sweep(output_dir)
    summary = {
        "validation": validation,
        "benchmark": benchmark,
        "threshold_sweep": threshold,
        "noise_ablation": noise,
        "fps_sweep": fps,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
