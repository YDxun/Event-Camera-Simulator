"""Reproducible CA experiments: threshold, FPS interpolation and noise ablation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2 as cv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evsim.config import SimulatorConfig
from evsim.demo import demo_frame
from evsim.events import EventStream
from evsim.simulator import EventSimulator
from evsim.visualization import EventVideoRenderer


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


def align_event_times(
    events: np.ndarray,
    reference: np.ndarray,
    gap_penalty_us: float = 5_000.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Order-preserving per-polarity alignment used for timestamp RMSE."""

    matched_actual: list[float] = []
    matched_reference: list[float] = []
    unmatched = 0
    for polarity in (-1, 1):
        actual_times = events["timestamp_us"][events["polarity"] == polarity].astype(
            np.float64
        )
        reference_times = reference["timestamp_us"][
            reference["polarity"] == polarity
        ].astype(np.float64)
        n = len(actual_times)
        m = len(reference_times)
        if n == 0 or m == 0:
            unmatched += n + m
            continue
        cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
        back = np.zeros((n + 1, m + 1), dtype=np.int8)
        cost[0, 0] = 0.0
        for i in range(1, n + 1):
            cost[i, 0] = i * gap_penalty_us
            back[i, 0] = 1
        for j in range(1, m + 1):
            cost[0, j] = j * gap_penalty_us
            back[0, j] = 2
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                candidates = (
                    cost[i - 1, j - 1]
                    + abs(actual_times[i - 1] - reference_times[j - 1]),
                    cost[i - 1, j] + gap_penalty_us,
                    cost[i, j - 1] + gap_penalty_us,
                )
                choice = int(np.argmin(candidates))
                cost[i, j] = candidates[choice]
                back[i, j] = choice
        i, j = n, m
        local_actual: list[int] = []
        local_reference: list[int] = []
        while i > 0 or j > 0:
            choice = int(back[i, j])
            if choice == 0:
                local_actual.append(i - 1)
                local_reference.append(j - 1)
                i -= 1
                j -= 1
            elif choice == 1:
                i -= 1
            else:
                j -= 1
        matched_actual.extend(actual_times[index] for index in reversed(local_actual))
        matched_reference.extend(
            reference_times[index] for index in reversed(local_reference)
        )
        unmatched += (n - len(local_actual)) + (m - len(local_reference))
    return (
        np.asarray(matched_actual, dtype=np.float64),
        np.asarray(matched_reference, dtype=np.float64),
        unmatched,
    )


def threshold_sweep(output_dir: Path) -> list[dict[str, float]]:
    thresholds = [0.10, 0.15, 0.20, 0.30, 0.40]
    frames = 160
    width, height = 160, 120
    duration_s = (frames - 1) / 960.0
    images = [demo_frame(width, height, index, frames) for index in range(frames)]
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        config = ideal_config()
        config.sensor.positive_threshold = threshold
        config.sensor.negative_threshold = threshold
        stream = simulate_frames(config, images)
        count = len(stream)
        rows.append(
            {
                "threshold": threshold,
                "event_count": count,
                "event_rate_over_input_hz": count / duration_s,
                "count_times_threshold": count * threshold,
            }
        )
    np.savetxt(
        output_dir / "threshold_sweep.csv",
        np.array(
            [
                [
                    row["threshold"],
                    row["event_count"],
                    row["event_rate_over_input_hz"],
                    row["count_times_threshold"],
                ]
                for row in rows
            ]
        ),
        delimiter=",",
        header="threshold,event_count,event_rate_over_input_hz,count_times_threshold",
        comments="",
        fmt=["%.4f", "%d", "%.3f", "%.3f"],
    )
    values = np.array([row["threshold"] for row in rows])
    counts = np.array([row["event_count"] for row in rows])
    normalized = np.array([row["count_times_threshold"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(values, counts, marker="o")
    axes[0].set_xlabel("Contrast threshold C")
    axes[0].set_ylabel("Event count")
    axes[0].set_title("Threshold sweep")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(values, normalized, marker="s")
    axes[1].set_xlabel("Contrast threshold C")
    axes[1].set_ylabel("Event count x C")
    axes[1].set_title("Approximate 1/C normalization")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "threshold_sweep.png", dpi=160)
    plt.close(fig)
    return rows


def noise_ablation(output_dir: Path) -> list[dict[str, float]]:
    frames = 160
    width, height = 120, 96
    duration_s = (frames - 1) / 960.0
    images = [demo_frame(width, height, index, frames) for index in range(frames)]
    variants = [
        ("A_ideal", False, 0.0, 0),
        ("B_threshold_mismatch", True, 0.0, 0),
        ("C_threshold_plus_background", True, 0.05, 0),
        ("D_plus_refractory", True, 0.05, 100),
    ]
    rows: list[dict[str, float]] = []
    for name, vary, background, refractory in variants:
        config = ideal_config()
        config.noise.enable_threshold_variation = vary
        config.noise.threshold_sigma = 0.03 if vary else 0.0
        config.noise.background_rate_hz = background
        config.sensor.refractory_period_us = refractory
        stream = simulate_frames(config, images)
        on_events = int(np.count_nonzero(stream.polarity > 0))
        off_events = int(np.count_nonzero(stream.polarity < 0))
        rows.append(
            {
                "variant": name,
                "event_count": len(stream),
                "on_events": on_events,
                "off_events": off_events,
                "on_off_ratio": on_events / off_events if off_events else 0.0,
                "event_rate_over_input_hz": len(stream) / duration_s,
            }
        )
    with (output_dir / "noise_ablation.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
        handle.write(chr(10))
    names = [row["variant"] for row in rows]
    counts = [row["event_count"] for row in rows]
    plt.figure(figsize=(8, 4))
    plt.bar(names, counts)
    plt.ylabel("Event count")
    plt.title("Simplified sensor non-ideality ablation")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "noise_ablation.png", dpi=160)
    plt.close()
    return rows


def accumulation_window_comparison(output_dir: Path) -> list[dict[str, float]]:
    frames = 160
    width, height = 160, 120
    images = [demo_frame(width, height, index, frames) for index in range(frames)]
    stream = simulate_frames(ideal_config(), images)
    start_us = int(0.05 * 1_000_000)
    reference_frame = images[min(frames - 1, int(0.05 * 960))]
    windows = [1_000, 5_000, 10_000, 20_000]
    rows: list[dict[str, float]] = []
    panels = []
    for window_us in windows:
        selected = stream.events[
            (stream.timestamp_us >= start_us)
            & (stream.timestamp_us < start_us + window_us)
        ]
        renderer = EventVideoRenderer(width, height, SimulatorConfig().visualization)
        renderer.add(selected, reference_frame, start_us + window_us)
        panel = renderer.render_representation_panel(reference_frame)
        cv.imwrite(str(output_dir / f"accumulation_{window_us}us.png"), panel)
        panels.append(panel)
        rows.append(
            {
                "accumulation_time_us": window_us,
                "event_count": int(len(selected)),
                "active_pixels": int(
                    len(
                        np.unique(
                            selected["y"].astype(np.int64) * width + selected["x"]
                        )
                    )
                ),
            }
        )
    montage = np.vstack(panels)
    cv.imwrite(str(output_dir / "accumulation_window_montage.png"), montage)
    with (output_dir / "accumulation_window.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(rows, handle, indent=2)
        handle.write(chr(10))
    return rows


def photometric_linearization_comparison(output_dir: Path) -> list[dict[str, float]]:
    frames = 120
    width, height = 160, 120
    duration_s = (frames - 1) / 960.0
    images = [demo_frame(width, height, index, frames) for index in range(frames)]
    rows: list[dict[str, float]] = []
    for name, linearize in (("direct_log", False), ("gamma_linearized", True)):
        config = ideal_config()
        config.input.linearize = linearize
        stream = simulate_frames(config, images)
        rows.append(
            {
                "variant": name,
                "event_count": len(stream),
                "event_rate_over_input_hz": len(stream) / duration_s,
            }
        )
    with (output_dir / "linearization_comparison.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(rows, handle, indent=2)
        handle.write(chr(10))
    plt.figure(figsize=(6, 4))
    plt.bar([row["variant"] for row in rows], [row["event_count"] for row in rows])
    plt.ylabel("Event count")
    plt.title("Photometric preprocessing comparison")
    plt.tight_layout()
    plt.savefig(output_dir / "linearization_comparison.png", dpi=160)
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
        matched_actual, matched_reference, unmatched = align_event_times(
            events, reference
        )
        if matched_actual.size:
            timing_rmse_us = float(
                np.sqrt(np.mean((matched_actual - matched_reference) ** 2))
            )
        else:
            timing_rmse_us = float("nan")
        relative_count_difference = (
            abs(count_error) / len(reference) if len(reference) else 0.0
        )
        rows.append(
            {
                "fps": fps,
                "event_count": len(events),
                "count_error_vs_3840": count_error,
                "relative_count_difference_vs_3840": relative_count_difference,
                "matched_events": int(matched_actual.size),
                "unmatched_events": int(unmatched),
                "timing_rmse_us_vs_3840": timing_rmse_us,
            }
        )
    np.savetxt(
        output_dir / "fps_sweep.csv",
        np.array(
            [
                [
                    row["fps"],
                    row["event_count"],
                    row["count_error_vs_3840"],
                    row["relative_count_difference_vs_3840"],
                    row["matched_events"],
                    row["unmatched_events"],
                    row["timing_rmse_us_vs_3840"],
                ]
                for row in rows
            ]
        ),
        delimiter=",",
        header="fps,event_count,count_error_vs_3840,relative_count_difference_vs_3840,matched_events,unmatched_events,timing_rmse_us_vs_3840",
        comments="",
        fmt=["%d", "%d", "%d", "%.6f", "%d", "%d", "%.6f"],
    )
    fps = np.array([row["fps"] for row in rows])
    rmse = np.array([row["timing_rmse_us_vs_3840"] for row in rows])
    relative_count = np.array(
        [row["relative_count_difference_vs_3840"] for row in rows]
    )
    plt.figure(figsize=(7, 4))
    plt.plot(fps, rmse, marker="o", label="Timestamp RMSE")
    plt.plot(
        fps,
        relative_count * 1000.0,
        marker="s",
        label="Relative count difference x 1000",
    )
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


def write_markdown_report(output_dir: Path, summary: dict) -> Path:
    root = output_dir.parent
    demo_path = root / "demo" / "statistics.json"
    demo = (
        json.loads(demo_path.read_text(encoding="utf-8")) if demo_path.exists() else {}
    )
    benchmark = summary["benchmark"]
    validation = summary["validation"]
    lines: list[str] = []
    lines.append("# Event Camera Simulator - Final Validation Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "High-FPS frames are converted to asynchronous `(x, y, t, p)` events using"
    )
    lines.append(
        "log-intensity contrast thresholding. No edge detection, tracking, SLAM or"
    )
    lines.append("downstream learning model is used.")
    lines.append("")
    lines.append("## Model conventions")
    lines.append("")
    lines.append("- `L = log(I + epsilon)`")
    lines.append("- Stateful pixel reference `L_ref`")
    lines.append("- Separate thresholds `C+` and `C-`")
    lines.append("- Multiple threshold crossings per frame interval")
    lines.append("- Piecewise-linear log-intensity interpolation inside each interval")
    lines.append("- Floor timestamp quantization; equal timestamps are allowed")
    lines.append("- Refractory check is applied after quantization")
    lines.append("- Background activity uses a simplified Poisson process")
    lines.append("")
    lines.append("## Correctness validation")
    lines.append("")
    lines.append("| Check | Passed | Details |")
    lines.append("|---|---:|---|")
    for check in validation["checks"]:
        details = json.dumps(check["details"], sort_keys=True)
        lines.append(f"| {check['name']} | {check['passed']} | {details} |")
    lines.append("")
    if demo:
        lines.append("## 960 FPS end-to-end demo")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        for key in (
            "frames_processed",
            "event_count",
            "on_events",
            "off_events",
            "input_duration_s",
            "active_event_duration_s",
            "event_rate_over_input_hz",
            "event_rate_over_active_hz",
            "monotonic_timestamps",
            "processing_fps",
        ):
            lines.append(f"| {key} | {demo.get(key)} |")
        lines.append("")
    lines.append("## Threshold sweep")
    lines.append("")
    lines.append("| C | Event count | Event rate over input | Count x C |")
    lines.append("|---:|---:|---:|---:|")
    for row in summary["threshold_sweep"]:
        lines.append(
            f"| {row['threshold']:.2f} | {row['event_count']:,} | "
            f"{row['event_rate_over_input_hz']:.1f} | {row['count_times_threshold']:.1f} |"
        )
    lines.append("")
    lines.append(
        "The event count decreases monotonically as `C` increases, while `N*C`"
    )
    lines.append("remains in the same order of magnitude, consistent with `N ~ 1/C`.")
    lines.append("")
    lines.append("## FPS convergence")
    lines.append("")
    lines.append(
        "| FPS | Events | Relative count diff | Matched | Unmatched | Timestamp RMSE vs 3840 FPS |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in summary["fps_sweep"]:
        lines.append(
            f"| {row['fps']} | {row['event_count']} | "
            f"{row['relative_count_difference_vs_3840']:.4f} | "
            f"{row['matched_events']} | {row['unmatched_events']} | "
            f"{row['timing_rmse_us_vs_3840']:.3f} us |"
        )
    lines.append("")
    lines.append("Matching method: events are aligned separately by polarity using an")
    lines.append(
        "order-preserving dynamic-programming sequence alignment. The match cost"
    )
    lines.append(
        "is absolute timestamp difference and the insertion/deletion penalty is"
    )
    lines.append(
        "5000 us. RMSE is computed only over matched pairs; unmatched events are"
    )
    lines.append("reported explicitly and excluded from RMSE.")
    lines.append("")
    lines.append("The timestamp error decreases as input FPS increases under the")
    lines.append("piecewise-linear interpolation assumption.")
    lines.append("")
    lines.append("## Simplified sensor non-idealities")
    lines.append("")
    lines.append("| Model | Events | ON | OFF | ON/OFF ratio | Rate over input |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in summary["noise_ablation"]:
        lines.append(
            f"| {row['variant']} | {row['event_count']:,} | {row['on_events']:,} | "
            f"{row['off_events']:,} | {row['on_off_ratio']:.4f} | "
            f"{row['event_rate_over_input_hz']:.1f} |"
        )
    lines.append("")
    lines.append("## Accumulation-window comparison")
    lines.append("")
    lines.append("| Window (us) | Events | Active pixels |")
    lines.append("|---:|---:|---:|")
    for row in summary["accumulation_window"]:
        lines.append(
            f"| {row['accumulation_time_us']} | {row['event_count']:,} | "
            f"{row['active_pixels']:,} |"
        )
    lines.append("")
    lines.append("Smaller windows preserve more temporal detail but are sparse. Larger")
    lines.append(
        "windows improve spatial visibility while reducing temporal resolution."
    )
    lines.append("Accumulation affects visualization only, not the raw event stream.")
    lines.append("")
    lines.append("## Photometric preprocessing")
    lines.append("")
    lines.append("| Preprocessing | Events | Rate over input |")
    lines.append("|---|---:|---:|")
    for row in summary["linearization_comparison"]:
        lines.append(
            f"| {row['variant']} | {row['event_count']:,} | "
            f"{row['event_rate_over_input_hz']:.1f} |"
        )
    lines.append("")
    lines.append("The source-camera response curve is generally unknown. Gamma")
    lines.append("linearization is an optional approximation, not a claim of exact")
    lines.append("radiometric calibration.")
    lines.append("")
    lines.append("## Efficiency")
    lines.append("")
    lines.append("| Backend | Runtime (s) | Frames/s | MPixel-frames/s | Events |")
    lines.append("|---|---:|---:|---:|---:|")
    for backend in ("vectorized", "loop"):
        row = benchmark[backend]
        lines.append(
            f"| {backend} | {row['elapsed_s']:.3f} | {row['frames_per_s']:.2f} | "
            f"{row['mega_pixel_frames_per_s']:.4f} | {row['events']:,} |"
        )
    lines.append("")
    lines.append(
        f"Measured speedup `T_loop / T_vectorized = "
        f"{benchmark['speedup_loop_over_vectorized']:.2f}x`."
    )
    lines.append("")
    lines.append("## Assumptions and limitations")
    lines.append("")
    lines.append("- Piecewise-linear log-intensity between consecutive frames")
    lines.append("- Independent pixels without readout arbitration")
    lines.append("- Fixed per-pixel Gaussian threshold mismatch")
    lines.append("- Simplified Poisson background activity")
    lines.append(
        "- Source motion blur, temporal aliasing and saturation are not recovered"
    )
    lines.append("- No tracking, detection, optical flow, SLAM, SNN or reconstruction")
    lines.append("")
    lines.append("## Verification artifacts")
    lines.append("")
    lines.append("- `validation.json`")
    lines.append("- `threshold_sweep.csv` and `threshold_sweep.png`")
    lines.append("- `fps_sweep.csv` and `fps_sweep.png`")
    lines.append("- `noise_ablation.json` and `noise_ablation.png`")
    lines.append("- `accumulation_window.json` and `accumulation_window_montage.png`")
    lines.append("- `linearization_comparison.json` and `linearization_comparison.png`")
    lines.append(
        "- `../demo/events.csv`, `../demo/events.npz` and `../demo/event_video.avi`"
    )
    lines.append("")
    report_path = root / "REPORT.md"
    report_path.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    return report_path


def main() -> int:
    from evsim.validation import benchmark_backends, run_core_validation

    output_dir = ROOT / "results_python" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = run_core_validation()
    benchmark = benchmark_backends(width=96, height=72, frames=30)
    threshold = threshold_sweep(output_dir)
    noise = noise_ablation(output_dir)
    fps = fps_sweep(output_dir)
    accumulation = accumulation_window_comparison(output_dir)
    linearization = photometric_linearization_comparison(output_dir)
    summary = {
        "validation": validation,
        "benchmark": benchmark,
        "threshold_sweep": threshold,
        "noise_ablation": noise,
        "fps_sweep": fps,
        "accumulation_window": accumulation,
        "linearization_comparison": linearization,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + chr(10), encoding="utf-8"
    )
    write_markdown_report(output_dir, summary)
    print(json.dumps(summary, indent=2))
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
