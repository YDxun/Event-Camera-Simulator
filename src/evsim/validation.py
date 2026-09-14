"""Analytic and cross-backend validation for the event model."""

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .config import SimulatorConfig
from .preprocessing import to_log_intensity
from .simulator import EventSimulator


@dataclass
class Check:
    name: str
    passed: bool
    details: dict[str, Any]


def _stream(config: SimulatorConfig, frames: list[np.ndarray]) -> np.ndarray:
    sim = EventSimulator(config)
    sim.initialize(frames[0], 0)
    parts = [
        sim.process(frame, index * 1000).events for index, frame in enumerate(frames[1:], start=1)
    ]
    return np.concatenate(parts) if parts else np.empty(0)


def _ideal_config(**overrides: Any) -> SimulatorConfig:
    config = SimulatorConfig()
    config.noise.enable_threshold_variation = False
    config.noise.background_rate_hz = 0.0
    for key, value in overrides.items():
        section, name = key.split(".", 1)
        setattr(getattr(config, section), name, value)
    config.validate()
    return config


def _log_value(dn: int, config: SimulatorConfig) -> np.float32:
    image = np.array([[dn]], dtype=np.uint8)
    return to_log_intensity(image, config.input, config.sensor)[0, 0]


def _expected_positive_times(
    config: SimulatorConfig, dn0: int, dn1: int, duration_us: int
) -> np.ndarray:
    log0 = _log_value(dn0, config)
    log1 = _log_value(dn1, config)
    delta = np.float32(log1 - log0)
    threshold = np.float32(config.sensor.positive_threshold)
    count = int(np.floor(float(np.float32(delta / threshold)) + 1e-12))
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    k = np.arange(1, count + 1, dtype=np.float32)
    levels = log0 + k * threshold
    alpha = (levels - log0) / delta
    quantized = np.floor(alpha * duration_us).astype(np.int64)
    resolution = config.sensor.timestamp_resolution_us
    return (quantized // resolution) * resolution


def run_core_validation(seed: int = 7) -> dict[str, Any]:
    checks: list[Check] = []
    static = np.full((8, 8), 120, dtype=np.uint8)
    events = _stream(_ideal_config(), [static, static])
    checks.append(Check("static_scene_no_events", len(events) == 0, {"events": len(events)}))

    black = np.zeros((1, 1), dtype=np.uint8)
    white = np.full((1, 1), 255, dtype=np.uint8)
    events = _stream(_ideal_config(), [black, white])
    positive = bool(len(events) > 1 and np.all(events["polarity"] == 1))
    checks.append(Check("bright_ramp_multiple_positive", positive, {"events": len(events)}))

    events = _stream(_ideal_config(), [white, black])
    negative = bool(len(events) > 1 and np.all(events["polarity"] == -1))
    checks.append(Check("dark_ramp_multiple_negative", negative, {"events": len(events)}))

    analytic_cfg = _ideal_config(
        **{"sensor.positive_threshold": 0.08, "sensor.negative_threshold": 0.08}
    )
    expected = _expected_positive_times(analytic_cfg, 5, 250, 1000)
    actual = _stream(
        analytic_cfg, [np.full((1, 1), 5, dtype=np.uint8), np.full((1, 1), 250, dtype=np.uint8)]
    )
    overlap = min(len(expected), len(actual))
    timestamp_errors = (
        np.abs(actual["timestamp_us"][:overlap] - expected[:overlap])
        if overlap
        else np.array([1000], dtype=np.int64)
    )
    analytic_pass = bool(
        len(expected) > 1
        and len(actual) == len(expected)
        and np.all(timestamp_errors <= analytic_cfg.sensor.timestamp_resolution_us)
    )
    checks.append(
        Check(
            "analytical_timestamp_alignment",
            analytic_pass,
            {
                "expected_events": len(expected),
                "actual_events": len(actual),
                "timestamp_mae_us": float(np.mean(timestamp_errors)),
                "timestamp_max_error_us": int(np.max(timestamp_errors)),
            },
        )
    )

    asym_cfg = _ideal_config(
        **{"sensor.positive_threshold": 0.15, "sensor.negative_threshold": 0.30}
    )
    positive = _stream(asym_cfg, [black, white])
    negative = _stream(asym_cfg, [white, black])
    asym_pass = bool(
        len(positive) > len(negative)
        and np.all(positive["polarity"] == 1)
        and np.all(negative["polarity"] == -1)
    )
    checks.append(
        Check(
            "asymmetric_threshold_validation",
            asym_pass,
            {"positive_events": len(positive), "negative_events": len(negative)},
        )
    )

    quant_cfg = _ideal_config(**{"sensor.timestamp_resolution_us": 100})
    events = _stream(quant_cfg, [black, white])
    quant_gaps = (
        np.diff(events["timestamp_us"]) if len(events) > 1 else np.array([], dtype=np.int64)
    )
    duplicate_bins = int(np.count_nonzero(quant_gaps == 0))
    quantized = bool(
        len(events)
        and np.all(events["timestamp_us"] % 100 == 0)
        and np.all(quant_gaps >= 0)
        and duplicate_bins > 0
    )
    checks.append(
        Check(
            "timestamp_quantization_floor_and_duplicates",
            quantized,
            {"events": len(events), "duplicate_timestamp_bins": duplicate_bins},
        )
    )

    rf_cfg = _ideal_config(**{"sensor.refractory_period_us": 100})
    events = _stream(rf_cfg, [black, white])
    gaps = np.diff(events["timestamp_us"]) if len(events) > 1 else np.array([100])
    valid = bool(len(events) <= 1 or np.all(gaps >= 100))
    checks.append(Check("refractory_period", valid, {"events": len(events)}))

    combined_cfg = _ideal_config(
        **{
            "sensor.timestamp_resolution_us": 100,
            "sensor.refractory_period_us": 100,
        }
    )
    combined_first = _stream(combined_cfg, [black, white])
    combined_second = _stream(combined_cfg, [black, white])
    combined_gaps = (
        np.diff(combined_first["timestamp_us"])
        if len(combined_first) > 1
        else np.array([100], dtype=np.int64)
    )
    combined_pass = bool(
        len(combined_first) > 0
        and np.all(combined_first["timestamp_us"] % 100 == 0)
        and np.all(combined_gaps >= 100)
        and np.array_equal(combined_first, combined_second)
    )
    checks.append(
        Check(
            "quantization_and_refractory_order",
            combined_pass,
            {
                "events": len(combined_first),
                "duplicate_bins": int(np.count_nonzero(combined_gaps == 0)),
            },
        )
    )

    ideal = _ideal_config()
    dark_sat_pass = bool(
        np.isfinite(_log_value(0, ideal))
        and len(_stream(ideal, [black, white])) > 1
        and len(_stream(ideal, [white, black])) > 1
        and len(_stream(ideal, [np.full((1, 1), 250, dtype=np.uint8), white])) == 0
        and len(_stream(ideal, [white, np.full((1, 1), 250, dtype=np.uint8)])) == 0
    )
    checks.append(
        Check(
            "dark_and_saturation_edge_cases",
            dark_sat_pass,
            {
                "black_log_finite": bool(np.isfinite(_log_value(0, ideal))),
                "near_saturation_events": 0,
            },
        )
    )

    rng = np.random.default_rng(seed)
    images = [rng.integers(0, 256, (16, 17), dtype=np.uint8) for _ in range(12)]
    vec_cfg = _ideal_config()
    loop_cfg = _ideal_config(**{"simulation.backend": "loop"})
    vec = _stream(vec_cfg, images)
    loop = _stream(loop_cfg, images)
    exact = bool(np.array_equal(vec, loop))
    checks.append(
        Check(
            "vectorized_equals_pixel_loop",
            exact,
            {"vectorized_events": len(vec), "loop_events": len(loop)},
        )
    )

    return {
        "passed": all(check.passed for check in checks),
        "checks": [asdict(check) for check in checks],
    }


def benchmark_backends(
    width: int = 320,
    height: int = 240,
    frames: int = 60,
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    images = [rng.integers(0, 256, (height, width), dtype=np.uint8) for _ in range(frames)]
    results: dict[str, Any] = {}
    for backend in ("vectorized", "loop"):
        config = _ideal_config(**{"simulation.backend": backend, "runtime.progress": False})
        sim = EventSimulator(config)
        sim.initialize(images[0], 0)
        started = time.perf_counter()
        total = 0
        for index, image in enumerate(images[1:], start=1):
            total += len(sim.process(image, index * 1000).events)
        elapsed = time.perf_counter() - started
        results[backend] = {
            "elapsed_s": elapsed,
            "frames_per_s": (frames - 1) / elapsed if elapsed > 0 else 0.0,
            "mega_pixel_frames_per_s": (frames - 1) * width * height / elapsed / 1e6
            if elapsed > 0
            else 0.0,
            "events": total,
        }
    vec = results["vectorized"]["elapsed_s"]
    loop = results["loop"]["elapsed_s"]
    results["speedup_loop_over_vectorized"] = loop / vec if vec > 0 else 0.0
    return results
