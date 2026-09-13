"""Analytic and cross-backend validation for the event model."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .config import SimulatorConfig
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
        sim.process(frame, index * 1000).events
        for index, frame in enumerate(frames[1:], start=1)
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

    quant_cfg = _ideal_config(**{"sensor.timestamp_resolution_us": 100})
    events = _stream(quant_cfg, [black, white])
    quantized = bool(len(events) and np.all(events["timestamp_us"] % 100 == 0))
    checks.append(Check("timestamp_quantization", quantized, {"events": len(events)}))

    rf_cfg = _ideal_config(**{"sensor.refractory_period_us": 100})
    events = _stream(rf_cfg, [black, white])
    gaps = np.diff(events["timestamp_us"]) if len(events) > 1 else np.array([100])
    valid = bool(len(events) <= 1 or np.all(gaps >= 100))
    checks.append(Check("refractory_period", valid, {"events": len(events)}))

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
            "mega_pixel_frames_per_s": (frames - 1) * width * height / elapsed / 1e6 if elapsed > 0 else 0.0,
            "events": total,
        }
    vec = results["vectorized"]["elapsed_s"]
    loop = results["loop"]["elapsed_s"]
    results["speedup_loop_over_vectorized"] = loop / vec if vec > 0 else 0.0
    return results
