"""Photometric preprocessing: grayscale DN -> linear intensity -> log intensity."""

from __future__ import annotations

import numpy as np

from .config import InputConfig, SensorConfig


def to_intensity(image: np.ndarray, config: InputConfig) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"Expected a grayscale image, got shape {image.shape}")
    max_dn = float((1 << config.bit_depth) - 1)
    intensity = image.astype(np.float32) / max_dn
    np.clip(intensity, 0.0, 1.0, out=intensity)
    if config.linearize:
        intensity = np.power(intensity, np.float32(config.gamma))
    return intensity


def to_log_intensity(image: np.ndarray, config: InputConfig, sensor: SensorConfig) -> np.ndarray:
    intensity = to_intensity(image, config)
    return np.log(intensity + np.float32(sensor.log_epsilon), dtype=np.float32)
