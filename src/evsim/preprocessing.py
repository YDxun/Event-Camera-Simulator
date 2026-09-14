"""Photometric preprocessing: grayscale DN -> linear intensity -> log intensity."""

import numpy as np

from .config import InputConfig, SensorConfig


def to_intensity(image: np.ndarray, config: InputConfig) -> np.ndarray:
    """Normalize raw digital numbers (DN) to normalized linear optical intensity [0.0, 1.0].

    If `config.linearize` is enabled, an inverse gamma power function is applied:
    `I_linear = (DN / max_dn) ** gamma`
    to approximate radiometric scene irradiance from gamma-encoded video.

    Args:
        image: 2D array of pixel values with shape (H, W).
        config: Input configuration specifying bit_depth, linearize, and gamma.

    Returns:
        Float32 array of normalized intensity in [0.0, 1.0].
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a grayscale image, got shape {image.shape}")
    max_dn = float((1 << config.bit_depth) - 1)
    intensity = image.astype(np.float32) / max_dn
    np.clip(intensity, 0.0, 1.0, out=intensity)
    if config.linearize:
        intensity = np.power(intensity, np.float32(config.gamma))
    return intensity


def to_log_intensity(
    image: np.ndarray, config: InputConfig, sensor: SensorConfig
) -> np.ndarray:
    """Transform image to logarithmic intensity: `L = ln(I + epsilon)`.

    Neuromorphic event pixels respond logarithmically to photocurrent.
    The constant epsilon models dark current and ensures mathematical stability
    at zero illumination.

    Args:
        image: 2D array of pixel values with shape (H, W).
        config: Input configuration.
        sensor: Sensor configuration containing `log_epsilon`.

    Returns:
        Float32 array of log intensity.
    """
    intensity = to_intensity(image, config)
    return np.log(intensity + np.float32(sensor.log_epsilon), dtype=np.float32)
