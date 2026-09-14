import numpy as np

from evsim.config import SimulatorConfig
from evsim.simulator import EventSimulator


def run(
    backend: str,
    images: list[np.ndarray],
    config: SimulatorConfig | None = None,
) -> np.ndarray:
    config = config or SimulatorConfig()
    config.simulation.backend = backend
    config.noise.enable_threshold_variation = False
    sim = EventSimulator(config)
    sim.initialize(images[0], 0)
    parts = [
        sim.process(image, index * 1000).events
        for index, image in enumerate(images[1:], start=1)
    ]
    return np.concatenate(parts)


def _backend_test_config(
    backend: str, interpolation: str = "linear"
) -> SimulatorConfig:
    config = SimulatorConfig()
    config.sensor.timestamp_resolution_us = 10
    config.sensor.refractory_period_us = 100
    config.simulation.backend = backend
    config.simulation.interpolation = interpolation
    config.noise.enable_threshold_variation = False
    config.noise.background_rate_hz = 0.0
    return config


def test_vectorized_matches_pixel_loop():
    rng = np.random.default_rng(2026)
    images = [rng.integers(0, 256, (20, 23), dtype=np.uint8) for _ in range(15)]
    vectorized = run("vectorized", images)
    loop = run("loop", images)
    assert np.array_equal(vectorized, loop)
    assert vectorized.size > 0


def test_vectorized_matches_pixel_loop_with_refractory_and_zero_delta():
    """Residual crossings in a zero-delta interval must use the same convention."""
    images = [np.full((1, 1), value, dtype=np.uint8) for value in (19, 67, 67)]
    configs = []
    for backend in ("vectorized", "loop"):
        config = SimulatorConfig()
        config.sensor.positive_threshold = 0.05
        config.sensor.negative_threshold = 0.08
        config.sensor.timestamp_resolution_us = 100
        config.sensor.refractory_period_us = 100
        config.simulation.backend = backend
        config.noise.enable_threshold_variation = False
        config.noise.background_rate_hz = 0.0
        configs.append(config)

    vectorized = run("vectorized", images, config=configs[0])
    loop = run("loop", images, config=configs[1])

    assert np.array_equal(vectorized, loop)
    assert vectorized["timestamp_us"][-1] == 2000


def test_vectorized_matches_pixel_loop_without_interpolation():
    rng = np.random.default_rng(2027)
    images = [rng.integers(0, 256, (6, 8), dtype=np.uint8) for _ in range(6)]
    vectorized = run(
        "vectorized",
        images,
        config=_backend_test_config("vectorized", interpolation="none"),
    )
    loop = run(
        "loop",
        images,
        config=_backend_test_config("loop", interpolation="none"),
    )
    assert np.array_equal(vectorized, loop)
