import numpy as np

from evsim.config import SimulatorConfig
from evsim.simulator import EventSimulator


def run(backend: str, images: list[np.ndarray]) -> np.ndarray:
    config = SimulatorConfig()
    config.simulation.backend = backend
    config.noise.enable_threshold_variation = False
    sim = EventSimulator(config)
    sim.initialize(images[0], 0)
    parts = [
        sim.process(image, index * 1000).events
        for index, image in enumerate(images[1:], start=1)
    ]
    return np.concatenate(parts)


def test_vectorized_matches_pixel_loop():
    rng = np.random.default_rng(2026)
    images = [rng.integers(0, 256, (20, 23), dtype=np.uint8) for _ in range(15)]
    vectorized = run("vectorized", images)
    loop = run("loop", images)
    assert np.array_equal(vectorized, loop)
    assert vectorized.size > 0
