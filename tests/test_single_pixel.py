import numpy as np

from evsim.config import SimulatorConfig
from evsim.preprocessing import to_log_intensity
from evsim.simulator import EventSimulator


def simulated(config: SimulatorConfig, images: list[np.ndarray]) -> np.ndarray:
    sim = EventSimulator(config)
    sim.initialize(images[0], 0)
    parts = [
        sim.process(image, index * 1000).events for index, image in enumerate(images[1:], start=1)
    ]
    return np.concatenate(parts) if parts else np.empty(0)


def test_constant_intensity_has_no_events():
    config = SimulatorConfig()
    image = np.full((12, 13), 120, dtype=np.uint8)
    assert simulated(config, [image, image]).size == 0


def test_multiple_positive_threshold_crossings_and_interpolation():
    config = SimulatorConfig()
    events = simulated(
        config,
        [np.zeros((1, 1), dtype=np.uint8), np.full((1, 1), 255, dtype=np.uint8)],
    )
    assert events.size > 10
    assert np.all(events["polarity"] == 1)
    assert np.all(np.diff(events["timestamp_us"]) > 0)
    assert events["timestamp_us"][0] > 0
    assert events["timestamp_us"][-1] < 1000


def test_dark_ramp_has_only_negative_events():
    config = SimulatorConfig()
    events = simulated(
        config,
        [np.full((1, 1), 255, dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8)],
    )
    assert events.size > 10
    assert np.all(events["polarity"] == -1)


def test_asymmetric_thresholds():
    config = SimulatorConfig()
    config.sensor.positive_threshold = 5.0
    config.sensor.negative_threshold = 0.2
    source = np.full((1, 1), 255, dtype=np.uint8)
    darker = np.full((1, 1), 80, dtype=np.uint8)
    events = simulated(config, [source, darker])
    assert events.size > 0
    assert np.all(events["polarity"] == -1)


def test_timestamp_resolution_and_refractory():
    config = SimulatorConfig()
    config.sensor.timestamp_resolution_us = 100
    config.sensor.refractory_period_us = 100
    events = simulated(
        config,
        [np.zeros((1, 1), dtype=np.uint8), np.full((1, 1), 255, dtype=np.uint8)],
    )
    assert events.size > 0
    assert np.all(events["timestamp_us"] % 100 == 0)
    assert np.all(np.diff(events["timestamp_us"]) >= 100)


def test_threshold_variation_is_reproducible():
    def run(seed: int) -> np.ndarray:
        config = SimulatorConfig()
        config.noise.enable_threshold_variation = True
        config.noise.threshold_sigma = 0.03
        config.noise.random_seed = seed
        images = [np.full((8, 8), value, dtype=np.uint8) for value in (20, 80, 160, 240, 120, 40)]
        return simulated(config, images)

    assert np.array_equal(run(123), run(123))
    assert not np.array_equal(run(123), run(124))


def test_analytical_timestamp_formula():
    config = SimulatorConfig()
    config.sensor.positive_threshold = 0.08
    config.sensor.negative_threshold = 0.08
    dark = np.full((1, 1), 5, dtype=np.uint8)
    bright = np.full((1, 1), 250, dtype=np.uint8)
    events = simulated(config, [dark, bright])
    log0 = float(to_log_intensity(dark, config.input, config.sensor)[0, 0])
    log1 = float(to_log_intensity(bright, config.input, config.sensor)[0, 0])
    count = int(np.floor((log1 - log0) / config.sensor.positive_threshold + 1e-12))
    expected = np.floor(
        (np.arange(1, count + 1) * config.sensor.positive_threshold) / (log1 - log0) * 1000
    ).astype(np.int64)
    assert len(events) == count
    assert np.all(np.abs(events["timestamp_us"] - expected) <= 1)
    assert np.mean(np.abs(events["timestamp_us"] - expected)) <= 1


def test_dark_saturation_and_duplicate_quantization_bins():
    config = SimulatorConfig()
    black = np.zeros((1, 1), dtype=np.uint8)
    white = np.full((1, 1), 255, dtype=np.uint8)
    near_white = np.full((1, 1), 250, dtype=np.uint8)
    assert np.isfinite(to_log_intensity(black, config.input, config.sensor)).all()
    assert len(simulated(config, [near_white, white])) == 0
    assert len(simulated(config, [white, near_white])) == 0

    config.sensor.timestamp_resolution_us = 100
    events = simulated(config, [black, white])
    gaps = np.diff(events["timestamp_us"])
    assert np.all(events["timestamp_us"] % 100 == 0)
    assert np.any(gaps == 0)
    assert np.all(gaps >= 0)
