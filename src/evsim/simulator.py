"""Vectorized and pixel-loop event generation from frame pairs.

Mathematical Formulation:
-------------------------
1. Photoreceptor Log Intensity:
   L(x, y, t) = ln(I(x, y, t) + epsilon)
   where epsilon prevents log(0) singularity and models dark current offset.

2. Temporal Contrast & Threshold Crossing:
   Between frame intervals [t0, t1], the log-intensity change relative to the
   pixel's stateful reference level L_ref is:
   Delta L = L(x, y, t1) - L_ref(x, y)

   An event fires when |Delta L| exceeds the contrast threshold:
   - ON event (polarity = +1):  Delta L >= C+
   - OFF event (polarity = -1): Delta L <= -C-

3. Multiple Crossings per Frame Interval:
   For high-speed motion, multiple events can fire within a single interval:
   k_max = floor(|Delta L| / C)
   Crossing levels: L_k = L_ref +/- k * C, for k in 1..k_max.

4. Piecewise-Linear Interpolation:
   Under the linear intensity assumption between t0 and t1:
   alpha_k = (L_k - L(t0)) / (L(t1) - L(t0))
   t_k^* = t0 + alpha_k * (t1 - t0)
   If the frame interval has no intensity change, the residual crossing is
   assigned to t1 as a deterministic model convention.

5. Quantization & Refractory Filter:
   - Timestamps are floor-quantized to sensor resolution:
     t_quant = floor(t_k / delta_t_res) * delta_t_res
   - Refractory period: events occurring within delta_t_refr from the
     last accepted event at that pixel are dropped.
   - For accepted events, the reference level updates: L_ref <- L_k.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimulatorConfig
from .events import EVENT_DTYPE, empty_events, sort_events
from .preprocessing import to_log_intensity


def _interpolation_alpha(step: float, numerator: float) -> float:
    """Return the fractional crossing position inside one frame interval.

    A zero-length interval has no temporal information. We use the explicit
    convention that a residual crossing is assigned to the end of the interval.
    """
    if abs(float(step)) <= 1e-12:
        return 1.0
    return min(1.0, max(0.0, float(numerator) / float(step)))


def _interpolation_alphas(step: np.ndarray, numerator: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of :func:`_interpolation_alpha`."""
    alpha = np.ones(step.shape, dtype=np.float64)
    nonzero = np.abs(step.astype(np.float64)) > 1e-12
    alpha[nonzero] = numerator[nonzero].astype(np.float64) / step[nonzero].astype(
        np.float64
    )
    return np.clip(alpha, 0.0, 1.0)


def _quantize_timestamp(value: float, resolution: int) -> int:
    """Apply pure floor quantization to one continuous timestamp."""
    return int(np.floor(float(value) / int(resolution)) * int(resolution))


def _quantize_timestamps(values: np.ndarray, resolution: int) -> np.ndarray:
    """Apply pure floor quantization to continuous timestamps in microseconds."""
    resolution = int(resolution)
    return (
        np.floor(values.astype(np.float64) / resolution).astype(np.int64) * resolution
    )


def _crossing_counts(delta: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """Calculate the number of threshold crossings for each pixel.

    Evaluates only pixels where delta >= threshold, avoiding floating-point
    divisions across the static scene background.
    """
    counts = np.zeros_like(delta, dtype=np.int64)
    active = delta >= threshold
    if np.any(active):
        counts[active] = np.floor(delta[active] / threshold[active] + 1e-12).astype(
            np.int64
        )
    return counts


@dataclass
class PairEvents:
    """Generated events and breakdown for a consecutive frame pair."""

    events: np.ndarray
    contrast_events: int
    background_events: int


class EventSimulator:
    """Stateful event-camera pixel model for a sequence of grayscale frames."""

    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.width = 0
        self.height = 0
        self.initialized = False
        self.last_timestamp_us = 0
        self.last_log_intensity: np.ndarray | None = None
        self.ref_log_intensity: np.ndarray | None = None
        self.pos_thresholds: np.ndarray | None = None
        self.neg_thresholds: np.ndarray | None = None
        self.last_event_time_us: np.ndarray | None = None
        self._current_log1: np.ndarray | None = None
        self.rng = np.random.default_rng(config.noise.random_seed)

    def initialize(self, image: np.ndarray, timestamp_us: int) -> None:
        if image.ndim != 2:
            raise ValueError(f"Expected grayscale frame, got shape {image.shape}")
        self.height, self.width = image.shape
        if (
            self.width > np.iinfo(np.uint16).max
            or self.height > np.iinfo(np.uint16).max
        ):
            raise ValueError("Sensor dimensions must fit in uint16")
        count = self.width * self.height
        self.rng = np.random.default_rng(self.config.noise.random_seed)

        sensor = self.config.sensor
        pos = np.full(count, sensor.positive_threshold, dtype=np.float32)
        neg = np.full(count, sensor.negative_threshold, dtype=np.float32)
        if (
            self.config.noise.enable_threshold_variation
            and self.config.noise.threshold_sigma > 0
        ):
            sigma = self.config.noise.threshold_sigma
            pos = self.rng.normal(sensor.positive_threshold, sigma, count).astype(
                np.float32
            )
            neg = self.rng.normal(sensor.negative_threshold, sigma, count).astype(
                np.float32
            )
            np.maximum(pos, np.float32(1e-4), out=pos)
            np.maximum(neg, np.float32(1e-4), out=neg)

        log0 = to_log_intensity(image, self.config.input, sensor).reshape(-1)
        self.last_log_intensity = log0.copy()
        self.ref_log_intensity = log0.copy()
        self.pos_thresholds = pos
        self.neg_thresholds = neg
        self.last_event_time_us = np.full(
            count,
            int(timestamp_us) - int(sensor.refractory_period_us) - 1,
            dtype=np.int64,
        )
        self.last_timestamp_us = int(timestamp_us)
        self.initialized = True

    def reset(self) -> None:
        """Reset the simulator internal state to uninitialized."""
        self.initialized = False
        self.last_log_intensity = None
        self.ref_log_intensity = None
        self.pos_thresholds = None
        self.neg_thresholds = None
        self.last_event_time_us = None
        self._current_log1 = None
        self.last_timestamp_us = 0

    def _build_candidates(
        self, log1: np.ndarray, timestamp_us: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate candidate events for the frame pair before refractory filtering."""
        assert self.last_log_intensity is not None
        assert self.ref_log_intensity is not None
        assert self.pos_thresholds is not None
        assert self.neg_thresholds is not None

        t0 = self.last_timestamp_us
        t1 = int(timestamp_us)
        log0 = self.last_log_intensity
        ref = self.ref_log_intensity
        delta = log1 - ref
        pos_counts = _crossing_counts(delta, self.pos_thresholds)
        neg_counts = _crossing_counts(-delta, self.neg_thresholds)
        contrast_counts = pos_counts + neg_counts
        total_contrast = int(contrast_counts.sum())
        limit = self.config.runtime.max_candidate_events_per_frame
        if limit and total_contrast > limit:
            raise RuntimeError(
                f"Frame pair would generate {total_contrast:,} candidate events, "
                f"exceeding runtime.max_candidate_events_per_frame={limit:,}. "
                "Raise the contrast threshold, enable a refractory period, or increase "
                "the configured limit (0 disables it)."
            )

        pixel_parts: list[np.ndarray] = []
        time_parts: list[np.ndarray] = []
        polarity_parts: list[np.ndarray] = []
        level_parts: list[np.ndarray] = []
        kind_parts: list[np.ndarray] = []

        if total_contrast:
            pixels = np.repeat(
                np.arange(contrast_counts.size, dtype=np.int64), contrast_counts
            )
            starts = np.cumsum(contrast_counts) - contrast_counts
            rank = np.arange(total_contrast, dtype=np.int64) - starts[pixels]
            k = rank + 1
            pos_for_pixel = pos_counts[pixels]
            is_positive = k <= pos_for_pixel
            steps = np.where(
                is_positive,
                k.astype(np.float32) * self.pos_thresholds[pixels],
                -k.astype(np.float32) * self.neg_thresholds[pixels],
            )
            levels = ref[pixels] + steps
            denominator = log1[pixels] - log0[pixels]
            numerator = levels - log0[pixels]
            alpha = _interpolation_alphas(denominator, numerator)
            if self.config.simulation.interpolation == "linear":
                raw_times = t0 + alpha * float(t1 - t0)
            else:
                raw_times = np.full(total_contrast, float(t1), dtype=np.float64)
            pixel_parts.append(pixels)
            time_parts.append(raw_times)
            polarity_parts.append(np.where(is_positive, np.int8(1), np.int8(-1)))
            level_parts.append(levels.astype(np.float32))
            kind_parts.append(np.zeros(total_contrast, dtype=np.int8))

        background_counts = np.zeros_like(contrast_counts)
        if self.config.noise.background_rate_hz > 0:
            dt_s = float(t1 - t0) / 1_000_000.0
            mean = self.config.noise.background_rate_hz * dt_s
            background_counts = self.rng.poisson(mean, size=contrast_counts.size)
            total_background = int(background_counts.sum())
            if limit and total_contrast + total_background > limit:
                raise RuntimeError(
                    f"Frame pair would generate {total_contrast + total_background:,} "
                    "candidate events including background activity, exceeding "
                    f"runtime.max_candidate_events_per_frame={limit:,}."
                )
            if total_background:
                bg_pixels = np.repeat(
                    np.arange(background_counts.size, dtype=np.int64),
                    background_counts,
                )
                bg_times = self.rng.integers(
                    t0, t1 + 1, size=total_background, dtype=np.int64
                )
                bg_polarity = self.rng.choice(
                    np.array([-1, 1], dtype=np.int8), size=total_background
                )
                pixel_parts.append(bg_pixels)
                time_parts.append(bg_times)
                polarity_parts.append(bg_polarity)
                level_parts.append(np.full(total_background, np.nan, dtype=np.float32))
                kind_parts.append(np.ones(total_background, dtype=np.int8))

        total = sum(part.size for part in pixel_parts)
        if total == 0:
            empty = empty_events()
            return (
                empty["timestamp_us"],
                empty["x"],
                empty["y"],
                empty["polarity"],
                np.empty(0, dtype=np.int8),
            )

        pixels = np.concatenate(pixel_parts)
        raw_times = np.concatenate(time_parts).astype(np.float64)
        polarity = np.concatenate(polarity_parts)
        levels = np.concatenate(level_parts)
        kind = np.concatenate(kind_parts)

        # Sort by continuous crossing time so quantization cannot reorder events
        # before refractory filtering. Then apply the one shared floor quantizer.
        order = np.lexsort((polarity, kind, raw_times, pixels))
        pixels = pixels[order]
        raw_times = raw_times[order]
        polarity = polarity[order]
        levels = levels[order]
        kind = kind[order]
        times = _quantize_timestamps(
            raw_times, self.config.sensor.timestamp_resolution_us
        )
        return pixels, times, polarity, levels, kind

    def _apply_refractory(
        self,
        pixels: np.ndarray,
        times: np.ndarray,
        polarity: np.ndarray,
        levels: np.ndarray,
        kind: np.ndarray,
    ) -> PairEvents:
        """Filter candidate events through the pixel refractory period.

        Candidates are processed rank-by-rank per pixel. An event is accepted
        if (t_event - t_last_event >= refractory_period_us).
        Accepted contrast events update the pixel's reference log-intensity L_ref.
        """
        assert self.last_event_time_us is not None
        assert self.ref_log_intensity is not None
        if pixels.size == 0:
            return PairEvents(empty_events(), 0, 0)

        counts = np.bincount(pixels, minlength=self.width * self.height)
        starts = np.cumsum(counts) - counts
        ranks = np.arange(pixels.size, dtype=np.int64) - starts[pixels]
        refractory = int(self.config.sensor.refractory_period_us)
        accepted_events: list[np.ndarray] = []
        accepted_contrast = 0
        accepted_background = 0

        for rank in range(int(ranks.max()) + 1):
            selected = ranks == rank
            if not np.any(selected):
                continue
            idx = np.flatnonzero(selected)
            pix = pixels[idx]
            timestamps = times[idx]
            keep = timestamps - self.last_event_time_us[pix] >= refractory
            if not np.any(keep):
                continue
            idx_keep = idx[keep]
            pix = pixels[idx_keep]
            timestamps = times[idx_keep]
            pol = polarity[idx_keep]
            lev = levels[idx_keep]
            candidate_kind = kind[idx_keep]

            self.last_event_time_us[pix] = timestamps
            contrast = candidate_kind == 0
            if np.any(contrast):
                self.ref_log_intensity[pix[contrast]] = lev[contrast]
                accepted_contrast += int(np.count_nonzero(contrast))
            accepted_background += int(np.count_nonzero(~contrast))

            block = np.empty(pix.size, dtype=EVENT_DTYPE)
            block["timestamp_us"] = timestamps
            block["x"] = pix % self.width
            block["y"] = pix // self.width
            block["polarity"] = pol
            accepted_events.append(block)

        if not accepted_events:
            return PairEvents(empty_events(), 0, 0)
        events = sort_events(np.concatenate(accepted_events))
        return PairEvents(events, accepted_contrast, accepted_background)

    def _process_pair_vectorized(
        self, log1: np.ndarray, timestamp_us: int
    ) -> PairEvents:
        """High-performance vectorized simulation path across all sensor pixels."""
        candidates = self._build_candidates(log1, timestamp_us)
        return self._apply_refractory(*candidates)

    def _process_pair_loop(self, log1: np.ndarray, timestamp_us: int) -> PairEvents:
        """Reference explicit nested-loop simulation path for verification and testing."""
        assert self.last_log_intensity is not None
        assert self.ref_log_intensity is not None
        assert self.pos_thresholds is not None
        assert self.neg_thresholds is not None
        assert self.last_event_time_us is not None

        t0 = self.last_timestamp_us
        t1 = int(timestamp_us)
        resolution = int(self.config.sensor.timestamp_resolution_us)
        refractory = int(self.config.sensor.refractory_period_us)
        dt_s = float(t1 - t0) / 1_000_000.0
        output: list[tuple[int, int, int]] = []
        accepted_contrast = 0
        accepted_background = 0

        for pixel in range(log1.size):
            candidates: list[tuple[float, int, int, float]] = []
            ref_value = np.float32(self.ref_log_intensity[pixel])
            delta = np.float32(np.float32(log1[pixel]) - ref_value)
            if delta >= self.pos_thresholds[pixel]:
                ratio = np.float32(delta / self.pos_thresholds[pixel])
                count = int(np.floor(float(ratio) + 1e-12))
                for k in range(1, count + 1):
                    level = np.float32(
                        ref_value + np.float32(k) * self.pos_thresholds[pixel]
                    )
                    candidates.append(
                        (self._interpolate_time(pixel, level, t0, t1), 0, 1, level)
                    )
            elif delta <= -self.neg_thresholds[pixel]:
                ratio = np.float32(-delta / self.neg_thresholds[pixel])
                count = int(np.floor(float(ratio) + 1e-12))
                for k in range(1, count + 1):
                    level = np.float32(
                        ref_value - np.float32(k) * self.neg_thresholds[pixel]
                    )
                    candidates.append(
                        (self._interpolate_time(pixel, level, t0, t1), 0, -1, level)
                    )

            if self.config.noise.background_rate_hz > 0:
                count = int(
                    self.rng.poisson(self.config.noise.background_rate_hz * dt_s)
                )
                for _ in range(count):
                    time = int(self.rng.integers(t0, t1 + 1))
                    polarity = int(self.rng.choice([-1, 1]))
                    candidates.append((time, 1, polarity, float("nan")))

            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            for raw_time, candidate_kind, polarity, level in candidates:
                time = _quantize_timestamp(raw_time, resolution)
                if time - self.last_event_time_us[pixel] < refractory:
                    continue
                self.last_event_time_us[pixel] = time
                if candidate_kind == 0:
                    self.ref_log_intensity[pixel] = level
                    accepted_contrast += 1
                else:
                    accepted_background += 1
                output.append((time, pixel, polarity))

        if not output:
            return PairEvents(empty_events(), 0, 0)
        raw = np.asarray(output, dtype=np.int64)
        events = np.empty(raw.shape[0], dtype=EVENT_DTYPE)
        events["timestamp_us"] = raw[:, 0]
        events["x"] = raw[:, 1] % self.width
        events["y"] = raw[:, 1] // self.width
        events["polarity"] = raw[:, 2]
        return PairEvents(sort_events(events), accepted_contrast, accepted_background)

    def _interpolate_time(self, pixel: int, level: float, t0: int, t1: int) -> float:
        """Compute a continuous linearly interpolated crossing time within [t0, t1]."""
        assert self.last_log_intensity is not None
        assert self._current_log1 is not None
        if self.config.simulation.interpolation == "none":
            return float(t1)
        step = np.float32(self._current_log1[pixel] - self.last_log_intensity[pixel])
        numerator = np.float32(np.float32(level) - self.last_log_intensity[pixel])
        alpha = _interpolation_alpha(float(step), float(numerator))
        return float(t0 + alpha * float(t1 - t0))

    def process(self, image: np.ndarray, timestamp_us: int) -> PairEvents:
        """Process the next video frame and return generated events.

        Args:
            image: Grayscale frame matching the sensor resolution (H, W).
            timestamp_us: Strictly increasing frame timestamp in microseconds.

        Returns:
            PairEvents containing the accepted events and statistics breakdown.
        """
        if not self.initialized:
            raise RuntimeError("Simulator must be initialized before processing frames")
        if timestamp_us <= self.last_timestamp_us:
            raise ValueError(
                f"Non-monotonic timestamp: {timestamp_us} <= {self.last_timestamp_us}"
            )
        if image.shape != (self.height, self.width):
            raise ValueError(
                f"Frame shape {image.shape} does not match sensor {(self.height, self.width)}"
            )

        log1 = to_log_intensity(image, self.config.input, self.config.sensor).reshape(
            -1
        )
        self._current_log1 = log1
        if self.config.simulation.backend == "vectorized":
            result = self._process_pair_vectorized(log1, timestamp_us)
        else:
            result = self._process_pair_loop(log1, timestamp_us)

        self.last_log_intensity = log1.copy()
        self.last_timestamp_us = int(timestamp_us)
        return result
