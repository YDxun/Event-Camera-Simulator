# Configuration Reference Guide

The event-camera simulator uses a modular JSON-based configuration model organized into logical subsections. This document details each configuration field, its physical unit, valid range, and practical recommendations.

---

## Configuration Sections

### 1. `input` — Input Data & Photometric Preprocessing

| Field | Type | Default | Unit / Range | Description |
|---|---|---|---|---|
| `fallback_fps` | `float` | `960.0` | `> 0` (Hz) | Frame rate assumed when processing image sequences without timestamp files or when video container metadata lacks valid FPS. |
| `linearize` | `bool` | `false` | `true` / `false` | If `true`, applies inverse gamma transformation before logarithm to approximate linear scene radiance. |
| `gamma` | `float` | `2.2` | `> 0` | Gamma exponent used when `linearize` is enabled (2.2 standard for sRGB/Rec.709). |
| `bit_depth` | `int` | `8` | `8` or `16` | Sensor bit depth used to normalize digital numbers (DN) to `[0.0, 1.0]`. |
| `timestamp_scale_us` | `float` | `1.0` | `> 0` | Multiplier applied to raw timestamp values from `ts_frame.txt` to convert them to microseconds. |
| `allow_timestamp_fallback` | `bool` | `false` | `true` / `false` | Permit a present but invalid timestamp file to fall back to fixed FPS. The default raises an error to protect timing integrity. |

### 2. `sensor` — Analog Pixel Model Parameters

| Field | Type | Default | Unit / Range | Description |
|---|---|---|---|---|
| `positive_threshold` | `float` | `0.20` | `> 0` (log-units) | Positive temporal contrast threshold $C^+$ required to trigger an ON event. Typically `0.10`–`0.40`. |
| `negative_threshold` | `float` | `0.20` | `> 0` (log-units) | Negative temporal contrast threshold $C^-$ required to trigger an OFF event. Real sensors often show slight asymmetry. |
| `log_epsilon` | `float` | `1e-3` | `> 0` | Offset $\epsilon$ in $L = \ln(I + \epsilon)$. Prevents $\ln(0)$ singularity and models sensor dark current. |
| `timestamp_resolution_us` | `int` | `1` | `≥ 1` ($\mu\text{s}$) | Floor-quantization temporal resolution. Event timestamps are emitted as integer multiples of this bin size. |
| `refractory_period_us` | `int` | `0` | `≥ 0` ($\mu\text{s}$) | Minimum time interval after an event before the same pixel can fire again. Models photoreceptor analog circuit reset dead-time (typically 50–200 $\mu\text{s}$). |

### 3. `simulation` — Algorithm & Backend Selection

| Field | Type | Default | Options | Description |
|---|---|---|---|---|
| `interpolation` | `string` | `"linear"` | `"linear"`, `"none"` | Sub-frame event timestamp interpolation. `"linear"` uses piecewise-linear log-intensity; `"none"` assigns all events the frame timestamp $t_1$. |
| `backend` | `string` | `"vectorized"` | `"vectorized"`, `"loop"` | Execution engine. `"vectorized"` uses broadcast NumPy arrays for high throughput. `"loop"` is an explicit reference loop for correctness verification. |

### 4. `noise` — Sensor Non-Idealities & Stochastic Perturbations

| Field | Type | Default | Unit / Range | Description |
|---|---|---|---|---|
| `enable_threshold_variation` | `bool` | `false` | `true` / `false` | If `true`, applies static Gaussian threshold variation across pixels to model analog transistor mismatch. |
| `threshold_sigma` | `float` | `0.03` | `≥ 0` | Standard deviation $\sigma_C$ of the Gaussian threshold distribution sampled at simulator initialization. |
| `background_rate_hz` | `float` | `0.0` | `≥ 0` (Hz) | Mean Poisson background / leakage event firing rate per pixel. Models thermal dark events. |
| `random_seed` | `int` | `42` | Any integer | Pseudo-random number generator seed ensuring deterministic, reproducible noise generation across runs. |

### 5. `visualization` — Accumulation & Rendering

| Field | Type | Default | Unit / Range | Description |
|---|---|---|---|---|
| `accumulation_time_us` | `int` | `10000` | `> 0` ($\mu\text{s}$) | Duration of temporal window used to accumulate asynchronous events into visual 2D frames (10,000 $\mu\text{s}$ = 10 ms). *Does not affect raw events.* |
| `playback_fps` | `float` | `30.0` | `> 0` (fps) | Frame rate of the rendered output video. |
| `overlay_opacity` | `float` | `0.65` | `[0.0, 1.0]` | Alpha blending opacity when superimposing color-coded events onto original grayscale video. |
| `output_video` | `string` | `""` | File path | Destination path for the 3-panel illustrative video (`.mp4` or `.avi`). |
| `display` | `bool` | `false` | `true` / `false` | Whether to display interactive windows during execution. |

### 6. `output` — Persistence & Serialization

| Field | Type | Default | Description |
|---|---|---|---|
| `csv_path` | `string` | `""` | Path to save the event stream as CSV (`timestamp_s,x,y,polarity`). |
| `npz_path` | `string` | `""` | Path to save the event stream as compressed NumPy NPZ archive (`events` key). |
| `statistics_path` | `string` | `""` | Path to save JSON summary metrics (event counts, rates, frame throughput). |

### 7. `runtime` — Execution Bounds

| Field | Type | Default | Description |
|---|---|---|---|
| `max_frames` | `int` | `0` | Maximum number of frames to process (`0` = process until end of source). |
| `progress` | `bool` | `true` | Display a terminal progress bar (`tqdm`) during simulation. |
| `max_candidate_events_per_frame` | `int` | `10000000` | Maximum candidate events expanded for one frame pair (`0` = unlimited). Prevents accidental memory exhaustion from extreme thresholds. |

## Timestamp and streaming behavior

Video frames use container presentation timestamps (PTS) when available. A missing
or non-monotonic PTS is repaired from the configured fallback FPS and recorded in
the result as `timestamp_source` and `timestamp_warning`. Image timestamp files are
strict by default because silently replacing experimental timing can invalidate the
event stream.

`evsim simulate --stream` keeps per-frame event chunks off the Python heap. CSV is
written incrementally; NPZ is assembled from temporary disk-backed chunks while
preserving the existing `events` array format. At least one event output path is
required in streaming mode.

---

## Included Configuration Presets

### `configs/ideal.json`
Deterministic, noiseless configuration ideal for baseline benchmarking, unit tests, and theoretical validation:
- $C^+ = C^- = 0.20$
- No threshold mismatch (`enable_threshold_variation: false`)
- No background noise (`background_rate_hz: 0.0`)
- No refractory dead-time (`refractory_period_us: 0`)
- Linear sub-frame temporal interpolation
- Vectorized execution

### `configs/enhanced.json`
Simplified enhanced model with phenomenological sensor non-idealities:
- Asymmetric contrast thresholds: $C^+ = 0.20$, $C^- = 0.18$
- Static Gaussian threshold mismatch ($\sigma = 0.02$)
- Poisson background activity ($0.02\text{ Hz/pixel}$)
- Pixel refractory period ($100\ \mu\text{s}$)
- Inverse gamma photometric linearization enabled ($\gamma = 2.2$)
- $5\text{ ms}$ accumulation window for crisp event preview
