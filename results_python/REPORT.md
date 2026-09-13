# Python Reimplementation and Validation Report

## What was rebuilt

The previous C++/CUDA implementation was replaced by a Python-first simulator.
The new pipeline directly implements the course requirement:

```text
High-FPS frames + sensor parameters
        -> log intensity
        -> temporal interpolation
        -> pixel threshold crossing
        -> (x, y, t, p) events
        -> raw file + illustrative video
```

No edge detector, tracking model or downstream vision model is used.

## End-to-end demo

Input:

- 960 FPS MJPG video
- 320 x 240 pixels
- 1 second, 960 frames
- synthetic moving bright and dark objects

Realistic simulation settings:

- `C+ = 0.20`
- `C- = 0.15`
- pixel-wise threshold variation `sigma = 0.02`
- background activity `0.02 Hz/pixel`
- refractory period `100 us`
- linear temporal interpolation
- 1 us timestamp resolution

Observed result:

| Metric | Result |
|---|---:|
| Events | 728,361 |
| ON events | 316,258 |
| OFF events | 412,103 |
| Event rate | 729,168 events/s |
| Unique event timestamps | 491,291 |
| Events exactly on input-frame timestamps | 0.52% |
| Monotonic timestamps | yes |
| Processing rate | 152.7 frames/s |
| Source frames | 960 |
| Event video frames | 192 |

The input contains only 960 sample timestamps, while the output contains
491,291 distinct microsecond timestamps. This verifies that temporal
interpolation is active rather than assigning every event to a frame timestamp.

Artifacts:

- `demo/input_960fps.avi`
- `demo/events.csv`
- `demo/events.npz`
- `demo/statistics.json`
- `demo/event_video.avi`
- `demo/event_preview.png`

## Unit and integration tests

`python -m pytest -q` result:

```text
12 passed
```

Covered checks include static scenes, positive/negative ramps, multiple
threshold crossings, interpolation, quantization, refractory behavior,
threshold variation reproducibility, image sequence loading, CSV/NPZ output,
video rendering and exact equivalence between the vectorized and pixel-loop
backends.

## Analytic validation

`python -m evsim validate` passes all checks.

| Check | Result |
|---|---|
| Constant intensity produces zero events | PASS |
| Bright ramp produces multiple positive events | PASS |
| Dark ramp produces multiple negative events | PASS |
| Timestamp quantization | PASS |
| Refractory period | PASS |
| Vectorized equals pixel-loop | PASS |

## Threshold sweep

The moving-scene event count decreases monotonically as the contrast threshold
increases, which agrees with the physical model.

| Threshold | Event count |
|---:|---:|
| 0.10 | 256,211 |
| 0.15 | 156,546 |
| 0.20 | 114,563 |
| 0.25 | 89,068 |
| 0.30 | 72,298 |
| 0.40 | 51,748 |

## FPS interpolation convergence

A nonlinear single-pixel brightness signal was sampled at several input FPS
values and compared with a 3840 FPS reference.

| Input FPS | Event count | Count error | Timestamp RMSE vs 3840 FPS |
|---:|---:|---:|---:|
| 120 | 35 | -3 | 38,967.9 us |
| 240 | 37 | -1 | 1,363.4 us |
| 480 | 38 | 0 | 790.5 us |
| 960 | 38 | 0 | 326.5 us |
| 1920 | 38 | 0 | 136.2 us |
| 3840 | 38 | 0 | 0.0 us |

The timestamp error decreases as the input sampling rate increases, showing the
expected convergence of linear interpolation.

## Noise ablation

| Variant | Event count |
|---|---:|
| Ideal | 69,767 |
| Threshold variation | 71,051 |
| Background activity | 69,865 |
| Combined | 71,150 |

Both non-idealities add a controlled number of events without changing the
underlying contrast-generation behavior.

## Efficiency

Benchmark configuration: 96 x 72 pixels, 30 frames.

| Backend | Runtime | Processing FPS | Events |
|---|---:|---:|---:|
| NumPy vectorized | 0.515 s | 56.3 | 889,455 |
| Pixel loop reference | 8.902 s | 3.3 | 889,455 |
| Speedup | 17.29x | 17.29x | identical |

The loop backend remains useful for correctness checking, while the vectorized
backend is the default production path.

## Main implementation files

- `evsim/simulator.py`: pixel model and both execution backends
- `evsim/preprocessing.py`: DN, linearization, log intensity
- `evsim/sources.py`: video and image-sequence readers
- `evsim/events.py`: event dtype and CSV/NPZ output
- `evsim/visualization.py`: event accumulation and video rendering
- `evsim/validation.py`: analytic and performance validation
- `evsim/cli.py`: command-line interface
