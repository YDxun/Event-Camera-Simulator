# Event Camera Simulator - Final Validation Report

Source revision: `b444b9e`

## Scope

High-FPS frames are converted to asynchronous `(x, y, t, p)` events using
log-intensity contrast thresholding. No edge detection, tracking, SLAM or
downstream learning model is used.

## Model conventions

- `L = log(I + epsilon)`
- Stateful pixel reference `L_ref`
- Separate thresholds `C+` and `C-`
- Multiple threshold crossings per frame interval
- Piecewise-linear log-intensity interpolation inside each interval
- Floor timestamp quantization; equal timestamps are allowed
- Refractory check is applied after quantization
- Background activity uses a simplified Poisson process

## Correctness validation

| Check | Passed | Details |
|---|---:|---|
| static_scene_no_events | True | {"events": 0} |
| bright_ramp_multiple_positive | True | {"events": 34} |
| dark_ramp_multiple_negative | True | {"events": 34} |
| analytical_timestamp_alignment | True | {"actual_events": 48, "expected_events": 48, "timestamp_mae_us": 0.0, "timestamp_max_error_us": 0} |
| asymmetric_threshold_validation | True | {"negative_events": 23, "positive_events": 46} |
| timestamp_quantization_floor_and_duplicates | True | {"duplicate_timestamp_bins": 24, "events": 34} |
| pure_floor_timestamp_quantization | True | {"events": 60, "first_timestamp_us": 0} |
| zero_delta_backend_consistency | True | {"last_timestamp_us": 2000, "loop_events": 11, "vectorized_events": 11} |
| refractory_period | True | {"events": 9} |
| quantization_and_refractory_order | True | {"duplicate_bins": 0, "events": 10} |
| dark_and_saturation_edge_cases | True | {"black_log_finite": true, "near_saturation_events": 0} |
| vectorized_equals_pixel_loop | True | {"loop_events": 12630, "vectorized_events": 12630} |

## 960 FPS end-to-end demo

| Metric | Value |
|---|---:|
| frames_processed | 960 |
| event_count | 1730424 |
| on_events | 814385 |
| off_events | 916039 |
| input_duration_s | 0.998958 |
| active_event_duration_s | 0.998921 |
| event_rate_over_input_hz | 1732228.9825998691 |
| event_rate_over_active_hz | 1732293.1443027027 |
| monotonic_timestamps | True |
| processing_fps | 96.47181858193868 |

## Threshold sweep

| C | Event count | Event rate over input | Count x C |
|---:|---:|---:|---:|
| 0.10 | 256,196 | 1546843.8 | 25619.6 |
| 0.15 | 156,535 | 945117.0 | 23480.2 |
| 0.20 | 114,555 | 691652.8 | 22911.0 |
| 0.30 | 72,293 | 436486.0 | 21687.9 |
| 0.40 | 51,744 | 312416.6 | 20697.6 |

The event count decreases monotonically as `C` increases, while `N*C`
remains in the same order of magnitude, consistent with `N ~ 1/C`.

## FPS convergence

| FPS | Events | Relative count diff | Matched | Unmatched | Timestamp RMSE vs 3840 FPS |
|---:|---:|---:|---:|---:|---:|
| 120 | 35 | 0.0789 | 35 | 3 | 2221.089 us |
| 240 | 37 | 0.0263 | 37 | 1 | 1363.397 us |
| 480 | 38 | 0.0000 | 38 | 0 | 790.724 us |
| 960 | 38 | 0.0000 | 38 | 0 | 326.346 us |
| 1920 | 38 | 0.0000 | 38 | 0 | 136.247 us |
| 3840 | 38 | 0.0000 | 38 | 0 | 0.000 us |

Matching method: events are aligned separately by polarity using an
order-preserving dynamic-programming sequence alignment. The match cost
is absolute timestamp difference and the insertion/deletion penalty is
5000 us. RMSE is computed only over matched pairs; unmatched events are
reported explicitly and excluded from RMSE.

The timestamp error decreases as input FPS increases under the
piecewise-linear interpolation assumption.

## Simplified sensor non-idealities

| Model | Events | ON | OFF | ON/OFF ratio | Rate over input |
|---|---:|---:|---:|---:|---:|
| A_ideal | 69,767 | 34,221 | 35,546 | 0.9627 | 421234.7 |
| B_threshold_mismatch | 71,051 | 34,823 | 36,228 | 0.9612 | 428987.2 |
| C_threshold_plus_background | 71,089 | 34,846 | 36,243 | 0.9615 | 429216.6 |
| D_plus_refractory | 66,689 | 32,864 | 33,825 | 0.9716 | 402650.6 |

## Accumulation-window comparison

| Window (us) | Events | Active pixels |
|---:|---:|---:|
| 1000 | 812 | 157 |
| 5000 | 4,807 | 1,343 |
| 10000 | 9,088 | 2,125 |
| 20000 | 17,677 | 3,180 |

Smaller windows preserve more temporal detail but are sparse. Larger
windows improve spatial visibility while reducing temporal resolution.
Accumulation affects visualization only, not the raw event stream.

## Photometric preprocessing

| Preprocessing | Events | Rate over input |
|---|---:|---:|
| direct_log | 114,519 | 923850.8 |
| gamma_linearized | 304,886 | 2459584.5 |

The source-camera response curve is generally unknown. Gamma
linearization is an optional approximation, not a claim of exact
radiometric calibration.

## Efficiency

| Backend | Runtime (s) | Frames/s | MPixel-frames/s | Events |
|---|---:|---:|---:|---:|
| vectorized | 0.345 | 84.09 | 0.5812 | 889,455 |
| loop | 5.827 | 4.98 | 0.0344 | 889,455 |

Measured speedup `T_loop / T_vectorized = 16.90x`.

## Assumptions and limitations

- Piecewise-linear log-intensity between consecutive frames
- Independent pixels without readout arbitration
- Fixed per-pixel Gaussian threshold mismatch
- Simplified Poisson background activity
- Source motion blur, temporal aliasing and saturation are not recovered
- No tracking, detection, optical flow, SLAM, SNN or reconstruction

## Verification artifacts

- `validation.json`
- `threshold_sweep.csv` and `threshold_sweep.png`
- `fps_sweep.csv` and `fps_sweep.png`
- `noise_ablation.json` and `noise_ablation.png`
- `accumulation_window.json` and `accumulation_window_montage.png`
- `linearization_comparison.json` and `linearization_comparison.png`
- `../demo/events.csv`, `../demo/events.npz` and `../demo/event_video.avi`

