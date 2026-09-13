# Event Camera Simulator - Final Validation Report

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
| analytical_timestamp_alignment | True | {"actual_events": 48, "expected_events": 48, "timestamp_mae_us": 0.5, "timestamp_max_error_us": 1} |
| asymmetric_threshold_validation | True | {"negative_events": 23, "positive_events": 46} |
| timestamp_quantization_floor_and_duplicates | True | {"duplicate_timestamp_bins": 24, "events": 34} |
| refractory_period | True | {"events": 9} |
| quantization_and_refractory_order | True | {"duplicate_bins": 0, "events": 10} |
| dark_and_saturation_edge_cases | True | {"black_log_finite": true, "near_saturation_events": 0} |
| vectorized_equals_pixel_loop | True | {"loop_events": 12630, "vectorized_events": 12630} |

## 960 FPS end-to-end demo

| Metric | Value |
|---|---:|
| frames_processed | 960 |
| event_count | 728361 |
| on_events | 316258 |
| off_events | 412103 |
| input_duration_s | 0.998958 |
| active_event_duration_s | 0.998893 |
| event_rate_over_input_hz | 729120.7438150552 |
| event_rate_over_active_hz | 729168.1891854282 |
| monotonic_timestamps | True |
| processing_fps | 128.7595664168783 |

## Threshold sweep

| C | Event count | Event rate over input | Count x C |
|---:|---:|---:|---:|
| 0.10 | 256,211 | 1546934.3 | 25621.1 |
| 0.15 | 156,546 | 945183.4 | 23481.9 |
| 0.20 | 114,563 | 691701.1 | 22912.6 |
| 0.30 | 72,298 | 436516.2 | 21689.4 |
| 0.40 | 51,748 | 312440.8 | 20699.2 |

The event count decreases monotonically as `C` increases, while `N*C`
remains in the same order of magnitude, consistent with `N ~ 1/C`.

## FPS convergence

| FPS | Events | Relative count diff | Timestamp RMSE vs 3840 FPS |
|---:|---:|---:|---:|
| 120 | 35 | 0.0789 | 38967.875 us |
| 240 | 37 | 0.0263 | 1363.356 us |
| 480 | 38 | 0.0000 | 790.464 us |
| 960 | 38 | 0.0000 | 326.466 us |
| 1920 | 38 | 0.0000 | 136.232 us |
| 3840 | 38 | 0.0000 | 0.000 us |

The timestamp error decreases as input FPS increases under the
piecewise-linear interpolation assumption.

## Simplified sensor non-idealities

| Model | Events | ON | OFF | ON/OFF ratio | Rate over input |
|---|---:|---:|---:|---:|---:|
| A_ideal | 69,767 | 34,221 | 35,546 | 0.9627 | 421234.7 |
| B_threshold_mismatch | 71,051 | 34,823 | 36,228 | 0.9612 | 428987.2 |
| C_threshold_plus_background | 71,150 | 34,865 | 36,285 | 0.9609 | 429584.9 |
| D_plus_refractory | 66,744 | 32,879 | 33,865 | 0.9709 | 402982.6 |

## Accumulation-window comparison

| Window (us) | Events | Active pixels |
|---:|---:|---:|
| 1000 | 812 | 157 |
| 5000 | 4,804 | 1,343 |
| 10000 | 9,088 | 2,125 |
| 20000 | 17,676 | 3,180 |

Smaller windows preserve more temporal detail but are sparse. Larger
windows improve spatial visibility while reducing temporal resolution.
Accumulation affects visualization only, not the raw event stream.

## Photometric preprocessing

| Preprocessing | Events | Rate over input |
|---|---:|---:|
| direct_log | 114,527 | 923915.3 |
| gamma_linearized | 304,904 | 2459729.7 |

The source-camera response curve is generally unknown. Gamma
linearization is an optional approximation, not a claim of exact
radiometric calibration.

## Efficiency

| Backend | Runtime (s) | Frames/s | MPixel-frames/s | Events |
|---|---:|---:|---:|---:|
| vectorized | 0.469 | 61.89 | 0.4278 | 889,455 |
| loop | 9.275 | 3.13 | 0.0216 | 889,455 |

Measured speedup `T_loop / T_vectorized = 19.79x`.

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
