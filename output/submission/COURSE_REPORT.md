# Event Camera Simulator - Course-Style Final Report

## Abstract

This project implements a configurable event-camera simulator that converts
high-frame-rate intensity frames into asynchronous `(x, y, t, p)` events. The
core model combines logarithmic intensity, stateful per-pixel reference levels,
contrast thresholds, temporal interpolation and timestamp quantization.
Simplified sensor non-idealities and a vectorized NumPy implementation are also
included. The simulator is validated analytically, cross-checked against an
independent pixel-loop backend, and evaluated with parameter, noise,
representation and efficiency experiments.

## 1. Motivation

Event cameras report asynchronous brightness changes rather than complete
frames. A simulator is useful because it allows controlled experiments with
thresholds, timestamps, resolution and non-idealities that are difficult to
sweep on physical hardware. The CA problem is therefore:

```text
High-FPS frames + sensor parameters -> simulated events + illustrative video
```

The project deliberately stays within frame-to-event simulation and does not
perform downstream detection, tracking, optical flow, SLAM or reconstruction.

## 2. Problem Formulation

Input frames are written as:

```text
I(x, y, t_k), k = 0, 1, ..., K
```

The output is an asynchronous event set:

```text
E = {(x_i, y_i, t_i, p_i)}
p_i in {-1, +1}
```

The objective is not to produce an edge image. The objective is to reconstruct
sub-frame threshold crossings from discrete high-FPS observations.

## 3. Event Generation Model

The photometric preprocessing is:

```text
L(x,y,t) = log(I(x,y,t) + epsilon)
```

Each pixel stores a reference log intensity `L_ref(x,y)`. For a new sample:

```text
Delta L = L - L_ref
```

ON event:

```text
Delta L >= C+ -> p = +1, L_ref <- L_ref + C+
```

OFF event:

```text
Delta L <= -C- -> p = -1, L_ref <- L_ref - C-
```

The number of crossings in a frame interval is:

```text
N+ = floor(max(L1 - L_ref, 0) / C+)
N- = floor(max(L_ref - L1, 0) / C-)
N  = N+ + N-
```

Between consecutive frames, log intensity is assumed piecewise linear:

```text
L(t) = L0 + (t - t0) / (t1 - t0) * (L1 - L0)
```

The crossing timestamp is:

```text
t_e = t0 + (L_e - L0) / (L1 - L0) * (t1 - t0)
```

The emitted timestamp is floor quantized:

```text
t_q = floor(t_e / Delta t_q) * Delta t_q
```

Equal timestamps are allowed; the stream is non-decreasing. Refractory
suppression is applied to quantized timestamps.

## 4. Proposed Framework

```text
High-FPS frames / video
          |
          v
Frame source and timestamps
          |
          v
Grayscale + optional linearization
          |
          v
Log intensity L = log(I + epsilon)
          |
          v
Piecewise-linear temporal interpolation
          |
          v
Stateful pixel model: L_ref, C+, C-, t_last
          |
          v
Threshold crossings and refractory check
          |
          v
(x, y, t, p) event stream
          |
    +-----+------+
    |            |
    v            v
CSV / NPZ   Event representation and video
```

## 5. Implementation

The default backend is vectorized with NumPy. It processes the complete pixel
array per frame interval. An independent pixel-loop backend is retained as a
reference implementation and is used to verify event-for-event equality under
the deterministic ideal model.

Implemented extensions include:

- asymmetric ON/OFF thresholds;
- fixed per-pixel Gaussian threshold mismatch;
- simplified Poisson background activity;
- refractory period;
- optional gamma linearization;
- binary event, event-count and overlay representations;
- configurable event accumulation window;
- CSV, NPZ and illustrative video output;
- a lightweight Streamlit demo that calls the same simulator core as the CLI.

## 6. Validation

The current validation suite contains 31 tests and 12 explicit model-consistency checks. The analytical test generates 48 expected threshold
crossings from a single-pixel ramp. The simulator generates 48 events with:

```text
timestamp MAE     = 0.5 us
maximum error     = 1.0 us
timestamp resolution = 1 us
```

This is a model-consistency validation, not a comparison against a physical
event-camera sensor. It verifies that preprocessing, state updates,
interpolation, crossing count and pure floor timestamp quantization are
internally consistent.

The asymmetric-threshold check uses `C+ = 0.15` and `C- = 0.30` and produces
46 ON events versus 23 OFF events for the same magnitude range, matching the
expected threshold ratio.

The vectorized and pixel-loop backends produce exactly 12,630 identical events
in the deterministic validation sequence. Additional regression checks cover
pure floor quantization at a clock boundary and a zero-delta residual crossing
combined with refractory filtering; those checks also pass event-for-event.

## 7. Experiments

### Threshold sensitivity

For `C = 0.10, 0.15, 0.20, 0.30, 0.40`, event count decreases from 256,196 to
51,744. The product `N * C` remains within the same order of magnitude,
supporting the expected `N ~ 1/C` trend.

### FPS convergence

A nonlinear single-pixel signal is sampled at 120 to 3840 FPS and compared with
the 3840 FPS output. Events are aligned separately by polarity using
order-preserving dynamic-programming sequence alignment. RMSE is computed only
over matched events; unmatched events are reported separately. As input FPS
increases, timestamp RMSE decreases from 2221 us at 120 FPS to 136 us at
1920 FPS and zero against the 3840 FPS reference.

This experiment demonstrates why high-FPS source data matters: linear
interpolation can estimate sub-frame timing, but the approximation improves as
sampling becomes denser.

### Simplified sensor non-idealities

Four models are compared:

1. ideal threshold model;
2. threshold mismatch;
3. threshold mismatch plus background activity;
4. previous model plus refractory period.

Threshold mismatch and background activity slightly increase event count.
Adding refractory suppression reduces event count from 71,150 to 66,744 by
enforcing a minimum time between accepted events in the same pixel.

### Event representation

The same event stream is accumulated over 1, 5, 10 and 20 ms. Short windows
preserve temporal detail but are sparse; longer windows improve spatial
visibility while losing temporal precision. Accumulation changes visualization
only and does not alter raw events.

### Photometric preprocessing

Direct log encoding produces 114,519 events, while the optional gamma-linearized
model produces 304,886. This large difference shows that event generation is
sensitive to the assumed photometric response. It does not prove that either
curve is the physically correct response of the source camera.

## 8. Limitations

- Log intensity is piecewise linear between consecutive source frames.
- Pixels are assumed independent and ideal readout is assumed.
- Background activity is a simplified Poisson approximation, not a complete
  physical model of leak and shot noise.
- Photoreceptor bandwidth, latency distribution and threshold drift are not
  modeled.
- Readout arbitration and transistor-level circuitry are outside scope.
- Motion blur, temporal aliasing and saturation already present in the source
  cannot be recovered.
- The validation is against the mathematical model, not against a physical
  event-camera dataset.

## 9. Conclusion

The implemented simulator covers the complete frame-to-event CA pipeline:
high-FPS input, photometric preprocessing, log intensity, stateful threshold
crossing, temporal interpolation, asynchronous timestamps, output and
visualization. It adds configurable non-idealities and an efficient vectorized
backend while retaining an independent reference implementation.

The project now provides quantitative correctness evidence, parameter
sensitivity analysis, non-ideality ablation, representation comparison and an
efficiency benchmark. Further work should prioritize presentation quality and
optional real-event qualitative comparison rather than expanding the simulator
with downstream computer-vision tasks.

## References

1. G. Gallego et al., "Event-based Vision: A Survey," IEEE TPAMI, 2020.
2. E. Mueggler et al., "The Event-Camera Dataset and Simulator," IJRR, 2017.
3. H. Rebecq, D. Gehrig, D. Scaramuzza, "ESIM: an Open Event Camera
   Simulator," CoRL, 2018.
4. Y. Hu, S.-C. Liu, T. Delbruck, "v2e: From Video Frames to Realistic DVS
   Events," CVPR Workshops, 2021.
5. EE5110/EE6110 Lecture 1, "From Frames to Events: Theory and Applications
   of Event-based Vision."
6. Event-based vision resources:
   https://github.com/uzh-rpg/event-based_vision_resources
7. Event-based datasets collection:
   https://github.com/lisiqi19971013/event-based-datasets
