# System Design

## Goal

Convert a high-FPS frame sequence into an asynchronous event stream

```text
(x, y, timestamp, polarity)
```

through a physically interpretable pixel model.

## Pipeline

```text
VideoFrameSource / ImageSequenceSource
                |
                v
grayscale conversion
                |
                v
optional gamma linearization
                |
                v
L = log(I + epsilon)
                |
                v
frame-pair temporal interpolation
                |
                v
pixel threshold state machine
                |
                v
vectorized pixel-array event generation
                |
       +--------+---------+
       |                  |
       v                  v
 CSV / NPZ          accumulation video
```

## Pixel state

Each pixel stores:

```text
L_ref(x,y)              reference log intensity after the last accepted event
last_event_time(x,y)    timestamp used by the refractory model
C_plus(x,y)             positive threshold
C_minus(x,y)            negative threshold
```

The vectorized backend keeps these as NumPy arrays. The loop backend keeps the
same mathematical state but processes one pixel at a time as an independent
reference implementation.

## Event generation

For an interval `[t0, t1]`, the temporal model is:

```text
L(t) = L0 + (L1 - L0) * (t - t0) / (t1 - t0)
```

Candidate positive and negative crossings are computed from the reference:

```text
L_ref + k C+    for k = 1..floor((L1 - L_ref) / C+)
L_ref - k C-    for k = 1..floor((L_ref - L1) / C-)
```

The event time is obtained by solving for `t`, then quantized to the configured
microsecond resolution. Candidate events are processed in timestamp order so
the refractory period and `L_ref` update are handled causally.

## Noise models

### Threshold mismatch

```text
C+(x,y) ~ Normal(C+, sigma_C^2)
C-(x,y) ~ Normal(C-, sigma_C^2)
```

The sampled maps are fixed for the whole sequence, matching sensor mismatch
rather than per-frame random noise.

### Background activity

For each pixel and frame interval `dt`, the number of spontaneous events is
sampled from:

```text
Poisson(background_rate_hz * dt)
```

Accepted background events undergo the same refractory check but do not change
`L_ref`.

## Correctness strategy

1. Test a single pixel analytically.
2. Verify constant intensity generates zero events.
3. Verify positive and negative ramps generate the correct polarity.
4. Verify multiple threshold crossings.
5. Verify interpolation timestamps lie inside the source interval.
6. Verify timestamp quantization and refractory behavior.
7. Verify the vectorized result exactly matches the independent pixel loop.
8. Verify input, output and rendering through integration tests.
