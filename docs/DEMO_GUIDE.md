# Demo Guide

## Primary live demo: Streamlit

Public URL:

https://event-camera-simulator-7mlnyhk52f9fp4ydasp4jd.streamlit.app/

Suggested 60-second demonstration:

1. Select `Enhanced`.
2. Keep `Built-in 960 FPS demo` as the source.
3. Press `Run Simulation`.
4. Show `Input | Events | Overlay`.
5. Change `C+` and `C-` from `0.20` to `0.40` and rerun; event density should decrease.
6. Change the visualization window from `1 ms` to `20 ms`; the rendered event frame should become denser while the raw event stream remains unchanged.
7. Download `events.npz` and `statistics.json`.

Useful sentence:

> The UI calls the same `simulate_source()` pipeline used by the CLI and quantitative experiments.

## Local fallback

Streamlit:

```bash
uv run streamlit run app.py
```

Built-in CLI demo:

```bash
uv run evsim demo --output-dir output/demo --fps 960 --seconds 1.0
```

Desktop GUI:

```bash
uv run evsim-gui
```

## Recommended presentation flow

| Time | Topic |
|---:|---|
| 0:00-1:30 | Why event cameras and why simulation |
| 1:30-3:30 | Problem: discrete high-FPS frames to asynchronous events |
| 3:30-6:00 | Pixel model: log intensity, `L_ref`, thresholds, interpolation |
| 6:00-8:00 | Pixel array and `(x,y,t,p)` event stream |
| 8:00-10:30 | Validation: pure floor quantization, zero-delta consistency, analytical timestamps |
| 10:30-12:30 | Experiments and efficiency |
| 12:30-14:00 | Limitations and related work |
| 14:00-15:00 | Conclusion and live UI demonstration |

## Claims to keep precise

- Validation is analytical model-consistency validation, not real-sensor validation.
- The noise model is a simplified phenomenological model.
- Pixels are modeled independently; readout arbitration is outside the simulator scope.
- The source video cannot recover information already lost to motion blur, aliasing or saturation.
