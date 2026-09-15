# Event Camera Simulator (`evsim`)

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/)
[![CI](https://github.com/YDxun/Event-Camera-Simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/YDxun/Event-Camera-Simulator/actions/workflows/ci.yml)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://event-camera-simulator-7mlnyhk52f9fp4ydasp4jd.streamlit.app/)

`evsim` is a team-built, verifiable simulator that converts high-frame-rate video
or image sequences into asynchronous event-camera streams `(x, y, t, p)`. It
models photoreceptor response and sensor timing while keeping the simulation core
independent from the command-line, desktop, and web interfaces.

## Why This Project

Conventional cameras record complete frames at fixed intervals. An event camera
reports only local brightness changes, producing sparse events with microsecond
timestamps. Real sensors and suitable datasets are not always available, so this
project provides a reproducible way to study event generation from familiar frame
inputs.

The simulator focuses on sensor transduction. It does not perform downstream tasks
such as object detection, tracking, optical flow, or SLAM.

## How It Works

```text
Video / image sequence + sensor configuration
                    |
                    v
        Timestamp-aware frame source
                    |
                    v
 Normalize -> optional linearization -> log intensity
                    |
                    v
 Stateful threshold crossings and sub-frame interpolation
                    |
                    v
 Timestamp quantization, refractory filtering, and noise
                    |
                    v
 Event stream + statistics + optional visualization
```

For each pixel, normalized intensity is transformed into log intensity:

```text
L(x, y, t) = ln(I(x, y, t) + epsilon)
```

An ON event is emitted when `L - L_ref >= C+`; an OFF event is emitted when
`L - L_ref <= -C-`. Multiple threshold crossings can occur between two frames.
Their timestamps are estimated with piecewise-linear interpolation, quantized to
the configured sensor clock, and filtered by a per-pixel refractory period.

The enhanced model can additionally simulate fixed per-pixel threshold mismatch
and Poisson background activity. See [Architecture and Mathematical Design](docs/design.md)
for the equations and modeling assumptions.

## Features

- Real presentation timestamps from PyAV, including variable-frame-rate video.
- Naturally ordered image sequences with strict timestamp-file validation.
- Fast vectorized NumPy backend plus a readable pixel-loop reference backend.
- Configurable thresholds, time resolution, refractory period, and sensor noise.
- Memory-bounded streaming output for long simulations.
- CSV and compressed NPZ event output with JSON diagnostics.
- Streamlit web UI, PySide6 desktop GUI, and a typed Tyro CLI.
- Analytical validation and backend-equivalence checks.

## Installation

Python 3.14 and [`uv`](https://docs.astral.sh/uv/) are recommended:

```bash
git clone https://github.com/YDxun/Event-Camera-Simulator.git
cd Event-Camera-Simulator
uv sync --all-extras --locked
```

For the core package only:

```bash
python -m pip install -e .
```

## Quick Start

Run the self-contained synthetic demo:

```bash
uv run evsim demo --output-dir output/demo --fps 960 --seconds 1.0
```

Simulate a video or image directory:

```bash
uv run evsim simulate \
  --input path/to/high_fps_video.mp4 \
  --config configs/enhanced.json \
  --output-csv output/events.csv \
  --output-npz output/events.npz \
  --video output/events.avi
```

For long inputs, add `--stream`. CSV data is appended incrementally and NPZ data
is assembled through disk-backed chunks instead of retaining the complete event
stream in memory.

```bash
uv run evsim simulate \
  --input path/to/input.mp4 \
  --output-npz output/events.npz \
  --stream
```

Inspect source timing or validate the mathematical model:

```bash
uv run evsim inspect --input path/to/input.mp4
uv run evsim validate --output-json output/validation.json
uv run evsim benchmark --frames 60
```

## Interactive Interfaces

Streamlit web application:

```bash
uv run streamlit run app.py
```

PySide6 desktop application:

```bash
uv run evsim-gui
```

Both interfaces delegate event generation to the same canonical simulation core.

## Event Format

Events use a compact NumPy structured dtype:

| Field | Type | Meaning |
|---|---|---|
| `timestamp_us` | `int64` | Event time in microseconds |
| `x` | `uint16` | Horizontal pixel coordinate |
| `y` | `uint16` | Vertical pixel coordinate |
| `polarity` | `int8` | `+1` for ON, `-1` for OFF |

CSV output uses `timestamp_s,x,y,polarity`; NPZ output stores the structured array
under the `events` key.

## Repository Layout

```text
src/evsim/             Simulation core, IO, CLI, GUI, and validation
configs/               Ideal and enhanced sensor presets
tests/                 Unit, integration, backend, CLI, web, and GUI tests
scripts/               Reproducible experiment automation
docs/                  Design, configuration, experiments, and demo guides
app.py                 Streamlit entry point
.github/workflows/     Automated quality and behavior checks
output/                Generated locally and intentionally not versioned
```

## Team Work Areas

The project is organized around three complementary work areas:

1. **Sensor model and algorithms**: photometric preprocessing, threshold crossing,
   interpolation, refractory behavior, and sensor non-idealities.
2. **Software system and interfaces**: timestamp-aware input, event serialization,
   streaming execution, CLI, desktop GUI, and web UI.
3. **Verification and evaluation**: analytical checks, backend comparison,
   performance experiments, automated tests, and reproducible demos.

This division keeps ownership clear while requiring all interfaces and experiments
to agree on one tested simulation core.

## Development

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run python scripts/run_experiments.py
```

Generated results are written under `output/` and can be recreated at any time.
The CI workflow runs formatting checks, tests, core validation, and UI smoke tests.

## Limitations

- Event timing between frames assumes piecewise-linear log-intensity change.
- Motion blur or temporal aliasing already present in the input cannot be recovered.
- Pixels are modeled independently; bus contention and sensor readout bandwidth are
  outside the current model.
- The noise model is phenomenological rather than transistor-level.

These boundaries are intentional and documented so results remain interpretable.

## Documentation

- [Architecture and Mathematical Design](docs/design.md)
- [Configuration Reference](docs/configuration.md)
- [Experimental Evaluation](docs/experiments.md)
- [Demo Guide](docs/DEMO_GUIDE.md)
- [Repository Guide](docs/REPOSITORY_GUIDE.md)
- [Related Work](docs/RELATED_WORK.md)
