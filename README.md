# Event Camera Simulator (`evsim`)

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://event-camera-simulator-7mlnyhk52f9fp4ydasp4jd.streamlit.app/)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/)
[![uv native](https://img.shields.io/badge/uv-native-purple.svg)](https://github.com/astral-sh/uv)

A high-performance, verifiable neuromorphic event camera simulator implemented in Python. It transforms high-frame-rate video or image sequences into asynchronous event streams `(x, y, t, p)` using a stateful log-intensity contrast-threshold model.

```text
High-FPS Video / Image Sequence + Sensor Config
                     │
                     ▼
           Event Camera Simulator
                     │
                     ▼
  (x, y, t, p) Event Stream + Illustrative Video
```

> [!NOTE]
> The simulator operates strictly at the photoreceptor transduction level (transducing optical intensity into events). It does not perform computer vision downstream tasks such as edge detection, tracking, or SLAM.

---

## Installation

`evsim` is native to [`uv`](https://docs.astral.sh/uv/) and targets modern Python (>=3.14).

```bash
# Clone the repository
git clone https://github.com/YDxun/Event-Camera-Simulator.git
cd Event-Camera-Simulator

# Install the complete locked development environment
uv sync --all-extras --locked

# Or with pip
python -m pip install -e ".[dev]"
```

---

## Quick Start & CLI

The CLI is structured and powered by [Tyro](https://brentyi.github.io/tyro/):

### 1. Run Built-in 960 FPS Synthetic Demo
Generates a 960 FPS synthetic test video, simulates events, and renders an illustrative side-by-side video:
```bash
uv run evsim demo --output-dir output/demo --fps 960 --seconds 1.0
```

### 2. Simulate Custom Video or Image Sequence
```bash
uv run evsim simulate \
  --input path/to/high_fps_video.mp4 \
  --config configs/enhanced.json \
  --output-csv output/events.csv \
  --output-npz output/events.npz \
  --video output/events.avi
```

For long inputs, add `--stream` with `--output-csv` and/or `--output-npz` to
write event chunks through disk instead of retaining the complete stream in RAM.
Video timestamps come from container PTS via PyAV, so variable-frame-rate input
is preserved. Image sequences with an invalid `ts_frame.txt` fail by default;
use `--allow-timestamp-fallback` only when fixed-FPS recovery is intentional.

### 3. Run Core Analytical Validation
Verifies consistency across 12 formal model-consistency checks (static scene suppression, ramp firing, microsecond alignment, refractory suppression, and backend equivalence):
```bash
uv run evsim validate --output-json output/validation.json
```

### 4. Benchmark Execution Backends
Evaluates throughput between the fast vectorized NumPy engine and the reference pixel-loop:
```bash
uv run evsim benchmark --frames 60
```

### 5. Inspect Input Metadata
```bash
uv run evsim inspect --input path/to/input.mp4
```

---

## Interactive Interfaces

### 1. Desktop GUI (PySide6)
A responsive desktop GUI featuring non-blocking background simulation, interactive parameter tuning, and real-time event visualization:
```bash
uv run evsim-gui
# or:
uv run evsim gui
```


### 2. Web UI (Streamlit)
A cloud-deployable browser interface available locally or on Streamlit Community Cloud:
```bash
uv run streamlit run app.py
```
Live demo: [Streamlit Public App](https://event-camera-simulator-7mlnyhk52f9fp4ydasp4jd.streamlit.app/)

---

## Architecture Overview

```text
Input Video / Frames (PyAV PTS / ImageSequence)
                     │
                     ▼
  Photometric Preprocessing (Normalized DN -> Optional Linearization -> Log Intensity)
                     │
                     ▼
  Stateful Pixel Processing (Vectorized NumPy or Reference Loop)
  - Threshold crossing detection: Delta L >= C+ (ON) or Delta L <= -C- (OFF)
  - Sub-frame piecewise-linear timestamp interpolation
  - Microsecond floor quantization & refractory dead-time filtering
  - Stateful reference level update: L_ref <- L_k
                     │
                     ▼
  Event Stream & Diagnostics
  ├── CSV ('timestamp_s,x,y,polarity') & Compressed NPZ ('events')
  ├── JSON Statistics (Event rates, ON/OFF ratio, processing FPS)
  └── Multi-panel Illustrative Video (Input | Events | Overlay)
```

---

## Documentation Directory

| Document | Description |
|---|---|
| [**Repository Guide**](docs/REPOSITORY_GUIDE.md) | Stable project layout, generated artifacts, and standard workflow. |
| [**Demo Guide**](docs/DEMO_GUIDE.md) | Public/local demo flow and the recommended presentation sequence. |
| [**Architecture & Design**](docs/design.md) | In-depth mathematical formulation, assumptions A1–A7, decoupled pipeline, and coordinate conventions. |
| [**Configuration Reference**](docs/configuration.md) | Complete reference of parameters, units, physical ranges, and presets (`ideal`, `enhanced`). |
| [**Experimental Verification**](docs/experiments.md) | Quantitative validation: threshold sweep, FPS convergence, noise ablation, and benchmarks. |
| [**Related Work**](docs/RELATED_WORK.md) | Academic positioning and comparison with ESIM, v2e, and the RPG Event-Camera Simulator. |
| [**Submission Workspace**](submission/README.md) | Final deliverable checklist and submission ZIP instructions. |
| [**Submission Checklist**](submission/SUBMISSION_CHECKLIST.md) | Coursework deliverables checklist and technical verification commands. |
| [**Coursework Report**](submission/COURSE_REPORT.md) | Research report detailing motivation, model derivations, and findings. |
| [**AI Use Report**](submission/AI_USE_AND_REVIEW_REPORT.md) | Formal documentation of AI assistance, review, and verification methodology. |

---

## Development & Testing

```bash
# Run full unit test suite
uv run pytest -v

# Run code style & lint checks
uv run ruff check
uv run ruff format --check

# Execute full experimental validation suite
uv run python scripts/run_experiments.py

# Assemble a clean submission ZIP after slides/video/report are ready
uv run python scripts/prepare_submission.py \
  --slides submission/slides.pdf \
  --video submission/presentation_video.mp4 \
  --ai-report submission/AI_USE_AND_REVIEW_REPORT.pdf
```
