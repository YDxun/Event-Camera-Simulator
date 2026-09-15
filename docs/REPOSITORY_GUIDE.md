# Repository Guide

This repository is organized around a single simulator core and several thin interfaces.

## Canonical locations

| Path | Purpose |
|---|---|
| `src/evsim/` | Canonical simulator implementation |
| `configs/` | Ideal and Enhanced parameter presets |
| `tests/` | Unit, backend, IO, CLI, Streamlit and GUI tests |
| `scripts/` | Reproducible experiment automation |
| `docs/` | Architecture, configuration, experiments, and demo guidance |
| `output/` | Reproducible generated results; safe to regenerate |
| `app.py` | Streamlit web entrypoint |
| `src/evsim/gui.py` | Optional PySide6 desktop entrypoint |

## Stable source versus generated output

Stable source and documentation:

```text
src/
configs/
scripts/
tests/
docs/
README.md
pyproject.toml
requirements.txt
packages.txt
```

Generated artifacts:

```text
output/validation.json
output/benchmark.json
output/experiments/
output/demo/
```

Do not hand-edit generated numeric tables when they can be regenerated. Run the workflows below instead.

## Standard workflow

```bash
uv sync --all-extras

uv run pytest -q
uv run ruff check .
uv run ruff format --check .

uv run evsim validate --output-json output/validation.json
uv run evsim demo --output-dir output/demo --fps 960 --seconds 1.0
uv run python scripts/run_experiments.py
```

## Interfaces

- CLI: reproducible execution and automation.
- Streamlit: primary public/demo interface.
- PySide6: optional local desktop interface.

The interfaces must delegate event generation to `src/evsim/`; do not duplicate the event-generation algorithm in a UI.
