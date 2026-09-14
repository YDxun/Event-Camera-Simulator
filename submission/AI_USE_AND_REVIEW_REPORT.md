# AI Use and Review Report

## 1. AI tools used

- OpenAI Codex / GPT-based coding assistant.
- Local Python, NumPy, OpenCV, pytest and Streamlit tools were used to execute and verify AI-assisted changes.

## 2. Purpose of AI use

AI assistance was used for the Python-first refactor, simulator module organization, test generation, validation design, documentation, experiment reporting and the lightweight Streamlit UI. AI was not used as a substitute for physical event-camera measurements.

## 3. Representative prompts

### Prompt 1: Python refactor

> Convert the existing C++/CUDA event-camera simulator into a Python-first project, keep the event-generation model, make it runnable on this computer, and remove unrelated CMake/CUDA environment files after validation.

Outcome: the project was rebuilt around `evsim`, NumPy, OpenCV and pytest. Old C++/CMake/vcpkg files were removed only after the Python tests and demo passed.

### Prompt 2: Validation review

> Review the current GitHub main branch and strengthen analytical timestamp validation, asymmetric thresholds, event-rate statistics, timestamp quantization/refractory conventions and dark/saturation edge cases.

Outcome: validation was expanded to 12 explicit model checks. The analytical result is 48 expected events versus 48 generated events, with MAE `0.5 us` and maximum error `1 us`.

### Prompt 3: Lightweight UI

> Build one lightweight Python UI for the existing simulator. Do not introduce React/FastAPI or duplicate simulator logic. The UI should expose only the main parameters and show Input, Events and Overlay changes interactively.

Outcome: a single-file Streamlit UI (`app.py`) was added. It calls the same `evsim` core used by the CLI and supports video, image-sequence ZIP and a built-in 960 FPS demo.

## 4. Useful AI feedback

- Separate active-duration event rate from input-duration event rate.
- Define timestamp floor quantization and refractory ordering explicitly.
- Validate analytical crossing counts and timestamps instead of relying only on visual output.
- Use an independent pixel-loop backend to check the vectorized backend.
- Keep the UI thin and reuse the simulator core.
- Avoid expanding into downstream tasks such as detection, tracking, SLAM or reconstruction.

## 5. Changes made after AI feedback

- Added `event_rate_over_active_hz` and `event_rate_over_input_hz`.
- Added analytical timestamp, asymmetric threshold, quantization and edge-case checks.
- Added per-polarity, order-preserving event matching for FPS RMSE.
- Added four-level non-ideality ablation.
- Added accumulation-window and linearization comparisons.
- Added binary event and event-count representations.
- Added Related Work, course report, submission checklist and interactive UI.
- Kept the event-generation simulator frozen after validation.

## 6. AI suggestion considered but not adopted

AI suggested adding real-event dataset comparison as a core requirement. This was not adopted as a P0 requirement because:

- the CA specifically asks for frames plus parameters to simulated events and illustrative video;
- the existing mathematical and implementation validation already covers the required pipeline;
- dataset integration would add data licensing, time alignment and sensor calibration work without changing the core simulator;
- real-event comparison is better treated as an optional P2 extension.

## 7. Reliability and limitations

AI-generated code and prose still require human review. References must be checked against the original papers. The simulator is validated against its mathematical model, not against a physical event-camera sensor. Experimental metrics depend on the stated assumptions and matching rules. The team remains responsible for every technical claim, figure, slide and submission artifact.

## Human review statement

The human author defined the CA scope, requested the Python-first refactor, selected the validation priorities, reviewed the implementation and results, and approved the final repository structure. The team is responsible for the final submission.
