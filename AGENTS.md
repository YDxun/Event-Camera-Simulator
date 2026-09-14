# evsim - Part B CA for EE5110

This is an ongoing guideline for agents.

Requirements may be added or modified during progressing.

## Overall

- Raw reqirements for CA are defined in: `specs/spec-sol.md`
  - which is extracted from `specs\slides\*.pdf`
  - `specs\slides\*.txt` provide text transcripts from pdfs
- In most cases, you SHOULD refer to the markdown
  - except for FINAL CHECK
- This is a CA work does not mean to mention it everywhere
  - design and implement and document as normal project
  - except for reqired reports and notes (for reporting and writing use)

## Engineering

- Python first, >=3.12
- Best practice, modern: layout, documentation comments, test/workflow
- Type hint: if possible
- `uv` native: should
  - use uv build backend
  - supports `uv run` for demo


## Architecture

- AI/Agent friendly dev:
  - may create a skill after finish
  - good organization
- frontend, backend decoupled
  - backend should consider correctness and performance
  - frontend should have 3 options:
    - web: streamlit, first class citizen
    - gui: focus on efficiency
      - does not mean to maximize performance
    - cli: no visaulization, focus on concise and clear

## Style and flavor

- `import cv2 as cv` in alignment with official OpenCV style
- use src-layout and editable install

### Ruff

Minimized config in `pyproject.toml`:

- no `target-version` explicit: from `requires-python`
- no `line-length`: default is OK
