# AI Use and Review Report

## Purpose

This draft describes AI assistance used during development. The team must review and edit it before submission.

## AI assistance

AI assisted with the Python conversion, module refactoring, tests, validation experiments, documentation and result generation. It did not replace physical event-camera measurements.

## Verification

- `python -m pytest -q`: 15 passed
- `python -m evsim validate`: all checks passed
- `python scripts/run_experiments.py`: exit code 0
- Analytical validation: 48 expected events, 48 actual events, MAE 0.5 us, maximum error 1 us
- Vectorized and pixel-loop outputs matched exactly in deterministic validation

## Human review

The human author defined the CA scope, requested the Python refactor, reviewed the implementation and results, and remains responsible for all claims and the final submission.

## Limitations

AI-generated code and prose require review. Literature references should be checked against original papers. The simulator is validated against its mathematical model, not a physical event-camera dataset.

## Team checklist

- [ ] Add team member names.
- [ ] Add submission date.
- [ ] Review slides and technical claims.
- [ ] Verify all references.
- [ ] Attach presentation video and slides.
