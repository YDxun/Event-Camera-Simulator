# Final Submission Checklist

## Required CA deliverables

- [ ] Presentation video, under 15 minutes.
- [ ] Presentation slides (`slides.pdf` or editable source plus PDF).
- [ ] Source code.
- [ ] AI Use and Review Report because AI assistance was used.
- [ ] Contribution table at the end of the presentation slides.

## Recommended archive layout

```text
README.md
app.py
pyproject.toml
requirements.txt
packages.txt
src/
configs/
scripts/
docs/
submission/
output/
|-- validation.json
|-- experiments/
`-- demo/
deliverables/
|-- slides.pdf
|-- presentation_video.mp4
`-- AI_USE_AND_REVIEW_REPORT.pdf
```

Large raw CSV files are useful in the repository for reproducibility but do not
need to be duplicated in the final presentation archive if the demo NPZ and
selected key figures are included.

## Final technical checks

- [ ] `uv run pytest -q`
- [ ] `uv run evsim validate --output-json output/validation.json`
- [ ] `uv run evsim demo --output-dir output/demo --fps 960 --seconds 1`
- [ ] `uv run python scripts/run_experiments.py`
- [ ] `uv run python scripts/prepare_submission.py --slides ... --video ... --ai-report ...`
- [ ] Verify figure units and axis labels.
- [ ] Explain that analytical validation is model-consistency validation, not real-sensor validation.
- [ ] Explain the FPS RMSE event-matching method.
- [ ] Verify all references.
