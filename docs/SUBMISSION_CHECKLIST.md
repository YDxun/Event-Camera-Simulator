# Final Submission Checklist

## Required CA deliverables

- [ ] Presentation video, under 15 minutes.
- [ ] Presentation slides (`slides.pdf` or editable source plus PDF).
- [ ] Source code.
- [ ] AI Use and Review Report because AI assistance was used.

## Recommended archive layout

```text
submission/
|-- src/
|   |-- evsim/
|   |-- configs/
|   |-- scripts/
|   `-- requirements.txt
|-- results/
|   |-- key_figures/
|   |-- validation.json
|   `-- demo_events.npz
|-- demo/
|   `-- illustrative_video.avi
|-- slides.pdf
`-- AI_use_and_review_report.pdf
```

Large raw CSV files are useful in the repository for reproducibility but do not
need to be duplicated in the final presentation archive if the demo NPZ and
selected key figures are included.

## Final technical checks

- [ ] `python -m pytest -q`
- [ ] `python -m evsim validate`
- [ ] `python -m evsim demo --output-dir results_python/demo --fps 960 --seconds 1`
- [ ] `python scripts/run_experiments.py`
- [ ] Verify figure units and axis labels.
- [ ] Explain that analytical validation is model-consistency validation, not real-sensor validation.
- [ ] Explain the FPS RMSE event-matching method.
- [ ] Verify all references.
