# Final Submission Workspace

This directory contains the stable submission documents. Generated experiment outputs remain under `output/`.

## Required deliverables

- Presentation video, under 15 minutes.
- Presentation slides, preferably `slides.pdf`.
- Source code.
- AI Use and Review Report as PDF because AI tools were used.
- Contribution table at the end of the slides.

Place final binary deliverables here before packaging:

```text
submission/
|-- slides.pdf
|-- presentation_video.mp4
|-- AI_USE_AND_REVIEW_REPORT.pdf
|-- COURSE_REPORT.md
|-- AI_USE_AND_REVIEW_REPORT.md
`-- SUBMISSION_CHECKLIST.md
```

## Build the submission ZIP

```bash
uv run python scripts/prepare_submission.py   --slides submission/slides.pdf   --video submission/presentation_video.mp4   --ai-report submission/AI_USE_AND_REVIEW_REPORT.pdf
```

The ZIP is written under `dist/`. It includes source, documentation, validation results, selected experiment artifacts and the 960 FPS demo video without duplicating the large CSV by default.

## Before packaging

- Follow `SUBMISSION_CHECKLIST.md`.
- Confirm the video is under 15 minutes.
- Confirm the contribution table is present at the end of the slides.
- Confirm PDF versions of slides and the AI report are included.
