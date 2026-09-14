"""Assemble a clean CA submission archive from source, results and deliverables."""

from __future__ import annotations

import argparse
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Entry:
    path: Path
    arcname: str


STATIC_FILES = (
    "README.md",
    "app.py",
    "pyproject.toml",
    "requirements.txt",
    "packages.txt",
    "uv.lock",
)

SOURCE_DIRS = (
    "src",
    "configs",
    "tests",
    "scripts",
    "docs",
    "submission",
)

RESULT_FILES = (
    "output/REPORT.md",
    "output/validation.json",
    "output/benchmark.json",
    "output/experiments/summary.json",
    "output/experiments/threshold_sweep.csv",
    "output/experiments/threshold_sweep.png",
    "output/experiments/fps_sweep.csv",
    "output/experiments/fps_sweep.png",
    "output/experiments/noise_ablation.json",
    "output/experiments/noise_ablation.png",
    "output/experiments/accumulation_window.json",
    "output/experiments/accumulation_window_montage.png",
    "output/experiments/linearization_comparison.json",
    "output/experiments/linearization_comparison.png",
)

DEMO_FILES = (
    "output/demo/config_used.json",
    "output/demo/statistics.json",
    "output/demo/event_preview.png",
    "output/demo/events.npz",
    "output/demo/event_video.avi",
    "output/demo/input_960fps.avi",
)


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except OSError, subprocess.CalledProcessError:
        return "unknown"


def _iter_tree(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )


def _add_directory(entries: dict[str, Entry], relative_dir: str) -> None:
    directory = ROOT / relative_dir
    if not directory.exists():
        return
    for path in _iter_tree(directory):
        arcname = path.relative_to(ROOT).as_posix()
        entries[arcname] = Entry(path=path, arcname=arcname)


def _add_file(
    entries: dict[str, Entry], relative_path: str, arcname: str | None = None
) -> None:
    path = ROOT / relative_path
    if path.is_file():
        target = arcname or path.relative_to(ROOT).as_posix()
        entries[target] = Entry(path=path, arcname=target)


def _resolve_user_file(path: str, label: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} file not found: {resolved}")
    return resolved


def _collect_entries(args: argparse.Namespace) -> list[Entry]:
    entries: dict[str, Entry] = {}

    for relative_path in STATIC_FILES:
        _add_file(entries, relative_path)
    for relative_dir in SOURCE_DIRS:
        _add_directory(entries, relative_dir)
    for relative_path in (*RESULT_FILES, *DEMO_FILES):
        _add_file(entries, relative_path)

    if args.include_csv:
        _add_file(entries, "output/demo/events.csv")

    optional = (
        (args.slides, "slides.pdf", "slides"),
        (args.video, "presentation_video.mp4", "presentation video"),
        (args.ai_report, "AI_USE_AND_REVIEW_REPORT.pdf", "AI review report"),
    )
    for supplied, destination, label in optional:
        if supplied:
            source = _resolve_user_file(supplied, label)
            arcname = f"deliverables/{destination}"
            entries[arcname] = Entry(path=source, arcname=arcname)

    if args.require_deliverables:
        missing = [label for supplied, _destination, label in optional if not supplied]
        if missing:
            raise ValueError(
                "Missing required deliverable arguments: " + ", ".join(missing)
            )

    return [entries[name] for name in sorted(entries)]


def _manifest(revision: str, entries: list[Entry]) -> str:
    lines = [
        "Event Camera Simulator Submission Manifest",
        f"Source revision: {revision}",
        f"Created UTC: {datetime.now(UTC).isoformat()}",
        "",
        f"Files: {len(entries)}",
        "",
    ]
    for entry in entries:
        lines.append(f"{entry.path.stat().st_size:>12,}  {entry.arcname}")
    return "\n".join(lines) + "\n"


def build_archive(args: argparse.Namespace) -> Path:
    entries = _collect_entries(args)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"{args.name}.zip"
    if archive.exists() and not args.force:
        raise FileExistsError(
            f"Archive already exists: {archive}. Use --force to replace it."
        )

    revision = _git_revision()
    manifest = _manifest(revision, entries)
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive_file:
        for entry in entries:
            archive_file.write(entry.path, entry.arcname)
        archive_file.writestr("SUBMISSION_MANIFEST.txt", manifest)

    print(f"Created {archive}")
    print(f"Source revision: {revision}")
    print(f"Files: {len(entries) + 1}")
    print(f"Size: {archive.stat().st_size / 1024 / 1024:.2f} MiB")
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="dist")
    parser.add_argument("--name", default="Event-Camera-Simulator-submission")
    parser.add_argument("--slides", help="Final slides PDF")
    parser.add_argument("--video", help="Presentation video, preferably MP4")
    parser.add_argument("--ai-report", help="Final AI Use and Review Report PDF")
    parser.add_argument(
        "--include-csv",
        action="store_true",
        help="Include the large demo events.csv file",
    )
    parser.add_argument(
        "--require-deliverables",
        action="store_true",
        help="Fail unless slides, video and AI report are supplied",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing ZIP"
    )
    args = parser.parse_args()
    build_archive(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
