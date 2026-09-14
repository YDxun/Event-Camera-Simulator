"""Lightweight Streamlit UI for the Python event-camera simulator."""

from __future__ import annotations

import shutil
import stat
import subprocess
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path

import cv2 as cv
import numpy as np
import streamlit as st

try:
    from imageio_ffmpeg import get_ffmpeg_exe
except ImportError:

    def get_ffmpeg_exe() -> str:
        """Fall back to a system ffmpeg when the Python helper is unavailable."""
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError(
                "ffmpeg is unavailable; install imageio-ffmpeg or system ffmpeg"
            )
        return executable


ROOT = Path(__file__).resolve().parent
# Streamlit Cloud runs the checkout directly using its existing Python runtime.
# Import from src so the web entrypoint does not install the Python >=3.14 package.
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evsim.config import SimulatorConfig
from evsim.demo import write_synthetic_video
from evsim.pipeline import SimulationResult, simulate_source
from evsim.sources import open_source
from evsim.visualization import EventVideoRenderer

MAX_ZIP_FILES = 10_000
MAX_ZIP_MEMBER_BYTES = 512 * 1024 * 1024
MAX_ZIP_TOTAL_BYTES = 2 * 1024 * 1024 * 1024


def _clear_previous_workdir(session_state) -> None:
    previous = session_state.pop("_workdir", None)
    if previous is not None:
        previous.cleanup()


def _safe_extract_zip(path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > MAX_ZIP_FILES:
            raise ValueError(f"ZIP contains more than {MAX_ZIP_FILES:,} entries")
        total_size = sum(member.file_size for member in members)
        if total_size > MAX_ZIP_TOTAL_BYTES:
            raise ValueError("ZIP expands beyond the 2 GiB safety limit")
        for member in members:
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(
                    f"ZIP symbolic links are not allowed: {member.filename}"
                )
            file_type = stat.S_IFMT(mode)
            if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(f"Unsupported ZIP member type: {member.filename}")
            if member.file_size > MAX_ZIP_MEMBER_BYTES:
                raise ValueError(
                    f"ZIP member exceeds the 512 MiB limit: {member.filename}"
                )
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    children = [item for item in destination.iterdir() if item.is_dir()]
    return children[0] if len(children) == 1 else destination


def _save_upload(upload, workdir: Path, default_suffix: str) -> Path:
    suffix = Path(upload.name).suffix.lower() or default_suffix
    path = workdir / f"input{suffix}"
    path.write_bytes(upload.getbuffer())
    return path


def _transcode_for_browser(source: Path, destination: Path) -> Path:
    """Convert a local video to browser-compatible H.264 MP4."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        get_ffmpeg_exe(),
        "-y",
        "-i",
        str(source),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not destination.exists():
        raise RuntimeError(
            "FFmpeg H.264 conversion failed: " + completed.stderr[-1000:]
        )
    return destination


def _preset_path(preset: str) -> Path:
    filename = "ideal.json" if preset == "Ideal" else "enhanced.json"
    return ROOT / "configs" / filename


def _build_config(
    preset: str,
    positive_threshold: float,
    negative_threshold: float,
    timestamp_resolution_us: int,
    refractory_period_us: int,
    threshold_variation: bool,
    background_activity: bool,
    linearization: bool,
    accumulation_time_us: int,
    max_frames: int,
    fallback_fps: float | None = None,
) -> SimulatorConfig:
    config = SimulatorConfig.load(_preset_path(preset))
    enhanced = SimulatorConfig.load(_preset_path("Enhanced"))
    config.sensor.positive_threshold = positive_threshold
    config.sensor.negative_threshold = negative_threshold
    config.sensor.timestamp_resolution_us = timestamp_resolution_us
    config.sensor.refractory_period_us = refractory_period_us
    config.noise.enable_threshold_variation = threshold_variation
    config.noise.threshold_sigma = (
        config.noise.threshold_sigma
        if threshold_variation and config.noise.threshold_sigma > 0
        else (enhanced.noise.threshold_sigma if threshold_variation else 0.0)
    )
    config.noise.background_rate_hz = (
        config.noise.background_rate_hz
        if background_activity and config.noise.background_rate_hz > 0
        else (enhanced.noise.background_rate_hz if background_activity else 0.0)
    )
    config.input.linearize = linearization
    if fallback_fps is not None:
        config.input.fallback_fps = fallback_fps
    config.visualization.accumulation_time_us = accumulation_time_us
    config.runtime.max_frames = max_frames
    config.runtime.progress = False
    config.validate()
    return config


def _run_ui_simulation(
    input_path: Path,
    config: SimulatorConfig,
    workdir: Path,
) -> tuple[SimulationResult, Path]:
    output_video = workdir / "event_video_raw.mp4"
    browser_video = workdir / "event_video_h264.mp4"
    config.visualization.output_video = str(output_video)
    config.output.npz_path = str(workdir / "events.npz")
    config.output.statistics_path = str(workdir / "statistics.json")
    config.output.csv_path = ""

    source = open_source(
        input_path,
        fallback_fps=config.input.fallback_fps,
        timestamp_scale_us=config.input.timestamp_scale_us,
        allow_timestamp_fallback=config.input.allow_timestamp_fallback,
    )
    try:
        renderer = EventVideoRenderer(source.width, source.height, config.visualization)
        renderer.open(output_video)
        result = simulate_source(
            source,
            config,
            max_frames=config.runtime.max_frames or None,
            progress=False,
            renderer=renderer,
        )
        result.statistics["source_fps"] = source.fps
    finally:
        source.close()
    result.event_stream.save_npz(config.output.npz_path)
    Path(config.output.statistics_path).write_text(
        __import__("json").dumps(result.statistics, indent=2) + chr(10),
        encoding="utf-8",
    )
    raw_path = renderer.actual_output_path or output_video
    _transcode_for_browser(raw_path, browser_video)
    return result, browser_video


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def _render_preview(
    input_path: Path,
    events: np.ndarray,
    width: int,
    height: int,
    accumulation_time_us: int,
    overlay_opacity: float,
    processed_frames: int | None = None,
    fallback_fps: float = 960.0,
    timestamp_scale_us: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = open_source(
        input_path,
        fallback_fps=fallback_fps,
        timestamp_scale_us=timestamp_scale_us,
    )
    try:
        total = source.total_frames or 1
        limit = processed_frames or total
        target_index = max(min(limit, total) // 2, 0)
        target_frame = None
        for frame in source:
            if frame.index == target_index:
                target_frame = frame
                break
    finally:
        source.close()
    if target_frame is None:
        raise RuntimeError("Could not obtain a preview frame")

    end_us = target_frame.timestamp_us
    start_us = max(end_us - accumulation_time_us, 0)
    timestamps = events["timestamp_us"]
    left = int(np.searchsorted(timestamps, start_us, side="left"))
    right = int(np.searchsorted(timestamps, end_us, side="right"))
    selected = events[left:right]

    config = SimulatorConfig()
    config.visualization.accumulation_time_us = accumulation_time_us
    config.visualization.overlay_opacity = overlay_opacity
    renderer = EventVideoRenderer(width, height, config.visualization)
    renderer.add(selected, target_frame.image, end_us)
    return (
        _to_rgb(target_frame.image),
        _to_rgb(renderer.render_event_frame()),
        _to_rgb(renderer.render_overlay(target_frame.image)),
    )


def _prepare_input(
    mode: str,
    uploaded_video,
    uploaded_zip,
    workdir: Path,
) -> Path:
    if mode == "Built-in 960 FPS demo":
        demo_path = ROOT / "output" / "demo" / "input_960fps.avi"
        if not demo_path.exists():
            write_synthetic_video(demo_path, 960.0, 1.0, 320, 240)
        return demo_path
    if mode == "Upload video":
        if uploaded_video is None:
            raise ValueError("Please choose a video file")
        return _save_upload(uploaded_video, workdir, ".mp4")
    if uploaded_zip is None:
        raise ValueError("Please choose an image-sequence ZIP file")
    zip_path = _save_upload(uploaded_zip, workdir, ".zip")
    return _safe_extract_zip(zip_path, workdir / "sequence")


def main() -> None:
    st.set_page_config(page_title="Event Camera Simulator", layout="wide")
    st.title("Event Camera Simulator")
    st.caption(
        "High-FPS frames -> log intensity -> temporal interpolation -> "
        "stateful threshold crossing -> (x, y, t, p)"
    )

    preset = st.radio("Preset", ["Ideal", "Enhanced"], horizontal=True)
    preset_config = SimulatorConfig.load(_preset_path(preset))
    defaults = (
        preset_config.sensor.positive_threshold,
        preset_config.sensor.negative_threshold,
        preset_config.sensor.timestamp_resolution_us,
        preset_config.sensor.refractory_period_us,
        preset_config.noise.enable_threshold_variation,
        preset_config.noise.background_rate_hz > 0,
        preset_config.input.linearize,
        max(1, int(preset_config.visualization.accumulation_time_us // 1000)),
    )
    source_mode = st.radio(
        "Input source",
        ["Built-in 960 FPS demo", "Upload video", "Upload image-sequence ZIP"],
        horizontal=True,
    )
    timestamp_mode = st.radio(
        "Input timestamp mode",
        ["Use source timing", "Override fallback FPS"],
        horizontal=True,
        help=(
            "Video FPS metadata and valid ts_frame.txt files remain authoritative. "
            "Override applies only to image sequences without timestamps."
        ),
    )

    with st.form("simulation_form"):
        uploaded_video = None
        uploaded_zip = None
        if source_mode == "Upload video":
            uploaded_video = st.file_uploader(
                "Input video", type=["mp4", "avi", "mov", "mkv", "m4v"]
            )
        elif source_mode == "Upload image-sequence ZIP":
            uploaded_zip = st.file_uploader("Input image-sequence ZIP", type=["zip"])

        left, right = st.columns(2)
        with left:
            positive_threshold = st.number_input(
                "Positive threshold C+", 0.01, 1.0, float(defaults[0]), 0.01
            )
            timestamp_resolution_us = st.selectbox(
                "Timestamp resolution (us)",
                [1, 10, 100, 1000],
                index=[1, 10, 100, 1000].index(int(defaults[2])),
            )
            threshold_variation = st.checkbox(
                "Threshold variation", value=bool(defaults[4])
            )
            linearization = st.checkbox("Linearization", value=bool(defaults[6]))
        with right:
            negative_threshold = st.number_input(
                "Negative threshold C-", 0.01, 1.0, float(defaults[1]), 0.01
            )
            refractory_period_us = st.number_input(
                "Refractory period (us)", 0, 100_000, int(defaults[3]), 10
            )
            background_activity = st.checkbox(
                "Background activity", value=bool(defaults[5])
            )
            accumulation_ms = st.select_slider(
                "Visualization window (ms)", [1, 5, 10, 20], value=int(defaults[7])
            )
            fallback_fps = None
            if timestamp_mode == "Override fallback FPS":
                fallback_fps = st.number_input(
                    "Fallback FPS for sequences without timestamps",
                    min_value=1.0,
                    max_value=10000.0,
                    value=960.0,
                    step=10.0,
                )

        with st.expander("Execution control"):
            max_frames = st.number_input(
                "Maximum frames (0 = all)",
                0,
                1_000_000,
                240,
                10,
                help="240 frames keeps the hosted demo responsive. Use 0 only for short clips when full processing is required.",
            )

        submitted = st.form_submit_button(
            "Run Simulation", type="primary", use_container_width=True
        )

    if submitted:
        _clear_previous_workdir(st.session_state)
        temporary = tempfile.TemporaryDirectory(prefix="evsim_ui_")
        workdir = Path(temporary.name)
        try:
            input_path = _prepare_input(
                source_mode, uploaded_video, uploaded_zip, workdir
            )
            config = _build_config(
                preset=preset,
                positive_threshold=positive_threshold,
                negative_threshold=negative_threshold,
                timestamp_resolution_us=timestamp_resolution_us,
                refractory_period_us=int(refractory_period_us),
                threshold_variation=threshold_variation,
                background_activity=background_activity,
                linearization=linearization,
                accumulation_time_us=int(accumulation_ms) * 1000,
                max_frames=int(max_frames),
                fallback_fps=fallback_fps,
            )
            with st.spinner("Running event simulation..."):
                input_playback_path = None
                if input_path.is_file():
                    if input_path.suffix.lower() == ".mp4":
                        input_playback_path = input_path
                    else:
                        input_playback_path = _transcode_for_browser(
                            input_path, workdir / "input_h264.mp4"
                        )
                result, video_path = _run_ui_simulation(input_path, config, workdir)
                input_preview, event_preview, overlay_preview = _render_preview(
                    input_path,
                    result.events,
                    int(result.statistics["width"]),
                    int(result.statistics["height"]),
                    int(accumulation_ms) * 1000,
                    config.visualization.overlay_opacity,
                    processed_frames=result.frames_processed,
                    fallback_fps=config.input.fallback_fps,
                    timestamp_scale_us=config.input.timestamp_scale_us,
                )
            st.session_state["ui_result"] = {
                "input_path": str(input_path),
                "input_playback_path": (
                    str(input_playback_path) if input_playback_path else None
                ),
                "video_path": str(video_path),
                "npz_path": config.output.npz_path,
                "statistics_path": config.output.statistics_path,
                "statistics": result.statistics,
                "input_preview": input_preview,
                "event_preview": event_preview,
                "overlay_preview": overlay_preview,
                "source_mode": source_mode,
                "accumulation_ms": int(accumulation_ms),
                "preset": preset,
                "timestamp_mode": timestamp_mode,
                "fallback_fps": fallback_fps,
            }
            st.session_state["_workdir"] = temporary
        except Exception as exc:  # noqa: BLE001
            temporary.cleanup()
            st.error(f"Simulation failed: {exc}")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())

    ui_result = st.session_state.get("ui_result")
    if not ui_result:
        st.info("Choose a source and press Run Simulation.")
        return

    stats = ui_result["statistics"]
    st.subheader("Result Preview")
    col_input, col_events, col_overlay = st.columns(3)
    with col_input:
        st.markdown("**Input**")
        st.image(ui_result["input_preview"])
    with col_events:
        st.markdown("**Events**")
        st.image(ui_result["event_preview"])
    with col_overlay:
        st.markdown("**Overlay**")
        st.image(ui_result["overlay_preview"])

    event_rate = float(stats.get("event_rate_over_input_hz", 0.0)) / 1000.0
    metric_cols = st.columns(4)
    metric_cols[0].metric("Events", f"{int(stats.get('event_count', 0)):,}")
    metric_cols[1].metric("ON", f"{int(stats.get('on_events', 0)):,}")
    metric_cols[2].metric("OFF", f"{int(stats.get('off_events', 0)):,}")
    metric_cols[3].metric("Event rate", f"{event_rate:,.1f} kEvents/s")
    processing_cols = st.columns(4)
    processing_cols[0].metric(
        "Processing", f"{float(stats.get('processing_fps', 0.0)):,.1f} FPS"
    )
    source_fps = stats.get("source_fps")
    processing_cols[1].metric(
        "Source FPS", f"{float(source_fps):,.1f}" if source_fps else "n/a"
    )
    processing_cols[2].metric(
        "Input duration", f"{float(stats.get('input_duration_s', 0.0)):,.4f} s"
    )
    processing_cols[3].metric("Accumulation", f"{ui_result['accumulation_ms']} ms")

    st.caption(
        f"Preset: {ui_result['preset']} | source: {ui_result['source_mode']} | "
        f"timestamp mode: {ui_result['timestamp_mode']} | "
        "Lower thresholds produce more events; larger visualization windows "
        "produce denser event frames without changing raw events."
    )

    with st.expander("Model / Physics"):
        st.latex(r"L(x,y,t)=\log(I(x,y,t)+\epsilon)")
        st.latex(r"\Delta L=L-L_{ref}")
        st.markdown(
            "**ON:** $\\Delta L \\ge C_+$  \n"
            "**OFF:** $\\Delta L \\le -C_-$  \n"
            "The raw event stream depends only on source timing, thresholds and pixel state; "
            "the visualization window changes rendering only."
        )

    video_path = Path(ui_result["video_path"])
    playback_cols = st.columns(2)
    with playback_cols[0]:
        st.markdown("**Input video**")
        input_playback = ui_result.get("input_playback_path")
        if input_playback and Path(input_playback).exists():
            st.video(input_playback)
        else:
            st.info(
                "The current source is an image sequence, so only the input preview is shown."
            )
    with playback_cols[1]:
        st.markdown("**Event video (H.264)**")
        if video_path.exists():
            st.video(str(video_path))

    download_cols = st.columns(3)
    npz_path = Path(ui_result["npz_path"])
    if npz_path.exists():
        download_cols[0].download_button(
            "Download events.npz",
            data=npz_path.read_bytes(),
            file_name="events.npz",
            mime="application/octet-stream",
            use_container_width=True,
        )
    statistics_path = Path(ui_result["statistics_path"])
    if statistics_path.exists():
        download_cols[1].download_button(
            "Download statistics.json",
            data=statistics_path.read_bytes(),
            file_name="statistics.json",
            mime="application/json",
            use_container_width=True,
        )
    if video_path.exists():
        download_cols[2].download_button(
            "Download event video",
            data=video_path.read_bytes(),
            file_name=video_path.name,
            mime="video/mp4" if video_path.suffix == ".mp4" else "video/x-msvideo",
            use_container_width=True,
        )

    with st.expander("Raw statistics"):
        st.json(stats)


if __name__ == "__main__":
    main()
