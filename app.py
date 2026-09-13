"""Lightweight Streamlit UI for the Python event-camera simulator."""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from evsim.config import SimulatorConfig
from evsim.demo import write_synthetic_video
from evsim.pipeline import SimulationResult, simulate_source
from evsim.sources import open_source
from evsim.visualization import EventVideoRenderer

ROOT = Path(__file__).resolve().parent


def _safe_extract_zip(path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
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
) -> SimulatorConfig:
    preset_path = ROOT / "configs" / ("ideal.json" if preset == "Ideal" else "realistic.json")
    config = SimulatorConfig.load(preset_path)
    config.sensor.positive_threshold = positive_threshold
    config.sensor.negative_threshold = negative_threshold
    config.sensor.timestamp_resolution_us = timestamp_resolution_us
    config.sensor.refractory_period_us = refractory_period_us
    config.noise.enable_threshold_variation = threshold_variation
    config.noise.threshold_sigma = 0.03 if threshold_variation else 0.0
    config.noise.background_rate_hz = 0.05 if background_activity else 0.0
    config.input.linearize = linearization
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
    output_video = workdir / "event_video.mp4"
    config.visualization.output_video = str(output_video)
    config.output.npz_path = str(workdir / "events.npz")
    config.output.statistics_path = str(workdir / "statistics.json")
    config.output.csv_path = ""

    source = open_source(
        input_path,
        fallback_fps=config.input.fallback_fps,
        timestamp_scale_us=config.input.timestamp_scale_us,
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
    finally:
        source.close()
    result.event_stream.save_npz(config.output.npz_path)
    Path(config.output.statistics_path).write_text(
        __import__("json").dumps(result.statistics, indent=2) + chr(10),
        encoding="utf-8",
    )
    return result, renderer.actual_output_path or output_video

def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _render_preview(
    input_path: Path,
    events: np.ndarray,
    width: int,
    height: int,
    accumulation_time_us: int,
    overlay_opacity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = open_source(input_path)
    try:
        total = source.total_frames or 1
        target_index = max(total // 2, 0)
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
        demo_path = ROOT / "results_python" / "demo" / "input_960fps.avi"
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
    defaults = {
        "Ideal": (0.20, 0.20, 1, 0, False, False, False, 10),
        "Enhanced": (0.20, 0.18, 1, 100, True, True, False, 5),
    }[preset]
    source_mode = st.radio(
        "Input source",
        ["Built-in 960 FPS demo", "Upload video", "Upload image-sequence ZIP"],
        horizontal=True,
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
                "Accumulation window (ms)", [1, 5, 10, 20], value=int(defaults[7])
            )

        with st.expander("Execution control"):
            max_frames = st.number_input(
                "Maximum frames (0 = all)", 0, 1_000_000, 0, 10
            )

        submitted = st.form_submit_button(
            "Run Simulation", type="primary", use_container_width=True
        )

    if submitted:
        workdir = Path(tempfile.mkdtemp(prefix="evsim_ui_"))
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
            )
            with st.spinner("Running event simulation..."):
                result, video_path = _run_ui_simulation(input_path, config, workdir)
                input_preview, event_preview, overlay_preview = _render_preview(
                    input_path,
                    result.events,
                    int(result.statistics["width"]),
                    int(result.statistics["height"]),
                    int(accumulation_ms) * 1000,
                    config.visualization.overlay_opacity,
                )
            st.session_state["ui_result"] = {
                "input_path": str(input_path),
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
            }
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")

    ui_result = st.session_state.get("ui_result")
    if not ui_result:
        st.info("Choose a source and press Run Simulation.")
        return

    stats = ui_result["statistics"]
    st.subheader("Result Preview")
    col_input, col_events, col_overlay = st.columns(3)
    with col_input:
        st.markdown("**Input**")
        st.image(ui_result["input_preview"], use_column_width=True)
    with col_events:
        st.markdown("**Events**")
        st.image(ui_result["event_preview"], use_column_width=True)
    with col_overlay:
        st.markdown("**Overlay**")
        st.image(ui_result["overlay_preview"], use_column_width=True)

    event_rate = float(stats.get("event_rate_over_input_hz", 0.0)) / 1000.0
    metric_cols = st.columns(4)
    metric_cols[0].metric("Events", f"{int(stats.get('event_count', 0)):,}")
    metric_cols[1].metric("ON", f"{int(stats.get('on_events', 0)):,}")
    metric_cols[2].metric("OFF", f"{int(stats.get('off_events', 0)):,}")
    metric_cols[3].metric("Event rate", f"{event_rate:,.1f} kEvents/s")
    processing_cols = st.columns(3)
    processing_cols[0].metric(
        "Processing", f"{float(stats.get('processing_fps', 0.0)):,.1f} FPS"
    )
    processing_cols[1].metric(
        "Input duration", f"{float(stats.get('input_duration_s', 0.0)):,.4f} s"
    )
    processing_cols[2].metric(
        "Timestamp resolution", f"{ui_result['accumulation_ms']} ms accumulation"
    )

    st.caption(
        f"Preset: {ui_result['preset']} | source: {ui_result['source_mode']} | "
        "Lower thresholds produce more events; larger accumulation windows "
        "produce denser visualizations without changing raw events."
    )

    video_path = Path(ui_result["video_path"])
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
