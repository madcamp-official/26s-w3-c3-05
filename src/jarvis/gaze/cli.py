"""Operational CLI for gaze calibration, camera diagnostics, and evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from collections.abc import Iterator, Sequence
from typing import Any

from jarvis.calibration.profiles import load_profiles, save_profiles
from jarvis.calibration.session import CalibrationSession
from jarvis.gaze.evaluation import LabeledFrame, compute_target_selection_accuracy
from jarvis.gaze.features import FaceObservation


def _load_labeled_csv(path: Path) -> list[LabeledFrame]:
    """Load the documented frame-level evaluation columns from a CSV trace."""
    required = {"frame_id", "timestamp_ms", "predicted_target", "ground_truth_target"}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Evaluation CSV is missing columns: {', '.join(sorted(missing))}")
        frames = []
        for row_number, row in enumerate(reader, start=2):
            try:
                frames.append(
                    LabeledFrame(
                        frame_id=int(row["frame_id"]),
                        timestamp_ms=int(row["timestamp_ms"]),
                        predicted_target=row["predicted_target"],
                        ground_truth_target=row["ground_truth_target"],
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid evaluation row {row_number}: {exc}") from exc
    return frames


def _open_camera(camera_index: int) -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional vision install
        raise RuntimeError("Install the vision extra: pip install -e '.[vision]'") from exc
    # Windows의 기본 백엔드(MSMF)는 내부적으로 프레임을 버퍼링해 지연이 주기적으로
    # 쌓였다 풀리는 끊김을 유발한다 — DSHOW로 이를 피하고, 버퍼를 1프레임으로 제한해
    # 항상 최신 프레임을 읽게 한다(source.py의 OpenCVCameraSource와 동일한 조치).
    if sys.platform == "win32":
        camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open camera index {camera_index}")
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


def _observation_stream(model: Path, camera_index: int) -> Iterator[FaceObservation]:
    """Yield observations from one camera while keeping vision imports optional."""
    import cv2

    from jarvis.gaze.landmarks import FaceLandmarkerAdapter

    camera = _open_camera(camera_index)
    start = time.monotonic()
    frame_id = 0
    try:
        with FaceLandmarkerAdapter(model) as adapter:
            while True:
                ok, bgr_frame = camera.read()
                if not ok:
                    raise RuntimeError("Camera frame capture failed")
                timestamp_ms = int((time.monotonic() - start) * 1000)
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                yield adapter.process(rgb_frame, timestamp_ms, frame_id)
                frame_id += 1
    finally:
        camera.release()


def _run_calibrate(args: argparse.Namespace) -> int:
    output = Path(args.output)
    existing = load_profiles(output) if output.is_file() else []
    session = CalibrationSession(args.device_id)
    print(f"Look naturally at {args.device_id}. Capturing for {args.duration_seconds:.1f}s.")
    deadline = time.monotonic() + args.duration_seconds
    for observation in _observation_stream(Path(args.model), args.camera_index):
        session.add_observation(observation)
        if time.monotonic() >= deadline:
            break
    if session.frame_count < args.minimum_frames:
        raise RuntimeError(
            f"Only {session.frame_count} valid frames captured; need {args.minimum_frames}. "
            "Check lighting, camera position, and face visibility."
        )
    profile = session.finalize()
    profiles = [item for item in existing if item.device_id != profile.device_id]
    profiles.append(profile)
    save_profiles(profiles, output)
    print(json.dumps({"saved_to": str(output), "profile": profile.device_id}, ensure_ascii=False))
    return 0


def _run_inspect(args: argparse.Namespace) -> int:
    print("Move left/right and up/down. Press Ctrl+C after verifying signs and axes.")
    last_printed_ms = -args.interval_ms
    try:
        for observation in _observation_stream(Path(args.model), args.camera_index):
            if observation.timestamp_ms - last_printed_ms < args.interval_ms:
                continue
            last_printed_ms = observation.timestamp_ms
            print(
                f"yaw={observation.head_yaw_deg:7.2f} "
                f"pitch={observation.head_pitch_deg:7.2f} "
                f"roll={observation.head_roll_deg:7.2f} "
                f"left_iris={observation.left_iris_relative} "
                f"right_iris={observation.right_iris_relative}"
            )
    except KeyboardInterrupt:
        return 0
    return 0


def _run_diagnose_composition(args: argparse.Namespace) -> int:
    from jarvis.gaze.composition_diagnostics import analyze_fixation_sweep, summarize
    from jarvis.gaze.config import GazeConfig

    config = GazeConfig()
    print(
        "한 지점(문제가 되는 물체의 중앙)에서 눈을 떼지 말고 고개만 움직이세요.\n"
        f"  1) 좌↔우로 천천히 끝까지 (yaw), 2) 상↓하로 천천히 끝까지 (pitch).\n"
        f"{args.duration_seconds:.0f}초 캡처합니다. 먼저 끝나면 Ctrl+C."
    )
    observations = []
    deadline = time.monotonic() + args.duration_seconds
    last_printed_ms = -args.interval_ms
    try:
        for observation in _observation_stream(Path(args.model), args.camera_index):
            observations.append(observation)
            if observation.timestamp_ms - last_printed_ms >= args.interval_ms:
                last_printed_ms = observation.timestamp_ms
                mean_x = (
                    observation.left_iris_relative[0] + observation.right_iris_relative[0]
                ) / 2.0
                mean_y = (
                    observation.left_iris_relative[1] + observation.right_iris_relative[1]
                ) / 2.0
                clamp_metric = max(abs(mean_x), abs(mean_y))
                marker = "  <-- CLAMP" if clamp_metric > config.max_valid_eye_offset else ""
                remaining = max(0.0, deadline - time.monotonic())
                print(
                    f"[{remaining:4.1f}s] head_yaw={observation.head_yaw_deg:+6.1f} "
                    f"head_pitch={observation.head_pitch_deg:+6.1f} "
                    f"iris=({mean_x:+.2f}, {mean_y:+.2f}){marker}"
                )
            if time.monotonic() >= deadline:
                break
    except KeyboardInterrupt:
        print("캡처를 중단하고 지금까지의 프레임으로 분석합니다.")

    diagnostics = analyze_fixation_sweep(observations, config)
    if diagnostics.valid_frames < args.minimum_frames:
        raise RuntimeError(
            f"Only {diagnostics.valid_frames} valid frames captured; "
            f"need {args.minimum_frames}."
        )

    if args.csv_output:
        csv_path = Path(args.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "timestamp_ms",
                    "frame_id",
                    "head_yaw_deg",
                    "head_pitch_deg",
                    "head_roll_deg",
                    "left_iris_x",
                    "left_iris_y",
                    "right_iris_x",
                    "right_iris_y",
                    "eyes_open",
                    "face_detected",
                    "tracking_confidence",
                ]
            )
            for item in observations:
                writer.writerow(
                    [
                        item.timestamp_ms,
                        item.frame_id,
                        item.head_yaw_deg,
                        item.head_pitch_deg,
                        item.head_roll_deg,
                        item.left_iris_relative[0],
                        item.left_iris_relative[1],
                        item.right_iris_relative[0],
                        item.right_iris_relative[1],
                        item.eyes_open,
                        item.face_detected,
                        min(item.eye_tracking_confidence, item.face_tracking_confidence),
                    ]
                )
        print(f"frames written to {csv_path}")

    print(json.dumps(asdict(diagnostics), indent=2, ensure_ascii=False))
    for line in summarize(diagnostics):
        print(f"- {line}")
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    frames = _load_labeled_csv(Path(args.input))
    result = compute_target_selection_accuracy(frames, args.dataset_id, args.conditions)
    payload = asdict(result) | {"accuracy": result.accuracy, "meets_target": result.accuracy >= 0.90}
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jarvis-gaze")
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser("calibrate", help="capture a per-device gaze profile")
    calibrate.add_argument("device_id")
    calibrate.add_argument("--model", default="models/face_landmarker.task")
    calibrate.add_argument("--output", default="data/calibration/profiles.json")
    calibrate.add_argument("--camera-index", type=int, default=0)
    calibrate.add_argument("--duration-seconds", type=float, default=3.0)
    calibrate.add_argument("--minimum-frames", type=int, default=30)
    calibrate.set_defaults(handler=_run_calibrate)

    inspect = subparsers.add_parser("inspect-head-pose", help="print live head and iris axes")
    inspect.add_argument("--model", default="models/face_landmarker.task")
    inspect.add_argument("--camera-index", type=int, default=0)
    inspect.add_argument("--interval-ms", type=int, default=250)
    inspect.set_defaults(handler=_run_inspect)

    diagnose = subparsers.add_parser(
        "diagnose-composition",
        help="fixation sweep: estimate per-user head weights and iris clamp saturation",
    )
    diagnose.add_argument("--model", default="models/face_landmarker.task")
    diagnose.add_argument("--camera-index", type=int, default=0)
    diagnose.add_argument("--duration-seconds", type=float, default=20.0)
    diagnose.add_argument("--minimum-frames", type=int, default=60)
    diagnose.add_argument("--interval-ms", type=int, default=500)
    diagnose.add_argument("--csv-output", help="optional per-frame CSV dump path")
    diagnose.set_defaults(handler=_run_diagnose_composition)

    evaluate = subparsers.add_parser("evaluate", help="measure Target Selection Accuracy")
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--dataset-id", required=True)
    evaluate.add_argument("--conditions", required=True)
    evaluate.add_argument("--output")
    evaluate.set_defaults(handler=_run_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
