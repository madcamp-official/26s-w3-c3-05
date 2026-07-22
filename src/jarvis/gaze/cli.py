"""Operational CLI for gaze calibration, camera diagnostics, and evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
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


def _run_verify_target(args: argparse.Namespace) -> int:
    """재등록-검증 루프의 한 회차: 등록된 target을 응시 스윕으로 재검증한다.

    사용법: 재등록 **직후** `--label after-registration --output <a.json>`으로
    1회, 시간이 지나거나 자세·조명이 바뀐 뒤 `--compare <a.json>`으로 1회 더
    실행한다. 직후부터 OUT이면 등록 수집 문제, 직후엔 IN인데 나중에 OUT이면
    세션 드리프트다(target_verification.py 판정 문구 참고).
    """
    import datetime as _datetime

    from jarvis.calibration.registry import TargetRegistry
    from jarvis.gaze.classifier import TargetClassifier
    from jarvis.gaze.config import GazeConfig
    from jarvis.gaze.direction import direction_to_yaw_pitch
    from jarvis.gaze.feature_profile import TargetFeatureSample
    from jarvis.gaze.smoothing import GazeSmoother
    from jarvis.gaze.features import compose_gaze_vector
    from jarvis.gaze.target_verification import (
        compare_verifications,
        verify_target_samples,
    )

    config = GazeConfig()
    registry = TargetRegistry(Path(args.profiles))
    records = registry.records
    if not records:
        raise RuntimeError(f"no registered targets in {args.profiles}")
    if args.target_id:
        record = registry.get(args.target_id)
        if record is None:
            known = ", ".join(item.target_id for item in records)
            raise RuntimeError(f"unknown target '{args.target_id}' (registered: {known})")
    elif len(records) == 1:
        record = records[0]
    else:
        known = ", ".join(item.target_id for item in records)
        raise RuntimeError(f"multiple targets registered ({known}) — pass target_id")

    classifier = TargetClassifier(config)
    for item in records:
        classifier.register_profile(
            item.to_profile(),
            geometry_3d=item.to_geometry_3d(),
            feature_profile=item.feature_profile,
            area_profile=item.area_profile,
            pose_correction=item.pose_correction,
        )
    area_profile = classifier.area_profiles.get(record.target_id)
    if area_profile is None:
        raise RuntimeError(f"'{record.target_id}' has no traced area profile — re-register it")

    print(
        f"'{record.name}' 중앙 한 점에서 눈을 떼지 말고 고개만 움직이세요.\n"
        f"  좌로 끝까지 → 우로 끝까지 → 위·아래 → 카메라 가까이·멀리.\n"
        f"{args.duration_seconds:.0f}초 캡처합니다. 먼저 끝나면 Ctrl+C.",
        flush=True,
    )
    smoother = GazeSmoother(config)
    samples: list[TargetFeatureSample] = []
    deadline = time.monotonic() + args.duration_seconds
    last_printed_ms = -args.interval_ms
    try:
        for observation in _observation_stream(Path(args.model), args.camera_index):
            # deadline 검사는 유효성 필터보다 먼저 한다 — 얼굴 미검출/눈 감음
            # 프레임이 continue로 빠지면 루프 끝의 검사를 영영 만나지 못해
            # 캡처가 종료되지 않는다.
            if time.monotonic() >= deadline:
                break
            if not observation.eyes_open:
                continue
            gaze_vector = compose_gaze_vector(observation, config)
            smoothed = smoother.update(gaze_vector)
            if smoothed is None:
                continue
            left_eye = observation.left_eye_center_normalized
            right_eye = observation.right_eye_center_normalized
            if left_eye is None or right_eye is None:
                continue
            face_scale = math.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
            if not math.isfinite(face_scale) or face_scale <= 0.0:
                continue
            gaze_yaw, gaze_pitch = direction_to_yaw_pitch(smoothed.direction)
            try:
                sample = TargetFeatureSample(
                    gaze_yaw=gaze_yaw,
                    gaze_pitch=gaze_pitch,
                    head_yaw=observation.head_yaw_deg,
                    head_pitch=observation.head_pitch_deg,
                    head_roll=observation.head_roll_deg,
                    face_scale=face_scale,
                    face_center_x=(left_eye[0] + right_eye[0]) * 0.5,
                    face_center_y=(left_eye[1] + right_eye[1]) * 0.5,
                )
            except ValueError:
                continue
            samples.append(sample)
            if observation.timestamp_ms - last_printed_ms >= args.interval_ms:
                last_printed_ms = observation.timestamp_ms
                distance, _gy, _gp = classifier.area_distance_and_gaze(
                    record.target_id, area_profile, sample
                )
                status = "IN " if distance <= config.target_match_tolerance else "OUT"
                remaining = max(0.0, deadline - time.monotonic())
                print(
                    f"[{remaining:4.1f}s] head_yaw={observation.head_yaw_deg:+6.1f} "
                    f"area=x{distance:4.2f} {status}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("캡처를 중단하고 지금까지의 프레임으로 판정합니다.")

    if len(samples) < args.minimum_frames:
        raise RuntimeError(f"Only {len(samples)} valid frames captured; need {args.minimum_frames}.")

    summary = verify_target_samples(classifier, record.target_id, samples, config)
    payload = {
        "target_id": record.target_id,
        "target_name": record.name,
        "label": args.label,
        "captured_at": _datetime.datetime.now().isoformat(timespec="seconds"),
        "profiles_path": str(args.profiles),
        "summary": asdict(summary),
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"saved to {output_path}")

    if args.save_samples:
        from jarvis.gaze.target_verification import export_sweep_samples

        export_sweep_samples(
            args.save_samples,
            target_id=record.target_id,
            name=record.name,
            center_yaw_pitch=(area_profile.center_yaw, area_profile.center_pitch),
            samples=samples,
            label=args.label,
        )
        print(f"raw samples saved to {args.save_samples}")

    if args.compare:
        earlier_payload = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        earlier_bins = earlier_payload.get("summary", {}).get("bins", [])
        print(f"\n-- {earlier_payload.get('label', args.compare)} 실행과 비교 --")
        for line in compare_verifications(earlier_bins, [dict(b) for b in payload["summary"]["bins"]]):
            print(f"- {line}")
    return 0


def _load_residual_dataset(path: Path):
    import numpy as np

    from jarvis.gaze.feature_profile import TargetFeatureSample
    from jarvis.gaze.ridge_residual import ResidualDataset

    payload = json.loads(path.read_text(encoding="utf-8"))
    return ResidualDataset(
        target_id=str(payload.get("target_id", path.stem)),
        center_yaw_pitch=(
            float(payload["center_yaw_pitch"][0]),
            float(payload["center_yaw_pitch"][1]),
        ),
        samples=tuple(
            TargetFeatureSample.from_array(np.asarray(row, dtype=np.float64))
            for row in payload["samples"]
        ),
    )


def _run_ab_residual(args: argparse.Namespace) -> int:
    """저장된 원시 샘플(등록/스윕 export)로 residual 보정 후보를 오프라인 평가한다.

    두 모드:
    - 단일 파일(positional): leave-one-yaw-bin-out — 같은 세션 안의 관대한 시험.
    - `--train`/`--eval`: 교차 세션 관문 — 세션 A로 학습해 세션 B에서만 평가.
      런타임 채택 판정은 반드시 이쪽이며, 서로 다른 날 eval 2개 PASS가 기준이다.
    """
    from jarvis.gaze.config import GazeConfig

    if args.train or args.eval:
        if not (args.train and args.eval):
            raise RuntimeError("--train and --eval must be given together")
        from jarvis.gaze.ridge_residual import evaluate_cross_session

        train_sets = [_load_residual_dataset(Path(p)) for p in args.train]
        eval_sets = [_load_residual_dataset(Path(p)) for p in args.eval]
        report = evaluate_cross_session(
            train_sets,
            eval_sets,
            GazeConfig(),
            ridge_lambda=args.ridge_lambda,
            kernel_bandwidth=args.kernel_bandwidth,
        )
        train_n = sum(len(d.samples) for d in train_sets)
        eval_n = sum(len(d.samples) for d in eval_sets)
        print(
            f"train: {len(train_sets)} files / {train_n} samples "
            f"({', '.join(d.target_id for d in train_sets)})"
        )
        print(f"eval:  {len(eval_sets)} files / {eval_n} samples")
        print("bin         n    raw    bin-table  ridge   kernel  (eval median error, deg)")
        for item in report.bins:
            table = (
                f"{item.table_error_deg:6.2f}" if item.table_error_deg is not None else "  n/a "
            )
            print(
                f"{item.label:11s} {item.frame_count:4d} {item.raw_error_deg:6.2f}  "
                f"{table}    {item.ridge_error_deg:6.2f}  {item.kernel_error_deg:6.2f}"
            )
        for line in report.verdict_lines:
            print(f"- {line}")
        return 0

    if not args.samples:
        raise RuntimeError("pass a samples file, or --train/--eval for the cross-session gate")
    from jarvis.gaze.ridge_residual import evaluate_leave_one_bin_out

    dataset = _load_residual_dataset(Path(args.samples))
    report = evaluate_leave_one_bin_out(
        list(dataset.samples),
        dataset.center_yaw_pitch,
        GazeConfig(),
        ridge_lambda=args.ridge_lambda,
    )
    print(
        f"target={dataset.target_id} samples={len(dataset.samples)} lambda={args.ridge_lambda}"
    )
    print("bin         n    raw      bin-table  ridge   (held-out median error, deg)")
    for item in report.bins:
        table = f"{item.bin_table_error_deg:6.2f}" if item.bin_table_error_deg is not None else "  n/a "
        print(
            f"{item.label:11s} {item.frame_count:4d} {item.raw_error_deg:6.2f}   "
            f"{table}   {item.ridge_error_deg:6.2f}"
        )
    for line in report.verdict_lines:
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


def _run_report(args: argparse.Namespace) -> int:
    from jarvis.gaze.session_report import build_report, format_report, load_session

    path = Path(args.session)
    session = load_session(path)
    report = build_report(session, path=path, bin_width_deg=args.bin_width)
    print(format_report(report))
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

    verify = subparsers.add_parser(
        "verify-target",
        help="fixation sweep against a registered target; --compare splits collection bug vs session drift",
    )
    verify.add_argument("target_id", nargs="?", help="registry target id (optional if only one)")
    verify.add_argument("--profiles", default="data/calibration/profiles.json")
    verify.add_argument("--model", default="models/face_landmarker.task")
    verify.add_argument("--camera-index", type=int, default=0)
    verify.add_argument("--duration-seconds", type=float, default=20.0)
    verify.add_argument("--minimum-frames", type=int, default=60)
    verify.add_argument("--interval-ms", type=int, default=500)
    verify.add_argument("--label", default="verify", help="run label stored in the JSON output")
    verify.add_argument("--output", help="save this run's JSON here (for a later --compare)")
    verify.add_argument("--compare", help="earlier run's JSON to diff against (collection bug vs drift)")
    verify.add_argument(
        "--save-samples",
        help="save this sweep's raw feature samples here (session data for ab-residual --train/--eval)",
    )
    verify.set_defaults(handler=_run_verify_target)

    ab_residual = subparsers.add_parser(
        "ab-residual",
        help="offline residual A/B: single-file leave-bin-out, or --train/--eval cross-session gate",
    )
    ab_residual.add_argument(
        "samples", nargs="?", help="single raw sample JSON for leave-one-bin-out"
    )
    ab_residual.add_argument(
        "--train", nargs="+", help="session-A sample files (registration or --save-samples exports)"
    )
    ab_residual.add_argument("--eval", nargs="+", help="session-B sample files (held-out session)")
    ab_residual.add_argument("--ridge-lambda", type=float, default=1.0)
    ab_residual.add_argument("--kernel-bandwidth", type=float, default=1.0)
    ab_residual.set_defaults(handler=_run_ab_residual)

    evaluate = subparsers.add_parser("evaluate", help="measure Target Selection Accuracy")
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--dataset-id", required=True)
    evaluate.add_argument("--conditions", required=True)
    evaluate.add_argument("--output")
    evaluate.set_defaults(handler=_run_evaluate)

    report = subparsers.add_parser(
        "report",
        help="aggregate a labeled monitoring session (JSONL) into an accuracy/bias report",
    )
    report.add_argument("session", help="session .jsonl recorded by the monitoring app")
    report.add_argument("--bin-width", type=float, default=10.0, help="head-yaw bin width (deg)")
    report.set_defaults(handler=_run_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Windows 콘솔(cp949)이 표현 못 하는 문자가 리포트에 섞여도 크래시 대신
    # 대체 문자로 출력한다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = _build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
