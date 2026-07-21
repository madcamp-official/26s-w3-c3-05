"""세션 레코더(JSONL)와 `jarvis-gaze report` 집계의 라운드트립 테스트.

실제 파이프라인 evaluate()가 만든 GazeSnapshot을 레코더로 기록한 파일이
헤더(config + target 프로필)만으로 다시 해석·집계되는지 — 즉 세션 파일이
자기완결적인지 — 를 검증한다. mediapipe/카메라는 필요 없다.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from jarvis.calibration.registry import TargetDirection, TargetRecord, TargetSpread
from jarvis.gaze.classifier import DeviceGazeProfile, TargetClassifier
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.feature_profile import (
    PoseCorrectionPoint,
    TargetAreaProfile,
    TargetPoseCorrection,
)
from jarvis.gaze.features import FaceObservation
from jarvis.gaze.lock import GazeLockStateMachine
from jarvis.gaze.session_report import (
    NO_TARGET_LABEL,
    build_report,
    format_report,
    load_session,
)
from jarvis.gaze.smoothing import GazeSmoother
from jarvis.monitoring.gaze_probe import evaluate
from jarvis.monitoring.session_recorder import GazeSessionRecorder


TARGET_ID = "target_001"


def _observation(
    *,
    frame_id: int,
    timestamp_ms: int,
    yaw: float = 0.0,
    pitch: float = 0.0,
) -> FaceObservation:
    return FaceObservation(
        timestamp_ms=timestamp_ms,
        frame_id=frame_id,
        left_iris_relative=(0.0, 0.0),
        right_iris_relative=(0.0, 0.0),
        head_yaw_deg=yaw,
        head_pitch_deg=pitch,
        head_roll_deg=0.0,
        eye_tracking_confidence=1.0,
        face_tracking_confidence=1.0,
        face_detected=True,
        eyes_open=True,
        left_eye_center_normalized=(0.4, 0.4),
        right_eye_center_normalized=(0.6, 0.4),
    )


def _area_profile() -> TargetAreaProfile:
    return TargetAreaProfile(
        center_yaw=0.0,
        center_pitch=0.0,
        radius_yaw=4.0,
        radius_pitch=4.0,
        sample_count=100,
    )


def _pose_correction() -> TargetPoseCorrection:
    return TargetPoseCorrection(
        points=(
            PoseCorrectionPoint(
                head_yaw_deg=-20.0, offset_yaw_deg=0.0, offset_pitch_deg=0.0, sample_count=30
            ),
            PoseCorrectionPoint(
                head_yaw_deg=0.0, offset_yaw_deg=0.0, offset_pitch_deg=0.0, sample_count=60
            ),
        ),
        reference_head_yaw_deg=0.0,
    )


def _target_record() -> TargetRecord:
    return TargetRecord(
        target_id=TARGET_ID,
        name="monitor",
        device_type="UNKNOWN",
        direction=TargetDirection(yaw=0.0, pitch=0.0),
        spread=TargetSpread(yaw=4.0, pitch=4.0),
        device_id=TARGET_ID,
        area_profile=_area_profile(),
        pose_correction=_pose_correction(),
    )


def _classifier(config: GazeConfig) -> TargetClassifier:
    classifier = TargetClassifier(config)
    classifier.register_profile(
        DeviceGazeProfile(
            device_id=TARGET_ID,
            mean_direction=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            variance=math.radians(4.0) ** 2,
        ),
        area_profile=_area_profile(),
        pose_correction=_pose_correction(),
    )
    return classifier


def _record_session(path: Path, frame_count: int = 12) -> GazeSessionRecorder:
    config = GazeConfig()
    smoother = GazeSmoother(config)
    classifier = _classifier(config)
    lock = GazeLockStateMachine(config)
    recorder = GazeSessionRecorder()
    recorder.start(path, config=config, targets=[_target_record()], started_timestamp_ms=0)
    for index in range(frame_count):
        snapshot = evaluate(
            _observation(frame_id=index, timestamp_ms=index * 33),
            smoother=smoother,
            classifier=classifier,
            lock=lock,
            config=config,
        )
        recorder.record(snapshot, TARGET_ID if index % 2 == 0 else NO_TARGET_LABEL)
    return recorder


def test_recorder_writes_self_contained_session(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    recorder = _record_session(path)
    footer = recorder.stop()

    assert footer["frames"] == 12
    session = load_session(path)
    assert session.header["config"]["target_match_tolerance"] == pytest.approx(1.10)
    assert session.target_names == {TARGET_ID: "monitor"}
    stored = session.target_record(TARGET_ID)
    assert stored is not None
    assert stored["pose_correction"]["points"][0]["head_yaw_deg"] == pytest.approx(-20.0)
    assert len(session.frames) == 12

    frame = session.frames[-1]
    assert frame["obs"]["head"] == [0.0, 0.0, 0.0]
    assert frame["gaze"]["smoothed"] is not None
    assert TARGET_ID in frame["targets"]
    assert frame["targets"][TARGET_ID]["area_nd"] <= 1.0
    assert frame["targets"][TARGET_ID]["correction_applied"] is False


def test_recorder_rejects_double_start_and_records_labels(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    recorder = _record_session(path)
    with pytest.raises(ValueError):
        recorder.start(tmp_path / "other.jsonl", config=GazeConfig(), targets=[])
    footer = recorder.stop()
    assert footer["labels"][TARGET_ID] == 6
    assert footer["labels"][NO_TARGET_LABEL] == 6
    with pytest.raises(ValueError):
        recorder.stop()


def test_report_aggregates_accuracy_per_label(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    _record_session(path).stop()
    session = load_session(path)
    report = build_report(session, path=path)

    assert report.total_frames == 12
    assert report.labeled_frames == 12
    by_label = {item.label: item for item in report.labels}
    assert set(by_label) == {TARGET_ID, NO_TARGET_LABEL}

    target = by_label[TARGET_ID]
    # 정면 응시 프레임은 등록 area 중앙에 있으므로 대부분 target으로 분류돼야 한다.
    assert target.accuracy_percent > 50.0
    assert target.bins and target.bins[0].frame_count == 6
    assert target.bins[0].in_area_percent == pytest.approx(100.0)
    assert target.bins[0].stored_offset_yaw == pytest.approx(0.0)

    # 같은 방향을 보는 "none" 라벨은 UNKNOWN이 아니므로 정확도가 낮게 잡힌다 —
    # 라벨별 기대값(UNKNOWN)이 분리 집계되는지 확인.
    none_label = by_label[NO_TARGET_LABEL]
    assert none_label.accuracy_percent < 50.0

    rendered = format_report(report)
    assert "monitor" in rendered
    assert "head yaw bin" in rendered


def test_report_warns_when_measured_bias_diverges_from_stored_offset(tmp_path: Path) -> None:
    """실측 편향과 저장된 보정 오프셋이 3도 이상 어긋난 bin은 경고로 드러난다."""
    path = tmp_path / "session.jsonl"
    config = GazeConfig()
    smoother = GazeSmoother(config)
    classifier = _classifier(config)
    lock = GazeLockStateMachine(config)
    recorder = GazeSessionRecorder()
    recorder.start(path, config=config, targets=[_target_record()], started_timestamp_ms=0)
    # head yaw -15도에서 저장된 보정은 0인데, 합성 gaze는 head 성분만으로도
    # (+15*0.25)*-1 = +3.75도 편향된다 → mismatch 경고 대상.
    for index in range(10):
        snapshot = evaluate(
            _observation(frame_id=index, timestamp_ms=index * 33, yaw=-15.0),
            smoother=smoother,
            classifier=classifier,
            lock=lock,
            config=config,
        )
        recorder.record(snapshot, TARGET_ID)
    recorder.stop()

    report = build_report(load_session(path), path=path)
    target = next(item for item in report.labels if item.label == TARGET_ID)
    assert target.warnings, "expected a correction-mismatch warning"
    assert "실측 편향" in target.warnings[0]
