"""등록 린터와 세션 리플레이(what-if)의 단위 테스트.

린터는 2026-07-22에 실제로 겪은 등록 실패 패턴(좁은 스윕 커버리지, clamp에 걸린
오프셋, 희박 bin)을 저장값만으로 잡아내야 하고, 리플레이는 녹화된 세션을 설정만
바꿔 다시 판정했을 때 결과가 그에 맞게 달라져야 한다.
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
from jarvis.gaze.registration_lint import lint_records, lint_target_record
from jarvis.gaze.session_report import build_report, load_session
from jarvis.gaze.smoothing import GazeSmoother
from jarvis.monitoring.gaze_probe import evaluate
from jarvis.monitoring.session_recorder import GazeSessionRecorder
from jarvis.monitoring.session_replay import (
    config_from_header,
    parse_override,
    replay_session,
)

TARGET_ID = "target_001"


def _area_profile() -> TargetAreaProfile:
    return TargetAreaProfile(
        center_yaw=0.0,
        center_pitch=0.0,
        radius_yaw=4.0,
        radius_pitch=4.0,
        sample_count=100,
        boundary_polygon=((-4.0, -4.0), (4.0, -4.0), (4.0, 4.0), (-4.0, 4.0)),
    )


def _correction(
    points: tuple[PoseCorrectionPoint, ...],
    reference: float | None = 0.0,
) -> TargetPoseCorrection:
    return TargetPoseCorrection(points=points, reference_head_yaw_deg=reference)


def _record(
    correction: TargetPoseCorrection | None,
    area: TargetAreaProfile | None = None,
) -> TargetRecord:
    return TargetRecord(
        target_id=TARGET_ID,
        name="monitor",
        device_type="UNKNOWN",
        direction=TargetDirection(yaw=0.0, pitch=0.0),
        spread=TargetSpread(yaw=4.0, pitch=4.0),
        device_id=TARGET_ID,
        area_profile=area if area is not None else _area_profile(),
        pose_correction=correction,
    )


def _healthy_correction() -> TargetPoseCorrection:
    return _correction(
        points=(
            PoseCorrectionPoint(-30.0, -2.0, 0.5, 80),
            PoseCorrectionPoint(-15.0, -1.0, 0.2, 90),
            PoseCorrectionPoint(0.0, 0.0, 0.0, 120),
            PoseCorrectionPoint(15.0, 1.2, -0.3, 85),
            PoseCorrectionPoint(30.0, 2.1, -0.6, 70),
        ),
    )


# --- registration lint --------------------------------------------------------


def test_lint_passes_healthy_registration() -> None:
    assert lint_target_record(_record(_healthy_correction()), GazeConfig()) == []


def test_lint_flags_missing_correction_and_area() -> None:
    config = GazeConfig()
    warnings = lint_target_record(_record(None), config)
    assert any("pose 보정 없음" in warning for warning in warnings)

    record = TargetRecord(
        target_id=TARGET_ID,
        name="monitor",
        device_type="UNKNOWN",
        direction=TargetDirection(yaw=0.0, pitch=0.0),
        spread=TargetSpread(yaw=4.0, pitch=4.0),
        device_id=TARGET_ID,
    )
    warnings = lint_target_record(record, config)
    assert any("area profile 없음" in warning for warning in warnings)


def test_lint_flags_one_sided_narrow_coverage() -> None:
    """speaker 실패 재현: 스윕이 head yaw -21~-7만 커버하고 정면이 빠짐."""
    correction = _correction(
        points=(
            PoseCorrectionPoint(-21.0, 0.0, 0.0, 109),
            PoseCorrectionPoint(-16.0, 3.7, -1.5, 415),
            PoseCorrectionPoint(-7.0, 4.0, -4.3, 44),
        ),
        reference=-22.0,
    )
    warnings = lint_target_record(_record(correction), GazeConfig())
    text = "\n".join(warnings)
    assert "커버리지" in text and "정면(0°)이 없음" in text
    assert "기준 자세" in text  # reference -22° 가 커버리지(-21~-7) 밖
    assert "고개 돌린 자세" in text


def test_lint_flags_clamped_offset_and_sparse_bin() -> None:
    """monitor 실패 재현: +32° bin 오프셋이 clamp(+10°)에 걸리고 표본 22개."""
    config = GazeConfig()
    correction = _correction(
        points=(
            PoseCorrectionPoint(-30.0, -2.0, 0.5, 80),
            PoseCorrectionPoint(0.0, 0.0, 0.0, 120),
            PoseCorrectionPoint(
                32.0, 2.1, config.pose_correction_max_offset_deg, 22
            ),
        ),
    )
    warnings = lint_target_record(_record(correction), config)
    text = "\n".join(warnings)
    assert "clamp" in text
    assert "표본 22개" in text


def test_lint_flags_area_radius_cap() -> None:
    config = GazeConfig()
    area = TargetAreaProfile(
        center_yaw=0.0,
        center_pitch=0.0,
        radius_yaw=config.registration_max_area_radius_deg,
        radius_pitch=4.0,
        sample_count=100,
    )
    warnings = lint_target_record(_record(_healthy_correction(), area), config)
    assert any("cap" in warning for warning in warnings)


def test_lint_records_returns_entry_per_target() -> None:
    findings = lint_records([_record(_healthy_correction())], GazeConfig())
    assert findings == {TARGET_ID: []}


# --- session replay -----------------------------------------------------------


def _observation(
    *, frame_id: int, timestamp_ms: int, yaw: float = 0.0
) -> FaceObservation:
    return FaceObservation(
        timestamp_ms=timestamp_ms,
        frame_id=frame_id,
        left_iris_relative=(0.0, 0.0),
        right_iris_relative=(0.0, 0.0),
        head_yaw_deg=yaw,
        head_pitch_deg=0.0,
        head_roll_deg=0.0,
        eye_tracking_confidence=1.0,
        face_tracking_confidence=1.0,
        face_detected=True,
        eyes_open=True,
        left_eye_center_normalized=(0.4, 0.4),
        right_eye_center_normalized=(0.6, 0.4),
    )


def _record_session(path: Path, frame_count: int = 12, yaw: float = 0.0) -> None:
    config = GazeConfig()
    smoother = GazeSmoother(config)
    classifier = TargetClassifier(config)
    classifier.register_profile(
        DeviceGazeProfile(
            device_id=TARGET_ID,
            mean_direction=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            variance=math.radians(4.0) ** 2,
        ),
        area_profile=_area_profile(),
        pose_correction=_healthy_correction(),
    )
    lock = GazeLockStateMachine(config)
    recorder = GazeSessionRecorder()
    recorder.start(
        path,
        config=config,
        targets=[_record(_healthy_correction())],
        started_timestamp_ms=0,
    )
    for index in range(frame_count):
        snapshot = evaluate(
            _observation(frame_id=index, timestamp_ms=index * 33, yaw=yaw),
            smoother=smoother,
            classifier=classifier,
            lock=lock,
            config=config,
        )
        recorder.record(snapshot, TARGET_ID)
    recorder.stop()


def test_replay_without_overrides_reproduces_classification(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    _record_session(path)
    session = load_session(path)
    replayed = replay_session(session)

    assert len(replayed.frames) == len(session.frames)
    original_targets = [frame["cls"]["target"] for frame in session.frames]
    replayed_targets = [frame["cls"]["target"] for frame in replayed.frames]
    assert replayed_targets == original_targets
    assert replayed.header["replay"] == {
        "overrides": {},
        "disable_pose_correction": False,
    }


def test_replay_with_shrunken_area_rejects_offcenter_gaze(tmp_path: Path) -> None:
    """head yaw -15° 세션(합성 gaze +3.75°)은 반경 4°에선 IN, 1°로 줄이면 OUT."""
    path = tmp_path / "session.jsonl"
    _record_session(path, yaw=-15.0)
    session = load_session(path)

    baseline = build_report(session)
    baseline_target = next(item for item in baseline.labels if item.label == TARGET_ID)
    assert baseline_target.accuracy_percent > 50.0

    replayed = replay_session(
        session,
        overrides={
            "registration_min_spread_deg": 1.0,
            "registration_max_area_radius_deg": 1.0,
        },
    )
    report = build_report(replayed)
    target = next(item for item in report.labels if item.label == TARGET_ID)
    assert target.accuracy_percent == 0.0
    assert replayed.header["config"]["registration_max_area_radius_deg"] == pytest.approx(1.0)


def test_replay_rejects_unknown_override_field(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    _record_session(path)
    session = load_session(path)
    with pytest.raises(ValueError, match="unknown GazeConfig field"):
        replay_session(session, overrides={"not_a_field": 1.0})


def test_replay_can_disable_pose_correction(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    _record_session(path)
    session = load_session(path)
    replayed = replay_session(session, disable_pose_correction=True)
    assert replayed.header["replay"]["disable_pose_correction"] is True
    # 보정 없이도 정면 응시 세션은 area 안이어야 한다(회귀 안전망).
    assert any(
        frame["cls"]["target"] == TARGET_ID for frame in replayed.frames
    )


def test_config_from_header_roundtrip_preserves_tuples() -> None:
    """asdict 경유로 리스트가 된 tuple 필드가 복원되는지 확인한다."""
    from dataclasses import asdict

    config = GazeConfig()
    header = {
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(config).items()
        }
    }
    restored = config_from_header(header)
    assert restored.pose_correction_bin_edges_deg == config.pose_correction_bin_edges_deg


def test_parse_override_types() -> None:
    assert parse_override("target_match_tolerance=1.2") == ("target_match_tolerance", 1.2)
    assert parse_override("smoothing_window_frames=8") == ("smoothing_window_frames", 8)
    assert parse_override("enable_3d_target_matching=false") == (
        "enable_3d_target_matching",
        False,
    )
    with pytest.raises(ValueError):
        parse_override("no_equals_sign")
