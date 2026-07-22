"""조건 충족식 등록 coverage: 편향된 스윕은 완료되지 않고, 원시 샘플은 저장된다."""

from __future__ import annotations

import json

import pytest

from jarvis.calibration.target_registration import (
    PoseCoverageTracker,
    RegistrationPhase,
    TargetRegistrationSession,
)
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.direction import yaw_pitch_to_direction
from jarvis.gaze.feature_profile import TargetFeatureSample
from jarvis.gaze.smoothing import SmoothedGaze


def _feature(
    head_yaw: float = 0.0,
    head_pitch: float = 0.0,
    face_scale: float = 0.10,
    gaze_yaw: float = 0.0,
    gaze_pitch: float = 0.0,
) -> TargetFeatureSample:
    return TargetFeatureSample(
        gaze_yaw=gaze_yaw,
        gaze_pitch=gaze_pitch,
        head_yaw=head_yaw,
        head_pitch=head_pitch,
        head_roll=0.0,
        face_scale=face_scale,
    )


def test_tracker_counts_pose_and_scale_conditions() -> None:
    tracker = PoseCoverageTracker(GazeConfig(), minimum_frames=3)
    # near/far는 정면 기준 scale(10프레임)이 쌓이기 전에는 세지 않는다.
    tracker.add(_feature(face_scale=0.12))
    assert dict((c.key, c.count) for c in tracker.report())["near"] == 0
    for _ in range(10):
        tracker.add(_feature(face_scale=0.10))
    assert tracker.reference_face_scale == pytest.approx(0.10)
    for _ in range(3):
        tracker.add(_feature(head_yaw=-25.0))
        tracker.add(_feature(head_yaw=25.0))
        tracker.add(_feature(head_pitch=15.0))
        tracker.add(_feature(head_pitch=-15.0))
        tracker.add(_feature(face_scale=0.12))   # 1.2x -> near
        tracker.add(_feature(face_scale=0.085))  # 0.85x -> far
    assert tracker.complete()
    assert tracker.missing_labels() == []
    assert tracker.progress() == pytest.approx(1.0)


def test_tracker_reports_missing_conditions() -> None:
    tracker = PoseCoverageTracker(GazeConfig(), minimum_frames=3)
    for _ in range(12):
        tracker.add(_feature())
    missing = tracker.missing_labels()
    assert "정면" not in missing
    # 좌/우, 상/하, 근/원은 짝으로 묶여 "한쪽도 안 채웠다"는 하나의 힌트로 나온다
    # (둘 다 개별로 요구하면 물체가 한쪽에 있거나 멀리 있을 때 반대쪽이
    # 구조적으로 어려운 문제가 재발한다, 2026-07-22 2·3차 실측).
    assert "고개 왼쪽/고개 오른쪽" in missing
    assert "고개 위/고개 아래" in missing
    assert "가까이/멀리" in missing
    assert 0.0 < tracker.progress() < 1.0


def test_tracker_completes_with_only_one_side_of_each_pair() -> None:
    """물체가 카메라 한쪽에 있어 반대쪽 고개 회전이 어려운 경우 — 같은 쪽만
    채워도 완료된다(둘 다 요구하지 않는다, 2026-07-22 2차 실측)."""
    tracker = PoseCoverageTracker(GazeConfig(), minimum_frames=3)
    for _ in range(10):
        tracker.add(_feature(face_scale=0.10))
    for _ in range(3):
        tracker.add(_feature(head_yaw=25.0))  # 오른쪽만, 왼쪽은 한 번도 없음
        tracker.add(_feature(head_pitch=-15.0))  # 아래만, 위는 한 번도 없음
        tracker.add(_feature(face_scale=0.12))
        tracker.add(_feature(face_scale=0.085))
    assert tracker.complete()
    assert tracker.missing_labels() == []
    counts = {c.key: c.count for c in tracker.report()}
    assert counts["left"] == 0
    assert counts["up"] == 0


def test_tracker_completes_with_only_far_when_object_is_far_from_camera() -> None:
    """물체가 카메라에서 멀리 있어 "가까이"를 채우기 어려운 경우 — "멀리"만
    채워도 완료된다(둘 다 요구하지 않는다, 2026-07-22 3차 실측)."""
    tracker = PoseCoverageTracker(GazeConfig(), minimum_frames=3)
    for _ in range(10):
        tracker.add(_feature(face_scale=0.10))
    for _ in range(3):
        tracker.add(_feature(head_yaw=-25.0))
        tracker.add(_feature(head_yaw=25.0))
        tracker.add(_feature(head_pitch=15.0))
        tracker.add(_feature(head_pitch=-15.0))
        tracker.add(_feature(face_scale=0.085))  # 멀리만, 가까이는 한 번도 없음
    assert tracker.complete()
    assert tracker.missing_labels() == []
    counts = {c.key: c.count for c in tracker.report()}
    assert counts["near"] == 0
    assert counts["far"] == 3


def test_tracker_stays_incomplete_without_front_when_total_frames_are_low() -> None:
    """정면을 한 번도 못 채우면(총 프레임도 적으면) 완료되지 않는다 — 총량 우회의 기준선."""
    tracker = PoseCoverageTracker(GazeConfig(), minimum_frames=3)
    for _ in range(10):  # 총량 우회 문턱(3 * 10.0 = 30)에 한참 못 미침
        tracker.add(_feature(head_yaw=25.0))  # 전부 오른쪽만 — 정면은 0
    assert not tracker.complete()
    assert "정면" in tracker.missing_labels()


def test_tracker_completes_without_front_when_total_frames_are_large_enough() -> None:
    """정면·근/원을 한 번도 못 채워도 전체 유효 프레임이 충분히 쌓이면 완료로 본다
    (사용자 지시 2026-07-22 — "너무 빡빡하다"). 정면이 없으면 근/원 기준 scale
    자체가 안 생겨 두 조건이 함께 막히는 연쇄를 총량으로 우회한다."""
    tracker = PoseCoverageTracker(GazeConfig(), minimum_frames=3)
    for _ in range(3):
        tracker.add(_feature(head_yaw=25.0))
        tracker.add(_feature(head_yaw=25.0, head_pitch=-15.0))  # yaw도 커서 정면 아님
    # 정면·근/원은 건드리지 않고, 총 프레임 수만 우회 문턱(3 * 10.0 = 30) 위로 쌓는다.
    for _ in range(30):
        tracker.add(_feature(head_yaw=25.0))
    assert tracker.complete()
    assert tracker.missing_labels() == []
    counts = {c.key: c.count for c in tracker.report()}
    assert counts["front"] == 0
    assert counts["near"] == 0
    assert counts["far"] == 0


def test_total_frame_override_does_not_help_yaw_pitch_groups() -> None:
    """좌/우·상/하는 총량 우회 대상이 아니다 — 한쪽만 요구하는 기존 완화로 충분하다고 보고 뺐다."""
    tracker = PoseCoverageTracker(GazeConfig(), minimum_frames=3)
    for _ in range(40):  # 총량은 우회 문턱을 넘지만 전부 정면(head_yaw=pitch=0)뿐
        tracker.add(_feature())
    assert not tracker.complete()
    missing = tracker.missing_labels()
    assert "고개 왼쪽/고개 오른쪽" in missing
    assert "고개 위/고개 아래" in missing


def _gaze(frame: int) -> SmoothedGaze:
    return SmoothedGaze(yaw_pitch_to_direction(0.0, 0.0), 1.0, frame * 50, frame)


def test_session_blocks_phase2_until_coverage_complete(tmp_path) -> None:
    session = TargetRegistrationSession(
        "monitor", "모니터", "DISPLAY", "d1",
        minimum_valid_frames=3,
        coverage_min_frames=3,
        raw_sample_dir=tmp_path,
    )
    # 정면만 수집: 시간(20초)이 훌쩍 지나도(프레임 간격 1초) 1단계가 끝나지 않는다.
    for frame in range(30):
        assert session.add(
            SmoothedGaze(yaw_pitch_to_direction(0.0, 0.0), 1.0, frame * 1_000, frame),
            1.0,
            feature_sample=_feature(),
        )
    assert session.is_elapsed(30 * 1_000) is False
    assert session.phase == RegistrationPhase.CENTER
    with pytest.raises(ValueError, match="coverage"):
        session.start_boundary(31_000)

    frame = 31
    for sample in (
        [_feature(head_yaw=-25.0)] * 3
        + [_feature(head_yaw=25.0)] * 3
        + [_feature(head_pitch=15.0)] * 3
        + [_feature(head_pitch=-15.0)] * 3
        + [_feature(face_scale=0.12)] * 3
        + [_feature(face_scale=0.085)] * 3
    ):
        assert session.add(
            SmoothedGaze(yaw_pitch_to_direction(0.0, 0.0), 1.0, 31_000 + frame * 50, frame),
            1.0,
            face_scale=sample.face_scale,
            feature_sample=sample,
        )
        frame += 1
    # coverage가 차면 시간과 무관하게 즉시 2단계로 넘어간다.
    assert session.phase == RegistrationPhase.BOUNDARY

    for index, offset in enumerate(
        [(2.0, 0.0), (0.0, 2.0), (-2.0, 0.0), (0.0, -2.0), (1.5, 1.5), (-1.5, -1.5)]
    ):
        assert session.add(
            SmoothedGaze(
                yaw_pitch_to_direction(offset[0], offset[1]),
                1.0,
                35_000 + index * 50,
                frame + index,
            ),
            1.0,
            feature_sample=_feature(gaze_yaw=offset[0], gaze_pitch=offset[1]),
        )
    record = session.finalize()
    assert record.area_profile is not None

    exports = list(tmp_path.glob("monitor_phase1_*.json"))
    assert len(exports) == 1
    payload = json.loads(exports[0].read_text(encoding="utf-8"))
    assert payload["target_id"] == "monitor"
    assert len(payload["samples"]) == session.center_valid_frame_count
    assert len(payload["samples"][0]) == 8


def test_finalize_rejects_incomplete_coverage() -> None:
    session = TargetRegistrationSession(
        "monitor", "모니터", "DISPLAY", "d1", minimum_valid_frames=3, coverage_min_frames=3
    )
    for frame in range(12):
        assert session.add(_gaze(frame), 1.0, feature_sample=_feature())
    with pytest.raises(ValueError, match="coverage"):
        session.finalize()
    assert "coverage_missing" in session.diagnostic_summary()


def test_zero_coverage_min_frames_keeps_time_based_flow() -> None:
    session = TargetRegistrationSession(
        "monitor", "모니터", "DISPLAY", "d1", minimum_valid_frames=3
    )
    assert session.coverage is None
    for frame in range(3):
        assert session.add(_gaze(frame), 1.0)
    session.start_boundary(1_000)  # coverage 없이도 기존 흐름 그대로.
    assert session.phase == RegistrationPhase.BOUNDARY
