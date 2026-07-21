"""Pose-conditioned gaze 보정: 자세별 iris 포화 편향의 학습·영속화·런타임 적용.

2026-07-22 diagnose-composition 실측(documents/gaze.md)에서 |head yaw| 15도
밖 iris 추정이 포화·역행해 합성 시선이 자세 불변이 되지 못함을 확인했다.
이 테스트는 그 편향을 등록 1단계 샘플로 배워 area 판정 전에 되돌리는 경로를
검증한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.calibration.registry import TargetRegistry
from jarvis.calibration.target_registration import TargetRegistrationSession
from jarvis.gaze.classifier import DeviceGazeProfile, TargetClassifier
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.direction import yaw_pitch_to_direction
from jarvis.gaze.feature_profile import (
    PoseCorrectionPoint,
    TargetAreaProfile,
    TargetFeatureSample,
    TargetPoseCorrection,
    build_pose_correction,
)
from jarvis.gaze.smoothing import SmoothedGaze


def _sample(
    gaze_yaw: float,
    gaze_pitch: float = 0.0,
    head_yaw: float = 0.0,
) -> TargetFeatureSample:
    return TargetFeatureSample(
        gaze_yaw=gaze_yaw,
        gaze_pitch=gaze_pitch,
        head_yaw=head_yaw,
        head_pitch=0.0,
        head_roll=0.0,
        face_scale=0.10,
    )


def _phase1_samples() -> list[TargetFeatureSample]:
    """중립 자세 8프레임 + head yaw 25도에서 (-5, +3) 편향 8프레임."""
    neutral = [_sample(0.1 * i, 0.0, head_yaw=0.5 * i) for i in range(8)]
    turned = [_sample(-5.0 + 0.1 * i, 3.0, head_yaw=25.0 + 0.3 * i) for i in range(8)]
    return neutral + turned


BUILD_KWARGS = dict(
    center_yaw_pitch=(0.0, 0.0),
    reference_head_yaw_deg=0.0,
    bin_edges_deg=GazeConfig().pose_correction_bin_edges_deg,
    minimum_bin_samples=8,
    maximum_offset_deg=10.0,
)


def test_build_recovers_bin_bias_and_zeroes_reference_pose() -> None:
    correction = build_pose_correction(_phase1_samples(), **BUILD_KWARGS)
    assert correction is not None
    offset_neutral = correction.offset_for(0.0)
    assert offset_neutral[0] == pytest.approx(0.0, abs=0.5)
    assert offset_neutral[1] == pytest.approx(0.0, abs=0.5)
    offset_turned = correction.offset_for(26.0)
    assert offset_turned[0] == pytest.approx(-5.0, abs=0.7)
    assert offset_turned[1] == pytest.approx(3.0, abs=0.7)


def test_offset_interpolates_between_bins_and_holds_beyond_ends() -> None:
    correction = TargetPoseCorrection(
        points=(
            PoseCorrectionPoint(0.0, 0.0, 0.0, 8),
            PoseCorrectionPoint(25.0, -5.0, 3.0, 8),
        )
    )
    midway = correction.offset_for(12.5)
    assert midway[0] == pytest.approx(-2.5)
    assert midway[1] == pytest.approx(1.5)
    # 바깥으로는 외삽하지 않고 끝값을 유지한다.
    assert correction.offset_for(60.0) == pytest.approx((-5.0, 3.0))
    assert correction.offset_for(-40.0) == pytest.approx((0.0, 0.0))


def test_sparse_bins_are_skipped_and_single_bin_returns_none() -> None:
    samples = [_sample(0.1 * i, head_yaw=0.5 * i) for i in range(8)]
    samples.extend(_sample(-5.0, 3.0, head_yaw=25.0) for _ in range(7))  # 8 미만
    assert build_pose_correction(samples, **BUILD_KWARGS) is None


def test_offsets_are_clamped_to_maximum() -> None:
    samples = [_sample(0.1 * i, head_yaw=0.5 * i) for i in range(8)]
    samples.extend(_sample(-25.0, 0.0, head_yaw=25.0 + 0.3 * i) for i in range(8))
    correction = build_pose_correction(samples, **BUILD_KWARGS)
    assert correction is not None
    assert correction.offset_for(26.0)[0] == pytest.approx(-10.0, abs=0.6)


def test_unsorted_points_are_rejected() -> None:
    with pytest.raises(ValueError):
        TargetPoseCorrection(
            points=(
                PoseCorrectionPoint(25.0, -5.0, 3.0, 8),
                PoseCorrectionPoint(0.0, 0.0, 0.0, 8),
            )
        )


def _area_classifier(
    correction: TargetPoseCorrection | None,
) -> TargetClassifier:
    classifier = TargetClassifier()
    classifier.register_profile(
        DeviceGazeProfile("monitor", yaw_pitch_to_direction(0.0, 0.0), variance=0.05),
        area_profile=TargetAreaProfile(
            center_yaw=0.0, center_pitch=0.0, radius_yaw=6.0, radius_pitch=6.0, sample_count=30
        ),
        pose_correction=correction,
    )
    return classifier


def test_correction_recovers_target_at_turned_head_pose() -> None:
    correction = TargetPoseCorrection(
        points=(
            PoseCorrectionPoint(0.0, 0.0, 0.0, 8),
            PoseCorrectionPoint(30.0, -5.0, 9.0, 8),
        )
    )
    classifier = _area_classifier(correction)
    direction = yaw_pitch_to_direction(-5.0, 9.0)
    # 고개를 30도 돌린 채 모니터를 응시: 원시 gaze (-5, 9)는 area 밖이지만
    # 등록 때 배운 편향을 빼면 중심으로 돌아온다.
    turned = classifier.classify(
        direction, feature_sample=_sample(-5.0, 9.0, head_yaw=30.0)
    )
    assert turned.target == "monitor"
    # 같은 gaze라도 중립 자세에서는 보정이 0이므로 그대로 거부된다 —
    # 실제로 모니터 옆을 보는 시선을 보정이 끌어당기지 않는다.
    neutral = classifier.classify(
        direction, feature_sample=_sample(-5.0, 9.0, head_yaw=0.0)
    )
    assert neutral.target == GazeConfig().UNKNOWN_TARGET


def test_targets_without_correction_use_raw_gaze() -> None:
    classifier = _area_classifier(None)
    sample = _sample(-5.0, 9.0, head_yaw=30.0)
    assert classifier.corrected_gaze_for("monitor", sample) == (-5.0, 9.0)
    assert classifier.classify(
        yaw_pitch_to_direction(-5.0, 9.0), feature_sample=sample
    ).target == GazeConfig().UNKNOWN_TARGET


def _gaze(frame: int, yaw: float, pitch: float = 0.0) -> SmoothedGaze:
    return SmoothedGaze(yaw_pitch_to_direction(yaw, pitch), 1.0, frame * 50, frame)


def test_registration_finalize_builds_pose_correction() -> None:
    session = TargetRegistrationSession(
        "monitor", "모니터", "DISPLAY", "device-1", minimum_valid_frames=8
    )
    for index, sample in enumerate(_phase1_samples()):
        assert session.add(
            _gaze(index, sample.gaze_yaw, sample.gaze_pitch),
            1.0,
            face_scale=0.10,
            feature_sample=sample,
        )
    session.start_boundary(5_000)
    for index in range(12):
        angle = index / 12.0 * 2.0 * np.pi
        yaw = 2.0 * float(np.cos(angle))
        pitch = 2.0 * float(np.sin(angle))
        assert session.add(
            _gaze(200 + index, yaw, pitch),
            1.0,
            feature_sample=_sample(yaw, pitch, head_yaw=0.0),
        )
    record = session.finalize()
    assert record.pose_correction is not None
    assert record.pose_correction.reference_head_yaw_deg == pytest.approx(0.0)
    offset = record.pose_correction.offset_for(26.0)
    assert offset[0] == pytest.approx(-5.0, abs=0.7)
    assert offset[1] == pytest.approx(3.0, abs=0.7)
    assert record.pose_correction.offset_for(0.0)[0] == pytest.approx(0.0, abs=0.5)


def test_registry_round_trips_pose_correction(tmp_path) -> None:
    session = TargetRegistrationSession(
        "monitor", "모니터", "DISPLAY", "device-1", minimum_valid_frames=8
    )
    for index, sample in enumerate(_phase1_samples()):
        assert session.add(
            _gaze(index, sample.gaze_yaw, sample.gaze_pitch), 1.0, feature_sample=sample
        )
    session.start_boundary(5_000)
    for index in range(12):
        angle = index / 12.0 * 2.0 * np.pi
        yaw, pitch = 2.0 * float(np.cos(angle)), 2.0 * float(np.sin(angle))
        assert session.add(
            _gaze(200 + index, yaw, pitch), 1.0, feature_sample=_sample(yaw, pitch)
        )
    record = session.finalize()
    assert record.pose_correction is not None

    path = tmp_path / "targets.json"
    TargetRegistry(path).upsert(record)
    reloaded = TargetRegistry(path).get("monitor")
    assert reloaded is not None
    assert reloaded.pose_correction == record.pose_correction

    # rename은 보정을 보존한다.
    registry = TargetRegistry(path)
    renamed = registry.rename("monitor", "left monitor")
    assert renamed.pose_correction == record.pose_correction


def test_legacy_record_without_field_loads_as_none(tmp_path) -> None:
    path = tmp_path / "targets.json"
    path.write_text(
        """[{"target_id": "lamp", "name": "lamp", "device_type": "LIGHT",
            "direction": {"yaw": 0.0, "pitch": 0.0},
            "spread": {"yaw": 4.0, "pitch": 4.0}, "device_id": "lamp"}]""",
        encoding="utf-8",
    )
    record = TargetRegistry(path).get("lamp")
    assert record is not None
    assert record.pose_correction is None
