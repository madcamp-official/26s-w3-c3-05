"""재등록-검증 루프 분석(target_verification.py): bin 판정과 수집 버그/드리프트 구분."""

from __future__ import annotations

import pytest

from jarvis.gaze.classifier import DeviceGazeProfile, TargetClassifier
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.direction import yaw_pitch_to_direction
from jarvis.gaze.feature_profile import (
    PoseCorrectionPoint,
    TargetAreaProfile,
    TargetFeatureSample,
    TargetPoseCorrection,
)
from jarvis.gaze.target_verification import (
    compare_verifications,
    verify_target_samples,
)


def _sample(gaze_yaw: float, gaze_pitch: float = 0.0, head_yaw: float = 0.0) -> TargetFeatureSample:
    return TargetFeatureSample(
        gaze_yaw=gaze_yaw,
        gaze_pitch=gaze_pitch,
        head_yaw=head_yaw,
        head_pitch=0.0,
        head_roll=0.0,
        face_scale=0.10,
    )


def _classifier(correction: TargetPoseCorrection | None = None) -> TargetClassifier:
    classifier = TargetClassifier()
    classifier.register_profile(
        DeviceGazeProfile("monitor", yaw_pitch_to_direction(0.0, 0.0), variance=0.05),
        area_profile=TargetAreaProfile(
            center_yaw=0.0, center_pitch=0.0, radius_yaw=6.0, radius_pitch=6.0, sample_count=30
        ),
        pose_correction=correction,
    )
    return classifier


def test_bins_report_in_fraction_and_rescue_usage() -> None:
    correction = TargetPoseCorrection(
        points=(
            PoseCorrectionPoint(0.0, 0.0, 0.0, 8),
            PoseCorrectionPoint(25.0, -8.0, 0.0, 8),
        )
    )
    classifier = _classifier(correction)
    samples = (
        # 중립: 원시로 이미 IN — rescue 불필요.
        [_sample(0.5 + 0.1 * i, head_yaw=0.5 * i) for i in range(10)]
        # head 25도: 원시 (-8, 0)은 OUT이지만 보정이 중심으로 되돌린다.
        + [_sample(-8.0, 0.0, head_yaw=25.0) for _ in range(10)]
        # head 35도(양끝 open bin): 보정 끝값(-8)이 못 미치는 먼 gaze → OUT 유지.
        + [_sample(-20.0, 0.0, head_yaw=35.0) for _ in range(10)]
    )
    summary = verify_target_samples(classifier, "monitor", samples)
    by_label = {b.label: b for b in summary.bins}

    neutral = by_label["[-10,+10)"]
    assert neutral.in_fraction == 1.0
    assert neutral.rescue_fraction == 0.0

    turned = by_label["[+20,+30)"]
    assert turned.in_fraction == 1.0
    assert turned.rescue_fraction == 1.0
    assert turned.median_effective_distance < turned.median_raw_distance

    extreme = by_label["[+30,+inf)"]
    assert extreme.in_fraction == 0.0

    assert summary.total_samples == 30


def test_empty_bins_are_omitted_and_missing_area_raises() -> None:
    classifier = _classifier()
    summary = verify_target_samples(classifier, "monitor", [_sample(0.0)])
    assert [b.label for b in summary.bins] == ["[-10,+10)"]
    with pytest.raises(ValueError):
        verify_target_samples(classifier, "unregistered", [_sample(0.0)])


def _bin_dict(label: str, in_fraction: float, frame_count: int = 20) -> dict:
    return {
        "label": label,
        "median_head_yaw_deg": 0.0,
        "frame_count": frame_count,
        "median_raw_distance": 1.0,
        "median_effective_distance": 0.9 if in_fraction >= 0.7 else 1.6,
        "in_fraction": in_fraction,
        "rescue_fraction": 0.0,
    }


def test_compare_flags_collection_problem_when_out_from_the_start() -> None:
    earlier = [_bin_dict("[+20,+30)", 0.1)]
    later = [_bin_dict("[+20,+30)", 0.0)]
    lines = compare_verifications(earlier, later)
    assert any("등록 수집 문제" in line for line in lines)
    assert not any("세션 드리프트" in line for line in lines)


def test_compare_flags_drift_when_in_then_out() -> None:
    earlier = [_bin_dict("[+20,+30)", 0.95)]
    later = [_bin_dict("[+20,+30)", 0.2)]
    lines = compare_verifications(earlier, later)
    assert any("세션 드리프트" in line for line in lines)
    assert not any("등록 수집 문제" in line for line in lines)


def test_compare_reports_stable_and_skips_thin_bins() -> None:
    earlier = [_bin_dict("[-10,+10)", 0.98), _bin_dict("[+20,+30)", 0.9, frame_count=3)]
    later = [_bin_dict("[-10,+10)", 0.95), _bin_dict("[+20,+30)", 0.1, frame_count=3)]
    lines = compare_verifications(earlier, later)
    # 표본 3개짜리 bin은 판정에서 제외되므로 드리프트로 오판하지 않는다.
    assert any("유지되고" in line for line in lines)
    assert not any("세션 드리프트" in line for line in lines)


def test_export_sweep_samples_round_trips_registration_schema(tmp_path) -> None:
    """스윕 export가 등록 export와 같은 스키마라 ab-residual 로더로 그대로 읽힌다."""
    import json

    import numpy as np

    from jarvis.gaze.target_verification import export_sweep_samples

    samples = [_sample(1.0, 2.0, head_yaw=5.0), _sample(-1.0, 0.5, head_yaw=-15.0)]
    path = tmp_path / "sweep.json"
    export_sweep_samples(
        path,
        target_id="monitor",
        name="모니터",
        center_yaw_pitch=(0.5, 6.0),
        samples=samples,
        label="session-b",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["target_id"] == "monitor"
    assert payload["center_yaw_pitch"] == [0.5, 6.0]
    assert len(payload["samples"]) == 2
    restored = TargetFeatureSample.from_array(np.asarray(payload["samples"][0]))
    assert restored == samples[0]


def test_compare_with_no_overlap_asks_for_wider_sweep() -> None:
    lines = compare_verifications([], [_bin_dict("[-10,+10)", 1.0)])
    assert any("비교 가능한 bin이 없습니다" in line for line in lines)
