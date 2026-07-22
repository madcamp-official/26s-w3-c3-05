"""응시 고정 스윕 진단(composition_diagnostics.py): implied weight 복원과 클램프 통계."""

from __future__ import annotations

import math

import pytest

from jarvis.gaze.composition_diagnostics import analyze_fixation_sweep, summarize
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.features import FaceObservation


def _observation(
    frame_id: int,
    *,
    head_yaw: float = 0.0,
    head_pitch: float = 0.0,
    head_roll: float = 0.0,
    iris_x: float = 0.0,
    iris_y: float = 0.0,
    eyes_open: bool = True,
    confidence: float = 0.9,
    face_detected: bool = True,
) -> FaceObservation:
    return FaceObservation(
        timestamp_ms=frame_id * 33,
        frame_id=frame_id,
        left_iris_relative=(iris_x, iris_y),
        right_iris_relative=(iris_x, iris_y),
        head_yaw_deg=head_yaw,
        head_pitch_deg=head_pitch,
        head_roll_deg=head_roll,
        eye_tracking_confidence=confidence,
        face_tracking_confidence=confidence,
        face_detected=face_detected,
        eyes_open=eyes_open,
    )


def _yaw_sweep(weight: float, config: GazeConfig) -> list[FaceObservation]:
    """합성 yaw가 정확히 0으로 고정되도록 눈이 고개를 보상하는 스윕."""
    observations = []
    for index, head_yaw in enumerate(range(-40, 41, 2)):
        iris_x = -(weight * head_yaw) / config.max_eye_offset_deg
        observations.append(_observation(index, head_yaw=float(head_yaw), iris_x=iris_x))
    return observations


def test_recovers_exact_weight_from_perfect_compensation() -> None:
    config = GazeConfig()
    diagnostics = analyze_fixation_sweep(_yaw_sweep(config.head_yaw_weight, config), config)
    assert diagnostics.yaw is not None
    assert diagnostics.yaw.implied_weight == pytest.approx(config.head_yaw_weight, abs=1e-9)
    assert diagnostics.yaw.r_squared == pytest.approx(1.0, abs=1e-9)
    assert diagnostics.yaw.composed_std_current_deg == pytest.approx(0.0, abs=1e-9)
    assert diagnostics.yaw.head_range_deg == pytest.approx(80.0)
    # head pitch는 상수였으므로 pitch 회귀는 계산하지 않는다.
    assert diagnostics.pitch is None


def test_detects_miscalibrated_weight() -> None:
    """실제 보상 감도가 0.60인데 설정이 0.25이면 implied weight가 0.60으로 나온다."""
    config = GazeConfig()
    diagnostics = analyze_fixation_sweep(_yaw_sweep(0.60, config), config)
    assert diagnostics.yaw is not None
    assert diagnostics.yaw.implied_weight == pytest.approx(0.60, abs=1e-9)
    assert diagnostics.yaw.current_weight == pytest.approx(config.head_yaw_weight)
    # 잘못된 현재 가중치로는 응시 중에도 합성 yaw가 흔들리고, implied로는 사라진다.
    assert diagnostics.yaw.composed_std_current_deg > 5.0
    assert diagnostics.yaw.composed_std_implied_deg == pytest.approx(0.0, abs=1e-9)


def test_pitch_axis_uses_head_pitch_and_iris_y() -> None:
    config = GazeConfig()
    observations = [
        _observation(
            index,
            head_pitch=float(pitch),
            iris_y=-(0.5 * pitch) / config.max_eye_offset_deg,
        )
        for index, pitch in enumerate(range(-20, 21, 2))
    ]
    diagnostics = analyze_fixation_sweep(observations, config)
    assert diagnostics.pitch is not None
    assert diagnostics.pitch.implied_weight == pytest.approx(0.5, abs=1e-9)
    assert diagnostics.yaw is None


def test_roll_rotation_matches_runtime_composition() -> None:
    """roll 90°에서 y축 offset이 회전 후 x 보상이 된다 — 합성식과 같은 처리."""
    config = GazeConfig()
    observations = []
    for index, head_yaw in enumerate(range(-30, 31, 2)):
        compensation = -(config.head_yaw_weight * head_yaw) / config.max_eye_offset_deg
        # _rotate_2d(x, y, -90) == (y, -x)이므로 raw (0, c)가 회전 후 (c, 0)이 된다.
        observations.append(
            _observation(
                index,
                head_yaw=float(head_yaw),
                head_roll=90.0,
                iris_x=0.0,
                iris_y=compensation,
            )
        )
    diagnostics = analyze_fixation_sweep(observations, config)
    assert diagnostics.yaw is not None
    assert diagnostics.yaw.implied_weight == pytest.approx(config.head_yaw_weight, abs=1e-6)


def test_clamp_statistics_match_runtime_rejection_rule() -> None:
    config = GazeConfig()
    limit = config.max_valid_eye_offset
    observations = [
        _observation(0, head_yaw=-10.0, iris_x=limit + 0.05),  # rejected
        _observation(1, head_yaw=0.0, iris_x=limit - 0.01),  # near clamp
        _observation(2, head_yaw=10.0, iris_x=0.10),
        _observation(3, head_yaw=20.0, iris_x=0.10),
    ]
    diagnostics = analyze_fixation_sweep(observations, config)
    assert diagnostics.clamp is not None
    assert diagnostics.clamp.rejected_fraction == pytest.approx(0.25)
    assert diagnostics.clamp.near_clamp_fraction == pytest.approx(0.25)
    assert diagnostics.clamp.max_abs_offset == pytest.approx(limit + 0.05)


def test_invalid_frames_are_excluded() -> None:
    config = GazeConfig()
    observations = _yaw_sweep(config.head_yaw_weight, config)
    observations.append(_observation(900, eyes_open=False))
    observations.append(_observation(901, face_detected=False))
    observations.append(_observation(902, confidence=0.1))
    diagnostics = analyze_fixation_sweep(observations, config)
    assert diagnostics.total_frames == len(observations)
    assert diagnostics.valid_frames == len(observations) - 3


def test_empty_capture_reports_zero_frames_without_fits() -> None:
    diagnostics = analyze_fixation_sweep([])
    assert diagnostics.total_frames == 0
    assert diagnostics.valid_frames == 0
    assert diagnostics.clamp is None
    assert diagnostics.yaw is None and diagnostics.pitch is None
    assert any("유효 프레임" in line for line in summarize(diagnostics))


def test_summarize_flags_saturation_and_narrow_sweep() -> None:
    config = GazeConfig()
    limit = config.max_valid_eye_offset
    observations = [
        _observation(index, head_yaw=float(index % 3), iris_x=limit + 0.05)
        for index in range(20)
    ]
    lines = summarize(analyze_fixation_sweep(observations, config))
    assert any("클램프" in line and "거부" in line for line in lines)
    assert any("좁아" in line for line in lines)


def test_low_r_squared_recommends_nonlinear_correction() -> None:
    """잡음이 지배적인 스윕은 가중치 대신 자세별 보정이 필요하다고 알린다."""
    config = GazeConfig()
    observations = []
    for index, head_yaw in enumerate(range(-40, 41, 2)):
        noise = 0.4 * math.sin(index * 2.1)
        observations.append(_observation(index, head_yaw=float(head_yaw), iris_x=noise))
    diagnostics = analyze_fixation_sweep(observations, config)
    assert diagnostics.yaw is not None
    assert diagnostics.yaw.r_squared < 0.5
    lines = summarize(diagnostics)
    assert any("비선형" in line for line in lines)


def test_rejects_invalid_near_clamp_ratio() -> None:
    with pytest.raises(ValueError):
        analyze_fixation_sweep([], near_clamp_ratio=1.5)
