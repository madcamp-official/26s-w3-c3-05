"""README 7장 "시선 방향 벡터 합성"의 핵심 설계 규칙을 검증한다."""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.gaze.config import GazeConfig
from jarvis.gaze.features import FaceObservation, compose_gaze_vector


def _observation(
    *,
    left_iris_relative: tuple[float, float] = (0.0, 0.0),
    right_iris_relative: tuple[float, float] = (0.0, 0.0),
    head_yaw_deg: float = 0.0,
    head_pitch_deg: float = 0.0,
    head_roll_deg: float = 0.0,
    eye_tracking_confidence: float = 1.0,
    face_tracking_confidence: float = 1.0,
    face_detected: bool = True,
) -> FaceObservation:
    return FaceObservation(
        timestamp_ms=1_000,
        frame_id=1,
        left_iris_relative=left_iris_relative,
        right_iris_relative=right_iris_relative,
        head_yaw_deg=head_yaw_deg,
        head_pitch_deg=head_pitch_deg,
        head_roll_deg=head_roll_deg,
        eye_tracking_confidence=eye_tracking_confidence,
        face_tracking_confidence=face_tracking_confidence,
        face_detected=face_detected,
    )


def test_forward_gaze_is_unit_z_vector() -> None:
    gaze = compose_gaze_vector(_observation())
    assert gaze is not None
    np.testing.assert_allclose(gaze.direction, [0.0, 0.0, 1.0], atol=1e-9)
    assert gaze.direction @ gaze.direction == pytest.approx(1.0)


def test_direction_is_always_unit_length() -> None:
    gaze = compose_gaze_vector(
        _observation(head_yaw_deg=37.0, head_pitch_deg=-12.0, left_iris_relative=(0.4, -0.6))
    )
    assert gaze is not None
    assert float(np.linalg.norm(gaze.direction)) == pytest.approx(1.0)


def test_head_only_and_eye_only_rotation_compose_to_same_vector() -> None:
    """README: 등록(고개 돌림)과 실사용(눈짓만)의 조합 비율이 달라도 물리적으로
    같은 곳을 보면 거의 같은 벡터가 나와야 한다 — 핵심 설계 근거에 대한 회귀 테스트."""
    config = GazeConfig()

    head_only = compose_gaze_vector(_observation(head_yaw_deg=15.0))
    eye_offset = 15.0 / config.max_eye_offset_deg
    eye_only = compose_gaze_vector(
        _observation(left_iris_relative=(eye_offset, 0.0), right_iris_relative=(eye_offset, 0.0))
    )

    assert head_only is not None and eye_only is not None
    np.testing.assert_allclose(head_only.direction, eye_only.direction, atol=1e-9)


def test_head_and_eye_contributions_add() -> None:
    combined = compose_gaze_vector(
        _observation(
            head_yaw_deg=10.0,
            left_iris_relative=(0.5, 0.0),
            right_iris_relative=(0.5, 0.0),
        )
    )
    config = GazeConfig()
    equivalent_head_only = compose_gaze_vector(
        _observation(head_yaw_deg=10.0 + 0.5 * config.max_eye_offset_deg)
    )
    assert combined is not None and equivalent_head_only is not None
    np.testing.assert_allclose(combined.direction, equivalent_head_only.direction, atol=1e-9)


def test_head_roll_derotates_eye_offset() -> None:
    """머리를 90도 옆으로 기울이면 이미지상 수평 눈 오프셋은 실제로는 수직 방향이어야 한다."""
    config = GazeConfig()
    rolled = compose_gaze_vector(
        _observation(
            head_roll_deg=90.0,
            left_iris_relative=(0.5, 0.0),
            right_iris_relative=(0.5, 0.0),
        )
    )
    upright_vertical = compose_gaze_vector(
        _observation(left_iris_relative=(0.0, -0.5), right_iris_relative=(0.0, -0.5))
    )
    assert rolled is not None and upright_vertical is not None
    np.testing.assert_allclose(rolled.direction, upright_vertical.direction, atol=1e-6)
    assert config.max_eye_offset_deg > 0  # sanity: config actually used above


def test_face_not_detected_returns_none() -> None:
    assert compose_gaze_vector(_observation(face_detected=False)) is None


def test_low_tracking_confidence_returns_none() -> None:
    config = GazeConfig(minimum_tracking_confidence=0.5)
    gaze = compose_gaze_vector(_observation(eye_tracking_confidence=0.1), config)
    assert gaze is None


def test_confidence_is_minimum_of_eye_and_face() -> None:
    gaze = compose_gaze_vector(
        _observation(eye_tracking_confidence=0.9, face_tracking_confidence=0.6)
    )
    assert gaze is not None
    assert gaze.confidence == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("head_yaw_deg", float("nan")),
        ("eye_tracking_confidence", 1.1),
        ("left_iris_relative", (1.1, 0.0)),
    ],
)
def test_face_observation_rejects_invalid_model_values(field: str, value: object) -> None:
    values: dict[str, object] = {
        "timestamp_ms": 1_000,
        "frame_id": 1,
        "left_iris_relative": (0.0, 0.0),
        "right_iris_relative": (0.0, 0.0),
        "head_yaw_deg": 0.0,
        "head_pitch_deg": 0.0,
        "head_roll_deg": 0.0,
        "eye_tracking_confidence": 1.0,
        "face_tracking_confidence": 1.0,
        "face_detected": True,
    }
    values[field] = value
    with pytest.raises(ValueError):
        FaceObservation(**values)  # type: ignore[arg-type]
