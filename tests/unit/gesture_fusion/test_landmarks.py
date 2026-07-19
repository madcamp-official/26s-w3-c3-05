"""README 8장 처리 과정 2·3단계(손목 기준·손바닥 크기 정규화)를 검증한다.

정규화 로직은 mediapipe 없이 순수하게 테스트할 수 있어야 한다는 설계 목표
(landmarks.py: 소스 교체 가능성)에 대한 회귀 테스트이기도 하다.
"""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.gesture_fusion.config import (
    HAND_LANDMARK_COUNT,
    LANDMARK_DIMS,
    MIDDLE_FINGER_MCP,
    WRIST,
    GestureConfig,
)
from jarvis.gesture_fusion.landmarks import (
    HandObservation,
    RawHandLandmarks,
    normalize_hand,
)


def _points_with_scale(scale: float, offset: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """손목=offset, 중지 MCP가 손목에서 x축으로 `scale`만큼 떨어진 21점 배열.

    필러 점들도 `scale`에 비례하게 두어, 서로 다른 scale의 배열이 손목 기준·손바닥
    크기 정규화 후 동일한 모양으로 수렴하도록 한다(스케일 불변성 테스트용).
    """
    points = np.zeros((HAND_LANDMARK_COUNT, LANDMARK_DIMS), dtype=np.float64)
    for i in range(HAND_LANDMARK_COUNT):
        points[i] = [offset[0] + i * 0.05 * scale, offset[1]]
    points[WRIST] = offset
    points[MIDDLE_FINGER_MCP] = [offset[0] + scale, offset[1]]
    return points


def _raw(
    points: np.ndarray,
    *,
    detection_confidence: float = 0.9,
    handedness_score: float = 0.95,
) -> RawHandLandmarks:
    return RawHandLandmarks(
        timestamp_ms=1_000,
        frame_id=1,
        points=points,
        handedness="Right",
        detection_confidence=detection_confidence,
        handedness_score=handedness_score,
    )


def test_wrist_becomes_origin() -> None:
    raw = _raw(_points_with_scale(0.2, offset=(0.5, 0.3)))
    obs = normalize_hand(raw)
    assert obs.hand_detected
    np.testing.assert_allclose(obs.landmarks[WRIST], [0.0, 0.0], atol=1e-9)


def test_scale_invariance_same_shape_different_distance() -> None:
    """카메라 거리(스케일)만 다르고 손 모양이 같으면 정규화 결과가 같아야 한다."""
    near = normalize_hand(_raw(_points_with_scale(0.4, offset=(0.1, 0.1))))
    far = normalize_hand(_raw(_points_with_scale(0.1, offset=(0.6, 0.2))))
    np.testing.assert_allclose(near.landmarks, far.landmarks, atol=1e-9)


def test_palm_scale_is_root_to_tip_distance() -> None:
    obs = normalize_hand(_raw(_points_with_scale(0.25)))
    assert obs.palm_scale == pytest.approx(0.25)
    # 정규화 후 중지 MCP는 손바닥 크기로 나뉘어 x=1 위치에 온다.
    np.testing.assert_allclose(obs.landmarks[MIDDLE_FINGER_MCP], [1.0, 0.0], atol=1e-9)


def test_low_detection_confidence_is_lost_tracking() -> None:
    raw = _raw(_points_with_scale(0.2), detection_confidence=0.1)
    obs = normalize_hand(raw)
    assert not obs.hand_detected
    assert obs.detection_confidence == 0.0
    assert obs.handedness_score == 0.0
    np.testing.assert_array_equal(obs.landmarks, np.zeros((HAND_LANDMARK_COUNT, LANDMARK_DIMS)))


def test_degenerate_palm_scale_is_lost_tracking() -> None:
    """손목과 중지 MCP가 겹쳐 손바닥 크기가 0이면 좌표 폭주 대신 추적 손실로 처리한다."""
    points = _points_with_scale(0.2)
    points[MIDDLE_FINGER_MCP] = points[WRIST]  # scale → 0
    obs = normalize_hand(_raw(points))
    assert not obs.hand_detected


def test_rotation_is_preserved_not_normalized() -> None:
    """볼륨 조절(손목 회전) 제스처를 위해 회전은 정규화하지 않는다."""
    upright = _points_with_scale(0.2)
    rotated = upright.copy()
    # 중지 MCP를 x축에서 y축으로 90도 돌린다: 정규화가 회전을 지운다면 두 결과가 같아진다.
    rotated[MIDDLE_FINGER_MCP] = [rotated[WRIST][0], rotated[WRIST][1] + 0.2]
    a = normalize_hand(_raw(upright))
    b = normalize_hand(_raw(rotated))
    assert not np.allclose(a.landmarks[MIDDLE_FINGER_MCP], b.landmarks[MIDDLE_FINGER_MCP])


def test_configurable_palm_reference_indices() -> None:
    """손바닥 크기 기준 랜드마크를 config로 바꿀 수 있다(파라미터 교체 가능성)."""
    config = GestureConfig(palm_scale_root_index=WRIST, palm_scale_tip_index=1)
    points = _points_with_scale(0.2)
    points[1] = [points[WRIST][0] + 0.5, points[WRIST][1]]
    obs = normalize_hand(_raw(points), config)
    assert obs.palm_scale == pytest.approx(0.5)


def test_origin_index_is_independent_of_palm_scale_root() -> None:
    """좌표 원점(origin_index)은 스케일 기준(palm_scale_root_index)과 분리돼 있다.

    스케일 기준만 손목으로 두고 원점을 다른 랜드마크로 옮기면, 그 랜드마크가
    (스케일 기준이 아님에도) 원점(0,0,0)에 와야 한다.
    """
    config = GestureConfig(
        origin_index=MIDDLE_FINGER_MCP,
        palm_scale_root_index=WRIST,
        palm_scale_tip_index=MIDDLE_FINGER_MCP,
    )
    obs = normalize_hand(_raw(_points_with_scale(0.2, offset=(0.5, 0.3))), config)
    # 원점으로 지정한 중지 MCP가 (0,0,0)에 온다.
    np.testing.assert_allclose(obs.landmarks[MIDDLE_FINGER_MCP], [0.0, 0.0], atol=1e-9)
    # 손목은 스케일 기준일 뿐 원점이 아니므로 (0,0,0)이 아니다.
    assert not np.allclose(obs.landmarks[WRIST], [0.0, 0.0])


def test_handedness_score_is_propagated_separately() -> None:
    """handedness_score는 검출 신뢰도와 별개로 관측값까지 전파된다."""
    obs = normalize_hand(_raw(_points_with_scale(0.2), detection_confidence=0.8, handedness_score=0.6))
    assert obs.detection_confidence == pytest.approx(0.8)
    assert obs.handedness_score == pytest.approx(0.6)


def test_raw_landmarks_reject_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        RawHandLandmarks(
            timestamp_ms=0,
            frame_id=0,
            points=np.zeros((10, LANDMARK_DIMS), dtype=np.float64),
            handedness="Right",
            detection_confidence=0.9,
            handedness_score=0.9,
        )


def test_observation_rejects_non_finite_landmarks() -> None:
    bad = np.zeros((HAND_LANDMARK_COUNT, LANDMARK_DIMS), dtype=np.float64)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        HandObservation(
            timestamp_ms=0,
            frame_id=0,
            landmarks=bad,
            handedness="Right",
            palm_scale=0.2,
            detection_confidence=0.9,
            handedness_score=0.9,
            hand_detected=True,
        )
