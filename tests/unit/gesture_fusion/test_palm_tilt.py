"""손 기울기 게이트 — 각도 계산과 임계 판정.

배경(2026-07-20 실측): 손바닥 축이 카메라 쪽으로 기울면 2D에서 단축되어 자세 정보가
실제로 소실된다. 구간별 6클래스 분류 정확도는 0~10° 90.8%, 10~20° 76.6%, 20~30° 47.3%,
30° 초과 37.0%였고, 기울기 구간만 따로 학습해도 20° 초과에서 25.9%(우연 17%)라
데이터로 해결되지 않는다. 그래서 20°를 넘으면 판정을 거부한다.
"""

import math

import numpy as np
import pytest

from jarvis.gesture_fusion.config import DEFAULT_GESTURE_CONFIG, GestureConfig
from jarvis.gesture_fusion.landmarks import (
    HandObservation,
    RawHandLandmarks,
    is_palm_tilted,
    normalize_hand,
    palm_tilt_degrees,
)

WRIST, MID_MCP = 0, 9


def _points_3d(tilt_deg: float, length: float = 0.2) -> np.ndarray:
    """손목→중지 MCP 축이 화면 평면과 `tilt_deg`를 이루는 21점을 만든다."""
    pts = np.zeros((21, 3), dtype=np.float64)
    rad = math.radians(tilt_deg)
    pts[MID_MCP] = [0.0, -length * math.cos(rad), -length * math.sin(rad)]
    return pts


@pytest.mark.parametrize("angle", [0.0, 10.0, 20.0, 45.0, 80.0])
def test_tilt_angle_recovered(angle: float) -> None:
    """2D 단축비의 arccos가 실제 기울기각을 복원한다."""
    assert palm_tilt_degrees(_points_3d(angle)) == pytest.approx(angle, abs=1e-6)


def test_tilt_is_sign_independent() -> None:
    """카메라 쪽으로 눕히든 반대로 젖히든 같은 각으로 잰다(단축량만이 신호)."""
    assert palm_tilt_degrees(_points_3d(30.0)) == pytest.approx(
        palm_tilt_degrees(_points_3d(-30.0))
    )


def test_degenerate_input_returns_none_not_zero() -> None:
    """길이 0·NaN·잘못된 shape에서는 각을 지어내지 않는다.

    0°로 지어내면 퇴화 프레임이 게이트를 **통과**해 버린다 — 알 수 없음을 안전한
    값으로 위조하지 않는다(development-principles.md 2절).
    """
    assert palm_tilt_degrees(np.zeros((21, 3))) is None
    assert palm_tilt_degrees(np.full((21, 3), np.nan)) is None
    assert palm_tilt_degrees(np.zeros((21, 2))) is None


def _observation(tilt: float | None) -> HandObservation:
    pts = np.zeros((21, 2), dtype=np.float64)
    pts[MID_MCP] = [0.0, -0.2]
    raw = RawHandLandmarks(
        timestamp_ms=1, frame_id=1, points=pts, handedness="Right",
        detection_confidence=1.0, handedness_score=1.0, palm_tilt_degrees=tilt,
    )
    return normalize_hand(raw)


def test_gate_rejects_beyond_threshold() -> None:
    assert not is_palm_tilted(_observation(19.9))
    assert not is_palm_tilted(_observation(20.0))   # 임계값 자체는 통과
    assert is_palm_tilted(_observation(20.1))


def test_unknown_tilt_does_not_block() -> None:
    """z를 못 내는 소스에서는 게이트가 걸리지 않는다 — 시스템이 통째로 멈추면 안 된다."""
    assert not is_palm_tilted(_observation(None))


def test_gate_disabled_by_zero_threshold() -> None:
    assert not is_palm_tilted(_observation(89.0), GestureConfig(max_palm_tilt_degrees=0.0))


def test_tilt_propagates_through_normalization() -> None:
    """정규화가 각도를 삼키지 않는다 — 게이트는 하류에서 걸린다."""
    assert _observation(33.0).palm_tilt_degrees == pytest.approx(33.0)


def test_threshold_is_twenty_degrees() -> None:
    """실측으로 고른 값이라 우연히 바뀌면 실패해야 한다(20° = 판정률 85.1% / 정확도 88.7%)."""
    assert DEFAULT_GESTURE_CONFIG.max_palm_tilt_degrees == 20.0


@pytest.mark.parametrize("bad", [-1.0, 91.0, float("nan")])
def test_invalid_threshold_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="max_palm_tilt_degrees"):
        GestureConfig(max_palm_tilt_degrees=bad)


# --- 분류기 입력 특징 ---

def test_pose_features_include_fingertip_distances() -> None:
    """좌표 뒤에 손끝 쌍거리 10개가 붙는다.

    좌표만 주면 작은 MLP가 손끝 사이의 *관계*를 스스로 뽑지 못한다 — 엄지-중지끝
    거리는 단독으로도 index_point/pinch_middle을 오류율 10.9%로 가르는데, 좌표만 준
    모델의 index_point 재현율은 50.2%였다. 명시적으로 더하자 94.0%가 됐다.
    """
    from jarvis.gesture_fusion.pose_protocol import (
        FINGERTIPS,
        pose_feature_dimension,
        pose_features,
    )

    points = np.zeros((21, 2), dtype=np.float64)
    points[FINGERTIPS[0]] = [0.0, 0.0]   # 엄지끝
    points[FINGERTIPS[1]] = [3.0, 4.0]   # 검지끝 — 거리 5
    features = pose_features(points)

    assert features.shape == (pose_feature_dimension(21, 2),) == (52,)
    assert np.allclose(features[:42], points.reshape(-1))
    assert features[42] == pytest.approx(5.0)  # 첫 쌍 = 엄지-검지


def test_pose_features_reject_flat_input() -> None:
    """평탄화된 좌표를 넘기면 쌍거리를 계산할 수 없어 조용히 틀린 값을 내면 안 된다."""
    from jarvis.gesture_fusion.pose_protocol import pose_features

    with pytest.raises(ValueError, match="2-D"):
        pose_features(np.zeros(42, dtype=np.float64))


def test_two_fingers_tilt_limit_is_forty_degrees() -> None:
    """30~40°에 표본 162개가 있어 근거가 있는 한계선(40° 초과는 21개뿐이라 열지 않음)."""
    from jarvis.gesture_fusion.pose_protocol import DEFAULT_POSE_TILT_LIMITS

    assert DEFAULT_POSE_TILT_LIMITS["two_fingers"] == 40.0
    assert DEFAULT_POSE_TILT_LIMITS["index_point"] == 20.0
