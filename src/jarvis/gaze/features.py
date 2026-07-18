"""Gaze feature normalization: compose the unified gaze-direction vector.

README 7장 "시선 방향 벡터 합성"의 핵심 규칙을 구현한다: 머리 yaw/pitch와 눈동자
오프셋을 별도 feature로 이어붙이지 않고, 하나의 시선 방향 단위 벡터로 합성한다.

    시선 방향 벡터 = 머리 회전(yaw, pitch) ⊕ 눈-머리 상대 오프셋

이렇게 만든 벡터만 calibration·target classifier에 쓰인다 (features.py 밖으로는
원본 홍채 위치·머리 각도를 노출하지 않는다).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from jarvis.gaze.config import GazeConfig

Vector3 = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FaceObservation:
    """단일 프레임의 정규화된 얼굴·홍채 관측값 (README 7장 "입력 정보").

    좌표계: 홍채 상대 위치는 각 눈 소켓 안에서 -1..1로 정규화된 좌표다
    (x: 눈 안쪽(-1)~바깥쪽(+1), y: 아래(-1)~위(+1)). 각도는 "카메라를 정면으로
    바라볼 때" 0도를 기준으로 한 degree 단위이며, yaw는 +가 오른쪽, pitch는 +가
    위쪽, roll은 +가 시계 방향(카메라 기준)이다.
    """

    timestamp_ms: int
    frame_id: int
    left_iris_relative: tuple[float, float]
    right_iris_relative: tuple[float, float]
    head_yaw_deg: float
    head_pitch_deg: float
    head_roll_deg: float
    eye_tracking_confidence: float
    face_tracking_confidence: float
    face_detected: bool


@dataclass(frozen=True, slots=True)
class GazeVector:
    """합성된 시선 방향 단위 벡터와 이 프레임의 신뢰도."""

    direction: Vector3
    confidence: float
    timestamp_ms: int
    frame_id: int


def _rotate_2d(x: float, y: float, angle_deg: float) -> tuple[float, float]:
    """(x, y)를 원점 기준 angle_deg만큼 반시계 방향으로 회전한다."""
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a


def _direction_from_yaw_pitch(yaw_deg: float, pitch_deg: float) -> Vector3:
    """yaw/pitch(도)를 카메라 좌표계의 단위 벡터로 변환한다.

    정면 응시(yaw=pitch=0)는 (0, 0, 1)이 되도록 한다 — README 7장 등록 예시
    (`mean_direction: [0.12, -0.04, 0.99]`)처럼 z 성분이 지배적인 것과 일치시키기
    위함이다.
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    x = math.sin(yaw) * math.cos(pitch)
    y = -math.sin(pitch)
    z = math.cos(yaw) * math.cos(pitch)
    return np.array([x, y, z], dtype=np.float64)


def compose_gaze_vector(
    observation: FaceObservation, config: GazeConfig = GazeConfig()
) -> GazeVector | None:
    """단일 프레임의 관측값을 시선 방향 단위 벡터로 합성한다.

    얼굴을 잃었거나 tracking confidence가 `config.minimum_tracking_confidence`
    미만이면 신뢰할 수 없는 프레임으로 보고 None을 반환한다 (추적 손실 — 값을
    지어내지 않는다, development-principles.md 1·2절).
    """
    confidence = min(observation.eye_tracking_confidence, observation.face_tracking_confidence)
    if not observation.face_detected or confidence < config.minimum_tracking_confidence:
        return None

    left_x, left_y = observation.left_iris_relative
    right_x, right_y = observation.right_iris_relative
    eye_offset_x = (left_x + right_x) / 2.0
    eye_offset_y = (left_y + right_y) / 2.0

    # 머리가 기울어진(roll) 만큼 눈 오프셋을 head-upright 좌표계로 되돌린다.
    eye_offset_x, eye_offset_y = _rotate_2d(eye_offset_x, eye_offset_y, -observation.head_roll_deg)

    eye_yaw_offset_deg = eye_offset_x * config.max_eye_offset_deg
    eye_pitch_offset_deg = eye_offset_y * config.max_eye_offset_deg

    total_yaw_deg = observation.head_yaw_deg + eye_yaw_offset_deg
    total_pitch_deg = observation.head_pitch_deg + eye_pitch_offset_deg

    direction = _direction_from_yaw_pitch(total_yaw_deg, total_pitch_deg)
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm == 0.0:
        return None
    direction = direction / norm

    return GazeVector(
        direction=direction,
        confidence=confidence,
        timestamp_ms=observation.timestamp_ms,
        frame_id=observation.frame_id,
    )
