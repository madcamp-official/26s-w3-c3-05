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
    left_eye_center_normalized: tuple[float, float] | None = None
    right_eye_center_normalized: tuple[float, float] | None = None
    eyes_open: bool = True
    head_position_mm: Vector3 | None = None
    """머리(얼굴 모델 원점)의 카메라 기준 3D 위치 근사값 — MediaPipe facial
    transformation matrix의 translation 성분(`transform[:3, 3]`)에서 얻는다.
    실제 눈금으로 검증된 계량 값이 아니라 MediaPipe의 표준 얼굴 모델 크기 가정에
    기반한 근사치다(models/README.md 참고). 얻을 수 없으면 None이며, 3D 삼각측량
    (calibration/triangulation.py)에서만 쓰이고 각도 기반 경로에는 영향을 주지
    않는다."""

    def __post_init__(self) -> None:
        if self.timestamp_ms < 0 or self.frame_id < 0:
            raise ValueError("timestamp_ms and frame_id must be non-negative")
        numeric_values = (
            *self.left_iris_relative,
            *self.right_iris_relative,
            self.head_yaw_deg,
            self.head_pitch_deg,
            self.head_roll_deg,
            self.eye_tracking_confidence,
            self.face_tracking_confidence,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("FaceObservation numeric values must be finite")
        if not all(
            -1.0 <= value <= 1.0 for value in (*self.left_iris_relative, *self.right_iris_relative)
        ):
            raise ValueError("iris relative positions must be within [-1, 1]")
        if not 0.0 <= self.eye_tracking_confidence <= 1.0:
            raise ValueError("eye_tracking_confidence must be within [0, 1]")
        if not 0.0 <= self.face_tracking_confidence <= 1.0:
            raise ValueError("face_tracking_confidence must be within [0, 1]")
        for name, center in (
            ("left_eye_center_normalized", self.left_eye_center_normalized),
            ("right_eye_center_normalized", self.right_eye_center_normalized),
        ):
            if center is None:
                continue
            if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in center):
                raise ValueError(f"{name} must contain normalized coordinates within [0, 1]")
        if self.head_position_mm is not None:
            if self.head_position_mm.shape != (3,) or not np.all(
                np.isfinite(self.head_position_mm)
            ):
                raise ValueError("head_position_mm must contain exactly three finite values")


@dataclass(frozen=True, slots=True)
class GazeVector:
    """합성된 시선 방향 단위 벡터와 이 프레임의 신뢰도."""

    direction: Vector3
    confidence: float
    timestamp_ms: int
    frame_id: int
    origin: Vector3 | None = None
    source: str = "head+iris"
    """시선 광선의 원점(카메라 기준 머리 위치, mm 근사) — FaceObservation의
    head_position_mm을 그대로 옮긴 값. 없으면 None(3D 삼각측량에 쓰이지 않고
    각도 기반 매칭으로만 처리된다)."""

    def __post_init__(self) -> None:
        if self.timestamp_ms < 0 or self.frame_id < 0:
            raise ValueError("timestamp_ms and frame_id must be non-negative")
        if not 0.0 <= self.confidence <= 1.0 or not math.isfinite(self.confidence):
            raise ValueError("confidence must be finite and within [0, 1]")
        if self.direction.shape != (3,) or not np.all(np.isfinite(self.direction)):
            raise ValueError("direction must contain exactly three finite values")
        norm = float(np.linalg.norm(self.direction))
        if not math.isclose(norm, 1.0, abs_tol=1e-6):
            raise ValueError(f"direction must be a unit vector, got norm={norm}")
        if self.origin is not None:
            if self.origin.shape != (3,) or not np.all(np.isfinite(self.origin)):
                raise ValueError("origin must contain exactly three finite values")
        if self.source not in {"head+iris", "head-only"}:
            raise ValueError("gaze source must be 'head+iris' or 'head-only'")


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
    return _compose_gaze_vector(observation, config, use_iris=observation.eyes_open)


def compose_head_vector(
    observation: FaceObservation, config: GazeConfig = GazeConfig()
) -> GazeVector | None:
    """Compose a low-confidence fallback from face/head pose only."""
    return _compose_gaze_vector(observation, config, use_iris=False)


def _compose_gaze_vector(
    observation: FaceObservation,
    config: GazeConfig,
    *,
    use_iris: bool,
) -> GazeVector | None:
    """단일 프레임의 관측값을 시선 방향 단위 벡터로 합성한다.

    얼굴을 잃었거나 tracking confidence가 `config.minimum_tracking_confidence`
    미만이면 신뢰할 수 없는 프레임으로 보고 None을 반환한다 (추적 손실 — 값을
    지어내지 않는다, development-principles.md 1·2절).
    """
    confidence = min(observation.eye_tracking_confidence, observation.face_tracking_confidence)
    if not observation.face_detected or confidence < config.minimum_tracking_confidence:
        return None

    if use_iris:
        left_x, left_y = observation.left_iris_relative
        right_x, right_y = observation.right_iris_relative
        eye_offset_x = (left_x + right_x) / 2.0
        eye_offset_y = (left_y + right_y) / 2.0
    else:
        eye_offset_x = 0.0
        eye_offset_y = 0.0
        confidence *= config.head_only_confidence_scale

    # 머리가 기울어진(roll) 만큼 눈 오프셋을 head-upright 좌표계로 되돌린다.
    eye_offset_x, eye_offset_y = _rotate_2d(eye_offset_x, eye_offset_y, -observation.head_roll_deg)

    eye_yaw_offset_deg = eye_offset_x * config.max_eye_offset_deg
    eye_pitch_offset_deg = eye_offset_y * config.max_eye_offset_deg

    total_yaw_deg = (
        observation.head_yaw_deg * config.head_yaw_weight + eye_yaw_offset_deg
    ) * config.horizontal_axis_sign
    total_pitch_deg = observation.head_pitch_deg * config.head_pitch_weight + eye_pitch_offset_deg

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
        origin=observation.head_position_mm,
        source="head+iris" if use_iris else "head-only",
    )
