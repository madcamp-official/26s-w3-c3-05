"""Hand feature engineering — HandObservation 시퀀스 → 프레임별 feature 벡터.

README 8장 처리 과정의 세 번째 단계를 구현한다:

    손바닥 크기 정규화 → 속도·가속도·관절 각도 생성 → Causal TCN/GRU

`HandFeatureExtractor`는 **causal streaming** 추출기다 — 현재와 과거 프레임만 쓰고
미래 프레임을 보지 않는다(development-principles.md 5.3: 온라인 추론은 causal). 속도는
직전 프레임과의 차분, 가속도는 속도의 차분으로 만들며, 시간 간격은 계약의 monotonic
`timestamp_ms`를 그대로 쓴다(자체 시계로 다시 재지 않는다).

feature 벡터의 구성(위치·관절각·속도·가속도)과 각 그룹의 on/off는 GestureConfig가
정한다. 그래서 뒤에 붙는 추론 모델(TCN/GRU 등)을 갈아끼우거나 입력 차원을 줄일 때
이 모듈이 아니라 config만 바꾸면 된다. 이 모듈은 mediapipe에 의존하지 않으므로 카메라·
모델 없이 단위 테스트할 수 있다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from jarvis.gesture_fusion.config import (
    DEFAULT_GESTURE_CONFIG,
    HAND_LANDMARK_COUNT,
    JOINT_ANGLE_TRIPLETS,
    GestureConfig,
)
from jarvis.gesture_fusion.landmarks import HandObservation
from jarvis.gesture_fusion.smoothing import OneEuroFilter

FloatArray = npt.NDArray[np.float64]

_POSITION_DIMS = HAND_LANDMARK_COUNT * 3


def compute_joint_angles(
    landmarks: FloatArray,
    triplets: tuple[tuple[int, int, int], ...] = JOINT_ANGLE_TRIPLETS,
) -> FloatArray:
    """각 (a, b, c) 삼각에서 꼭짓점 b의 굴곡각(radian, 0..π)을 잰다.

    두 뼈 벡터(b→a, b→c) 중 하나라도 길이가 0에 가까우면(랜드마크 겹침) 각을
    정의할 수 없으므로 0을 넣는다 — NaN을 흘려보내지 않는다
    (development-principles.md 7.2: 모델/입력의 비정상값은 실행이 아니라 안전값으로).
    """
    angles = np.zeros(len(triplets), dtype=np.float64)
    for i, (a, b, c) in enumerate(triplets):
        v1 = landmarks[a] - landmarks[b]
        v2 = landmarks[c] - landmarks[b]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos = float(np.dot(v1, v2) / (n1 * n2))
        angles[i] = math.acos(max(-1.0, min(1.0, cos)))
    return angles


@dataclass(frozen=True, slots=True)
class FrameFeatures:
    """한 프레임의 모델 입력 feature 벡터.

    `vector`는 config가 켠 그룹만 위치→관절각→속도→가속도 순으로 이어붙인 1차원
    배열이다. `hand_detected=False`면 추적 손실 프레임으로, `vector`는 0으로 채운
    같은 길이의 배열이며 downstream은 이를 실행이 아니라 대기/거부로 다뤄야 한다.
    """

    timestamp_ms: int
    frame_id: int
    vector: FloatArray
    hand_detected: bool

    def __post_init__(self) -> None:
        if self.timestamp_ms < 0 or self.frame_id < 0:
            raise ValueError("timestamp_ms and frame_id must be non-negative")
        if self.vector.ndim != 1:
            raise ValueError("feature vector must be one-dimensional")
        if not np.all(np.isfinite(self.vector)):
            raise ValueError("feature vector must be finite")


def feature_dimension(config: GestureConfig = DEFAULT_GESTURE_CONFIG) -> int:
    """켜진 그룹으로 만들어질 feature 벡터의 길이. 모델 입력층 크기와 맞춘다."""
    dims = 0
    if config.include_positions:
        dims += _POSITION_DIMS
    if config.include_joint_angles:
        dims += len(JOINT_ANGLE_TRIPLETS)
    if config.include_velocity:
        dims += _POSITION_DIMS
    if config.include_acceleration:
        dims += _POSITION_DIMS
    return dims


class HandFeatureExtractor:
    """HandObservation을 하나씩 밀어넣어 프레임별 FrameFeatures를 얻는 causal 추출기.

    상태(직전 좌표·직전 속도·직전 timestamp)를 들고 속도·가속도를 온라인으로
    계산한다. 추적 손실 프레임이나 `max_frame_gap_ms`를 넘는 공백 뒤에는 history를
    리셋해, 공백을 가로지르는 큰 좌표 점프를 허위 속도로 만들지 않는다. 리셋 직후
    첫 유효 프레임의 속도·가속도는 0이다.
    """

    def __init__(self, config: GestureConfig = DEFAULT_GESTURE_CONFIG) -> None:
        self._config = config
        self._dimension = feature_dimension(config)
        self._prev_landmarks: FloatArray | None = None
        self._prev_velocity: FloatArray | None = None
        self._prev_timestamp_ms: int | None = None
        # Smooth the landmark positions before differencing so per-frame jitter
        # is not amplified into the velocity/acceleration features. Disabled by
        # config for tests that isolate the raw differentiation math.
        self._smoother: OneEuroFilter | None = (
            OneEuroFilter(
                min_cutoff=config.smoothing_min_cutoff,
                beta=config.smoothing_beta,
                d_cutoff=config.smoothing_d_cutoff,
            )
            if config.smooth_landmarks
            else None
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def reset(self) -> None:
        """속도·가속도 history와 평활화 상태를 비운다(추적 손실·시퀀스 경계에서 호출)."""
        self._prev_landmarks = None
        self._prev_velocity = None
        self._prev_timestamp_ms = None
        if self._smoother is not None:
            self._smoother.reset()

    def push(self, observation: HandObservation) -> FrameFeatures:
        """관측값 하나를 처리해 이 프레임의 feature를 반환한다(과거만 사용)."""
        if not observation.hand_detected:
            self.reset()
            return self._empty_features(observation)

        # 공백·역전 판정을 먼저 해 필요하면 평활화 상태까지 함께 리셋한 뒤 평활화한다.
        dt_ms = self._delta_ms(observation.timestamp_ms)
        landmarks = observation.landmarks
        if self._smoother is not None:
            landmarks = self._smoother.filter(landmarks, observation.timestamp_ms)
        flat = landmarks.reshape(-1)

        if dt_ms is None or self._prev_landmarks is None:
            velocity = np.zeros(_POSITION_DIMS, dtype=np.float64)
        else:
            dt_s = dt_ms / 1000.0
            velocity = (flat - self._prev_landmarks) / dt_s

        if dt_ms is None or self._prev_velocity is None:
            acceleration = np.zeros(_POSITION_DIMS, dtype=np.float64)
        else:
            dt_s = dt_ms / 1000.0
            acceleration = (velocity - self._prev_velocity) / dt_s

        vector = self._assemble(landmarks, velocity, acceleration)

        self._prev_landmarks = flat
        self._prev_velocity = velocity
        self._prev_timestamp_ms = observation.timestamp_ms

        return FrameFeatures(
            timestamp_ms=observation.timestamp_ms,
            frame_id=observation.frame_id,
            vector=vector,
            hand_detected=True,
        )

    def _delta_ms(self, timestamp_ms: int) -> int | None:
        """직전 프레임과의 시간 간격(ms). history가 없거나 공백이 크면 None(=리셋)."""
        if self._prev_timestamp_ms is None:
            return None
        dt = timestamp_ms - self._prev_timestamp_ms
        if dt <= 0 or dt > self._config.max_frame_gap_ms:
            # 순서 역전·중복 timestamp나 큰 공백은 신뢰할 수 없다 → 차분을 건너뛰고
            # history를 리셋한다(다음 프레임부터 다시 쌓는다).
            self.reset()
            return None
        return dt

    def _assemble(
        self,
        landmarks: FloatArray,
        velocity: FloatArray,
        acceleration: FloatArray,
    ) -> FloatArray:
        parts: list[FloatArray] = []
        if self._config.include_positions:
            parts.append(landmarks.reshape(-1))
        if self._config.include_joint_angles:
            parts.append(compute_joint_angles(landmarks))
        if self._config.include_velocity:
            parts.append(velocity)
        if self._config.include_acceleration:
            parts.append(acceleration)
        return np.concatenate(parts).astype(np.float64, copy=False)

    def _empty_features(self, observation: HandObservation) -> FrameFeatures:
        return FrameFeatures(
            timestamp_ms=observation.timestamp_ms,
            frame_id=observation.frame_id,
            vector=np.zeros(self._dimension, dtype=np.float64),
            hand_detected=False,
        )
