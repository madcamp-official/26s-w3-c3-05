"""Hand feature engineering — HandObservation 시퀀스 → 프레임별 feature 벡터.

README 8장 처리 과정의 세 번째 단계를 구현한다:

    손바닥 크기 정규화 → 속도·관절 각도 생성 → Causal TCN/GRU

`HandFeatureExtractor`는 **causal streaming** 추출기다 — 현재와 과거 프레임만 쓰고
미래 프레임을 보지 않는다(development-principles.md 5.3: 온라인 추론은 causal). 속도는
직전 프레임과의 차분으로 만들며, 시간 간격은 계약의 monotonic `timestamp_ms`를 그대로
쓴다(자체 시계로 다시 재지 않는다).

feature 벡터의 구성(위치·관절각·속도)과 각 그룹의 on/off는 GestureConfig가 정한다.
그래서 뒤에 붙는 추론 모델(TCN/GRU 등)을 갈아끼우거나 입력 차원을 줄일 때 이 모듈이
아니라 config만 바꾸면 된다. 이 모듈은 mediapipe에 의존하지 않으므로 카메라·모델
없이 단위 테스트할 수 있다.

손가락 관절 위치의 가속도(2026-07-19 이전 `include_acceleration`)는 모델 입력에서
제거했다(2026-07-19 결정, documents/decisions.md) — 순수 위치 기반 신호라 손목
평행이동 가속도(아래 `wrist_acceleration`, swipe 판별용 별개 신호)와는 무관하며
이 제거로 영향받지 않는다.
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
    LANDMARK_DIMS,
    GestureConfig,
)
from jarvis.gesture_fusion.landmarks import HandObservation
from jarvis.gesture_fusion.smoothing import OneEuroFilter

FloatArray = npt.NDArray[np.float64]

_POSITION_DIMS = HAND_LANDMARK_COUNT * LANDMARK_DIMS
_WRIST_DIMS = LANDMARK_DIMS  # 손목 평행이동 벡터(x, y) 한 개의 차원 (z 제외)


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

    `vector`는 config가 켠 그룹만 위치→관절각→속도 순으로 이어붙인 1차원
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
    if config.include_wrist_translation:
        dims += 2 * _WRIST_DIMS  # 손목 평행이동 속도 + 가속도
    return dims


class HandFeatureExtractor:
    """HandObservation을 하나씩 밀어넣어 프레임별 FrameFeatures를 얻는 causal 추출기.

    상태(직전 좌표·직전 timestamp)를 들고 속도를 온라인으로 계산한다(손가락 관절
    위치의 가속도는 2026-07-19에 모델 입력에서 제거했다 — 손목 평행이동 가속도
    `wrist_acceleration`은 별개 신호라 아래에서 계속 계산한다). 추적 손실 프레임이나
    `max_frame_gap_ms`를 넘는 공백 뒤에는 history를 리셋해, 공백을 가로지르는 큰
    좌표 점프를 허위 속도로 만들지 않는다. 리셋 직후 첫 유효 프레임의 속도는 0이다.
    """

    def __init__(self, config: GestureConfig = DEFAULT_GESTURE_CONFIG) -> None:
        self._config = config
        self._dimension = feature_dimension(config)
        self._prev_landmarks: FloatArray | None = None
        self._prev_timestamp_ms: int | None = None
        # Wrist translation history — differenced with the SAME dt as the landmark
        # features so wrist velocity/acceleration share their timing and reset rules.
        self._prev_wrist_position: FloatArray | None = None
        self._prev_wrist_velocity: FloatArray | None = None
        # The exact (smoothed, if enabled) landmarks fed to the model this frame,
        # exposed read-only so a debugging view can show the real model input
        # rather than a separate approximation. None until the first valid frame
        # and after a reset/tracking loss.
        self._last_landmarks: FloatArray | None = None
        # The wrist translation velocity/acceleration actually fed to the model this
        # frame, exposed read-only for the same debugging view. None until the first
        # valid frame and after a reset/tracking loss.
        self._last_wrist_velocity: FloatArray | None = None
        self._last_wrist_acceleration: FloatArray | None = None
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
        # A separate One-Euro state for the wrist translation signal: it must be
        # smoothed before differencing for the same reason as the landmarks (palm_scale
        # and the wrist point both jitter), but it is a different (2,) signal so it
        # cannot share the landmark filter's per-coordinate state.
        self._wrist_smoother: OneEuroFilter | None = (
            OneEuroFilter(
                min_cutoff=config.smoothing_min_cutoff,
                beta=config.smoothing_beta,
                d_cutoff=config.smoothing_d_cutoff,
            )
            if config.smooth_landmarks
            else None
        )
        # palm_scale smoother (2026-07-19, 손목 평행이동 잡음 수정 — GestureConfig의
        # smooth_palm_scale 문서 참조). wrist_position = origin/palm_scale은 분자(절대
        # 위치)가 커서 palm_scale의 프레임별 잡음이 크게 증폭되는데, 위 _wrist_smoother는
        # 나눗셈 *이후* 값만 다뤄 이 증폭을 못 잡는다. palm_scale 자체를 평활화해 그
        # 증폭의 원인을 줄인다(실측: 정지 시 손목 속도 잡음이 약 3.85배 감소).
        self._palm_scale_smoother: OneEuroFilter | None = (
            OneEuroFilter(
                min_cutoff=config.palm_scale_smoothing_min_cutoff,
                beta=config.palm_scale_smoothing_beta,
                d_cutoff=config.palm_scale_smoothing_d_cutoff,
            )
            if config.smooth_landmarks and config.smooth_palm_scale
            else None
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def last_landmarks(self) -> FloatArray | None:
        """이 프레임 모델에 실제로 들어간 (평활화 여부 반영) 정규화 랜드마크 (21, 2).

        추적 손실·리셋 이후에는 None이다. 디버깅 뷰가 "모델이 실제로 보는 정점"을
        그대로 표시하는 데 쓴다(별도 근사가 아니라 같은 값).
        """
        return None if self._last_landmarks is None else self._last_landmarks.copy()

    @property
    def last_wrist_velocity(self) -> FloatArray | None:
        """이 프레임 모델에 들어간 손목 평행이동 속도 (2,). 추적 손실·리셋 후 None.

        `last_landmarks`와 같은 목적 — 디버깅 뷰가 "모델이 실제로 보는 손목 이동 속도"를
        그대로 표시한다(평활화 여부 반영). `include_wrist_translation`가 꺼져 있어도
        값 자체는 계산해 노출하지만, 그 경우 feature 벡터에는 들어가지 않는다.
        """
        return None if self._last_wrist_velocity is None else self._last_wrist_velocity.copy()

    @property
    def last_wrist_acceleration(self) -> FloatArray | None:
        """이 프레임 모델에 들어간 손목 평행이동 가속도 (2,). 추적 손실·리셋 후 None."""
        return (
            None
            if self._last_wrist_acceleration is None
            else self._last_wrist_acceleration.copy()
        )

    def reset(self) -> None:
        """속도 history와 평활화 상태를 비운다(추적 손실·시퀀스 경계에서 호출)."""
        self._prev_landmarks = None
        self._prev_timestamp_ms = None
        self._prev_wrist_position = None
        self._prev_wrist_velocity = None
        self._last_landmarks = None
        self._last_wrist_velocity = None
        self._last_wrist_acceleration = None
        if self._smoother is not None:
            self._smoother.reset()
        if self._wrist_smoother is not None:
            self._wrist_smoother.reset()
        if self._palm_scale_smoother is not None:
            self._palm_scale_smoother.reset()

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

        # 손목 평행이동도 같은 dt로 causal 차분한다. 미분 전에 (설정 시) 평활화해
        # palm_scale·손목 점의 지터가 속도로 증폭되지 않게 한다 — 랜드마크와 동일 규칙.
        wrist_position = observation.wrist_position
        if self._palm_scale_smoother is not None:
            # wrist_position(=origin/raw palm_scale)을 평활화된 palm_scale 기준으로
            # 재조정한다. raw/smoothed 비율을 곱하면 origin에 직접 접근하지 않고도
            # origin/palm_scale(smoothed)를 정확히 재현할 수 있다(2026-07-19 결정,
            # documents/decisions.md — 정지 시 손목 속도 잡음 약 3.85배 감소 실측).
            smoothed_palm_scale = float(
                self._palm_scale_smoother.filter(observation.palm_scale, observation.timestamp_ms)
            )
            if math.isfinite(smoothed_palm_scale) and smoothed_palm_scale > 0.0:
                wrist_position = wrist_position * (observation.palm_scale / smoothed_palm_scale)
        if self._wrist_smoother is not None:
            wrist_position = self._wrist_smoother.filter(
                wrist_position, observation.timestamp_ms
            )

        if dt_ms is None or self._prev_wrist_position is None:
            wrist_velocity = np.zeros(_WRIST_DIMS, dtype=np.float64)
        else:
            dt_s = dt_ms / 1000.0
            wrist_velocity = (wrist_position - self._prev_wrist_position) / dt_s

        if dt_ms is None or self._prev_wrist_velocity is None:
            wrist_acceleration = np.zeros(_WRIST_DIMS, dtype=np.float64)
        else:
            dt_s = dt_ms / 1000.0
            wrist_acceleration = (wrist_velocity - self._prev_wrist_velocity) / dt_s

        vector = self._assemble(landmarks, velocity, wrist_velocity, wrist_acceleration)

        self._prev_landmarks = flat
        self._prev_wrist_position = wrist_position
        self._prev_wrist_velocity = wrist_velocity
        self._prev_timestamp_ms = observation.timestamp_ms
        self._last_landmarks = landmarks
        self._last_wrist_velocity = wrist_velocity
        self._last_wrist_acceleration = wrist_acceleration

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
        wrist_velocity: FloatArray,
        wrist_acceleration: FloatArray,
    ) -> FloatArray:
        parts: list[FloatArray] = []
        if self._config.include_positions:
            parts.append(landmarks.reshape(-1))
        if self._config.include_joint_angles:
            parts.append(compute_joint_angles(landmarks))
        if self._config.include_velocity:
            parts.append(velocity)
        # 손목 평행이동 그룹은 기존 3개 그룹 뒤에 순수 추가한다(속도 → 가속도 순).
        # 기존 그룹의 오프셋을 바꾸지 않아 학습된 가중치·기존 슬라이스가 그대로 유효하다.
        if self._config.include_wrist_translation:
            parts.append(wrist_velocity)
            parts.append(wrist_acceleration)
        return np.concatenate(parts).astype(np.float64, copy=False)

    def _empty_features(self, observation: HandObservation) -> FrameFeatures:
        return FrameFeatures(
            timestamp_ms=observation.timestamp_ms,
            frame_id=observation.frame_id,
            vector=np.zeros(self._dimension, dtype=np.float64),
            hand_detected=False,
        )
