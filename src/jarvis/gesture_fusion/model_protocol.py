"""Gesture 분류 모델의 교체 가능한 경계 — torch에 의존하지 않는 순수 모듈.

`landmarks.py`(순수) / `mediapipe_hands.py`(torch 상당의 무거운 의존성) 관계와 같은
구조다: 이 파일은 `GestureModel` Protocol과 그 입출력 타입만 정의해, 모델을 쓰는
쪽(gesture spotting, task 4)이 torch 없이도 타입 체크·단위 테스트를 할 수 있게 한다.
실제 torch 구현은 `model.py`(`ml` extra 필요)에 있다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from jarvis.contracts.messages import GesturePhase

FloatArray = npt.NDArray[np.float64]

# Phase는 계약상 닫힌 enum(IDLE/ONSET/ACTIVE/ENDING)이라 클래스 순서를 고정한다.
# 값 추가·삭제는 계약 변경 절차(interface-contract.md)를 따른다.
PHASE_LABELS: tuple[GesturePhase, ...] = (
    GesturePhase.IDLE,
    GesturePhase.ONSET,
    GesturePhase.ACTIVE,
    GesturePhase.ENDING,
)

# README 8장 "지원 제스처"의 6개 동적 제스처. Pinch·주먹은 README가 명시한 확장
# 기능이라 MVP 기본 label에서 뺀다(추가 시 이 튜플만 확장하면 됨 — 열린 문자열 키
# 원칙과 일관). "none"은 명확한 제스처가 없는 구간(IDLE 등)을 위한 배경 클래스로,
# 이게 없으면 분류기가 매 프레임 억지로 6개 중 하나를 골라 오탐을 늘린다.
DEFAULT_GESTURE_LABELS: tuple[str, ...] = (
    "none",
    "swipe_up",
    "swipe_down",
    "swipe_left",
    "swipe_right",
    "rotate_clockwise",
    "rotate_counter_clockwise",
)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Causal TCN 아키텍처 파라미터.

    `GestureConfig`(전처리 임계값)와 분리한다 — 이 값들은 모델 가중치의 shape을
    결정하므로, 저장된 가중치와 함께 버전 관리해야 하는 별개의 관심사다
    (development-principles.md 7.3).

    torch에 의존하지 않는 순수 값 타입이라(`ModelPrediction`·`ModelMetadata`와 같이)
    여기 torch-free 경계에 둔다 — `ml` extra 없이도 검증 규칙을 단위 테스트할 수 있다.
    """

    feature_dim: int
    """입력 feature 벡터 차원. `features.feature_dimension(GestureConfig)`와 일치해야 한다."""

    gesture_labels: tuple[str, ...] = DEFAULT_GESTURE_LABELS
    """gesture 분류 head의 출력 클래스 순서. 열린 문자열 키(interface-contract.md)."""

    channels: tuple[int, ...] = (32, 32, 32)
    """각 temporal block의 채널 수. 층이 늘수록(dilation이 커질수록) 더 긴 과거를 본다."""

    kernel_size: int = 3
    """각 causal conv의 시간축 커널 크기."""

    dropout: float = 0.2
    """temporal block 내부 dropout 비율 (0=off, 학습 시에만 적용)."""

    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if len(self.gesture_labels) < 2:
            raise ValueError("gesture_labels must contain at least two classes")
        if len(set(self.gesture_labels)) != len(self.gesture_labels):
            raise ValueError("gesture_labels must not contain duplicates")
        if not self.channels or any(c <= 0 for c in self.channels):
            raise ValueError("channels must be a non-empty tuple of positive ints")
        if self.kernel_size < 2:
            raise ValueError("kernel_size must be at least 2 for causal padding to be meaningful")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be within [0, 1)")

    @property
    def receptive_field(self) -> int:
        """이 아키텍처가 인과적으로 볼 수 있는 최대 과거 프레임 수(현재 프레임 포함).

        각 temporal block은 동일 dilation의 causal conv 두 개를 직렬로 쓰므로,
        block i(0-indexed, dilation=2**i)가 늘리는 시야는 `2 * (kernel_size-1) * 2**i`.
        스트리밍 추론에 필요한 최소 window 길이로 쓰인다.
        """
        span = 0
        for i in range(len(self.channels)):
            dilation = 2**i
            span += 2 * (self.kernel_size - 1) * dilation
        return span + 1


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    """모델 한 번의 추론 결과 (윈도우의 마지막 시점 기준).

    timestamp_ms·frame_id는 모델이 모르는 값이라 담지 않는다 — 호출자(gesture
    spotting)가 원본 프레임의 값을 그대로 붙여 `jarvis.contracts.GestureEstimate`를
    조립한다(공통 규칙: 각 모듈이 자체 시계로 timestamp를 다시 만들지 않는다).
    """

    gesture: str
    gesture_confidence: float
    phase: GesturePhase
    phase_confidence: float
    uncertainty: float

    def __post_init__(self) -> None:
        for name, value in (
            ("gesture_confidence", self.gesture_confidence),
            ("phase_confidence", self.phase_confidence),
            ("uncertainty", self.uncertainty),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1], got {value}")


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """학습된 가중치와 함께 다니는 메타데이터 (development-principles.md 7.3).

    `trained=False`(기본값)는 무작위 초기화 가중치라는 뜻이다 — 이 상태의 예측은
    fusion·safe commit 등 실제 실행 경로에 쓰지 않는다.
    """

    version: str = "untrained"
    trained: bool = False
    training_data_source: str = ""
    evaluation_notes: str = ""


class GestureModel(Protocol):
    """window(과거~현재 feature 시퀀스) → 마지막 시점의 gesture·phase 예측.

    MVP는 `model.CausalTCNGestureModel`이 구현한다. 나중에 GRU, 다른 아키텍처,
    원격 추론 서버로 바꾸더라도 downstream은 이 Protocol만 바라보면 된다
    (2026-07-18 결정: 추론 위치를 교체 가능한 경계로 분리).
    """

    @property
    def labels(self) -> tuple[str, ...]:
        """gesture 분류 head가 인식하는 label 순서."""
        ...

    @property
    def window_size(self) -> int:
        """이 모델이 인과적으로 필요로 하는 최소 시퀀스 길이(프레임 수)."""
        ...

    def predict(self, window: FloatArray) -> ModelPrediction:
        """window: (T, feature_dim), 시간순(가장 오래된 것이 index 0)."""
        ...


def normalized_entropy(probs: FloatArray) -> float:
    """확률 분포의 엔트로피를 [0, 1]로 정규화한 불확실성 지표.

    클래스가 균등분포에 가까울수록(어느 gesture인지 모호할수록) 1에 가깝고, 한
    클래스에 확신이 쏠릴수록 0에 가깝다. 클래스 수가 1이면(정의상 엔트로피 없음) 0.
    """
    num_classes = probs.shape[0]
    if num_classes <= 1:
        return 0.0
    safe_probs = np.clip(probs, 1e-12, 1.0)
    entropy = float(-np.sum(safe_probs * np.log(safe_probs)))
    max_entropy = math.log(num_classes)
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


@dataclass(slots=True)
class SlidingFeatureWindow:
    """causal 스트리밍용 고정 길이 feature 윈도우 — 가장 오래된 프레임이 앞에 온다.

    `HandFeatureExtractor`가 프레임마다 내는 벡터를 여기 채워 `GestureModel.predict`에
    바로 넘길 수 있는 (window_size, feature_dim) 배열을 유지한다. 손 추적이 끊기면
    (`push(None)`) 윈도우를 리셋한다 — 끊긴 구간 이전 손 모양이 새 시퀀스에 섞여
    들어가는 것을 막는다(HandFeatureExtractor의 리셋과 같은 이유).
    """

    window_size: int
    feature_dim: int
    _buffer: list[FloatArray] = field(default_factory=list)

    def reset(self) -> None:
        self._buffer.clear()

    def push(self, vector: FloatArray | None) -> FloatArray:
        """feature 벡터 하나를 밀어넣고 현재 윈도우 스냅샷을 반환한다.

        `vector=None`은 추적 손실 프레임을 뜻하며 윈도우를 리셋한 뒤 0벡터로 채운
        스냅샷을 반환한다 — 호출자가 이 프레임에서 예측을 신뢰하지 않도록 한다.
        """
        if vector is None:
            self.reset()
            return np.zeros((self.window_size, self.feature_dim), dtype=np.float64)
        if vector.shape != (self.feature_dim,):
            raise ValueError(f"vector must have shape ({self.feature_dim},), got {vector.shape}")
        self._buffer.append(vector)
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)
        return np.stack(self._buffer, axis=0)
