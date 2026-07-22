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

# 인식 대상 제스처 8종 + 배경 "none"(2026-07-20 결정: 사용자가 지정한 목록만
# 학습 — swipe 포함 나머지 18개 Jester 클래스는 이번엔 제외한다. README 8장
# "지원 제스처"의 swipe 4종·주석과 다르다는 점을 인지하고 있음, README는 이후
# 갱신 예정). "No gesture"만 배경 "none"에 대응하고, 나머지는 각자 고유 label을
# 갖는다. **"none"은 반드시 index 0을 유지한다**(spotting.py가 배경 label로
# `DEFAULT_GESTURE_LABELS[0]`을 참조). 열린 문자열 키(interface-contract.md)라 새
# 제스처는 이 튜플만 확장하면 된다.
DEFAULT_GESTURE_LABELS: tuple[str, ...] = (
    "none",
    # 손목 회전 (2)
    "rotate_clockwise",
    "rotate_counter_clockwise",
    # 두 손가락 슬라이드 (4)
    "slide_two_fingers_up",
    "slide_two_fingers_down",
    "slide_two_fingers_left",
    "slide_two_fingers_right",
    # 정적 손 모양 (1)
    "drumming_fingers",
    # 정의된 제스처 어디에도 안 속하는 동작(Jester "Doing other things") (1)
    "doing_other_things",
    # 정지 명령 — 활짝 편 손바닥 정적 포즈(Jester "Stop Sign") (1). 배경이 아니라
    # 액션을 유발하는 전경 제스처다(2026-07-20 추가). drumming_fingers와 달리
    # 손가락이 거의 완전히 펴진 고유 포즈라 판별이 쉽다(평균 관절각 3.04rad,
    # 검출율 99.1%).
    "stop_sign",
)

# 액션을 유발하지 않는 "배경" 클래스들 (2026-07-20 결정).
#
# **학습은 이 셋을 각각 다른 클래스로 한다.** 가만히 있는 손·손가락 두드리기·아무
# 동작은 물리적으로 전혀 다른 신호라, 하나로 묶어 "배경이란 무엇인가"를 뭉뚱그린
# 분포로 배우게 하는 것보다 각각을 명시적으로 배우게 하는 편이 특징을 더 깨끗하게
# 만든다(보조 과제 효과). 합치는 것은 **결정 단계에서만** 한다 —
# `collapse_background_probabilities` 참고.
#
# 이 구조의 실질적 이점: 나중에 어떤 배경 동작을 진짜 제스처로 승격하려면 이
# 집합에서 빼기만 하면 된다. 라벨 집합·가중치 shape이 그대로라 **재학습이 필요
# 없다**. 라벨 자체를 합쳐버리면 되살릴 때 처음부터 다시 학습해야 한다.
DEFAULT_BACKGROUND_LABELS: frozenset[str] = frozenset(
    {
        "none",
        "drumming_fingers",
        "doing_other_things",
    }
)

# 모델이 가정하는 입력 프레임레이트(fps). Jester 사전학습이 12fps 프레임 시퀀스라
# TCN의 고정 receptive field(프레임 수)가 그 cadence에서 특정 실시간 길이를 덮도록
# 학습됐다. 따라서 (1) 웹캠 파인튜닝 클립은 저장 전 이 fps로 리샘플하고
# (`training.augment.resample_clip_to_fps`), (2) 실시간 인식도 이 fps로 프레임을
# 솎아 feed해야(`FrameRateLimiter`) velocity·receptive field가 정합한다. 학습·추론
# cadence가 어긋나면 두 선택지 중 최악이 되므로 한 곳에서 정의해 공유한다.
# 값은 Jester 추출 fps(`training/extract/extract_jester.py`의 _FPS)와 같아야 한다.
EXPECTED_INPUT_FPS: float = 12.0

# 실시간 추론이 모델에 넣는 윈도우 길이(프레임). 아키텍처의 receptive field(29프레임 =
# 2.42초)보다 **짧게** 둔다 — 모델의 "기억"을 실질적으로 줄이는 값이다.
#
# 왜 짧게 두는가(2026-07-22 실측): 웹캠 클립의 전경 제스처는 중앙값 11프레임(0.92초),
# p90 15, 최대 21(rotate_clockwise)이다. 29프레임 윈도우를 꽉 채우면 중앙값 제스처는
# 그중 38%만 차지하고 **나머지 62%는 직전 동작이나 배경**이다. 그래서 동작이 끝난 뒤에도
# 모델이 한동안 이전 제스처를 계속 예측한다.
#
# 짧게 넣어도 안전한 이유: 학습은 클립을 자연 길이 그대로 먹이므로(대부분 21프레임 이하)
# 학습 시 결정 프레임도 `[내부 0패딩] + [짧은 클립]`을 본다. 즉 29프레임을 실제 데이터로
# 꽉 채운 상태가 오히려 학습 분포 밖이고, 윈도우를 줄이면 학습과 **더** 가까워진다.
# `CausalTCNGestureModel._pad_to_window`가 짧은 입력의 앞을 0으로 채워 주므로 가중치·
# shape은 전혀 바뀌지 않는다 — 재학습이 필요 없는 순수 런타임 노브다.
#
# 15를 고른 근거는 추측이 아니라 실측이다(2026-07-22, 학습된 체크포인트로 캐시 클립을
# 실제 파이프라인에 재생). 두 가지를 같은 잣대로 쟀다:
#
#   (A) 제스처 A를 한 직후 제스처 B를 할 때, B 구간이 A로 오인되는 프레임 비율
#       w=29: 19.2%   w=21: 18.7%   w=18: 15.3%   w=15: 10.4%
#   (B) 그 반대급부 — 단일 클립 후반부가 정답으로 예측되는 비율(인식 정확도 대용)
#       w=29: 36.6%   w=21: 36.6%   w=18: 36.6%   w=15: 36.5%
#
# 즉 윈도우를 줄이면 잔류는 절반으로 떨어지는데 인식 손해는 측정되지 않는다. 단일
# 제스처는 대부분 15프레임 이하라(중앙 11) 윈도우를 다 채우지 못해, 짧게 잡아도 잃을
# 것이 없기 때문이다. 가장 긴 rotate_clockwise(최대 21프레임)조차 87.9% -> 87.1%로
# 사실상 변화가 없었다.
#
# 주의(측정의 한계): (A)는 클립을 **틈 없이** 이어 붙여 잰 값이라 실사용보다 비관적이다
# — 실제로는 동작 사이 멈춤이 윈도우를 비워 준다. (B)의 절대값이 낮은 것은 클립 후반의
# 동작 종료 구간까지 세기 때문이며, 윈도우 간 상대 비교로만 의미가 있다.
#
# 남는 문제: 잔류의 대부분은 회전 한 종류다(w=15에서도 rotate_clockwise 42%,
# rotate_counter_clockwise 12.8%, 나머지 5종은 모두 7% 미만). 회전은 순환 동작이라
# 윈도우 길이로는 끝까지 해결되지 않는다 — 별도 대응이 필요하다.
INFERENCE_WINDOW_FRAMES: int = 15


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

    background_labels: frozenset[str] = DEFAULT_BACKGROUND_LABELS
    """액션을 유발하지 않는 배경 클래스들. `gesture_labels`에 없는 이름은 무시한다.

    가중치 shape에는 영향이 없지만 **같은 가중치로도 다른 예측을 내게 하므로**
    (배경 확률 합산 여부가 argmax 결과를 바꾼다) 체크포인트와 함께 버전 관리해야
    하는 값이다 — `gesture_labels`와 같은 이유로 여기 둔다.
    """

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
        # 배경 이름 검사는 `gesture_labels` 하나가 아니라 **표준 라벨 집합과의 합집합**을
        # 기준으로 한다. 축소된 label 튜플로 만드는 설정(테스트·실험)에서 기본 배경
        # 집합을 그대로 쓸 수 있어야 하고(없는 이름은 무시), 동시에 표준 라벨을 개명하고
        # 이 집합을 안 고치면 그 동작이 조용히 전경 제스처로 바뀌는 것은 잡아야 한다 —
        # 우리가 지금 고치고 있는 버그가 정확히 그 종류다.
        known_labels = set(DEFAULT_GESTURE_LABELS) | set(self.gesture_labels)
        unknown_background = self.background_labels - known_labels
        if unknown_background:
            raise ValueError(
                f"background_labels contains unknown label(s): {sorted(unknown_background)} — "
                f"라벨을 개명했다면 DEFAULT_BACKGROUND_LABELS도 함께 고쳐야 한다"
            )
        if self.gesture_labels[0] not in self.background_labels:
            raise ValueError(
                f"gesture_labels[0]={self.gesture_labels[0]!r} must be a background label — "
                "배경 확률을 합산한 뒤 대표 label로 index 0을 쓴다(spotting.py도 같은 규약)"
            )
        if not set(self.gesture_labels) - self.background_labels:
            raise ValueError("at least one non-background (actionable) gesture label is required")
        if not self.channels or any(c <= 0 for c in self.channels):
            raise ValueError("channels must be a non-empty tuple of positive ints")
        if self.kernel_size < 2:
            raise ValueError("kernel_size must be at least 2 for causal padding to be meaningful")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be within [0, 1)")

    @property
    def background_indices(self) -> tuple[int, ...]:
        """배경 클래스의 출력 index들(오름차순). 첫 원소가 대표 배경(라벨 index 0)이다."""
        return tuple(
            i for i, label in enumerate(self.gesture_labels) if label in self.background_labels
        )

    @property
    def foreground_indices(self) -> tuple[int, ...]:
        """액션을 유발할 수 있는(배경이 아닌) 클래스의 출력 index들(오름차순)."""
        return tuple(
            i for i, label in enumerate(self.gesture_labels) if label not in self.background_labels
        )

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


# 배경 vs 최선 제스처 동점 판정 허용 오차. `background_prob`(합산)과
# `best_prob`(단일 원소)은 서로 다른 부동소수점 연산 경로를 거치므로, 수학적으로
# 정확히 같아야 하는 값도 float32 softmax에서는 미세하게(~1e-7) 어긋난다 —
# 실측: 배경 3클래스 각 1/6일 때 배경합 0.5 vs 전경 0.5000000596046448
# (diff≈6e-8, 2026-07-20 발견). 엄격한 `>=` 비교는 이런 경우 "동점이면 배경"
# 규칙이 부동소수점 잡음 때문에 사실상 발동하지 않는다 — 이 규칙의 취지
# 자체가 "razor-thin 차이로는 액션을 내지 않는다"이므로, 이 정도 잡음은 동점으로
# 봐야 규칙이 의도대로 작동한다.
BACKGROUND_TIE_TOLERANCE = 1e-6


def collapse_background_probabilities(
    probs: FloatArray,
    background_indices: tuple[int, ...],
    foreground_indices: tuple[int, ...],
) -> tuple[int, float, FloatArray]:
    """배경 클래스 확률을 하나로 합산한 뒤 "배경 vs 최선의 제스처"를 결정한다.

    반환값은 `(선택된 원본 클래스 index, 그 확신도, 합산 후 분포)`. 배경이 이기면
    index는 대표 배경(`background_indices[0]`)이다.

    **argmax를 전체 클래스에 그대로 쓰지 않는 이유.** 배경을 여러 클래스로 나눠
    학습하면(`DEFAULT_BACKGROUND_LABELS` 주석 참고) 진짜 배경 구간에서 확률이 그
    클래스들로 쪼개진다. 그러면 질량 합계로는 배경이 압도적인데도 제스처 하나가 더
    낮은 절대 확률로 argmax를 이기는 구간이 생긴다(예: 배경 0.25/0.25/0.20 = 0.70 대
    제스처 0.30). 비교 전에 합산하면 이 표 분산이 사라진다.

    합산 후 분포는 `[배경, 제스처1, …]` 순서이며, 불확실성은 **이 분포에서** 재야
    한다 — 원본 분포에서 재면 "어느 배경인지 모호함"이 불확실성으로 새어 들어간다.

    동점이면 배경이 이긴다 — 액션은 확신이 있을 때만 일으킨다(2.2 알 수 없으면 거부).
    """
    if not background_indices or not foreground_indices:
        raise ValueError("both background_indices and foreground_indices must be non-empty")

    background_prob = float(np.sum(probs[list(background_indices)]))
    foreground_probs = probs[list(foreground_indices)]
    collapsed = np.concatenate(([background_prob], foreground_probs)).astype(np.float64)

    best = int(np.argmax(foreground_probs))
    best_prob = float(foreground_probs[best])
    if background_prob >= best_prob - BACKGROUND_TIE_TOLERANCE:
        return background_indices[0], background_prob, collapsed
    return foreground_indices[best], best_prob, collapsed


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


@dataclass(slots=True)
class FrameRateLimiter:
    """스트림을 목표 fps로 솎는 causal 게이트 — 실시간 인식을 학습 cadence에 맞춘다.

    파인튜닝 클립이 `EXPECTED_INPUT_FPS`로 정규화되면(웹캠 30fps → 리샘플) 추론도 같은
    fps로 feed해야 velocity·receptive field가 정합한다. 매 프레임을 파이프라인에 넣는
    대신, 직전 채택 프레임과 target 간격(`1000/target_fps` ms) 이상 벌어졌을 때만
    채택한다. 미래 프레임을 못 보므로 보간이 아니라 **솎기**(causal)이며, velocity는
    실제 dt로 계산되므로 격자 지터(예: 30fps에서 66~99ms)에 강건하다.

    `SlidingFeatureWindow`와 같은 feed 계층 유틸이라 여기(torch-free 경계)에 둔다 —
    모니터·런타임 인식 경로가 공유하고, mediapipe·torch 없이 단위 테스트할 수 있다.
    """

    target_fps: float
    _last_accepted_ms: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.target_fps) or self.target_fps <= 0.0:
            raise ValueError("target_fps must be finite and positive")

    def reset(self) -> None:
        """다음 프레임을 무조건 채택하도록 상태를 비운다(추적 세션 경계 등에서)."""
        self._last_accepted_ms = None

    def should_accept(self, timestamp_ms: int) -> bool:
        """이 프레임을 인식 파이프라인에 넣을지 판정한다. **채택(True) 시 상태를 갱신한다.**

        첫 프레임은 항상 채택한다. 이후에는 직전 채택 이후 target 간격 이상 지났을
        때만 채택해 유효 프레임레이트를 `target_fps` 이하로 유지한다.
        """
        interval_ms = 1000.0 / self.target_fps
        if (
            self._last_accepted_ms is None
            or timestamp_ms - self._last_accepted_ms >= interval_ms
        ):
            self._last_accepted_ms = timestamp_ms
            return True
        return False
