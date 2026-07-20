"""Gesture spotting 상태 머신 — 노이즈 있는 프레임별 phase 예측을 하나의 깨끗한
제스처 이벤트로 만든다.

README 8장: "제스처가 여러 프레임에서 검출되더라도 ENDING과 상태 머신을 이용해
하나의 이벤트로 만든다." 모델(model.py, task 3)은 프레임마다 독립적으로 phase를
추측하므로 그 자체로 흔들릴 수 있다(ONSET↔IDLE 반복, 단계 건너뛰기 등). 이 모듈은
raw 모델 출력을 디바운스해 진짜 전이만 통과시키고, 안정된 phase 신호를 매 프레임
`jarvis.contracts.GestureEstimate`로 낸다(interface-contract.md §2, 밀집 스트림).
Fusion(task 5·6)은 이 스트림에서 `ENDING`으로의 전이를 감지해 커밋 조건을
판단한다(README 9장).

이 모듈은 torch나 `GestureModel` 구현에 의존하지 않는다 — `model_protocol.
ModelPrediction`(순수 값)만 입력으로 받아, 실제 모델과 별개로 상태 머신 로직을
단위 테스트할 수 있다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from jarvis.contracts.messages import GestureEstimate, GesturePhase
from jarvis.gesture_fusion.model_protocol import DEFAULT_GESTURE_LABELS, ModelPrediction

# 각 상태에서 디바운스로 확정됐을 때 실제로 받아들이는 다음 상태 집합. 여기 없는
# 전이(예: IDLE→ACTIVE처럼 단계를 건너뜀)는 모델 노이즈로 보고 무시한다 — 모델
# 출력을 그대로 믿지 않는다(development-principles.md 7.2).
_ALLOWED_TRANSITIONS: dict[GesturePhase, frozenset[GesturePhase]] = {
    GesturePhase.IDLE: frozenset({GesturePhase.ONSET}),
    GesturePhase.ONSET: frozenset({GesturePhase.ACTIVE, GesturePhase.IDLE}),
    GesturePhase.ACTIVE: frozenset({GesturePhase.ENDING, GesturePhase.IDLE}),
    GesturePhase.ENDING: frozenset({GesturePhase.IDLE}),
}


@dataclass(frozen=True, slots=True)
class SpotterConfig:
    """Gesture spotting 디바운스·게이팅 파라미터."""

    min_consecutive_frames: int = 2
    """이 수만큼 raw phase가 연속으로 같아야 상태 전이를 확정한다(단일 프레임 노이즈 억제)."""

    min_onset_gesture_confidence: float = 0.5
    """ONSET을 확정할 때 gesture 분류 확신도가 이 값 미만이면 제스처 시작으로 인정하지 않는다."""

    background_labels: frozenset[str] = frozenset({DEFAULT_GESTURE_LABELS[0]})
    """"제스처 없음"을 뜻하는 배경 label 집합. ONSET 확정 시 이 중 하나면 거부한다.

    `DEFAULT_GESTURE_LABELS[0]`("none")만 기본값으로 두지만, 학습 라벨 구성에
    따라 "타겟 제스처가 아닌 동작"을 별도 label(예: "doing_other_things")로 두는
    경우가 있다 — capability map에 매핑이 없어 런타임 결과는 "동작 없음"으로
    none과 동일한데, 이 필드가 문자열 하나뿐이면 그 label의 ONSET을 걸러내지
    못해 진짜 제스처 시작과 경합하는 채로 label lock을 잡는다(2026-07-20 발견).
    호출자가 학습 라벨 집합에 맞는 배경 label들을 명시적으로 넘겨야 한다."""

    def __post_init__(self) -> None:
        if self.min_consecutive_frames < 1:
            raise ValueError("min_consecutive_frames must be at least 1")
        if not math.isfinite(self.min_onset_gesture_confidence) or not (
            0.0 <= self.min_onset_gesture_confidence <= 1.0
        ):
            raise ValueError("min_onset_gesture_confidence must be finite and within [0, 1]")
        if not self.background_labels:
            raise ValueError("background_labels must not be empty")
        if any(not label for label in self.background_labels):
            raise ValueError("background_labels must not contain empty strings")


DEFAULT_SPOTTER_CONFIG = SpotterConfig()


class GestureSpotter:
    """프레임별 `ModelPrediction`을 밀어넣고 디바운스된 `GestureEstimate`를 얻는다.

    한 제스처당 `phase=ENDING`은 정확히 한 프레임만 나온다 — 그 프레임을 낸 즉시
    내부 상태를 IDLE로 리셋해, 같은 제스처가 여러 이벤트로 중복 집계되지 않는다
    (development-principles.md 2.3: 한 gesture event는 최대 하나의 intent를 만든다).
    """

    def __init__(self, config: SpotterConfig = DEFAULT_SPOTTER_CONFIG) -> None:
        self._config = config
        self._state = GesturePhase.IDLE
        self._pending_phase: GesturePhase | None = None
        self._pending_streak = 0
        self._locked_gesture: str | None = None

    @property
    def state(self) -> GesturePhase:
        """디바운스로 확정된 현재 spotter 상태(raw 모델 출력이 아님)."""
        return self._state

    @property
    def is_tracking_gesture(self) -> bool:
        """IDLE이 아니면 True.

        `pointer/` 모듈이 커서 스트림을 일시정지할지 판단하는 신호다(2026-07-18
        결정: 노트북 Lock 중 ONSET 감지 시 커서 일시정지, IDLE 복귀 시 커서 모드 복귀).
        """
        return self._state != GesturePhase.IDLE

    def reset(self) -> None:
        """상태를 IDLE로 되돌리고 디바운스·lock 상태를 모두 비운다."""
        self._state = GesturePhase.IDLE
        self._pending_phase = None
        self._pending_streak = 0
        self._locked_gesture = None

    def push(
        self, prediction: ModelPrediction | None, timestamp_ms: int, frame_id: int
    ) -> GestureEstimate:
        """모델 예측 하나를 처리해 이 프레임의 `GestureEstimate`를 반환한다.

        `prediction=None`은 손 추적 손실을 뜻한다 — 진행 중이던 제스처를 안전하게
        포기하고 즉시 IDLE로 되돌린다(development-principles.md 2.2: 추적 손실은 거부).
        """
        if prediction is None:
            self.reset()
            return GestureEstimate(
                timestamp_ms=timestamp_ms,
                frame_id=frame_id,
                # 손 추적 손실 시 보고할 자리표시자 값 — 어떤 배경 label이 예측됐는지와
                # 무관한 합성 상태이므로, background_labels 중 임의의 하나가 아니라
                # 계약상 정본인 DEFAULT_GESTURE_LABELS[0]("none")을 쓴다.
                gesture=DEFAULT_GESTURE_LABELS[0],
                gesture_confidence=0.0,
                phase=GesturePhase.IDLE,
                phase_confidence=0.0,
                uncertainty=1.0,
            )

        confirmed_phase = self._debounce(prediction.phase)
        self._advance(confirmed_phase, prediction)

        estimate = GestureEstimate(
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            gesture=self._locked_gesture or prediction.gesture,
            gesture_confidence=prediction.gesture_confidence,
            phase=self._state,
            phase_confidence=prediction.phase_confidence,
            uncertainty=prediction.uncertainty,
        )

        if self._state == GesturePhase.ENDING:
            self.reset()

        return estimate

    def _debounce(self, raw_phase: GesturePhase) -> GesturePhase:
        """raw phase가 `min_consecutive_frames` 연속으로 같을 때만 확정값으로 인정한다.

        아직 확정되지 않았으면(스트릭이 임계값 미만) 현재 안정 상태를 그대로 반환해
        `_advance`가 전이를 시도하지 않게 한다.
        """
        if raw_phase == self._pending_phase:
            self._pending_streak += 1
        else:
            self._pending_phase = raw_phase
            self._pending_streak = 1

        if self._pending_streak >= self._config.min_consecutive_frames:
            return raw_phase
        return self._state

    def _advance(self, confirmed_phase: GesturePhase, prediction: ModelPrediction) -> None:
        if confirmed_phase == self._state:
            return
        if confirmed_phase not in _ALLOWED_TRANSITIONS[self._state]:
            return

        if confirmed_phase == GesturePhase.ONSET:
            if (
                prediction.gesture in self._config.background_labels
                or prediction.gesture_confidence < self._config.min_onset_gesture_confidence
            ):
                # 배경 클래스거나 확신이 낮으면 제스처 시작으로 인정하지 않는다.
                # 다음 프레임에 더 나은 예측이 오면 다시 시도할 수 있도록 상태는
                # 그대로 두고 이번 전이만 건너뛴다.
                return
            self._locked_gesture = prediction.gesture

        self._state = confirmed_phase
        if confirmed_phase == GesturePhase.IDLE:
            # ONSET/ACTIVE에서 중단(abort)되어 IDLE로 돌아간 경우에도 lock을 비워야
            # 다음 제스처 시도가 이전 label을 이어받지 않는다.
            self._locked_gesture = None
