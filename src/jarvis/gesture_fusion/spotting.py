"""Gesture spotting 상태 머신 — 노이즈 있는 프레임별 예측을 하나의 깨끗한
제스처 이벤트로 만든다.

README 8장: "제스처가 여러 프레임에서 검출되더라도 ENDING과 상태 머신을 이용해
하나의 이벤트로 만든다." 모델(model.py, task 3)은 프레임마다 독립적으로 예측하므로
그 자체로 흔들릴 수 있다. 이 모듈은 그 출력을 디바운스해 IDLE→ONSET→ACTIVE→ENDING
lifecycle을 만들고, 안정된 phase 신호를 매 프레임 `jarvis.contracts.GestureEstimate`로
낸다(interface-contract.md §2, 밀집 스트림). Fusion(task 5·6)은 이 스트림에서
`ENDING`으로의 전이를 감지해 커밋 조건을 판단한다(README 9장).

**2026-07-21: lifecycle을 phase head가 아니라 gesture 활성 신호로 구동한다.**
학습된 phase head는 ONSET을 실측 0%(0/1432 프레임) 예측한다 — ONSET 라벨(클립 앞
15%)이 스트리밍 윈도우의 0-패딩 시작부와 겹쳐 IDLE과 구분이 안 되기 때문이다. 그래서
이전 phase 기반 전이는 `IDLE→ONSET`이 절대 안 일어나 모든 동적 제스처의 이벤트를
0개 냈다(raw 분류는 정상). 대신 "비배경 제스처가 확신을 갖고 지속되면 시작, 배경으로
돌아가거나 확신이 떨어지면 끝"으로 구동한다(아래 상수 주석 참조).

이 모듈은 torch나 `GestureModel` 구현에 의존하지 않는다 — `model_protocol.
ModelPrediction`(순수 값)만 입력으로 받아, 실제 모델과 별개로 상태 머신 로직을
단위 테스트할 수 있다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from jarvis.contracts.messages import GestureEstimate, GesturePhase
from jarvis.gesture_fusion.model_protocol import DEFAULT_GESTURE_LABELS, ModelPrediction

# 2026-07-21: 이벤트 판정을 **phase head가 아니라 gesture 활성 신호**로 구동한다.
# 이전에는 모델의 phase 출력(IDLE→ONSET→ACTIVE→ENDING)을 디바운스해 상태를 전이했는데,
# 학습된 phase head가 ONSET을 실측 0%(0/1432 프레임) 예측한다 — ONSET 라벨(클립 앞
# 15%)이 스트리밍 윈도우의 0-패딩 시작부와 겹쳐 IDLE과 구분이 안 되기 때문이다. 그
# 결과 `IDLE→ONSET`이 절대 안 일어나 모든 동적 제스처가 이벤트를 0개 냈다(raw 분류는
# 정상인데 스포터에서 전부 막힘). 그래서 "비배경 제스처가 확신을 갖고 지속되면 시작,
# 배경으로 돌아가거나 확신이 떨어지면 끝"이라는 gesture 기반 lifecycle로 바꾼다.
# phase head 출력은 더는 전이를 구동하지 않는다(신뢰 불가). 디바운스(min_consecutive_
# frames)는 그대로라 단일 프레임 노이즈는 여전히 억제한다.


@dataclass(frozen=True, slots=True)
class SpotterConfig:
    """Gesture spotting 디바운스·게이팅 파라미터."""

    min_consecutive_frames: int = 2
    """gesture **활성** 신호가 이 수만큼 연속으로 같아야 전이를 확정한다(단일 프레임
    노이즈 억제). 활성 = 비배경 label + 확신도 임계 이상."""

    min_release_frames: int = 1
    """gesture **비활성**(종료) 확정에 필요한 연속 프레임 수 — 진입과 비대칭이다.

    ENDING은 "제스처가 끝났다"가 아니라 "더 이상 감지되지 않는다"로 정의되므로
    (`_advance`), 이 값이 곧 "동작을 멈춘 뒤 명령이 나가기까지의 고정 지연"이다.
    12fps 추론에서 2프레임이면 약 167ms가 그대로 반응 지연에 실린다. 진입을 느리게
    두는 이유(오발 방지)는 종료에는 해당하지 않는다 — 종료를 한 프레임 일찍 인정해서
    생기는 최악은 "이미 하던 제스처가 조금 일찍 확정되는 것"이라 위험이 비대칭이다."""

    min_onset_gesture_confidence: float = 0.5
    """제스처를 "활성"으로 인정할 gesture 분류 확신도 하한. 이 값 미만이면 배경과
    같이 비활성으로 본다(시작 인정 안 함, 진행 중이면 종료 신호로 취급)."""

    background_labels: frozenset[str] = frozenset({DEFAULT_GESTURE_LABELS[0]})
    """"제스처 없음"을 뜻하는 배경 label 집합. 이 중 하나면 비활성으로 본다.

    `DEFAULT_GESTURE_LABELS[0]`("none")만 기본값으로 두지만, 학습 라벨 구성에
    따라 "타겟 제스처가 아닌 동작"을 별도 label(예: "doing_other_things")로 두는
    경우가 있다 — capability map에 매핑이 없어 런타임 결과는 "동작 없음"으로
    none과 동일한데, 이 필드가 문자열 하나뿐이면 그 label의 ONSET을 걸러내지
    못해 진짜 제스처 시작과 경합하는 채로 label lock을 잡는다(2026-07-20 발견).
    호출자가 학습 라벨 집합에 맞는 배경 label들을 명시적으로 넘겨야 한다."""

    def __post_init__(self) -> None:
        if self.min_consecutive_frames < 1:
            raise ValueError("min_consecutive_frames must be at least 1")
        if self.min_release_frames < 1:
            raise ValueError("min_release_frames must be at least 1")
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
        # 디바운스 대상은 phase가 아니라 "활성(비배경+확신)" 불리언이다.
        self._pending_active: bool | None = None
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
        self._pending_active = None
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

        confirmed_active = self._debounce(self._is_active(prediction))
        self._advance(confirmed_active, prediction)

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

    def _is_active(self, prediction: ModelPrediction) -> bool:
        """이 프레임이 "제스처 진행 중"인가 — 비배경 label이면서 확신도가 임계 이상.

        배경 label이거나 확신이 낮으면 비활성(시작 안 함, 진행 중이면 종료 쪽)."""
        return (
            prediction.gesture not in self._config.background_labels
            and prediction.gesture_confidence >= self._config.min_onset_gesture_confidence
        )

    def _debounce(self, active: bool) -> bool | None:
        """같은 신호가 연속으로 충분히 반복될 때만 확정값으로 인정한다.

        임계는 **방향에 따라 다르다**: 활성(진입)은 `min_consecutive_frames`, 비활성
        (종료)은 `min_release_frames`. 종료 임계가 곧 반응 지연이라 더 짧게 둔다
        (`SpotterConfig.min_release_frames` 참조). 아직 확정되지 않았으면 `None`을
        반환해 `_advance`가 전이를 시도하지 않게 한다(단일 프레임 깜빡임 억제).
        """
        if active == self._pending_active:
            self._pending_streak += 1
        else:
            self._pending_active = active
            self._pending_streak = 1

        required = (
            self._config.min_consecutive_frames if active else self._config.min_release_frames
        )
        if self._pending_streak >= required:
            return active
        return None

    def _advance(self, confirmed_active: bool | None, prediction: ModelPrediction) -> None:
        """확정된 활성 신호로 lifecycle을 전이한다.

        IDLE→ONSET→ACTIVE→ENDING을 gesture 활성으로 구동한다. phase head 출력은
        쓰지 않는다(ONSET을 신뢰성 있게 못 내므로 — 모듈 상단 주석 참조).
        """
        if confirmed_active is None:
            return  # 아직 디바운스 미확정 — 상태 유지

        if self._state == GesturePhase.IDLE:
            if confirmed_active:
                # 비배경 제스처가 확신을 갖고 지속됨 → 시작. 이 프레임의 label을 lock해
                # 이벤트 내내 유지한다(도중에 raw 예측이 흔들려도 바뀌지 않음).
                self._locked_gesture = prediction.gesture
                self._state = GesturePhase.ONSET
        elif self._state == GesturePhase.ONSET:
            # ONSET은 한 프레임짜리 진입 신호 — 계속 활성이면 ACTIVE로, 아니면 중단.
            if confirmed_active:
                self._state = GesturePhase.ACTIVE
            else:
                self._state = GesturePhase.IDLE
                self._locked_gesture = None
        elif self._state == GesturePhase.ACTIVE:
            # 배경으로 돌아가거나 확신이 떨어져 비활성이 확정되면 제스처 종료.
            if not confirmed_active:
                self._state = GesturePhase.ENDING  # 이벤트 방출 후 push()가 즉시 reset
