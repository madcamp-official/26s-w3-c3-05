"""Fusion confidence·safe commit — README 9장 결합 점수·Commit 조건 4·5, Intent
상태 머신의 `INTENT_CANDIDATE`→`COMMITTED`→`COOLDOWN` 전이.

Task 5(`alignment.py`)가 Commit 조건 1·2·3·6(Target Lock·시간 관계)을 판정한 뒤,
이 모듈이 나머지를 담당한다:

    S = P(target) × P(gesture) × gaze_stability × (1 − uncertainty)

- Commit 조건 4(target confidence 기준)·5(gesture confidence 기준)
- 결합 점수 threshold 판정
- 커밋 직후 연속 오발을 막는 COOLDOWN

Commit 조건 7(동일 이벤트 재실행 방지)과 `intent_id` 결정적 생성은 task 7
(`dedup.py`)이 맡고, 이 모듈이 문턱값을 모두 통과한 커밋 직전에 그 결과를
반영한다. 실제 `jarvis.contracts.Intent` 조립(capability/operation/value 매핑)은
task 8이 담당한다 — 이 모듈은 "커밋해도 되는가"만 결정하고 무엇을 커밋할지는
만들지 않는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from jarvis.contracts.messages import GestureEstimate, GesturePhase, TargetEstimate
from jarvis.gesture_fusion.alignment import (
    DEFAULT_ALIGNMENT_CONFIG,
    AlignmentConfig,
    TargetLockState,
    TemporalAligner,
)
from jarvis.gesture_fusion.dedup import IntentDeduplicator


class IntentPhase(StrEnum):
    """README 9장 Intent 상태 머신.

    `jarvis.contracts.GesturePhase`와 달리 이 값은 모듈 경계를 넘지 않는 Fusion
    내부 상태라 `jarvis.contracts`가 아니라 여기 둔다. `INTENT_CANDIDATE`·
    `COMMITTED`는 제스처 `ENDING` 처리 한 번 안에서 동기적으로 지나가는 순간
    상태라 `FusionEngine.phase`로는 관측되지 않는다(커밋되면 곧바로 `COOLDOWN`,
    거부되면 `TARGET_LOCKED`/`IDLE`로 돌아간다) — `CommitDecision`으로만 드러난다.
    """

    IDLE = "IDLE"
    TARGET_CANDIDATE = "TARGET_CANDIDATE"
    TARGET_LOCKED = "TARGET_LOCKED"
    GESTURE_TRACKING = "GESTURE_TRACKING"
    INTENT_CANDIDATE = "INTENT_CANDIDATE"
    COMMITTED = "COMMITTED"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """Safe commit 임계값. threshold 변경은 documents/decisions.md에 기록한다
    (development-principles.md 8절)."""

    commit_threshold: float = 0.5
    """결합 점수 S가 이 값 이상이어야 커밋한다. MVP 초기값 — 실 데이터로 재보정 대상."""

    min_target_confidence: float = 0.80
    """Commit 조건 4: 이 값 미만이면 target confidence 부족으로 거부한다."""

    min_gesture_confidence: float = 0.80
    """Commit 조건 5: 이 값 미만이면 gesture confidence 부족으로 거부한다."""

    cooldown_ms: int = 500
    """커밋 직후 이 시간 동안 추가 커밋을 막는다(연속 오발 방지).

    Task 7의 intent 단위 중복 방지(동일 이벤트 재실행 금지, Commit 조건 7)와는
    다른 관심사다 — 여기서는 "너무 가까운 시간 안에 또 커밋하지 말라"는 시간 기반
    최소 간격만 보장한다.
    """

    dedup_history_size: int = 256
    """Commit 조건 7 판정에 기억해 둘 최근 커밋 frame_id 개수(`IntentDeduplicator`)."""

    def __post_init__(self) -> None:
        if not math.isfinite(self.commit_threshold) or not 0.0 <= self.commit_threshold <= 1.0:
            raise ValueError("commit_threshold must be finite and within [0, 1]")
        for name, value in (
            ("min_target_confidence", self.min_target_confidence),
            ("min_gesture_confidence", self.min_gesture_confidence),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")
        if self.cooldown_ms < 0:
            raise ValueError("cooldown_ms must be non-negative")
        if self.dedup_history_size < 1:
            raise ValueError("dedup_history_size must be at least 1")


DEFAULT_FUSION_CONFIG = FusionConfig()


@dataclass(frozen=True, slots=True)
class FusionScore:
    """결합 점수의 구성요소와 최종값. 각 항을 그대로 남겨 trace·디버깅에 쓴다."""

    target_confidence: float
    gesture_confidence: float
    gaze_stability: float
    uncertainty: float
    value: float

    def __post_init__(self) -> None:
        for name, val in (
            ("target_confidence", self.target_confidence),
            ("gesture_confidence", self.gesture_confidence),
            ("gaze_stability", self.gaze_stability),
            ("uncertainty", self.uncertainty),
            ("value", self.value),
        ):
            if not math.isfinite(val) or not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1], got {val}")


def compute_fusion_score(
    target_confidence: float,
    gesture_confidence: float,
    gaze_stability: float,
    uncertainty: float,
) -> FusionScore:
    """S = P(target) × P(gesture) × gaze_stability × (1 − uncertainty) (README 9장).

    네 입력 모두 [0, 1] 범위가 아니면 `FusionScore.__post_init__`이 거부한다 —
    모델·정렬 계층의 출력을 그대로 믿지 않는다(development-principles.md 7.2).
    """
    value = target_confidence * gesture_confidence * gaze_stability * (1.0 - uncertainty)
    return FusionScore(
        target_confidence=target_confidence,
        gesture_confidence=gesture_confidence,
        gaze_stability=gaze_stability,
        uncertainty=uncertainty,
        value=value,
    )


@dataclass(frozen=True, slots=True)
class CommitDecision:
    """제스처가 완결된 순간의 최종 커밋 판정 — task 7·8이 이 값을 받아 Intent를 만든다."""

    committed: bool
    reason: str
    """거부 이유(trace용) 또는 성공 시 "committed"."""
    target: str | None
    gesture: str | None
    score: FusionScore | None
    timestamp_ms: int
    frame_id: int
    intent_id: str | None = None
    """`committed=True`일 때만 채워진다. task 8이 `jarvis.contracts.Intent.intent_id`로
    그대로 쓴다 — Commit 조건 7을 통과한, 이 이벤트만의 결정적 식별자(dedup.py)."""


_GESTURE_IN_PROGRESS_PHASES = frozenset({GesturePhase.ONSET, GesturePhase.ACTIVE, GesturePhase.ENDING})


class FusionEngine:
    """Gaze·Gesture 스트림을 받아 안전한 commit 여부를 판정한다.

    내부적으로 `TemporalAligner`(task 5)를 감싸 Commit 조건 1·2·3·6을 위임하고,
    조건 4·5·결합 점수·threshold·COOLDOWN을 이 레이어에서 더한다.
    """

    def __init__(
        self,
        config: FusionConfig = DEFAULT_FUSION_CONFIG,
        alignment_config: AlignmentConfig = DEFAULT_ALIGNMENT_CONFIG,
    ) -> None:
        self._config = config
        self._aligner = TemporalAligner(alignment_config)
        self._dedup = IntentDeduplicator(config.dedup_history_size)
        self._last_gesture_phase = GesturePhase.IDLE
        self._cooldown_until_ms: int | None = None

    @property
    def lock_state(self) -> TargetLockState:
        return self._aligner.lock_state

    @property
    def phase(self) -> IntentPhase:
        """현재 안정 상태(README 9장 Intent 상태 머신).

        `INTENT_CANDIDATE`·`COMMITTED`는 여기 나타나지 않는다 — 위 클래스 docstring 참고.
        """
        if self._cooldown_until_ms is not None:
            return IntentPhase.COOLDOWN
        lock = self._aligner.lock_state
        if lock.locked and self._last_gesture_phase in _GESTURE_IN_PROGRESS_PHASES:
            return IntentPhase.GESTURE_TRACKING
        if lock.locked:
            return IntentPhase.TARGET_LOCKED
        if lock.candidate is not None:
            return IntentPhase.TARGET_CANDIDATE
        return IntentPhase.IDLE

    def push_target(self, estimate: TargetEstimate) -> None:
        """Gaze→Fusion 스트림(§1) 프레임 하나를 반영한다."""
        self._expire_cooldown(estimate.timestamp_ms)
        self._aligner.push_target(estimate)

    def push_gesture(self, estimate: GestureEstimate) -> CommitDecision | None:
        """Gesture→Fusion 스트림(§2) 프레임 하나를 반영한다.

        제스처가 이번 프레임에 `ENDING`으로 완결되지 않았으면 `None`을 반환한다.
        완결됐으면(정렬 실패 포함) 항상 `CommitDecision`을 반환한다 — "판단하지
        않음"과 "거부함"을 구분해 trace에 남기기 위함이다.
        """
        self._expire_cooldown(estimate.timestamp_ms)
        self._last_gesture_phase = estimate.phase

        alignment = self._aligner.push_gesture(estimate)
        if alignment is None:
            return None

        if self._cooldown_until_ms is not None:
            return self._reject(estimate, "cooldown active", alignment.target)
        if not alignment.aligned:
            return self._reject(estimate, alignment.reason, alignment.target)
        if alignment.target_confidence < self._config.min_target_confidence:
            return self._reject(estimate, "target confidence below minimum", alignment.target)
        if estimate.gesture_confidence < self._config.min_gesture_confidence:
            return self._reject(estimate, "gesture confidence below minimum", alignment.target)

        score = compute_fusion_score(
            target_confidence=alignment.target_confidence,
            gesture_confidence=estimate.gesture_confidence,
            gaze_stability=alignment.gaze_stability,
            uncertainty=estimate.uncertainty,
        )
        if score.value < self._config.commit_threshold:
            return CommitDecision(
                committed=False,
                reason="fusion score below commit threshold",
                target=alignment.target,
                gesture=estimate.gesture,
                score=score,
                timestamp_ms=estimate.timestamp_ms,
                frame_id=estimate.frame_id,
            )

        # Commit 조건 7: 이 프레임에서 이미 커밋한 적이 있으면(재전송·재생) 두 번째
        # Intent를 만들지 않는다. 여기까지 온 이벤트는 이미 조건 1~6을 통과했으므로
        # cooldown은 새로 걸지 않는다 — 새로운 일이 일어난 게 아니라 같은 일의 재생이다.
        intent_id = self._dedup.register(estimate.frame_id)
        if intent_id is None:
            return CommitDecision(
                committed=False,
                reason="duplicate event (frame already committed)",
                target=alignment.target,
                gesture=estimate.gesture,
                score=score,
                timestamp_ms=estimate.timestamp_ms,
                frame_id=estimate.frame_id,
            )

        self._cooldown_until_ms = estimate.timestamp_ms + self._config.cooldown_ms
        return CommitDecision(
            committed=True,
            reason="committed",
            target=alignment.target,
            gesture=estimate.gesture,
            score=score,
            timestamp_ms=estimate.timestamp_ms,
            frame_id=estimate.frame_id,
            intent_id=intent_id,
        )

    def _expire_cooldown(self, timestamp_ms: int) -> None:
        if self._cooldown_until_ms is not None and timestamp_ms >= self._cooldown_until_ms:
            self._cooldown_until_ms = None

    def _reject(self, estimate: GestureEstimate, reason: str, target: str | None) -> CommitDecision:
        return CommitDecision(
            committed=False,
            reason=reason,
            target=target,
            gesture=estimate.gesture,
            score=None,
            timestamp_ms=estimate.timestamp_ms,
            frame_id=estimate.frame_id,
        )
