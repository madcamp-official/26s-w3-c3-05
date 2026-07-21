"""Gaze Lock state machine (README 7장 "Gaze Lock 상태 머신").

    SEARCHING → CANDIDATE → TARGET_LOCKED → GESTURE_WAIT → EXPIRED 또는 COMMITTED

`GESTURE_WAIT`/`COMMITTED`로의 전이는 Gesture & Intent Fusion의 판단에 달려 있고
그 내부 구현은 Gaze 모듈이 알지 못한다(documents/repository-structure.md 의존 방향:
세 핵심 모듈은 서로의 내부 파일을 직접 import하지 않는다). 따라서 이 상태 머신은
`notify_gesture_started`/`notify_committed`라는 좁은 이벤트 훅만 외부(Runtime의
composition root)에 제공하고, Fusion의 내부 로직은 참조하지 않는다.
"""

from __future__ import annotations

from enum import StrEnum

from jarvis.gaze.classifier import ClassificationResult
from jarvis.gaze.config import GazeConfig


class GazeLockState(StrEnum):
    SEARCHING = "SEARCHING"
    CANDIDATE = "CANDIDATE"
    TARGET_LOCKED = "TARGET_LOCKED"
    GESTURE_WAIT = "GESTURE_WAIT"
    EXPIRED = "EXPIRED"
    COMMITTED = "COMMITTED"


def _is_confident(classification: ClassificationResult, config: GazeConfig) -> bool:
    """README 7장 초기 기준의 minimum_probability·minimum_margin을 모두 만족하는지."""
    if classification.target == config.UNKNOWN_TARGET:
        return False
    margin = classification.probability - classification.second_best_probability
    return (
        classification.probability >= config.minimum_probability
        and margin >= config.minimum_margin
    )


class GazeLockStateMachine:
    """시선 후보 dwell과 Target Lock TTL을 관리하는 상태 머신."""

    def __init__(self, config: GazeConfig = GazeConfig()) -> None:
        self._config = config
        self._state = GazeLockState.SEARCHING
        self._locked_device: str | None = None
        self._candidate_device: str | None = None
        self._candidate_started_at_ms: int | None = None
        self._candidate_elapsed_ms = 0
        self._lock_expires_at_ms: int | None = None

    @property
    def state(self) -> GazeLockState:
        return self._state

    @property
    def locked_device(self) -> str | None:
        """TARGET_LOCKED 또는 GESTURE_WAIT 상태일 때만 값을 가진다."""
        if self._state in (GazeLockState.TARGET_LOCKED, GazeLockState.GESTURE_WAIT):
            return self._locked_device
        return None

    @property
    def candidate_device(self) -> str | None:
        """Target currently accumulating dwell time, before confirmation."""
        return self._candidate_device if self._state == GazeLockState.CANDIDATE else None

    @property
    def candidate_elapsed_ms(self) -> int:
        """Continuous dwell accumulated for the current candidate."""
        return self._candidate_elapsed_ms

    @property
    def dwell_progress(self) -> float:
        """Normalized 0..1 progress toward the configured gaze confirmation."""
        if self._state in (GazeLockState.TARGET_LOCKED, GazeLockState.GESTURE_WAIT):
            return 1.0
        if self._state != GazeLockState.CANDIDATE:
            return 0.0
        if self._config.dwell_time_ms == 0:
            return 1.0
        return min(1.0, self._candidate_elapsed_ms / self._config.dwell_time_ms)

    def is_locked_to(self, device_id: str) -> bool:
        """Cursor Control Mapper 게이트(README 6장) 등에서 쓰는 편의 함수."""
        return self.locked_device == device_id

    def reset(self) -> None:
        """추적 손실 등으로 처음부터 다시 탐색해야 할 때."""
        self._state = GazeLockState.SEARCHING
        self._locked_device = None
        self._candidate_device = None
        self._candidate_started_at_ms = None
        self._candidate_elapsed_ms = 0
        self._lock_expires_at_ms = None

    def update(self, timestamp_ms: int, classification: ClassificationResult) -> GazeLockState:
        """한 프레임의 분류 결과를 반영해 상태를 전이시키고 새 상태를 반환한다.

        SEARCHING→CANDIDATE 승격과 CANDIDATE→TARGET_LOCKED 승격(dwell 만족 시)은
        같은 호출 안에서 이어질 수 있다 — 그래야 `dwell_time_ms=0`처럼 즉시 잠기는
        경계값도 별도 프레임 없이 올바르게 동작한다.
        """
        confident = _is_confident(classification, self._config)

        if self._state in (GazeLockState.EXPIRED, GazeLockState.COMMITTED):
            # EXPIRED·COMMITTED는 한 프레임짜리 이벤트다 — 그다음 프레임부터
            # SEARCHING/CANDIDATE로 새로 시작한다.
            self.reset()

        if self._state == GazeLockState.SEARCHING:
            if not confident:
                return self._state
            self._state = GazeLockState.CANDIDATE
            self._candidate_device = classification.target
            self._candidate_started_at_ms = timestamp_ms
            self._candidate_elapsed_ms = 0
            # 곧바로 dwell 조건을 확인하기 위해 아래 CANDIDATE 분기로 이어진다.

        if self._state == GazeLockState.CANDIDATE:
            if not confident:
                self.reset()
                return self._state
            if classification.target != self._candidate_device:
                self._candidate_device = classification.target
                self._candidate_started_at_ms = timestamp_ms
                self._candidate_elapsed_ms = 0
                return self._state
            assert self._candidate_started_at_ms is not None
            dwell_elapsed_ms = timestamp_ms - self._candidate_started_at_ms
            self._candidate_elapsed_ms = max(0, dwell_elapsed_ms)
            if dwell_elapsed_ms < self._config.dwell_time_ms:
                return self._state
            self._state = GazeLockState.TARGET_LOCKED
            self._locked_device = self._candidate_device
            self._candidate_elapsed_ms = self._config.dwell_time_ms
            self._lock_expires_at_ms = timestamp_ms + self._config.target_lock_ttl_ms
            return self._state

        if self._state in (GazeLockState.TARGET_LOCKED, GazeLockState.GESTURE_WAIT):
            assert self._lock_expires_at_ms is not None
            if confident and classification.target == self._locked_device:
                # 계속 같은 기기를 보고 있다 — 유예 시간을 갱신해 Lock을 유지한다.
                self._lock_expires_at_ms = timestamp_ms + self._config.target_lock_ttl_ms
            elif timestamp_ms >= self._lock_expires_at_ms:
                self._state = GazeLockState.EXPIRED
                return self._state
            # 다른 곳을 보거나 추적이 불안정해도 TTL 안에서는 Lock을 유지한다
            # (README 7장: "손을 보기 위해 시선을 잠깐 이동해도 선택을 일정 시간 유지").
            return self._state

        return self._state

    def notify_gesture_started(self, timestamp_ms: int) -> GazeLockState:
        """TARGET_LOCKED 상태에서 Fusion이 gesture 시작을 감지했을 때 호출한다."""
        if self._state == GazeLockState.TARGET_LOCKED:
            assert self._lock_expires_at_ms is not None
            if timestamp_ms >= self._lock_expires_at_ms:
                self._state = GazeLockState.EXPIRED
                return self._state
            self._state = GazeLockState.GESTURE_WAIT
        return self._state

    def notify_committed(self, timestamp_ms: int) -> GazeLockState:
        """GESTURE_WAIT 상태에서 Fusion이 intent를 commit했을 때 호출한다."""
        if self._state == GazeLockState.GESTURE_WAIT:
            assert self._lock_expires_at_ms is not None
            if timestamp_ms >= self._lock_expires_at_ms:
                self._state = GazeLockState.EXPIRED
                return self._state
            self._state = GazeLockState.COMMITTED
        return self._state
