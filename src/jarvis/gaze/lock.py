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
        self._unknown_started_at_ms: int | None = None
        self._unknown_elapsed_ms = 0
        self._candidate_gap_started_at_ms: int | None = None

    @property
    def state(self) -> GazeLockState:
        return self._state

    @property
    def locked_device(self) -> str | None:
        """Last confirmed target, retained through handoff and gesture events."""
        if self._state in (
            GazeLockState.TARGET_LOCKED,
            GazeLockState.GESTURE_WAIT,
            GazeLockState.EXPIRED,
            GazeLockState.COMMITTED,
        ):
            return self._locked_device
        return None

    @property
    def candidate_device(self) -> str | None:
        """Target currently accumulating dwell time, before confirmation."""
        if self._state in (GazeLockState.CANDIDATE, GazeLockState.TARGET_LOCKED):
            return self._candidate_device
        return None

    @property
    def candidate_elapsed_ms(self) -> int:
        """Continuous dwell accumulated for the current candidate."""
        return self._candidate_elapsed_ms

    @property
    def dwell_progress(self) -> float:
        """Normalized 0..1 progress toward the configured gaze confirmation."""
        if self.candidate_device is None:
            return 1.0 if self.locked_device is not None else 0.0
        if self._config.dwell_time_ms == 0:
            return 1.0
        return min(1.0, self._candidate_elapsed_ms / self._config.dwell_time_ms)

    @property
    def unknown_elapsed_ms(self) -> int:
        """Continuous UNKNOWN time while retaining a previously confirmed target."""
        return self._unknown_elapsed_ms

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
        self._candidate_gap_started_at_ms = None
        self._clear_unknown_timer()

    def update(self, timestamp_ms: int, classification: ClassificationResult) -> GazeLockState:
        """한 프레임의 분류 결과를 반영해 상태를 전이시키고 새 상태를 반환한다.

        SEARCHING→CANDIDATE 승격과 CANDIDATE→TARGET_LOCKED 승격(dwell 만족 시)은
        같은 호출 안에서 이어질 수 있다 — 그래야 `dwell_time_ms=0`처럼 즉시 잠기는
        경계값도 별도 프레임 없이 올바르게 동작한다.
        """
        confident = _is_confident(classification, self._config)

        if self._state in (GazeLockState.EXPIRED, GazeLockState.COMMITTED):
            # Gesture lifecycle events are one-frame events, but they do not
            # discard the user's confirmed selection.
            if self._locked_device is not None:
                self._state = GazeLockState.TARGET_LOCKED
                self._lock_expires_at_ms = timestamp_ms + self._config.target_lock_ttl_ms
            else:
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
                # 깜빡임·회복 프레임의 순간 UNKNOWN 한 번으로 3초 dwell을 0으로
                # 되돌리면 자연 깜빡임 주기(2~5초)보다 dwell이 길어 영원히 확정되지
                # 않는다(2026-07-22 실사용). blink hold(300ms)가 못 덮는 꼬리를
                # candidate_grace_ms까지 유예하고, 그 이상 지속될 때만 리셋한다.
                if self._candidate_gap_started_at_ms is None:
                    self._candidate_gap_started_at_ms = timestamp_ms
                if (
                    timestamp_ms - self._candidate_gap_started_at_ms
                    > self._config.candidate_grace_ms
                ):
                    self.reset()
                return self._state
            self._candidate_gap_started_at_ms = None
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
            self._clear_candidate()
            self._clear_unknown_timer()
            self._lock_expires_at_ms = timestamp_ms + self._config.target_lock_ttl_ms
            return self._state

        if self._state == GazeLockState.TARGET_LOCKED:
            assert self._lock_expires_at_ms is not None
            if classification.target == self._config.UNKNOWN_TARGET:
                self._clear_candidate()
                if self._unknown_started_at_ms is None:
                    self._unknown_started_at_ms = timestamp_ms
                self._unknown_elapsed_ms = max(
                    0, timestamp_ms - self._unknown_started_at_ms
                )
                if self._unknown_elapsed_ms >= self._config.confirmed_unknown_timeout_ms:
                    self.reset()
                    return self._state
                self._lock_expires_at_ms = timestamp_ms + self._config.target_lock_ttl_ms
                return self._state
            self._clear_unknown_timer()
            if confident and classification.target == self._locked_device:
                self._clear_candidate()
                self._lock_expires_at_ms = timestamp_ms + self._config.target_lock_ttl_ms
                return self._state
            if not confident:
                # Cancel only the replacement attempt; retain the last confirmed target.
                self._clear_candidate()
                self._lock_expires_at_ms = timestamp_ms + self._config.target_lock_ttl_ms
                return self._state
            if classification.target != self._candidate_device:
                self._start_candidate(classification.target, timestamp_ms)
            assert self._candidate_started_at_ms is not None
            self._candidate_elapsed_ms = max(0, timestamp_ms - self._candidate_started_at_ms)
            self._lock_expires_at_ms = timestamp_ms + self._config.target_lock_ttl_ms
            if self._candidate_elapsed_ms >= self._config.dwell_time_ms:
                self._locked_device = classification.target
                self._clear_candidate()
            return self._state

        if self._state == GazeLockState.GESTURE_WAIT:
            assert self._lock_expires_at_ms is not None
            if confident and classification.target == self._locked_device:
                self._lock_expires_at_ms = timestamp_ms + self._config.target_lock_ttl_ms
            elif timestamp_ms >= self._lock_expires_at_ms:
                self._state = GazeLockState.EXPIRED
            return self._state

        return self._state

    def notify_gesture_started(self, timestamp_ms: int) -> GazeLockState:
        """TARGET_LOCKED 상태에서 Fusion이 gesture 시작을 감지했을 때 호출한다."""
        if self._state == GazeLockState.TARGET_LOCKED:
            self._clear_candidate()
            self._clear_unknown_timer()
            self._lock_expires_at_ms = timestamp_ms + self._config.target_lock_ttl_ms
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

    def _start_candidate(self, device_id: str, timestamp_ms: int) -> None:
        self._candidate_device = device_id
        self._candidate_started_at_ms = timestamp_ms
        self._candidate_elapsed_ms = 0

    def _clear_candidate(self) -> None:
        self._candidate_device = None
        self._candidate_started_at_ms = None
        self._candidate_elapsed_ms = 0

    def _clear_unknown_timer(self) -> None:
        self._unknown_started_at_ms = None
        self._unknown_elapsed_ms = 0
