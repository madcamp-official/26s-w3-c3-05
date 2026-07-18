"""시선·제스처 temporal alignment — README 9장 Commit 조건 1·2·3·6.

Fusion은 계약(interface-contract.md §1)으로 원시 프레임별 `TargetEstimate`만 받는다 —
lock 여부를 담은 필드가 없다. 그래서 Intent 상태 머신의 `TARGET_CANDIDATE→
TARGET_LOCKED` 전이를 Fusion이 독립적으로 추적해야 한다. 이는 Gaze 모듈(1인)이
커서 통과 게이팅용으로 갖는 자체 Gaze Lock(README 7장)과는 별개의 관심사다 — 모듈
경계 규칙(repository-structure.md: "세 핵심 모듈은 서로의 내부 파일을 직접 import하지
않는다")에 따라 Gaze 내부 구현을 재사용하지 않고, 같은 원시 확률 신호에서 Fusion
자신의 lock을 계산한다.

`TemporalAligner`는 Gaze→Fusion 스트림(§1)과 Gesture→Fusion 스트림(§2, spotting.py
출력)을 각각 독립적으로 받아 최신 상태를 유지하다가, 제스처가 `ENDING`으로 완결되는
순간에만 Commit 조건 1·2·3·6을 판정한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from jarvis.contracts.messages import GestureEstimate, GesturePhase, TargetEstimate


@dataclass(frozen=True, slots=True)
class AlignmentConfig:
    """Fusion 자체 Target Lock·정렬 판정에 쓰는 튜너블 파라미터.

    README 7장 "초기 기준"과 같은 실세계 상수(dwell 500ms, TTL 1500ms, 확률 0.80,
    margin 0.20)를 기본값으로 쓰지만, Gaze 모듈의 `GazeConfig`와는 독립된 값이다 —
    두 모듈이 같은 원시 신호에서 각자의 목적(커서 게이팅 vs intent commit)으로 따로
    lock을 추적하므로, 한쪽 임계값을 바꿔도 다른 쪽에 영향이 없어야 한다.
    """

    target_dwell_ms: int = 500
    """TARGET_CANDIDATE가 TARGET_LOCKED로 승격하기 전 유지해야 하는 최소 시간."""

    target_lock_ttl_ms: int = 1500
    """lock이 계속 갱신되지 않을 때(시선이 흔들리거나 벗어날 때) 만료까지의 유예 시간.

    lock이 붙어 있는 동안 유효한 프레임이 들어올 때마다 `timestamp_ms + 이 값`으로
    갱신되는 슬라이딩 윈도우다 — 한 대상을 계속 응시하는데 1.5초마다 재-dwell해야
    하는 것은 비현실적이라는 판단(documents/decisions.md에 기록).
    """

    min_target_probability: float = 0.80
    """대상 후보로 인정하거나 lock을 유지하기 위한 최소 top-1 확률."""

    min_target_margin: float = 0.20
    """candidate 인정·lock 유지에 필요한 top-1과 top-2 확률의 최소 차이."""

    unknown_target: str = "UNKNOWN"
    """어떤 기기도 보고 있지 않다는 뜻의 target 값(interface-contract.md 공통 규칙)."""

    def __post_init__(self) -> None:
        if self.target_dwell_ms < 0:
            raise ValueError("target_dwell_ms must be non-negative")
        if self.target_lock_ttl_ms <= 0:
            raise ValueError("target_lock_ttl_ms must be positive")
        for name, value in (
            ("min_target_probability", self.min_target_probability),
            ("min_target_margin", self.min_target_margin),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1], got {value}")
        if not self.unknown_target:
            raise ValueError("unknown_target must not be empty")


DEFAULT_ALIGNMENT_CONFIG = AlignmentConfig()


@dataclass(frozen=True, slots=True)
class TargetLockState:
    """Fusion이 추적하는 Target Lock의 현재 상태 (Commit 조건 1의 판정 근거)."""

    locked: bool
    target: str | None
    locked_at_ms: int | None
    expires_at_ms: int | None
    target_confidence: float
    stability: float
    candidate: str | None = None
    """아직 dwell을 못 채워 lock되지 않았지만 후보로 쌓이는 중인 대상(있으면).

    `locked=False`일 때 `IDLE`과 `TARGET_CANDIDATE`(README 9장 Intent 상태 머신)를
    구분하는 데 쓴다(task 6). `locked=True`일 때는 항상 None이다.
    """


_UNLOCKED = TargetLockState(
    locked=False, target=None, locked_at_ms=None, expires_at_ms=None,
    target_confidence=0.0, stability=0.0, candidate=None,
)


class TargetLockTracker:
    """`TargetEstimate` 스트림에서 Fusion의 TARGET_LOCKED 상태를 추적한다."""

    def __init__(self, config: AlignmentConfig = DEFAULT_ALIGNMENT_CONFIG) -> None:
        self._config = config
        self._candidate_target: str | None = None
        self._candidate_since_ms: int | None = None
        self._locked_target: str | None = None
        self._locked_at_ms: int | None = None
        self._expires_at_ms: int | None = None
        self._last_confidence = 0.0
        self._last_stability = 0.0

    @property
    def state(self) -> TargetLockState:
        if self._locked_target is None:
            if self._candidate_target is None:
                return _UNLOCKED
            return TargetLockState(
                locked=False, target=None, locked_at_ms=None, expires_at_ms=None,
                target_confidence=self._last_confidence, stability=self._last_stability,
                candidate=self._candidate_target,
            )
        return TargetLockState(
            locked=True,
            target=self._locked_target,
            locked_at_ms=self._locked_at_ms,
            expires_at_ms=self._expires_at_ms,
            target_confidence=self._last_confidence,
            stability=self._last_stability,
            candidate=None,
        )

    def reset(self) -> None:
        self._candidate_target = None
        self._candidate_since_ms = None
        self._release_lock()

    def push(self, estimate: TargetEstimate) -> TargetLockState:
        """Gaze→Fusion 스트림 프레임 하나를 반영하고 갱신된 lock 상태를 반환한다."""
        valid = (
            estimate.target != self._config.unknown_target
            and estimate.probability >= self._config.min_target_probability
            and (estimate.probability - estimate.second_best_probability)
            >= self._config.min_target_margin
        )

        if self._locked_target is not None:
            self._push_while_locked(estimate, valid)
        else:
            self._push_while_unlocked(estimate, valid)

        return self.state

    def _push_while_locked(self, estimate: TargetEstimate, valid: bool) -> None:
        assert self._expires_at_ms is not None  # locked 상태에서는 항상 설정됨
        if estimate.timestamp_ms > self._expires_at_ms:
            self._release_lock()
            # 만료 시점 자체도 새 candidate의 시작점이 될 수 있으므로 이어서 평가한다.
            self._push_while_unlocked(estimate, valid)
            return
        if valid and estimate.target == self._locked_target:
            self._expires_at_ms = estimate.timestamp_ms + self._config.target_lock_ttl_ms
            self._last_confidence = estimate.probability
            self._last_stability = estimate.stability
            return
        if valid and estimate.target != self._locked_target:
            # 다른 기기로 확실히 옮겨감 — 기존 lock을 놓고 새 candidate부터 다시 쌓는다.
            self._release_lock()
            self._start_candidate(estimate)
            return
        # invalid하지만 아직 TTL 안 — 유예 기간이므로 lock은 유지하되 confidence는
        # 갱신하지 않는다(이 프레임이 실제로 그 대상을 본 게 아니므로).

    def _push_while_unlocked(self, estimate: TargetEstimate, valid: bool) -> None:
        if not valid:
            self._candidate_target = None
            self._candidate_since_ms = None
            self._last_confidence = estimate.probability
            self._last_stability = estimate.stability
            return

        if self._candidate_target != estimate.target:
            self._start_candidate(estimate)
        else:
            assert self._candidate_since_ms is not None  # candidate_target이 설정됐으므로 항상 같이 설정됨
            dwell = estimate.timestamp_ms - self._candidate_since_ms
            if dwell >= self._config.target_dwell_ms:
                self._locked_target = estimate.target
                self._locked_at_ms = estimate.timestamp_ms
                self._expires_at_ms = estimate.timestamp_ms + self._config.target_lock_ttl_ms

        self._last_confidence = estimate.probability
        self._last_stability = estimate.stability

    def _start_candidate(self, estimate: TargetEstimate) -> None:
        self._candidate_target = estimate.target
        self._candidate_since_ms = estimate.timestamp_ms

    def _release_lock(self) -> None:
        self._locked_target = None
        self._locked_at_ms = None
        self._expires_at_ms = None


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """Commit 조건 1·2·3·6의 판정 결과 — 결합 점수를 낼지 말지 결정하는 데 쓴다(task 6)."""

    aligned: bool
    reason: str
    """정렬 실패 이유(trace용, development-principles.md 5.5). 성공 시 "aligned"."""
    target: str | None
    target_confidence: float
    gaze_stability: float


def check_alignment(
    lock: TargetLockState, onset_timestamp_ms: int, ending_timestamp_ms: int
) -> AlignmentResult:
    """제스처 ONSET·ENDING 시각과 Target Lock 상태로 Commit 조건 1·2·3·6을 검사한다.

    조건 6("시간 관계가 유효함")은 별도 필드로 추가 검증할 게 없다 — 제스처가 lock
    시작 이후 시작하고(조건 2) lock 만료 전에 끝났다면(조건 3) 그 자체로 시간 관계가
    유효하므로, 코드에서는 2·3의 결합으로 자연히 표현된다(documents/decisions.md에 기록).
    """
    if not lock.locked or lock.target is None:
        return AlignmentResult(False, "target not locked", None, 0.0, 0.0)  # 조건 1
    assert lock.locked_at_ms is not None and lock.expires_at_ms is not None  # locked=True면 항상 설정됨
    if onset_timestamp_ms < lock.locked_at_ms:
        return AlignmentResult(
            False, "gesture started before target lock", lock.target,
            lock.target_confidence, lock.stability,
        )  # 조건 2
    if ending_timestamp_ms > lock.expires_at_ms:
        return AlignmentResult(
            False, "gesture completed after target lock ttl", lock.target,
            lock.target_confidence, lock.stability,
        )  # 조건 3
    return AlignmentResult(True, "aligned", lock.target, lock.target_confidence, lock.stability)


class TemporalAligner:
    """Gaze·Gesture 두 스트림을 시간축으로 정렬해 Commit 조건 1·2·3·6을 판정한다.

    두 스트림은 서로 다른 프레임레이트·타이밍으로 들어올 수 있어(README 9장 "시선과
    제스처의 시간 관계가 유효함") 각 push 메서드가 독립적으로 최신 상태를 유지하다가,
    제스처가 `ENDING`으로 완결되는 순간에만("as-of" 조인) 정렬을 평가한다.
    """

    def __init__(self, config: AlignmentConfig = DEFAULT_ALIGNMENT_CONFIG) -> None:
        self._lock_tracker = TargetLockTracker(config)
        self._onset_timestamp_ms: int | None = None

    @property
    def lock_state(self) -> TargetLockState:
        return self._lock_tracker.state

    def push_target(self, estimate: TargetEstimate) -> TargetLockState:
        """Gaze→Fusion 스트림(§1) 프레임 하나를 반영한다."""
        return self._lock_tracker.push(estimate)

    def push_gesture(self, estimate: GestureEstimate) -> AlignmentResult | None:
        """Gesture→Fusion 스트림(§2, spotting.py 출력) 프레임 하나를 반영한다.

        `phase=ONSET`이면 이 제스처 시도의 시작 시각을 기억해 둔다(조건 2 검사용).
        `phase=ENDING`이면 그 시작 시각과 현재 lock 상태로 정렬을 평가해 반환한다.
        그 외 phase는 아직 평가할 완결된 이벤트가 없어 `None`을 반환한다.
        """
        if estimate.phase == GesturePhase.ONSET:
            self._onset_timestamp_ms = estimate.timestamp_ms
            return None
        if estimate.phase != GesturePhase.ENDING:
            return None

        onset_ms = self._onset_timestamp_ms
        self._onset_timestamp_ms = None  # 한 이벤트당 한 번만 소비
        if onset_ms is None:
            # ONSET을 보지 못한 채 ENDING만 들어온 비정상 스트림 — 정렬 불가로
            # 안전하게 거부한다(development-principles.md 2.2: 불확실하면 거부).
            return AlignmentResult(False, "missing onset timestamp", None, 0.0, 0.0)

        return check_alignment(self._lock_tracker.state, onset_ms, estimate.timestamp_ms)
