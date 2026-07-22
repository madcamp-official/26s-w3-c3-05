"""README 7장 "Gaze Lock 상태 머신": SEARCHING → CANDIDATE → TARGET_LOCKED →
GESTURE_WAIT → EXPIRED 또는 COMMITTED."""

from __future__ import annotations

from jarvis.gaze.classifier import ClassificationResult
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.lock import GazeLockState, GazeLockStateMachine

UNKNOWN = GazeConfig().UNKNOWN_TARGET


def _confident(target: str, probability: float = 0.9, second_best: float = 0.05) -> ClassificationResult:
    return ClassificationResult(target=target, probability=probability, second_best_probability=second_best)


def _unknown() -> ClassificationResult:
    return ClassificationResult(target=UNKNOWN, probability=0.1, second_best_probability=0.08)


def test_starts_in_searching() -> None:
    lock = GazeLockStateMachine()
    assert lock.state == GazeLockState.SEARCHING
    assert GazeConfig().dwell_time_ms == 3000
    assert lock.dwell_progress == 0.0


def test_unknown_keeps_searching() -> None:
    lock = GazeLockStateMachine()
    state = lock.update(0, _unknown())
    assert state == GazeLockState.SEARCHING


def test_confident_target_becomes_candidate() -> None:
    lock = GazeLockStateMachine()
    state = lock.update(0, _confident("laptop"))
    assert state == GazeLockState.CANDIDATE
    assert lock.candidate_device == "laptop"
    assert lock.candidate_elapsed_ms == 0


def test_candidate_exposes_three_second_progress() -> None:
    lock = GazeLockStateMachine(GazeConfig(dwell_time_ms=3000))
    lock.update(1_000, _confident("laptop"))

    lock.update(2_500, _confident("laptop"))

    assert lock.candidate_device == "laptop"
    assert lock.candidate_elapsed_ms == 1_500
    assert lock.dwell_progress == 0.5


def test_low_margin_does_not_promote_to_candidate() -> None:
    config = GazeConfig(minimum_margin=0.2)
    lock = GazeLockStateMachine(config)
    state = lock.update(0, _confident("laptop", probability=0.85, second_best=0.75))
    assert state == GazeLockState.SEARCHING


def test_dwell_promotes_candidate_to_target_locked() -> None:
    config = GazeConfig(dwell_time_ms=500)
    lock = GazeLockStateMachine(config)
    lock.update(0, _confident("laptop"))
    assert lock.state == GazeLockState.CANDIDATE

    state = lock.update(499, _confident("laptop"))
    assert state == GazeLockState.CANDIDATE

    state = lock.update(500, _confident("laptop"))
    assert state == GazeLockState.TARGET_LOCKED
    assert lock.locked_device == "laptop"
    assert lock.is_locked_to("laptop")
    assert lock.candidate_device is None
    assert lock.dwell_progress == 1.0


def test_switching_candidate_device_resets_dwell_timer() -> None:
    config = GazeConfig(dwell_time_ms=500)
    lock = GazeLockStateMachine(config)
    lock.update(0, _confident("laptop"))
    lock.update(400, _confident("room.bulb"))
    # 새 후보로 바뀌었으니 dwell이 다시 시작되어 400ms로는 잠기지 않는다.
    state = lock.update(400 + 400, _confident("room.bulb"))
    assert state == GazeLockState.CANDIDATE
    state = lock.update(400 + 500, _confident("room.bulb"))
    assert state == GazeLockState.TARGET_LOCKED
    assert lock.locked_device == "room.bulb"


def test_candidate_drops_to_searching_after_grace_expires() -> None:
    """순간 UNKNOWN은 깜빡임일 수 있어 유예하고, 지속되면 탐색으로 돌아간다."""
    lock = GazeLockStateMachine(GazeConfig(candidate_grace_ms=600))
    lock.update(0, _confident("laptop"))
    # 유예 안: 후보와 dwell이 유지된다.
    state = lock.update(100, _unknown())
    assert state == GazeLockState.CANDIDATE
    assert lock.candidate_device == "laptop"
    # 유예를 넘겨 계속 UNKNOWN이면 리셋된다.
    state = lock.update(100 + 700, _unknown())
    assert state == GazeLockState.SEARCHING


def test_blink_length_unknown_gap_does_not_reset_dwell() -> None:
    """깜빡임 길이의 UNKNOWN 공백(<= grace)은 3초 dwell을 0으로 되돌리지 않는다 —
    깜빡임마다 리셋되면 자연 깜빡임 주기(2~5초)보다 dwell이 길어 영원히 확정되지
    않는다(2026-07-22 실사용)."""
    config = GazeConfig(dwell_time_ms=3000, candidate_grace_ms=600)
    lock = GazeLockStateMachine(config)
    lock.update(0, _confident("laptop"))
    lock.update(1_000, _confident("laptop"))
    # 1.0~1.4초 구간 깜빡임 → UNKNOWN 프레임들.
    lock.update(1_200, _unknown())
    state = lock.update(1_400, _unknown())
    assert state == GazeLockState.CANDIDATE
    # 회복 후 같은 target이면 dwell이 처음부터가 아니라 이어서 적립된다.
    state = lock.update(2_999, _confident("laptop"))
    assert state == GazeLockState.CANDIDATE
    state = lock.update(3_000, _confident("laptop"))
    assert state == GazeLockState.TARGET_LOCKED
    assert lock.locked_device == "laptop"


def test_second_blink_gap_gets_fresh_grace() -> None:
    """유예 타이머는 확신 프레임이 돌아올 때마다 초기화된다 — 깜빡임 두 번이
    누적으로 grace를 넘겨도 각각이 짧으면 dwell은 유지된다."""
    config = GazeConfig(dwell_time_ms=3000, candidate_grace_ms=600)
    lock = GazeLockStateMachine(config)
    lock.update(0, _confident("laptop"))
    lock.update(500, _unknown())
    assert lock.state == GazeLockState.CANDIDATE
    lock.update(900, _confident("laptop"))
    lock.update(1_800, _unknown())
    state = lock.update(2_200, _confident("laptop"))
    assert state == GazeLockState.CANDIDATE
    state = lock.update(3_000, _confident("laptop"))
    assert state == GazeLockState.TARGET_LOCKED


def test_locked_persists_through_brief_look_away_within_ttl() -> None:
    config = GazeConfig(dwell_time_ms=100, target_lock_ttl_ms=1000)
    lock = GazeLockStateMachine(config)
    lock.update(0, _confident("laptop"))
    lock.update(100, _confident("laptop"))
    assert lock.state == GazeLockState.TARGET_LOCKED

    # 손을 보려고 잠깐 다른 곳을 보거나 추적이 흔들려도 TTL 안에서는 유지된다.
    state = lock.update(500, _unknown())
    assert state == GazeLockState.TARGET_LOCKED
    assert lock.locked_device == "laptop"


def test_locked_selection_releases_after_two_continuous_unknown_seconds() -> None:
    config = GazeConfig(
        dwell_time_ms=100,
        target_lock_ttl_ms=1000,
        confirmed_unknown_timeout_ms=2000,
    )
    lock = GazeLockStateMachine(config)
    lock.update(0, _confident("laptop"))
    lock.update(100, _confident("laptop"))
    assert lock.state == GazeLockState.TARGET_LOCKED

    state = lock.update(200, _unknown())
    assert state == GazeLockState.TARGET_LOCKED
    assert lock.locked_device == "laptop"
    assert lock.unknown_elapsed_ms == 0

    state = lock.update(2199, _unknown())
    assert state == GazeLockState.TARGET_LOCKED
    assert lock.locked_device == "laptop"
    assert lock.unknown_elapsed_ms == 1999

    state = lock.update(2200, _unknown())
    assert state == GazeLockState.SEARCHING
    assert lock.locked_device is None
    assert lock.unknown_elapsed_ms == 0


def test_known_target_interrupts_unknown_release_timer() -> None:
    config = GazeConfig(dwell_time_ms=100, confirmed_unknown_timeout_ms=2000)
    lock = GazeLockStateMachine(config)
    lock.update(0, _confident("monitor"))
    lock.update(100, _confident("monitor"))

    lock.update(200, _unknown())
    lock.update(1900, _unknown())
    assert lock.unknown_elapsed_ms == 1700
    lock.update(1950, _confident("monitor"))
    assert lock.unknown_elapsed_ms == 0

    lock.update(2000, _unknown())
    state = lock.update(3999, _unknown())
    assert state == GazeLockState.TARGET_LOCKED
    assert lock.locked_device == "monitor"


def test_replacement_candidate_keeps_confirmed_target_until_dwell() -> None:
    config = GazeConfig(dwell_time_ms=3000, target_lock_ttl_ms=1000)
    lock = GazeLockStateMachine(config)
    lock.update(0, _confident("monitor"))
    lock.update(3000, _confident("monitor"))

    state = lock.update(3100, _confident("speaker"))
    assert state == GazeLockState.TARGET_LOCKED
    assert lock.locked_device == "monitor"
    assert lock.candidate_device == "speaker"

    lock.update(5000, _confident("speaker"))
    assert lock.locked_device == "monitor"
    assert lock.candidate_device == "speaker"
    assert lock.candidate_elapsed_ms == 1900


def test_cancelled_replacement_keeps_previous_confirmed_target() -> None:
    config = GazeConfig(dwell_time_ms=3000)
    lock = GazeLockStateMachine(config)
    lock.update(0, _confident("monitor"))
    lock.update(3000, _confident("monitor"))
    lock.update(3100, _confident("speaker"))

    state = lock.update(4000, _unknown())
    assert state == GazeLockState.TARGET_LOCKED
    assert lock.locked_device == "monitor"
    assert lock.candidate_device is None


def test_replacement_handoff_is_atomic_after_dwell() -> None:
    config = GazeConfig(dwell_time_ms=3000)
    lock = GazeLockStateMachine(config)
    lock.update(0, _confident("monitor"))
    lock.update(3000, _confident("monitor"))
    lock.update(3100, _confident("speaker"))

    state = lock.update(6100, _confident("speaker"))
    assert state == GazeLockState.TARGET_LOCKED
    assert lock.locked_device == "speaker"
    assert lock.candidate_device is None


def test_reconfirming_locked_target_refreshes_ttl() -> None:
    config = GazeConfig(dwell_time_ms=100, target_lock_ttl_ms=1000)
    lock = GazeLockStateMachine(config)
    lock.update(0, _confident("laptop"))
    lock.update(100, _confident("laptop"))

    lock.update(900, _confident("laptop"))  # 만료 전에 다시 확인 -> TTL 갱신
    state = lock.update(900 + 999, _confident("laptop"))
    assert state == GazeLockState.TARGET_LOCKED


def test_gesture_started_only_from_target_locked() -> None:
    lock = GazeLockStateMachine()
    # SEARCHING에서는 무시된다.
    assert lock.notify_gesture_started(0) == GazeLockState.SEARCHING

    config = GazeConfig(dwell_time_ms=0)
    lock = GazeLockStateMachine(config)
    lock.update(0, _confident("laptop"))
    assert lock.state == GazeLockState.TARGET_LOCKED

    state = lock.notify_gesture_started(10)
    assert state == GazeLockState.GESTURE_WAIT
    assert lock.locked_device == "laptop"


def test_gesture_start_refreshes_sticky_selection_ttl() -> None:
    config = GazeConfig(dwell_time_ms=0, target_lock_ttl_ms=1000)
    lock = GazeLockStateMachine(config)
    lock.update(100, _confident("laptop"))

    state = lock.notify_gesture_started(1100)

    assert state == GazeLockState.GESTURE_WAIT
    assert lock.locked_device == "laptop"


def test_committed_only_from_gesture_wait_and_restores_selection_next_tick() -> None:
    config = GazeConfig(dwell_time_ms=0)
    lock = GazeLockStateMachine(config)
    lock.update(0, _confident("laptop"))
    lock.notify_gesture_started(10)

    state = lock.notify_committed(20)
    assert state == GazeLockState.COMMITTED

    state = lock.update(21, _unknown())
    assert state == GazeLockState.TARGET_LOCKED
    assert lock.locked_device == "laptop"


def test_commit_after_gesture_wait_ttl_is_rejected_but_selection_is_retained() -> None:
    config = GazeConfig(dwell_time_ms=0, target_lock_ttl_ms=1000)
    lock = GazeLockStateMachine(config)
    lock.update(100, _confident("laptop"))
    lock.notify_gesture_started(500)

    state = lock.notify_committed(1500)

    assert state == GazeLockState.EXPIRED
    assert lock.locked_device == "laptop"


def test_reset_clears_candidate_and_lock() -> None:
    config = GazeConfig(dwell_time_ms=0)
    lock = GazeLockStateMachine(config)
    lock.update(0, _confident("laptop"))
    assert lock.state == GazeLockState.TARGET_LOCKED

    lock.reset()
    assert lock.state == GazeLockState.SEARCHING
    assert lock.locked_device is None
