"""시선·제스처 temporal alignment를 검증한다 (README 9장 Commit 조건 1·2·3·6)."""

from __future__ import annotations

from jarvis.contracts.messages import GestureEstimate, GesturePhase, TargetEstimate
from jarvis.gesture_fusion.alignment import (
    AlignmentConfig,
    TargetLockTracker,
    TemporalAligner,
    check_alignment,
)


def _target(
    timestamp_ms: int,
    *,
    target: str = "room.bulb",
    probability: float = 0.9,
    second_best_probability: float = 0.05,
    stability: float = 0.9,
    frame_id: int = 0,
) -> TargetEstimate:
    return TargetEstimate(
        timestamp_ms=timestamp_ms,
        frame_id=frame_id,
        target=target,
        probability=probability,
        second_best_probability=second_best_probability,
        stability=stability,
    )


def _gesture(
    timestamp_ms: int, phase: GesturePhase, *, gesture: str = "swipe_down", frame_id: int = 0
) -> GestureEstimate:
    return GestureEstimate(
        timestamp_ms=timestamp_ms,
        frame_id=frame_id,
        gesture=gesture,
        gesture_confidence=0.9,
        phase=phase,
        phase_confidence=0.9,
        uncertainty=0.1,
    )


def _config(**overrides: object) -> AlignmentConfig:
    defaults: dict[str, object] = dict(target_dwell_ms=200, target_lock_ttl_ms=1000)
    defaults.update(overrides)
    return AlignmentConfig(**defaults)  # type: ignore[arg-type]


# --- TargetLockTracker: dwell → lock ---


def test_default_target_confirmation_requires_three_seconds() -> None:
    assert AlignmentConfig().target_dwell_ms == 3000


def test_below_dwell_time_stays_unlocked() -> None:
    tracker = TargetLockTracker(_config(target_dwell_ms=200))
    tracker.push(_target(0))
    state = tracker.push(_target(100))  # dwell=100ms < 200ms
    assert not state.locked


def test_dwell_time_reached_locks_target() -> None:
    tracker = TargetLockTracker(_config(target_dwell_ms=200))
    tracker.push(_target(0))
    state = tracker.push(_target(200))  # dwell=200ms
    assert state.locked
    assert state.target == "room.bulb"
    assert state.locked_at_ms == 200


def test_low_probability_does_not_start_candidate() -> None:
    tracker = TargetLockTracker(_config(target_dwell_ms=200))
    tracker.push(_target(0, probability=0.5))
    state = tracker.push(_target(200, probability=0.5))
    assert not state.locked


def test_low_margin_does_not_start_candidate() -> None:
    tracker = TargetLockTracker()
    state = tracker.push(_target(0, probability=0.85, second_best_probability=0.80))  # margin=0.05
    assert not state.locked


def test_unknown_target_never_locks() -> None:
    tracker = TargetLockTracker(_config(target_dwell_ms=0))
    state = tracker.push(_target(0, target="UNKNOWN", probability=0.99, second_best_probability=0.0))
    assert not state.locked


def test_switching_target_before_dwell_restarts_candidate() -> None:
    tracker = TargetLockTracker(_config(target_dwell_ms=200))
    tracker.push(_target(0, target="room.bulb"))
    tracker.push(_target(100, target="laptop"))  # 대상이 바뀌어 candidate 리셋
    state = tracker.push(_target(250, target="laptop"))  # laptop 기준 dwell=150ms < 200ms
    assert not state.locked


# --- lock 유지·만료(TTL) ---


def test_lock_ttl_extends_while_target_held() -> None:
    tracker = TargetLockTracker(_config(target_dwell_ms=0, target_lock_ttl_ms=500))
    tracker.push(_target(0))
    locked = tracker.push(_target(0))
    assert locked.expires_at_ms == 500
    later = tracker.push(_target(400))  # 만료 전 갱신
    assert later.locked
    assert later.expires_at_ms == 900  # 400 + 500


def test_lock_expires_after_ttl_without_refresh() -> None:
    tracker = TargetLockTracker(_config(target_dwell_ms=0, target_lock_ttl_ms=500))
    tracker.push(_target(0))
    tracker.push(_target(0))
    # 500ms 넘게 아무 갱신도 없다가 늦게 도착
    state = tracker.push(_target(600, probability=0.0, second_best_probability=0.0))
    assert not state.locked


def test_brief_invalid_frame_within_ttl_keeps_lock() -> None:
    """짧은 시선 흔들림(유예 기간 안)은 lock을 깨지 않는다."""
    tracker = TargetLockTracker(_config(target_dwell_ms=0, target_lock_ttl_ms=500))
    tracker.push(_target(0))
    tracker.push(_target(0))
    state = tracker.push(_target(100, probability=0.1, second_best_probability=0.05))
    assert state.locked
    assert state.target == "room.bulb"


def test_switching_target_keeps_old_lock_during_replacement_dwell() -> None:
    tracker = TargetLockTracker(_config(target_dwell_ms=200, target_lock_ttl_ms=500))
    tracker.push(_target(0, target="room.bulb"))
    tracker.push(_target(200, target="room.bulb"))
    state = tracker.push(_target(250, target="laptop"))
    assert state.locked
    assert state.target == "room.bulb"
    assert state.candidate == "laptop"


def test_cancelled_replacement_keeps_old_fusion_lock() -> None:
    tracker = TargetLockTracker(_config(target_dwell_ms=200, target_lock_ttl_ms=500))
    tracker.push(_target(0, target="room.bulb"))
    tracker.push(_target(200, target="room.bulb"))
    tracker.push(_target(250, target="laptop"))

    state = tracker.push(_target(300, target="UNKNOWN"))
    assert state.locked
    assert state.target == "room.bulb"
    assert state.candidate is None


def test_replacement_fusion_handoff_is_atomic_after_dwell() -> None:
    tracker = TargetLockTracker(_config(target_dwell_ms=200, target_lock_ttl_ms=500))
    tracker.push(_target(0, target="room.bulb"))
    tracker.push(_target(200, target="room.bulb"))
    tracker.push(_target(250, target="laptop"))

    state = tracker.push(_target(450, target="laptop"))
    assert state.locked
    assert state.target == "laptop"
    assert state.candidate is None
    assert state.locked_at_ms == 450


def test_reset_clears_lock_and_candidate() -> None:
    tracker = TargetLockTracker(_config(target_dwell_ms=0))
    tracker.push(_target(0))
    tracker.push(_target(0))
    assert tracker.state.locked
    tracker.reset()
    assert not tracker.state.locked


# --- check_alignment: Commit 조건 1·2·3·6 ---


def test_alignment_rejected_when_not_locked() -> None:
    unlocked_state = TargetLockTracker().state  # 아직 아무 프레임도 안 받은 초기 상태
    result = check_alignment(unlocked_state, onset_timestamp_ms=100, ending_timestamp_ms=200)
    assert not result.aligned
    assert "not locked" in result.reason


def test_alignment_rejected_when_gesture_started_before_lock() -> None:
    tracker = TargetLockTracker(_config(target_dwell_ms=0, target_lock_ttl_ms=1000))
    tracker.push(_target(500))
    tracker.push(_target(500))
    lock = tracker.state
    result = check_alignment(lock, onset_timestamp_ms=100, ending_timestamp_ms=600)
    assert not result.aligned
    assert "before" in result.reason


def test_alignment_rejected_when_gesture_completes_after_ttl() -> None:
    tracker = TargetLockTracker(_config(target_dwell_ms=0, target_lock_ttl_ms=200))
    tracker.push(_target(0))
    tracker.push(_target(0))  # expires_at_ms = 200
    lock = tracker.state
    result = check_alignment(lock, onset_timestamp_ms=50, ending_timestamp_ms=300)
    assert not result.aligned
    assert "ttl" in result.reason


def test_alignment_accepted_when_gesture_within_lock_window() -> None:
    tracker = TargetLockTracker(_config(target_dwell_ms=0, target_lock_ttl_ms=1000))
    tracker.push(_target(0))
    tracker.push(_target(0))
    lock = tracker.state
    result = check_alignment(lock, onset_timestamp_ms=50, ending_timestamp_ms=500)
    assert result.aligned
    assert result.target == "room.bulb"


# --- TemporalAligner: 두 스트림 결합 ---


def test_aligner_returns_none_before_gesture_completes() -> None:
    aligner = TemporalAligner(_config(target_dwell_ms=0))
    aligner.push_target(_target(0))
    aligner.push_target(_target(0))
    assert aligner.push_gesture(_gesture(50, GesturePhase.ONSET)) is None
    assert aligner.push_gesture(_gesture(80, GesturePhase.ACTIVE)) is None


def test_aligner_full_success_path() -> None:
    aligner = TemporalAligner(_config(target_dwell_ms=0, target_lock_ttl_ms=1000))
    aligner.push_target(_target(0))
    aligner.push_target(_target(0))  # 잠금, expires_at_ms=1000
    aligner.push_gesture(_gesture(100, GesturePhase.ONSET))
    aligner.push_gesture(_gesture(150, GesturePhase.ACTIVE))
    result = aligner.push_gesture(_gesture(300, GesturePhase.ENDING))
    assert result is not None
    assert result.aligned
    assert result.target == "room.bulb"


def test_aligner_rejects_when_onset_missing() -> None:
    aligner = TemporalAligner(_config(target_dwell_ms=0))
    aligner.push_target(_target(0))
    aligner.push_target(_target(0))
    result = aligner.push_gesture(_gesture(300, GesturePhase.ENDING))  # ONSET 없이 바로 ENDING
    assert result is not None
    assert not result.aligned
    assert "onset" in result.reason


def test_aligner_consumes_onset_once_per_event() -> None:
    """ENDING 이후 다음 이벤트에서 이전 ONSET 시각이 재사용되지 않는다."""
    aligner = TemporalAligner(_config(target_dwell_ms=0, target_lock_ttl_ms=1000))
    aligner.push_target(_target(0))
    aligner.push_target(_target(0))
    aligner.push_gesture(_gesture(100, GesturePhase.ONSET))
    aligner.push_gesture(_gesture(200, GesturePhase.ENDING))
    # ONSET 없이 바로 ENDING이 다시 오면 이전 값이 재사용되지 않고 거부되어야 한다.
    result = aligner.push_gesture(_gesture(400, GesturePhase.ENDING))
    assert result is not None
    assert not result.aligned
    assert "onset" in result.reason


def test_aligner_target_switch_mid_gesture_is_rejected() -> None:
    """도중에 다른 기기로 재-lock되면(현재 lock의 시작이 gesture onset보다 늦음) 거부한다."""
    aligner = TemporalAligner(_config(target_dwell_ms=0, target_lock_ttl_ms=1000))
    aligner.push_target(_target(0, target="room.bulb"))
    aligner.push_target(_target(0, target="room.bulb"))
    aligner.push_gesture(_gesture(50, GesturePhase.ONSET))
    # 제스처 도중 다른 기기로 확실히 옮겨감 → 새 lock 시작 시각이 onset보다 늦어짐
    aligner.push_target(_target(100, target="laptop"))
    aligner.push_target(_target(100, target="laptop"))
    result = aligner.push_gesture(_gesture(300, GesturePhase.ENDING))
    assert result is not None
    assert not result.aligned


# --- TargetLockTracker: confirmation gate (OK사인 등 복귀 확인) ---


def test_first_ever_lock_on_gated_target_needs_no_confirmation() -> None:
    """아무것도 확정된 적이 없으면(첫 lock) 게이트된 target이어도 곧바로 잠긴다."""
    tracker = TargetLockTracker(
        _config(target_dwell_ms=200, confirmation_gated_targets=frozenset({"laptop"}))
    )
    tracker.push(_target(0, target="laptop"))
    state = tracker.push(_target(200, target="laptop"))
    assert state.locked
    assert state.target == "laptop"


def test_staying_on_gated_target_needs_no_confirmation() -> None:
    """게이트된 target에 이미 확정돼 있고 계속 그 target을 보는 중이면 걸리지 않는다."""
    tracker = TargetLockTracker(
        _config(
            target_dwell_ms=200,
            target_lock_ttl_ms=5000,
            confirmation_gated_targets=frozenset({"laptop"}),
        )
    )
    tracker.push(_target(0, target="laptop"))
    tracker.push(_target(200, target="laptop"))  # 첫 확정
    state = tracker.push(_target(2000, target="laptop"))
    assert state.locked
    assert state.target == "laptop"


def test_returning_to_gated_target_without_confirmation_stays_pending() -> None:
    """전구→노트북으로 돌아올 때 확인 신호가 없으면 dwell을 채워도 노트북으로 안 바뀐다.

    기존 lock(전구)은 sticky해서 새 후보의 dwell만으로는 안 풀린다 — 게이트가 그
    교체(atomic swap) 자체를 막아, 새 target이 확정되지 않은 채 계속 후보로 남는다.
    """
    tracker = TargetLockTracker(
        _config(
            target_dwell_ms=200,
            target_lock_ttl_ms=5000,
            confirmation_gated_targets=frozenset({"laptop"}),
        )
    )
    tracker.push(_target(0, target="room.bulb"))
    tracker.push(_target(200, target="room.bulb"))  # 전구로 첫 확정
    tracker.push(_target(300, target="laptop"))
    state = tracker.push(_target(1000, target="laptop"))  # dwell(700ms) 충분히 넘음
    assert state.locked
    assert state.target == "room.bulb"  # 아직 전구 lock 그대로
    assert state.candidate == "laptop"  # 후보로는 계속 쌓이되 확정은 안 됨


def test_confirmation_signal_unlocks_the_pending_gate() -> None:
    """확인 신호(OK사인) 이후에는 같은 후보가 곧바로 잠긴다."""
    tracker = TargetLockTracker(
        _config(
            target_dwell_ms=200,
            target_lock_ttl_ms=5000,
            confirmation_gated_targets=frozenset({"laptop"}),
        )
    )
    tracker.push(_target(0, target="room.bulb"))
    tracker.push(_target(200, target="room.bulb"))
    tracker.push(_target(300, target="laptop"))
    pending = tracker.push(_target(1000, target="laptop"))
    assert pending.target == "room.bulb"  # 신호 없이는 계속 대기
    tracker.note_confirmation_signal(1050)
    state = tracker.push(_target(1100, target="laptop"))
    assert state.locked
    assert state.target == "laptop"


def test_confirmation_signal_within_pre_roll_counts_even_before_candidate_started() -> None:
    """후보 시작보다 살짝 먼저 온 확인 신호도 pre-roll 안이면 유효하다."""
    tracker = TargetLockTracker(
        _config(
            target_dwell_ms=200,
            confirmation_gated_targets=frozenset({"laptop"}),
            confirmation_pre_roll_ms=300,
        )
    )
    tracker.push(_target(0, target="room.bulb"))
    tracker.push(_target(200, target="room.bulb"))
    tracker.note_confirmation_signal(250)  # 후보(300ms) 시작보다 50ms 먼저 — pre-roll(300ms) 안
    tracker.push(_target(300, target="laptop"))
    state = tracker.push(_target(500, target="laptop"))  # dwell=200ms
    assert state.locked
    assert state.target == "laptop"


def test_confirmation_signal_older_than_pre_roll_does_not_count() -> None:
    """pre-roll보다 오래된 확인 신호는 무효 — 무관한 과거 동작으로 본다."""
    tracker = TargetLockTracker(
        _config(
            target_dwell_ms=200,
            target_lock_ttl_ms=5000,
            confirmation_gated_targets=frozenset({"laptop"}),
            confirmation_pre_roll_ms=50,
        )
    )
    tracker.push(_target(0, target="room.bulb"))
    tracker.push(_target(200, target="room.bulb"))
    tracker.note_confirmation_signal(100)  # 후보 시작(300ms)보다 200ms 먼저 — pre-roll(50ms) 밖
    tracker.push(_target(300, target="laptop"))
    state = tracker.push(_target(500, target="laptop"))
    assert state.target == "room.bulb"  # 여전히 대기 — laptop으로 안 바뀜


def test_ungated_target_is_unaffected_by_gate() -> None:
    """게이트 대상이 아닌 target(전구)은 확인 신호 없이도 평소대로 잠긴다."""
    tracker = TargetLockTracker(
        _config(target_dwell_ms=200, confirmation_gated_targets=frozenset({"laptop"}))
    )
    tracker.push(_target(0, target="laptop"))
    tracker.push(_target(200, target="laptop"))  # 노트북 첫 확정
    tracker.push(_target(300, target="room.bulb"))
    state = tracker.push(_target(500, target="room.bulb"))  # 전구는 게이트 대상이 아님
    assert state.locked
    assert state.target == "room.bulb"
