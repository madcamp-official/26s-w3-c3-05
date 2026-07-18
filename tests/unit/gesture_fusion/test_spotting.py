"""Gesture spotting 상태 머신을 검증한다 (README 8장: 여러 프레임 검출을 하나의
이벤트로 만드는 디바운스 상태 머신).
"""

from __future__ import annotations

from jarvis.contracts.messages import GesturePhase
from jarvis.gesture_fusion.model_protocol import ModelPrediction
from jarvis.gesture_fusion.spotting import GestureSpotter, SpotterConfig


def _prediction(
    phase: GesturePhase,
    *,
    gesture: str = "swipe_down",
    gesture_confidence: float = 0.9,
    phase_confidence: float = 0.9,
    uncertainty: float = 0.1,
) -> ModelPrediction:
    return ModelPrediction(
        gesture=gesture,
        gesture_confidence=gesture_confidence,
        phase=phase,
        phase_confidence=phase_confidence,
        uncertainty=uncertainty,
    )


def _config(**overrides: object) -> SpotterConfig:
    defaults: dict[str, object] = dict(min_consecutive_frames=2)
    defaults.update(overrides)
    return SpotterConfig(**defaults)  # type: ignore[arg-type]


# --- 디바운스: 단일 프레임 노이즈 억제 ---


def test_single_frame_onset_does_not_transition() -> None:
    spotter = GestureSpotter(_config(min_consecutive_frames=2))
    estimate = spotter.push(_prediction(GesturePhase.ONSET), timestamp_ms=0, frame_id=0)
    assert estimate.phase == GesturePhase.IDLE
    assert spotter.state == GesturePhase.IDLE


def test_consecutive_onset_confirms_transition() -> None:
    spotter = GestureSpotter(_config(min_consecutive_frames=2))
    spotter.push(_prediction(GesturePhase.ONSET), timestamp_ms=0, frame_id=0)
    estimate = spotter.push(_prediction(GesturePhase.ONSET), timestamp_ms=10, frame_id=1)
    assert estimate.phase == GesturePhase.ONSET
    assert spotter.state == GesturePhase.ONSET


def test_flickering_raw_phase_never_confirms() -> None:
    spotter = GestureSpotter(_config(min_consecutive_frames=3))
    for i in range(6):
        phase = GesturePhase.ONSET if i % 2 == 0 else GesturePhase.IDLE
        estimate = spotter.push(_prediction(phase), timestamp_ms=i * 10, frame_id=i)
        assert estimate.phase == GesturePhase.IDLE
    assert spotter.state == GesturePhase.IDLE


# --- 게이팅: 배경 클래스·낮은 확신도는 ONSET으로 인정하지 않음 ---


def test_onset_rejected_for_background_label() -> None:
    spotter = GestureSpotter(_config(min_consecutive_frames=2, background_label="none"))
    for i in range(2):
        estimate = spotter.push(
            _prediction(GesturePhase.ONSET, gesture="none"), timestamp_ms=i * 10, frame_id=i
        )
    assert estimate.phase == GesturePhase.IDLE
    assert spotter.state == GesturePhase.IDLE


def test_onset_rejected_for_low_gesture_confidence() -> None:
    spotter = GestureSpotter(_config(min_consecutive_frames=2, min_onset_gesture_confidence=0.8))
    for i in range(2):
        estimate = spotter.push(
            _prediction(GesturePhase.ONSET, gesture_confidence=0.3), timestamp_ms=i * 10, frame_id=i
        )
    assert estimate.phase == GesturePhase.IDLE


def test_onset_accepted_once_confidence_recovers() -> None:
    spotter = GestureSpotter(_config(min_consecutive_frames=2, min_onset_gesture_confidence=0.8))
    spotter.push(_prediction(GesturePhase.ONSET, gesture_confidence=0.3), timestamp_ms=0, frame_id=0)
    spotter.push(_prediction(GesturePhase.ONSET, gesture_confidence=0.3), timestamp_ms=10, frame_id=1)
    # 여전히 streak≥2가 유지되는 상태에서 confidence가 회복되면 다음 프레임에 확정된다.
    estimate = spotter.push(
        _prediction(GesturePhase.ONSET, gesture_confidence=0.95), timestamp_ms=20, frame_id=2
    )
    assert estimate.phase == GesturePhase.ONSET


# --- 단계 건너뛰기 거부 (모델 출력을 그대로 믿지 않음) ---


def test_idle_to_active_jump_is_rejected() -> None:
    spotter = GestureSpotter(_config(min_consecutive_frames=2))
    for i in range(4):
        estimate = spotter.push(_prediction(GesturePhase.ACTIVE), timestamp_ms=i * 10, frame_id=i)
    assert estimate.phase == GesturePhase.IDLE
    assert spotter.state == GesturePhase.IDLE


def test_onset_to_ending_jump_is_rejected() -> None:
    spotter = GestureSpotter(_config(min_consecutive_frames=1))
    spotter.push(_prediction(GesturePhase.ONSET), timestamp_ms=0, frame_id=0)
    assert spotter.state == GesturePhase.ONSET
    estimate = spotter.push(_prediction(GesturePhase.ENDING), timestamp_ms=10, frame_id=1)
    assert estimate.phase == GesturePhase.ONSET  # 건너뛰기 거부, ONSET 유지
    assert spotter.state == GesturePhase.ONSET


# --- 전체 사이클: ENDING은 정확히 한 번 ---


def test_full_cycle_emits_ending_exactly_once() -> None:
    spotter = GestureSpotter(_config(min_consecutive_frames=1))
    phases_in_order = [
        GesturePhase.ONSET,
        GesturePhase.ACTIVE,
        GesturePhase.ENDING,
        GesturePhase.ENDING,  # 이후에도 raw가 계속 ENDING이어도 한 번만 나와야 함
        GesturePhase.ENDING,
    ]
    seen_phases = []
    for i, phase in enumerate(phases_in_order):
        estimate = spotter.push(_prediction(phase), timestamp_ms=i * 10, frame_id=i)
        seen_phases.append(estimate.phase)

    assert seen_phases.count(GesturePhase.ENDING) == 1
    # ENDING 방출 직후 즉시 IDLE로 리셋되어, 이후 raw=ENDING은 전이가 거부된다.
    assert seen_phases[-1] == GesturePhase.IDLE
    assert spotter.state == GesturePhase.IDLE


def test_locked_gesture_label_persists_through_event() -> None:
    """ONSET에서 lock된 label은 도중에 raw 예측이 바뀌어도 이벤트 내내 유지된다."""
    spotter = GestureSpotter(_config(min_consecutive_frames=1))
    spotter.push(_prediction(GesturePhase.ONSET, gesture="swipe_down"), timestamp_ms=0, frame_id=0)
    estimate = spotter.push(
        _prediction(GesturePhase.ACTIVE, gesture="swipe_left"), timestamp_ms=10, frame_id=1
    )
    assert estimate.gesture == "swipe_down"
    estimate = spotter.push(
        _prediction(GesturePhase.ENDING, gesture="rotate_clockwise"), timestamp_ms=20, frame_id=2
    )
    assert estimate.gesture == "swipe_down"


# --- 추적 손실: 안전하게 포기 ---


def test_tracking_loss_aborts_active_gesture() -> None:
    spotter = GestureSpotter(_config(min_consecutive_frames=1))
    spotter.push(_prediction(GesturePhase.ONSET), timestamp_ms=0, frame_id=0)
    spotter.push(_prediction(GesturePhase.ACTIVE), timestamp_ms=10, frame_id=1)
    assert spotter.state == GesturePhase.ACTIVE

    estimate = spotter.push(None, timestamp_ms=20, frame_id=2)
    assert estimate.phase == GesturePhase.IDLE
    assert estimate.gesture_confidence == 0.0
    assert estimate.uncertainty == 1.0
    assert spotter.state == GesturePhase.IDLE
    assert not spotter.is_tracking_gesture


def test_abort_to_idle_clears_lock_for_next_gesture() -> None:
    spotter = GestureSpotter(_config(min_consecutive_frames=1))
    spotter.push(_prediction(GesturePhase.ONSET, gesture="swipe_down"), timestamp_ms=0, frame_id=0)
    spotter.push(_prediction(GesturePhase.IDLE), timestamp_ms=10, frame_id=1)
    assert spotter.state == GesturePhase.IDLE

    estimate = spotter.push(
        _prediction(GesturePhase.ONSET, gesture="swipe_up"), timestamp_ms=20, frame_id=2
    )
    assert estimate.gesture == "swipe_up"


# --- pointer 모듈 연동 신호 ---


def test_is_tracking_gesture_reflects_non_idle_state() -> None:
    spotter = GestureSpotter(_config(min_consecutive_frames=1))
    assert not spotter.is_tracking_gesture
    spotter.push(_prediction(GesturePhase.ONSET), timestamp_ms=0, frame_id=0)
    assert spotter.is_tracking_gesture


def test_reset_clears_all_state() -> None:
    spotter = GestureSpotter(_config(min_consecutive_frames=1))
    spotter.push(_prediction(GesturePhase.ONSET, gesture="swipe_down"), timestamp_ms=0, frame_id=0)
    spotter.reset()
    assert spotter.state == GesturePhase.IDLE
    assert not spotter.is_tracking_gesture
    estimate = spotter.push(
        _prediction(GesturePhase.ONSET, gesture="swipe_up"), timestamp_ms=10, frame_id=1
    )
    assert estimate.gesture == "swipe_up"


# --- 타이밍 계약: timestamp_ms·frame_id를 그대로 전달 ---


def test_timestamp_and_frame_id_pass_through_unchanged() -> None:
    spotter = GestureSpotter()
    estimate = spotter.push(_prediction(GesturePhase.IDLE), timestamp_ms=123456, frame_id=42)
    assert estimate.timestamp_ms == 123456
    assert estimate.frame_id == 42
