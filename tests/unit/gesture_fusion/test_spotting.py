"""Gesture spotting 상태 머신을 검증한다 (README 8장: 여러 프레임 검출을 하나의
이벤트로 만드는 디바운스 상태 머신).

2026-07-21: 이벤트 판정을 phase head가 아니라 **gesture 활성 신호**(비배경 label +
확신도 임계)로 구동하도록 바꿨다(spotting.py 상단 주석 참조 — 학습된 phase head가
ONSET을 실측 0% 예측해 이전 phase 기반 전이가 모든 동적 제스처의 이벤트를 0개
냈다). 따라서 이 테스트들은 raw phase가 아니라 gesture/confidence를 조작한다.
"""

from __future__ import annotations

from jarvis.contracts.messages import GesturePhase
from jarvis.gesture_fusion.model_protocol import ModelPrediction
from jarvis.gesture_fusion.spotting import GestureSpotter, SpotterConfig

# 스포터는 이제 prediction.phase를 전이에 쓰지 않는다. 계약 타입을 채우기 위한
# 자리표시자로만 넣고, 활성 여부는 gesture·gesture_confidence가 결정한다.
_ANY_PHASE = GesturePhase.ACTIVE


def _prediction(
    *,
    gesture: str = "slide_two_fingers_left",
    gesture_confidence: float = 0.9,
    phase_confidence: float = 0.9,
    uncertainty: float = 0.1,
) -> ModelPrediction:
    """기본값은 "활성"(비배경 제스처 + 높은 확신)이다."""
    return ModelPrediction(
        gesture=gesture,
        gesture_confidence=gesture_confidence,
        phase=_ANY_PHASE,
        phase_confidence=phase_confidence,
        uncertainty=uncertainty,
    )


def _inactive(**overrides: object) -> ModelPrediction:
    """배경 label 예측(비활성) — 손은 보이지만 제스처가 아님."""
    return _prediction(gesture="none", **overrides)  # type: ignore[arg-type]


def _config(**overrides: object) -> SpotterConfig:
    defaults: dict[str, object] = dict(min_consecutive_frames=2)
    defaults.update(overrides)
    return SpotterConfig(**defaults)  # type: ignore[arg-type]


def _drive_to_active(spotter: GestureSpotter, gesture: str = "slide_two_fingers_left", n: int = 3) -> None:
    """min_consecutive_frames만큼 활성 프레임을 밀어 ACTIVE 상태까지 진입시킨다."""
    for i in range(n):
        spotter.push(_prediction(gesture=gesture), timestamp_ms=i * 10, frame_id=i)


# --- 디바운스: 단일 프레임 노이즈 억제 ---


def test_single_active_frame_does_not_transition() -> None:
    spotter = GestureSpotter(_config(min_consecutive_frames=2))
    estimate = spotter.push(_prediction(), timestamp_ms=0, frame_id=0)
    assert estimate.phase == GesturePhase.IDLE
    assert spotter.state == GesturePhase.IDLE


def test_consecutive_active_confirms_onset() -> None:
    spotter = GestureSpotter(_config(min_consecutive_frames=2))
    spotter.push(_prediction(), timestamp_ms=0, frame_id=0)
    estimate = spotter.push(_prediction(), timestamp_ms=10, frame_id=1)
    assert estimate.phase == GesturePhase.ONSET
    assert spotter.state == GesturePhase.ONSET


def test_flickering_activity_never_confirms() -> None:
    """활성/비활성이 매 프레임 번갈아 뜨면(스트릭이 임계값에 도달 못 함) 전이하지 않는다."""
    spotter = GestureSpotter(_config(min_consecutive_frames=3))
    for i in range(6):
        pred = _prediction() if i % 2 == 0 else _inactive()
        estimate = spotter.push(pred, timestamp_ms=i * 10, frame_id=i)
        assert estimate.phase == GesturePhase.IDLE
    assert spotter.state == GesturePhase.IDLE


# --- 게이팅: 배경 클래스·낮은 확신도는 활성으로 인정하지 않음 ---


def test_onset_rejected_for_background_label() -> None:
    spotter = GestureSpotter(_config(min_consecutive_frames=2, background_labels=frozenset({"none"})))
    for i in range(3):
        estimate = spotter.push(_inactive(), timestamp_ms=i * 10, frame_id=i)
    assert estimate.phase == GesturePhase.IDLE
    assert spotter.state == GesturePhase.IDLE


def test_onset_rejected_for_any_configured_background_label() -> None:
    """배경 label이 여럿(예: "none" + "doing_other_things")이면 전부 비활성이어야 한다.

    2026-07-20 발견: background_label이 문자열 하나뿐이던 시절엔 "타겟 제스처가
    아닌 동작"을 별도 label로 둔 학습 구성에서 그 label을 걸러내지 못했다.
    """
    spotter = GestureSpotter(
        _config(min_consecutive_frames=2, background_labels=frozenset({"none", "doing_other_things"}))
    )
    for i in range(3):
        estimate = spotter.push(
            _prediction(gesture="doing_other_things"), timestamp_ms=i * 10, frame_id=i
        )
    assert estimate.phase == GesturePhase.IDLE
    assert spotter.state == GesturePhase.IDLE


def test_onset_rejected_for_low_gesture_confidence() -> None:
    spotter = GestureSpotter(_config(min_consecutive_frames=2, min_onset_gesture_confidence=0.8))
    for i in range(3):
        estimate = spotter.push(
            _prediction(gesture_confidence=0.3), timestamp_ms=i * 10, frame_id=i
        )
    assert estimate.phase == GesturePhase.IDLE


def test_onset_confirmed_once_confidence_recovers() -> None:
    """낮은 확신(비활성)이 이어지다 확신이 회복되면, 활성 스트릭이 임계에 도달할 때 확정된다."""
    spotter = GestureSpotter(_config(min_consecutive_frames=2, min_onset_gesture_confidence=0.8))
    spotter.push(_prediction(gesture_confidence=0.3), timestamp_ms=0, frame_id=0)  # 비활성
    spotter.push(_prediction(gesture_confidence=0.95), timestamp_ms=10, frame_id=1)  # 활성 1
    estimate = spotter.push(_prediction(gesture_confidence=0.95), timestamp_ms=20, frame_id=2)  # 활성 2
    assert estimate.phase == GesturePhase.ONSET
    assert spotter.state == GesturePhase.ONSET


# --- lifecycle: 활성 지속 → ONSET → ACTIVE ---


def test_sustained_activity_advances_onset_to_active() -> None:
    spotter = GestureSpotter(_config(min_consecutive_frames=1))
    spotter.push(_prediction(), timestamp_ms=0, frame_id=0)  # ONSET
    assert spotter.state == GesturePhase.ONSET
    estimate = spotter.push(_prediction(), timestamp_ms=10, frame_id=1)  # ACTIVE
    assert estimate.phase == GesturePhase.ACTIVE
    assert spotter.state == GesturePhase.ACTIVE


# --- 전체 사이클: 활성 후 비활성 → ENDING 정확히 한 번 ---


def test_full_cycle_emits_ending_exactly_once() -> None:
    spotter = GestureSpotter(_config(min_consecutive_frames=1))
    # 활성 2프레임(ONSET→ACTIVE) 뒤 비활성이 이어지면 ENDING이 한 번 나오고 IDLE로 리셋된다.
    scripted = [_prediction(), _prediction(), _inactive(), _inactive(), _inactive()]
    seen_phases = []
    for i, pred in enumerate(scripted):
        estimate = spotter.push(pred, timestamp_ms=i * 10, frame_id=i)
        seen_phases.append(estimate.phase)

    assert seen_phases.count(GesturePhase.ENDING) == 1
    assert seen_phases[-1] == GesturePhase.IDLE
    assert spotter.state == GesturePhase.IDLE


def test_locked_gesture_label_persists_through_event() -> None:
    """시작에서 lock된 label은 도중에 raw 예측이 바뀌어도 이벤트 내내 유지된다."""
    spotter = GestureSpotter(_config(min_consecutive_frames=1))
    spotter.push(_prediction(gesture="slide_two_fingers_left"), timestamp_ms=0, frame_id=0)
    estimate = spotter.push(_prediction(gesture="slide_two_fingers_right"), timestamp_ms=10, frame_id=1)
    assert estimate.gesture == "slide_two_fingers_left"
    estimate = spotter.push(_prediction(gesture="rotate_clockwise"), timestamp_ms=20, frame_id=2)
    assert estimate.gesture == "slide_two_fingers_left"


# --- 추적 손실: 안전하게 포기(이벤트 없이) ---


def test_tracking_loss_aborts_active_gesture() -> None:
    spotter = GestureSpotter(_config(min_consecutive_frames=1))
    _drive_to_active(spotter, n=2)
    assert spotter.state == GesturePhase.ACTIVE

    estimate = spotter.push(None, timestamp_ms=20, frame_id=2)
    assert estimate.phase == GesturePhase.IDLE
    assert estimate.gesture_confidence == 0.0
    assert estimate.uncertainty == 1.0
    assert spotter.state == GesturePhase.IDLE
    assert not spotter.is_tracking_gesture


def test_inactive_during_onset_aborts_and_clears_lock() -> None:
    """ONSET 직후 활성이 끊기면(비활성 확정) IDLE로 되돌아가고 lock이 비워져야 한다."""
    spotter = GestureSpotter(_config(min_consecutive_frames=1))
    spotter.push(_prediction(gesture="slide_two_fingers_left"), timestamp_ms=0, frame_id=0)
    assert spotter.state == GesturePhase.ONSET
    spotter.push(_inactive(), timestamp_ms=10, frame_id=1)
    assert spotter.state == GesturePhase.IDLE

    # 다음 제스처가 이전 label을 이어받지 않는다.
    estimate = spotter.push(_prediction(gesture="slide_two_fingers_up"), timestamp_ms=20, frame_id=2)
    assert estimate.gesture == "slide_two_fingers_up"


# --- pointer 모듈 연동 신호 ---


def test_is_tracking_gesture_reflects_non_idle_state() -> None:
    spotter = GestureSpotter(_config(min_consecutive_frames=1))
    assert not spotter.is_tracking_gesture
    spotter.push(_prediction(), timestamp_ms=0, frame_id=0)
    assert spotter.is_tracking_gesture


def test_reset_clears_all_state() -> None:
    spotter = GestureSpotter(_config(min_consecutive_frames=1))
    spotter.push(_prediction(gesture="slide_two_fingers_left"), timestamp_ms=0, frame_id=0)
    spotter.reset()
    assert spotter.state == GesturePhase.IDLE
    assert not spotter.is_tracking_gesture
    estimate = spotter.push(_prediction(gesture="slide_two_fingers_up"), timestamp_ms=10, frame_id=1)
    assert estimate.gesture == "slide_two_fingers_up"


# --- 타이밍 계약: timestamp_ms·frame_id를 그대로 전달 ---


def test_timestamp_and_frame_id_pass_through_unchanged() -> None:
    spotter = GestureSpotter()
    estimate = spotter.push(_inactive(), timestamp_ms=123456, frame_id=42)
    assert estimate.timestamp_ms == 123456
    assert estimate.frame_id == 42
