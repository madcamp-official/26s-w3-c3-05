"""조기 발사(ACTIVE 중 커밋)와 비대칭 디바운스 검증.

지키는 불변식 두 가지:
1. 한 제스처 이벤트는 조기 발사돼도 **최대 한 번만** 실행된다(ENDING 중복 억제).
2. 허용 목록에 없는 제스처(예: power toggle 계열)는 조기 발사되지 않는다.
"""

from __future__ import annotations

from jarvis.contracts.messages import GestureEstimate, GesturePhase, TargetEstimate
from jarvis.gesture_fusion.alignment import AlignmentConfig
from jarvis.gesture_fusion.fusion import FusionConfig, FusionEngine
from jarvis.gesture_fusion.model_protocol import ModelPrediction
from jarvis.gesture_fusion.spotting import GestureSpotter, SpotterConfig

EARLY_GESTURES = frozenset({"rotate_clockwise", "slide_two_fingers_up"})


def _config(**overrides: object) -> FusionConfig:
    base: dict[str, object] = {
        "commit_threshold": 0.2,
        "min_target_confidence": 0.5,
        "min_gesture_confidence": 0.3,
        "cooldown_ms": 0,
        "early_dispatch_frames": 3,
        "early_dispatch_min_confidence": 0.8,
        "early_dispatch_gestures": EARLY_GESTURES,
    }
    base.update(overrides)
    return FusionConfig(**base)  # type: ignore[arg-type]


def _engine(config: FusionConfig | None = None) -> FusionEngine:
    engine = FusionEngine(
        config if config is not None else _config(),
        AlignmentConfig(target_dwell_ms=0, target_lock_ttl_ms=5000),
    )
    for _ in range(2):
        engine.push_target(
            TargetEstimate(
                timestamp_ms=0,
                frame_id=0,
                target="room.bulb",
                probability=0.95,
                second_best_probability=0.02,
                stability=0.95,
            )
        )
    return engine


def _gesture(
    timestamp_ms: int, phase: GesturePhase, gesture: str, confidence: float = 0.95
) -> GestureEstimate:
    return GestureEstimate(
        timestamp_ms=timestamp_ms,
        frame_id=timestamp_ms,
        gesture=gesture,
        gesture_confidence=confidence,
        phase=phase,
        phase_confidence=0.9,
        uncertainty=0.05,
    )


def _run(engine: FusionEngine, gesture: str, active_frames: int, confidence: float = 0.95):
    """ONSET → ACTIVE×N → ENDING을 흘리고 나온 결정들을 모은다."""
    decisions = []
    decisions.append(engine.push_gesture(_gesture(100, GesturePhase.ONSET, gesture, confidence)))
    for i in range(active_frames):
        decisions.append(
            engine.push_gesture(
                _gesture(200 + i * 80, GesturePhase.ACTIVE, gesture, confidence)
            )
        )
    decisions.append(
        engine.push_gesture(
            _gesture(200 + active_frames * 80 + 80, GesturePhase.ENDING, gesture, confidence)
        )
    )
    return [d for d in decisions if d is not None]


# --- 조기 발사 ---------------------------------------------------------------


def test_fires_early_after_enough_active_frames() -> None:
    decisions = _run(_engine(), "rotate_clockwise", active_frames=4)
    committed = [d for d in decisions if d.committed]
    assert len(committed) == 1
    assert committed[0].reason == "committed (early)"


def test_event_executes_exactly_once_even_though_ending_follows() -> None:
    """조기 발사한 이벤트는 ENDING에서 다시 실행되지 않는다(핵심 불변식)."""
    decisions = _run(_engine(), "rotate_clockwise", active_frames=5)
    assert sum(1 for d in decisions if d.committed) == 1
    assert any(d.reason == "already dispatched early" for d in decisions)


def test_not_enough_active_frames_falls_back_to_ending_commit() -> None:
    decisions = _run(_engine(), "rotate_clockwise", active_frames=1)
    committed = [d for d in decisions if d.committed]
    assert len(committed) == 1
    assert committed[0].reason == "committed"  # 조기 발사가 아니라 기존 경로


def test_gesture_outside_allowlist_never_fires_early() -> None:
    """power toggle 계열은 멱등이 아니라 조기 발사 대상이 아니다."""
    decisions = _run(_engine(), "stop_sign", active_frames=6)
    committed = [d for d in decisions if d.committed]
    assert len(committed) == 1
    assert committed[0].reason == "committed"  # ENDING에서만


def test_low_confidence_blocks_early_dispatch() -> None:
    # min_gesture_confidence(0.3)는 넘지만 early 임계(0.8)에는 못 미치는 확신도
    decisions = _run(_engine(), "rotate_clockwise", active_frames=6, confidence=0.5)
    committed = [d for d in decisions if d.committed]
    assert len(committed) == 1
    assert committed[0].reason == "committed"


def test_label_flip_resets_the_active_streak() -> None:
    """방향이 뒤집히면 스트릭이 초기화돼 반대쪽을 조기 발사하지 않는다."""
    engine = _engine()
    engine.push_gesture(_gesture(100, GesturePhase.ONSET, "rotate_clockwise"))
    # 흔들리는 구간: 시계/반시계가 번갈아 들어와 어느 쪽도 연속 3프레임을 못 채운다
    for i, label in enumerate(["rotate_clockwise", "rotate_counter_clockwise"] * 3):
        assert engine.push_gesture(_gesture(200 + i * 80, GesturePhase.ACTIVE, label)) is None


def test_disabled_by_default() -> None:
    """early_dispatch_frames 기본값 0 — 설정하지 않으면 기존 동작 그대로."""
    engine = FusionEngine(
        FusionConfig(commit_threshold=0.2, min_target_confidence=0.5, min_gesture_confidence=0.3),
        AlignmentConfig(target_dwell_ms=0, target_lock_ttl_ms=5000),
    )
    for _ in range(2):
        engine.push_target(
            TargetEstimate(
                timestamp_ms=0, frame_id=0, target="room.bulb",
                probability=0.95, second_best_probability=0.02, stability=0.95,
            )
        )
    decisions = _run(engine, "rotate_clockwise", active_frames=8)
    committed = [d for d in decisions if d.committed]
    assert len(committed) == 1
    assert committed[0].reason == "committed"


# --- 비대칭 디바운스 ---------------------------------------------------------


def test_release_debounce_is_shorter_than_entry_by_default() -> None:
    config = SpotterConfig()
    assert config.min_release_frames < config.min_consecutive_frames


def _prediction(gesture: str, confidence: float = 0.95) -> ModelPrediction:
    return ModelPrediction(
        gesture=gesture,
        gesture_confidence=confidence,
        phase=GesturePhase.ACTIVE,  # 스포터는 phase head를 쓰지 않는다(spotting.py 주석)
        phase_confidence=0.9,
        uncertainty=0.05,
    )


def test_ending_fires_one_frame_after_gesture_stops() -> None:
    """비대칭 디바운스의 실제 효과 — 멈춘 뒤 1프레임 만에 ENDING이 나온다.

    이 값이 곧 "동작을 멈춘 뒤 명령이 나가기까지의 고정 지연"이라, 프레임 수가
    그대로 반응성이다(12fps에서 1프레임 ≈ 83ms).
    """
    spotter = GestureSpotter(SpotterConfig(min_consecutive_frames=2, min_release_frames=1))
    phases = []
    for i in range(4):  # 제스처 유지 — ONSET을 거쳐 ACTIVE로
        phases.append(spotter.push(_prediction("rotate_clockwise"), i * 80, i).phase)
    assert phases[-1] == GesturePhase.ACTIVE

    # 손을 멈춘 첫 프레임(배경 label)에서 곧바로 ENDING이 나와야 한다
    ending = spotter.push(_prediction("none"), 400, 4)
    assert ending.phase == GesturePhase.ENDING


def test_slow_release_still_available_when_configured() -> None:
    """종료 디바운스를 2로 두면 예전처럼 한 프레임 더 기다린다(설정으로 되돌릴 수 있다)."""
    spotter = GestureSpotter(SpotterConfig(min_consecutive_frames=2, min_release_frames=2))
    for i in range(4):
        spotter.push(_prediction("rotate_clockwise"), i * 80, i)
    assert spotter.push(_prediction("none"), 400, 4).phase == GesturePhase.ACTIVE
    assert spotter.push(_prediction("none"), 480, 5).phase == GesturePhase.ENDING


def test_early_dispatch_allowlist_matches_capability_map() -> None:
    """허용 목록은 capability map의 **상대 연산 제스처와 정확히 일치**해야 한다.

    조기 발사는 되돌릴 수 없으므로, 매핑에 비멱등 연산(toggle/set)이 새로 생겼는데
    목록이 따라가지 않으면 그 제스처가 조용히 조기 발사된다. 그 드리프트를 여기서 막는다.
    """
    import json
    from pathlib import Path

    from jarvis.monitoring.demo_bridge import EARLY_DISPATCH_GESTURES

    map_path = Path(__file__).resolve().parents[3] / "configs" / "gesture_capability_map.json"
    devices = json.loads(map_path.read_text(encoding="utf-8"))["devices"]

    relative: set[str] = set()
    non_relative: set[str] = set()
    for gestures in devices.values():
        for gesture, action in gestures.items():
            target = relative if action["operation"] in ("increment", "decrement") else non_relative
            target.add(gesture)

    # 어느 기기에서든 비상대 연산에 쓰이면 조기 발사 대상에서 빠져야 한다.
    eligible = relative - non_relative
    assert EARLY_DISPATCH_GESTURES == eligible, (
        "조기 발사 허용 목록이 capability map과 어긋난다 — "
        f"목록에만: {sorted(EARLY_DISPATCH_GESTURES - eligible)}, "
        f"매핑에만: {sorted(eligible - EARLY_DISPATCH_GESTURES)}"
    )
    assert "stop_sign" not in EARLY_DISPATCH_GESTURES  # power toggle은 절대 조기 발사 금지


def test_release_frames_must_be_positive() -> None:
    try:
        SpotterConfig(min_release_frames=0)
    except ValueError as exc:
        assert "min_release_frames" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("0은 거부돼야 한다")
