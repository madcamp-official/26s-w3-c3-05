"""Intent 조립·출력을 검증한다 (README 9장 Intent 계약, gesture→capability 매핑)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.contracts.messages import GestureEstimate, GesturePhase, TargetEstimate
from jarvis.gesture_fusion.alignment import AlignmentConfig
from jarvis.gesture_fusion.fusion import CommitDecision, FusionConfig, FusionEngine, FusionScore
from jarvis.gesture_fusion.model_protocol import (
    DEFAULT_BACKGROUND_LABELS,
    DEFAULT_GESTURE_LABELS,
)
from jarvis.gesture_fusion.intent import (
    CapabilityAction,
    GestureCapabilityMap,
    IntentConfig,
    assemble_intent,
    validate_capability_map,
)

_REPO_MAP_PATH = Path(__file__).resolve().parents[3] / "configs" / "gesture_capability_map.json"


def _committed_decision(
    *,
    target: str = "room.bulb",
    gesture: str = "swipe_down",
    intent_id: str = "intent-42",
    target_confidence: float = 0.9,
    gesture_confidence: float = 0.85,
) -> CommitDecision:
    score = FusionScore(
        target_confidence=target_confidence,
        gesture_confidence=gesture_confidence,
        gaze_stability=0.9,
        uncertainty=0.1,
        value=0.6,
    )
    return CommitDecision(
        committed=True,
        reason="committed",
        target=target,
        gesture=gesture,
        score=score,
        timestamp_ms=300,
        frame_id=42,
        intent_id=intent_id,
    )


def _simple_map() -> GestureCapabilityMap:
    return GestureCapabilityMap(
        {
            "room.bulb": {
                "swipe_down": CapabilityAction("brightness", "decrement", 10),
                "rotate_clockwise": CapabilityAction("color_temperature", "increment", 100),
            },
            "laptop": {
                "swipe_down": CapabilityAction("scroll", "decrement", 3),
            },
        }
    )


# --- CapabilityAction ---


def test_capability_action_rejects_empty_capability() -> None:
    with pytest.raises(ValueError, match="capability"):
        CapabilityAction("", "set", 1)


def test_capability_action_rejects_non_finite_value() -> None:
    with pytest.raises(ValueError, match="finite"):
        CapabilityAction("brightness", "set", float("nan"))


def test_capability_action_allows_bool_value() -> None:
    action = CapabilityAction("power", "set", True)
    assert action.value is True


# --- GestureCapabilityMap ---


def test_lookup_returns_action_for_known_pair() -> None:
    capability_map = _simple_map()
    action = capability_map.lookup("room.bulb", "swipe_down")
    assert action == CapabilityAction("brightness", "decrement", 10)


def test_lookup_returns_none_for_unknown_gesture() -> None:
    capability_map = _simple_map()
    assert capability_map.lookup("room.bulb", "swipe_left") is None


def test_lookup_returns_none_for_unknown_device() -> None:
    capability_map = _simple_map()
    assert capability_map.lookup("unknown.device", "swipe_down") is None


def test_same_gesture_maps_differently_per_device() -> None:
    """같은 제스처(swipe_down)도 기기에 따라 다른 capability로 매핑된다."""
    capability_map = _simple_map()
    bulb_action = capability_map.lookup("room.bulb", "swipe_down")
    laptop_action = capability_map.lookup("laptop", "swipe_down")
    assert bulb_action is not None and laptop_action is not None
    assert bulb_action.capability != laptop_action.capability


def test_from_json_loads_repo_config() -> None:
    """configs/gesture_capability_map.json이 실제로 파싱 가능하고 필수 매핑을 담고 있다."""
    capability_map = GestureCapabilityMap.from_json(_REPO_MAP_PATH)
    bulb_action = capability_map.lookup("room.bulb", "slide_two_fingers_down")
    assert bulb_action is not None
    assert bulb_action.capability == "brightness"
    assert bulb_action.operation == "decrement"


def test_repo_config_is_consistent_with_the_training_label_set() -> None:
    """실제 저장소 config가 현재 라벨 집합과 맞는지 검사한다 — 이 커밋의 회귀 가드.

    `validate_capability_map`(8b65e6c)은 있었지만 저장소 config에 대고 돌리는 곳이
    없어서, 라벨에서 swipe 4종이 빠졌을 때 매핑 6개가 죽은 채로 남아도 아무도
    실패하지 않았다(scroll·window_switch·brightness 세 capability가 어떤 제스처로도
    도달 불가). 이 테스트가 그 연결을 만든다 — 라벨을 바꾸고 맵을 안 고치면 CI에서
    바로 걸린다.

    배경 label은 실제 배경 집합(`DEFAULT_BACKGROUND_LABELS`)을 넘겨 매핑 요구에서
    제외한다. 기본값 `{"none"}`만 쓰면 drumming_fingers·doing_other_things에도
    기기 동작을 요구하게 되는데, 이들은 "동작 없음"이 곧 의도다.
    """
    validate_capability_map(
        GestureCapabilityMap.from_json(_REPO_MAP_PATH),
        gesture_labels=DEFAULT_GESTURE_LABELS,
        background_labels=DEFAULT_BACKGROUND_LABELS,
    )


def test_every_actionable_gesture_reaches_at_least_one_device() -> None:
    """액션 가능한 7개 제스처가 전부 어딘가에 매핑돼 있어야 한다(문서화 겸 가드)."""
    registered = GestureCapabilityMap.from_json(_REPO_MAP_PATH).registered_gestures()
    actionable = set(DEFAULT_GESTURE_LABELS) - DEFAULT_BACKGROUND_LABELS
    assert actionable == {
        "rotate_clockwise",
        "rotate_counter_clockwise",
        "slide_two_fingers_up",
        "slide_two_fingers_down",
        "slide_two_fingers_left",
        "slide_two_fingers_right",
        "stop_sign",  # room.bulb 전원 토글 (2026-07-20 추가)
    }
    assert actionable <= registered


def test_registered_gestures_collects_across_devices() -> None:
    capability_map = _simple_map()
    assert capability_map.registered_gestures() == {"swipe_down", "rotate_clockwise"}


# --- validate_capability_map ---


def test_validate_capability_map_passes_when_consistent() -> None:
    capability_map = _simple_map()
    validate_capability_map(
        capability_map,
        gesture_labels=("none", "swipe_down", "rotate_clockwise"),
        background_labels=frozenset({"none"}),
    )  # 예외 없이 통과해야 한다.


def test_validate_capability_map_rejects_unknown_gesture() -> None:
    """맵이 참조하는 gesture가 현재 라벨 집합에 없으면(라벨 삭제 후 맵 미갱신) 실패해야 한다.

    2026-07-20 발견: gesture_capability_map.json이 라벨에서 빠진 swipe_up 등을
    그대로 참조해도 이를 잡아내는 검증이 전혀 없었다.
    """
    capability_map = _simple_map()  # "swipe_down"을 참조
    with pytest.raises(ValueError, match="swipe_down"):
        validate_capability_map(
            capability_map,
            gesture_labels=("none", "rotate_clockwise"),  # swipe_down 없음
            background_labels=frozenset({"none"}),
        )


def test_validate_capability_map_rejects_unmapped_non_background_label() -> None:
    """배경이 아닌 라벨에 매핑이 하나도 없으면 실패해야 한다(신규 라벨 추가 후 매핑 누락 방지)."""
    capability_map = _simple_map()
    with pytest.raises(ValueError, match="drumming_fingers"):
        validate_capability_map(
            capability_map,
            gesture_labels=("none", "swipe_down", "rotate_clockwise", "drumming_fingers"),
            background_labels=frozenset({"none"}),
        )


def test_validate_capability_map_exempts_background_labels_from_mapping_requirement() -> None:
    """배경 label은 매핑이 없어도(=동작 없음이 의도) 통과해야 한다."""
    capability_map = _simple_map()
    validate_capability_map(
        capability_map,
        gesture_labels=("none", "doing_other_things", "swipe_down", "rotate_clockwise"),
        background_labels=frozenset({"none", "doing_other_things"}),
    )  # 예외 없이 통과해야 한다.


def test_from_json_ignores_underscore_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "map.json"
    config_path.write_text(
        json.dumps(
            {
                "_comment": "should be ignored",
                "devices": {"laptop": {"swipe_up": {"capability": "scroll", "operation": "increment", "value": 1}}},
            }
        ),
        encoding="utf-8",
    )
    capability_map = GestureCapabilityMap.from_json(config_path)
    assert capability_map.lookup("laptop", "swipe_up") is not None


# --- assemble_intent ---


def test_assemble_intent_success() -> None:
    decision = _committed_decision()
    intent = assemble_intent(decision, _simple_map())
    assert intent is not None
    assert intent.intent_id == "intent-42"
    assert intent.target == "room.bulb"
    assert intent.gesture == "swipe_down"
    assert intent.capability == "brightness"
    assert intent.operation == "decrement"
    assert intent.value == 10
    assert intent.target_confidence == 0.9
    assert intent.gesture_confidence == 0.85


def test_assemble_intent_uses_configured_ttl() -> None:
    decision = _committed_decision()
    intent = assemble_intent(decision, _simple_map(), IntentConfig(default_expires_in_ms=2000))
    assert intent is not None
    assert intent.expires_in_ms == 2000


def test_assemble_intent_returns_none_when_not_committed() -> None:
    decision = _committed_decision()
    rejected = CommitDecision(
        committed=False,
        reason="cooldown active",
        target=decision.target,
        gesture=decision.gesture,
        score=None,
        timestamp_ms=decision.timestamp_ms,
        frame_id=decision.frame_id,
        intent_id=None,
    )
    assert assemble_intent(rejected, _simple_map()) is None


def test_assemble_intent_returns_none_for_unmapped_gesture() -> None:
    decision = _committed_decision(gesture="swipe_left")
    assert assemble_intent(decision, _simple_map()) is None


def test_assemble_intent_returns_none_for_unmapped_device() -> None:
    decision = _committed_decision(target="unregistered.device")
    assert assemble_intent(decision, _simple_map()) is None


# --- end-to-end: FusionEngine 커밋 → Intent 조립 ---


def _target(timestamp_ms: int, *, target: str = "room.bulb") -> TargetEstimate:
    return TargetEstimate(
        timestamp_ms=timestamp_ms,
        frame_id=0,
        target=target,
        probability=0.9,
        second_best_probability=0.05,
        stability=0.9,
    )


def _gesture(timestamp_ms: int, phase: GesturePhase, *, frame_id: int = 1) -> GestureEstimate:
    return GestureEstimate(
        timestamp_ms=timestamp_ms,
        frame_id=frame_id,
        gesture="swipe_down",
        gesture_confidence=0.9,
        phase=phase,
        phase_confidence=0.9,
        uncertainty=0.05,
    )


def test_end_to_end_commit_to_intent() -> None:
    engine = FusionEngine(
        FusionConfig(commit_threshold=0.3, min_target_confidence=0.5, min_gesture_confidence=0.5),
        AlignmentConfig(target_dwell_ms=0, target_lock_ttl_ms=2000),
    )
    engine.push_target(_target(0))
    engine.push_target(_target(0))
    engine.push_gesture(_gesture(100, GesturePhase.ONSET))
    decision = engine.push_gesture(_gesture(300, GesturePhase.ENDING))
    assert decision is not None and decision.committed

    intent = assemble_intent(decision, _simple_map())
    assert intent is not None
    assert intent.capability == "brightness"
    assert intent.intent_id == decision.intent_id
