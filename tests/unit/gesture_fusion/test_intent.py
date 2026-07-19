"""Intent 조립·출력을 검증한다 (README 9장 Intent 계약, gesture→capability 매핑)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.contracts.messages import GestureEstimate, GesturePhase, TargetEstimate
from jarvis.gesture_fusion.alignment import AlignmentConfig
from jarvis.gesture_fusion.fusion import CommitDecision, FusionConfig, FusionEngine, FusionScore
from jarvis.gesture_fusion.intent import (
    CapabilityAction,
    GestureCapabilityMap,
    IntentConfig,
    assemble_intent,
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
    bulb_action = capability_map.lookup("room.bulb", "swipe_down")
    assert bulb_action is not None
    assert bulb_action.capability == "brightness"
    assert bulb_action.operation == "decrement"


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
