"""시연 배선의 순수 코어 — Qt·카메라·네트워크 없이 판정 경로 전체를 검증한다.

`DemoBridge`가 흡수하는 세 가지 시연 전용 관심사(기기 id 치환·타깃 고정 폴백·
임계값 프리셋)와, "왜 실행되지 않았는가"를 삼키지 않는다는 계약을 고정한다.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from jarvis.contracts.messages import GestureEstimate, GesturePhase, TargetEstimate
from jarvis.monitoring.demo_bridge import (
    BLOCK_REASONS,
    BULB_DEVICE_ID,
    LAPTOP_DEVICE_ID,
    PRESET_LOOSE,
    PRESET_STRICT,
    UNKNOWN_TARGET,
    DemoBridge,
    DeviceMappingStore,
    describe_decision,
)


def _target(target: str, timestamp_ms: int, *, probability: float = 0.95) -> TargetEstimate:
    return TargetEstimate(
        timestamp_ms=timestamp_ms,
        frame_id=timestamp_ms // 10,
        target=target,
        probability=probability,
        second_best_probability=0.0,
        stability=0.95,
    )


def _gesture(
    phase: GesturePhase,
    timestamp_ms: int,
    *,
    gesture: str = "slide_two_fingers_down",
    confidence: float = 0.95,
) -> GestureEstimate:
    return GestureEstimate(
        timestamp_ms=timestamp_ms,
        frame_id=timestamp_ms // 10,
        gesture=gesture,
        gesture_confidence=confidence,
        phase=phase,
        phase_confidence=0.9,
        uncertainty=0.05,
    )


def _store(tmp_path: Path, mapping: dict[str, str] | None = None) -> DeviceMappingStore:
    store = DeviceMappingStore(tmp_path / "demo_device_map.json")
    for target_id, device_id in (mapping or {}).items():
        store.set(target_id, device_id)
    return store


# --- 기기 id 치환: 이게 없으면 모든 커밋이 NO_MAPPING으로 죽는다 ---


def test_mapped_target_resolves_to_runtime_device(tmp_path: Path) -> None:
    bridge = DemoBridge(mapping_store=_store(tmp_path, {"target_001": BULB_DEVICE_ID}))
    assert bridge.resolve_target("target_001") == BULB_DEVICE_ID


def test_unmapped_target_resolves_to_unknown(tmp_path: Path) -> None:
    """매핑 없는 물체가 우연히 기기로 해석되면 안 된다 — UNKNOWN이면 lock 자체가 안 걸린다."""
    bridge = DemoBridge(mapping_store=_store(tmp_path))
    assert bridge.resolve_target("target_042") == UNKNOWN_TARGET


def test_unknown_target_passes_through(tmp_path: Path) -> None:
    bridge = DemoBridge(mapping_store=_store(tmp_path, {"target_001": LAPTOP_DEVICE_ID}))
    assert bridge.resolve_target(UNKNOWN_TARGET) == UNKNOWN_TARGET


def test_bridge_without_store_never_resolves() -> None:
    bridge = DemoBridge(mapping_store=None)
    assert bridge.resolve_target("target_001") == UNKNOWN_TARGET


def test_unmapped_target_never_locks(tmp_path: Path) -> None:
    bridge = DemoBridge(mapping_store=_store(tmp_path))
    for ms in range(0, 5000, 100):
        bridge.push_target(_target("target_001", ms))
    assert bridge.locked_device is None
    assert bridge.candidate_device is None


def test_mapped_target_locks_after_dwell(tmp_path: Path) -> None:
    bridge = DemoBridge(mapping_store=_store(tmp_path, {"target_001": BULB_DEVICE_ID}))
    for ms in range(0, 2000, 100):
        bridge.push_target(_target("target_001", ms))
    assert bridge.locked_device == BULB_DEVICE_ID


def test_returning_to_laptop_needs_confirmation_signal(tmp_path: Path) -> None:
    """전구→노트북 복귀는 OK사인(확인 신호) 없이는 확정되지 않는다(사용자 지시 2026-07-22).

    노트북은 게이트된 target이라, dwell을 채워도 `note_confirmation_signal`이 최근에
    호출된 적 없으면 전구 lock이 그대로 유지된다.
    """
    bridge = DemoBridge(
        mapping_store=_store(
            tmp_path, {"target_001": BULB_DEVICE_ID, "target_002": LAPTOP_DEVICE_ID}
        )
    )
    for ms in range(0, 1000, 100):
        bridge.push_target(_target("target_001", ms))
    assert bridge.locked_device == BULB_DEVICE_ID

    for ms in range(1000, 2500, 100):  # dwell(800ms)을 한참 넘김
        bridge.push_target(_target("target_002", ms))
    assert bridge.locked_device == BULB_DEVICE_ID  # 확인 신호 없이는 그대로
    assert bridge.candidate_device == LAPTOP_DEVICE_ID

    bridge.note_confirmation_signal(2500)
    bridge.push_target(_target("target_002", 2600))
    assert bridge.locked_device == LAPTOP_DEVICE_ID


# --- 매핑 영속화 ---


def test_mapping_store_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "demo_device_map.json"
    store = DeviceMappingStore(path)
    store.set("target_001", BULB_DEVICE_ID)
    assert DeviceMappingStore(path).get("target_001") == BULB_DEVICE_ID


def test_mapping_store_clears_with_none(tmp_path: Path) -> None:
    path = tmp_path / "demo_device_map.json"
    store = DeviceMappingStore(path)
    store.set("target_001", LAPTOP_DEVICE_ID)
    store.set("target_001", None)
    assert DeviceMappingStore(path).get("target_001") is None
    assert DeviceMappingStore(path).has_selection("target_001") is True


def test_mapping_store_default_does_not_override_explicit_disconnect(tmp_path: Path) -> None:
    path = tmp_path / "demo_device_map.json"
    store = DeviceMappingStore(path)
    store.set("target_001", None)
    store.set_default("target_001", LAPTOP_DEVICE_ID)
    assert store.get("target_001") is None


def test_mapping_store_remove_forgets_deleted_target(tmp_path: Path) -> None:
    store = DeviceMappingStore(tmp_path / "demo_device_map.json")
    store.set("target_001", LAPTOP_DEVICE_ID)
    store.remove("target_001")
    assert store.has_selection("target_001") is False


def test_mapping_store_rejects_unknown_device(tmp_path: Path) -> None:
    store = DeviceMappingStore(tmp_path / "demo_device_map.json")
    with pytest.raises(ValueError):
        store.set("target_001", "toaster")


def test_mapping_store_survives_corrupt_file(tmp_path: Path) -> None:
    """손상 파일이 시연을 죽이지 않는다 — 빈 매핑으로 시작한다."""
    path = tmp_path / "demo_device_map.json"
    path.write_text("{not json", encoding="utf-8")
    assert DeviceMappingStore(path).mapping == {}


# --- 폴백(타깃 고정): gaze가 없어도 시연이 성립해야 한다 ---


def test_fallback_locks_without_any_gaze_stream() -> None:
    """`--no-gaze`로 실행해도 제스처 프레임만으로 lock이 잡혀야 한다."""
    bridge = DemoBridge(mapping_store=None)
    bridge.set_fallback(LAPTOP_DEVICE_ID)
    for ms in range(0, 2000, 100):
        bridge.push_gesture(_gesture(GesturePhase.IDLE, ms))
    assert bridge.locked_device == LAPTOP_DEVICE_ID


def test_fallback_overrides_real_gaze(tmp_path: Path) -> None:
    """고정이 켜져 있으면 시선이 다른 기기를 가리켜도 고정 기기로 간다."""
    bridge = DemoBridge(mapping_store=_store(tmp_path, {"target_001": BULB_DEVICE_ID}))
    bridge.set_fallback(LAPTOP_DEVICE_ID)
    for ms in range(0, 2000, 100):
        bridge.push_target(_target("target_001", ms))
    assert bridge.locked_device == LAPTOP_DEVICE_ID


def test_changing_fallback_resets_lock() -> None:
    bridge = DemoBridge(mapping_store=None)
    bridge.set_fallback(LAPTOP_DEVICE_ID)
    for ms in range(0, 2000, 100):
        bridge.push_gesture(_gesture(GesturePhase.IDLE, ms))
    assert bridge.locked_device == LAPTOP_DEVICE_ID
    bridge.set_fallback(BULB_DEVICE_ID)
    assert bridge.locked_device is None  # 새 기기로 dwell을 다시 쌓는다


def test_fallback_rejects_unknown_device() -> None:
    with pytest.raises(ValueError):
        DemoBridge(mapping_store=None).set_fallback("toaster")


# --- pose 경로 중재 ---


def test_pose_not_suppressed_without_lock() -> None:
    assert DemoBridge(mapping_store=None).should_suppress_pose is False


def test_pose_not_suppressed_when_locked_to_laptop() -> None:
    """노트북에 lock된 동안에는 커서·클릭이 그대로 살아 있어야 한다."""
    bridge = DemoBridge(mapping_store=None)
    bridge.set_fallback(LAPTOP_DEVICE_ID)
    for ms in range(0, 2000, 100):
        bridge.push_gesture(_gesture(GesturePhase.IDLE, ms))
    assert bridge.locked_device == LAPTOP_DEVICE_ID
    assert bridge.should_suppress_pose is False


def test_pose_suppressed_when_locked_to_bulb() -> None:
    bridge = DemoBridge(mapping_store=None)
    bridge.set_fallback(BULB_DEVICE_ID)
    for ms in range(0, 2000, 100):
        bridge.push_gesture(_gesture(GesturePhase.IDLE, ms))
    assert bridge.locked_device == BULB_DEVICE_ID
    assert bridge.should_suppress_pose is True


# --- 판정 흐름 ---


def _drive_to_lock(bridge: DemoBridge, until_ms: int = 2000) -> None:
    for ms in range(0, until_ms, 100):
        bridge.push_gesture(_gesture(GesturePhase.IDLE, ms))


def test_decision_only_at_ending() -> None:
    bridge = DemoBridge(mapping_store=None)
    bridge.set_fallback(LAPTOP_DEVICE_ID)
    _drive_to_lock(bridge)
    assert bridge.push_gesture(_gesture(GesturePhase.ONSET, 2000)) is None
    assert bridge.push_gesture(_gesture(GesturePhase.ACTIVE, 2100)) is None
    decision = bridge.push_gesture(_gesture(GesturePhase.ENDING, 2200))
    assert decision is not None


def test_full_cycle_commits_with_fallback() -> None:
    """폴백 + 확신 있는 제스처면 실제로 커밋까지 간다(배선이 살아 있다는 증거)."""
    bridge = DemoBridge(mapping_store=None)
    bridge.set_fallback(LAPTOP_DEVICE_ID)
    _drive_to_lock(bridge)
    bridge.push_gesture(_gesture(GesturePhase.ONSET, 2000))
    decision = bridge.push_gesture(_gesture(GesturePhase.ENDING, 2200))
    assert decision is not None
    assert decision.committed is True
    assert decision.target == LAPTOP_DEVICE_ID
    assert decision.intent_id is not None


def test_gesture_without_lock_is_blocked() -> None:
    """시나리오 6~7: 아무 기기도 보지 않고 제스처 → 명령이 나가지 않는다."""
    bridge = DemoBridge(mapping_store=None)
    bridge.push_gesture(_gesture(GesturePhase.ONSET, 0))
    decision = bridge.push_gesture(_gesture(GesturePhase.ENDING, 200))
    assert decision is not None
    assert decision.committed is False
    assert BLOCK_REASONS[decision.reason] == "바라보는 기기 없음"


def test_execution_is_off_by_default() -> None:
    """안전 기본값은 비실행 — 켜기 전에는 어떤 커밋도 기기로 나가지 않는다."""
    assert DemoBridge(mapping_store=None).execution_enabled is False


# --- 프리셋 ---


def test_strict_preset_needs_longer_dwell() -> None:
    """느슨 프리셋에서 잡히던 lock이 빡빡 프리셋에서는 같은 시간에 안 잡혀야 한다."""
    loose = DemoBridge(mapping_store=None, preset=PRESET_LOOSE)
    strict = DemoBridge(mapping_store=None, preset=PRESET_STRICT)
    for bridge in (loose, strict):
        bridge.set_fallback(LAPTOP_DEVICE_ID)
        _drive_to_lock(bridge, until_ms=1200)
    assert loose.locked_device == LAPTOP_DEVICE_ID
    assert strict.locked_device is None


def test_reconfigure_resets_lock() -> None:
    bridge = DemoBridge(mapping_store=None, preset=PRESET_LOOSE)
    bridge.set_fallback(LAPTOP_DEVICE_ID)
    _drive_to_lock(bridge)
    assert bridge.locked_device == LAPTOP_DEVICE_ID
    bridge.reconfigure(PRESET_STRICT)
    assert bridge.locked_device is None
    assert bridge.preset is PRESET_STRICT


def test_reconfigure_keeps_fallback() -> None:
    bridge = DemoBridge(mapping_store=None, preset=PRESET_LOOSE)
    bridge.set_fallback(BULB_DEVICE_ID)
    bridge.reconfigure(PRESET_STRICT)
    assert bridge.fallback_device == BULB_DEVICE_ID


# --- 차단 사유를 삼키지 않는다 ---

_REASON_PATTERNS = (
    re.compile(r'reason="([^"]+)"'),
    re.compile(r'AlignmentResult\(\s*False,\s*"([^"]+)"'),
    re.compile(r'_reject\(estimate,\s*"([^"]+)"'),
)


def _source_reasons() -> set[str]:
    """fusion.py·alignment.py가 실제로 내는 사유 문자열을 소스에서 긁어온다.

    목록을 테스트에 손으로 베껴 두면 새 사유가 추가돼도 조용히 통과한다. 소스를
    직접 읽어 "새 사유를 추가했는데 화면 문구를 안 만들었다"를 잡는다.
    """
    import jarvis.gesture_fusion.alignment as alignment_mod
    import jarvis.gesture_fusion.fusion as fusion_mod

    found: set[str] = set()
    for module in (fusion_mod, alignment_mod):
        source = Path(module.__file__).read_text(encoding="utf-8")  # type: ignore[arg-type]
        for pattern in _REASON_PATTERNS:
            found.update(pattern.findall(source))
    return found


def test_every_fusion_reason_has_a_korean_label() -> None:
    missing = _source_reasons() - set(BLOCK_REASONS)
    assert not missing, f"화면 문구가 없는 차단 사유: {sorted(missing)}"


def test_source_scan_actually_finds_reasons() -> None:
    """위 테스트가 빈 집합을 비교하며 늘 통과하는 것을 막는 가드."""
    reasons = _source_reasons()
    assert "target not locked" in reasons
    assert "cooldown active" in reasons


def test_describe_decision_reports_block_reason() -> None:
    bridge = DemoBridge(mapping_store=None)
    bridge.push_gesture(_gesture(GesturePhase.ONSET, 0))
    decision = bridge.push_gesture(_gesture(GesturePhase.ENDING, 200))
    assert decision is not None
    line = describe_decision(decision)
    assert "차단" in line
    assert "바라보는 기기 없음" in line


def test_describe_decision_reports_commit() -> None:
    bridge = DemoBridge(mapping_store=None)
    bridge.set_fallback(LAPTOP_DEVICE_ID)
    _drive_to_lock(bridge)
    bridge.push_gesture(_gesture(GesturePhase.ONSET, 2000))
    decision = bridge.push_gesture(_gesture(GesturePhase.ENDING, 2200))
    assert decision is not None
    assert describe_decision(decision).startswith("커밋")


def test_unknown_reason_is_passed_through_not_swallowed() -> None:
    """모르는 사유가 와도 빈 문자열이 아니라 원문이 보여야 한다."""
    from jarvis.gesture_fusion.fusion import CommitDecision

    decision = CommitDecision(
        committed=False,
        reason="some brand new reason",
        target=None,
        gesture="stop_sign",
        score=None,
        timestamp_ms=0,
        frame_id=0,
    )
    assert "some brand new reason" in describe_decision(decision)
