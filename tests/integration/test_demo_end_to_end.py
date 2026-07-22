"""시연 배선 end-to-end: 실시간 스트림 → DemoBridge → Fusion → Intent → adapter.

`test_fusion_to_command.py`가 `CommitDecision` 이후를 지킨다면, 이 테스트는 그
**앞단**을 지킨다 — 모니터가 매 프레임 내는 `TargetEstimate`·`GestureEstimate`가
실제로 기기 명령까지 도달하는가. 이 사슬이 끊겨 있던 것이 통합 전의 상태였다
(FusionEngine을 생성하는 프로덕션 코드가 없었다).

특히 **기기 id 치환**을 회귀로 고정한다: 물체 등록은 `target_001`을 발급하는데
capability 매핑은 `room.bulb`를 키로 쓰므로, 치환이 빠지면 모든 커밋이 조용히
NO_MAPPING으로 죽는다. 카메라·모델·네트워크는 쓰지 않는다.
"""

from __future__ import annotations

from pathlib import Path

from jarvis.contracts.messages import Command, GestureEstimate, GesturePhase, TargetEstimate
from jarvis.gesture_fusion.fusion import CommitDecision
from jarvis.monitoring.demo_bridge import (
    BULB_DEVICE_ID,
    LAPTOP_DEVICE_ID,
    DemoBridge,
    DeviceMappingStore,
)
from jarvis.runtime.devices import (
    BULB_ADAPTER,
    LAPTOP_ADAPTER,
    build_default_capability_map,
    build_default_registry,
)
from jarvis.runtime.executor import ExecutionStage, IntentExecutor
from jarvis.runtime_protocol.adapters.base import AdapterResult, AdapterStatus
from jarvis.runtime_protocol.capture.clock import RuntimeClock
from jarvis.runtime_protocol.protocol.capability import DeviceProfile
from jarvis.runtime_protocol.protocol.engine import ProtocolEngine


class FakeAdapter:
    """네트워크·OS 입력 없이 도달한 명령만 기록한다."""

    def __init__(self) -> None:
        self.calls: list[Command] = []

    def execute(self, command: Command, profile: DeviceProfile) -> AdapterResult:
        self.calls.append(command)
        return AdapterResult(AdapterStatus.ACKNOWLEDGED, "ok")


def _target(target_id: str, timestamp_ms: int) -> TargetEstimate:
    return TargetEstimate(
        timestamp_ms=timestamp_ms,
        frame_id=timestamp_ms,
        target=target_id,
        probability=0.95,
        second_best_probability=0.02,
        stability=0.95,
    )


def _gesture(phase: GesturePhase, timestamp_ms: int, gesture: str) -> GestureEstimate:
    return GestureEstimate(
        timestamp_ms=timestamp_ms,
        frame_id=timestamp_ms,
        gesture=gesture,
        gesture_confidence=0.95,
        phase=phase,
        phase_confidence=0.9,
        uncertainty=0.05,
    )


def _executor(adapters: dict[str, FakeAdapter]) -> IntentExecutor:
    registry = build_default_registry()
    return IntentExecutor(
        engine=ProtocolEngine(registry, RuntimeClock()),
        registry=registry,
        adapters=dict(adapters),
        capability_map=build_default_capability_map(),
    )


def _bridge(tmp_path: Path, mapping: dict[str, str]) -> DemoBridge:
    store = DeviceMappingStore(tmp_path / "demo_device_map.json")
    for target_id, device_id in mapping.items():
        store.set(target_id, device_id)
    return DemoBridge(mapping_store=store)


def _run_event(
    bridge: DemoBridge, target_id: str, gesture: str, *, dwell_ms: int = 1500
) -> CommitDecision | None:
    """물체를 응시해 lock을 잡은 뒤 제스처 한 번을 완결시킨다."""
    for ms in range(0, dwell_ms, 100):
        bridge.push_target(_target(target_id, ms))
    bridge.push_gesture(_gesture(GesturePhase.ONSET, dwell_ms, gesture))
    return bridge.push_gesture(_gesture(GesturePhase.ENDING, dwell_ms + 200, gesture))


def test_registered_target_reaches_bulb_adapter(tmp_path: Path) -> None:
    """등록 물체 target_001을 room.bulb로 이어 주면 밝기 명령이 실제 adapter까지 간다."""
    bridge = _bridge(tmp_path, {"target_001": BULB_DEVICE_ID})
    decision = _run_event(bridge, "target_001", "slide_two_fingers_down")
    assert decision is not None
    assert decision.committed is True

    adapter = FakeAdapter()
    outcome = _executor({BULB_ADAPTER: adapter}).execute(decision)
    assert outcome.stage is ExecutionStage.DISPATCHED
    assert outcome.executed is True
    assert len(adapter.calls) == 1
    command = adapter.calls[0]
    assert command.device_id == BULB_DEVICE_ID
    assert command.capability == "brightness"
    assert command.operation == "decrement"


def test_same_gesture_reaches_laptop_as_scroll(tmp_path: Path) -> None:
    """같은 slide_down이 노트북에서는 스크롤로 간다 — 기획안의 핵심 주장."""
    bridge = _bridge(tmp_path, {"target_002": LAPTOP_DEVICE_ID})
    decision = _run_event(bridge, "target_002", "slide_two_fingers_down")
    assert decision is not None

    adapter = FakeAdapter()
    outcome = _executor({LAPTOP_ADAPTER: adapter}).execute(decision)
    assert outcome.executed is True
    assert adapter.calls[0].capability == "scroll"
    assert adapter.calls[0].operation == "decrement"


def test_unmapped_target_never_reaches_any_adapter(tmp_path: Path) -> None:
    """매핑하지 않은 물체를 보며 제스처해도 명령이 나가지 않는다(시나리오 6~7)."""
    bridge = _bridge(tmp_path, {})
    decision = _run_event(bridge, "target_009", "slide_two_fingers_down")
    assert decision is not None
    assert decision.committed is False

    adapter = FakeAdapter()
    outcome = _executor({BULB_ADAPTER: adapter, LAPTOP_ADAPTER: adapter}).execute(
        decision
    )
    assert outcome.stage is ExecutionStage.NOT_COMMITTED
    assert adapter.calls == []


def test_fallback_reaches_adapter_without_any_gaze(tmp_path: Path) -> None:
    """타깃 고정 폴백이면 시선 스트림이 전혀 없어도 명령이 나간다(--no-gaze 대비)."""
    bridge = _bridge(tmp_path, {})
    bridge.set_fallback(LAPTOP_DEVICE_ID)
    for ms in range(0, 1500, 100):
        bridge.push_gesture(_gesture(GesturePhase.IDLE, ms, "none"))
    bridge.push_gesture(_gesture(GesturePhase.ONSET, 1500, "rotate_clockwise"))
    decision = bridge.push_gesture(_gesture(GesturePhase.ENDING, 1700, "rotate_clockwise"))
    assert decision is not None and decision.committed

    adapter = FakeAdapter()
    outcome = _executor({LAPTOP_ADAPTER: adapter}).execute(decision)
    assert outcome.executed is True
    assert adapter.calls[0].capability == "volume"
    assert adapter.calls[0].operation == "increment"


def test_slide_left_right_reach_the_bulb_as_brightness(tmp_path: Path) -> None:
    """좌우 슬라이드도 전구에서는 밝기다 — 상하가 잘 안 잡힐 때의 이중 경로."""
    for gesture, operation in (
        ("slide_two_fingers_left", "decrement"),
        ("slide_two_fingers_right", "increment"),
    ):
        bridge = _bridge(tmp_path, {"target_001": BULB_DEVICE_ID})
        decision = _run_event(bridge, "target_001", gesture)
        assert decision is not None and decision.committed

        adapter = FakeAdapter()
        outcome = _executor({BULB_ADAPTER: adapter}).execute(decision)
        assert outcome.executed is True
        assert adapter.calls[0].capability == "brightness"
        assert adapter.calls[0].operation == operation


def test_laptop_slide_left_right_no_longer_maps_dynamically(tmp_path: Path) -> None:
    """노트북 데스크톱 전환은 정적 two_fingers 스와이프로 갈아끼웠다(2026-07-22).

    그래서 동적 slide_left/right는 더 이상 노트북 capability에 매핑되지 않는다 —
    커밋은 되지만 노트북에 capability 매핑이 없어 NO_MAPPING으로 떨어지고 어떤
    adapter에도 도달하지 않는다. 실제 데스크톱 전환은 pose_state._track_swipe →
    pose_control.switch_desktop 경로가 담당한다(capability map을 타지 않음)."""
    for gesture in ("slide_two_fingers_left", "slide_two_fingers_right"):
        bridge = _bridge(tmp_path, {"target_002": LAPTOP_DEVICE_ID})
        decision = _run_event(bridge, "target_002", gesture)
        assert decision is not None and decision.committed

        adapter = FakeAdapter()
        outcome = _executor({LAPTOP_ADAPTER: adapter}).execute(decision)
        assert outcome.stage is ExecutionStage.NO_MAPPING
        assert adapter.calls == []


def test_rotate_reaches_the_bulb_as_color(tmp_path: Path) -> None:
    """회전은 색온도가 아니라 색상이다(2026-07-22 변경)."""
    bridge = _bridge(tmp_path, {"target_001": BULB_DEVICE_ID})
    decision = _run_event(bridge, "target_001", "rotate_clockwise")
    assert decision is not None

    adapter = FakeAdapter()
    outcome = _executor({BULB_ADAPTER: adapter}).execute(decision)
    assert outcome.executed is True
    assert adapter.calls[0].capability == "color"
    assert adapter.calls[0].operation == "increment"
