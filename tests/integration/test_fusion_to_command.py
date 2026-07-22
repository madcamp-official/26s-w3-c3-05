"""End-to-end 배선 검증: Fusion CommitDecision → Intent → Command → adapter.

카메라·모델 없이, 실제 `FusionEngine`이 만든 커밋을 실제 gesture→capability 매핑
(`configs/gesture_capability_map.json`)과 실제 device capability model로 흘려보내,
명령이 adapter까지 정확한 capability/operation/value로 도달하는지 확인한다. 이전에는
Fusion의 Intent를 Protocol에 넘기는 배선이 없어 이 사슬이 끊겨 있었다 — 이 테스트가
그 배선을 회귀로 지킨다.
"""

from __future__ import annotations

from jarvis.contracts.messages import Command, GestureEstimate, GesturePhase, TargetEstimate
from jarvis.gesture_fusion.alignment import AlignmentConfig
from jarvis.gesture_fusion.fusion import FusionConfig, FusionEngine
from jarvis.runtime.devices import (
    BULB_ADAPTER,
    build_default_capability_map,
    build_default_registry,
)
from jarvis.runtime.executor import ExecutionStage, IntentExecutor
from jarvis.runtime_protocol.adapters.base import AdapterResult, AdapterStatus
from jarvis.runtime_protocol.capture.clock import RuntimeClock
from jarvis.runtime_protocol.protocol.capability import DeviceProfile
from jarvis.runtime_protocol.protocol.engine import ProtocolEngine


class FakeAdapter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[Command] = []

    def execute(self, command: Command, profile: DeviceProfile) -> AdapterResult:
        self.calls.append(command)
        return AdapterResult(AdapterStatus.ACKNOWLEDGED, "ok")


def _target(timestamp_ms: int, target: str) -> TargetEstimate:
    return TargetEstimate(
        timestamp_ms=timestamp_ms,
        frame_id=0,
        target=target,
        probability=0.95,
        second_best_probability=0.02,
        stability=0.95,
    )


def _gesture(timestamp_ms: int, phase: GesturePhase, gesture: str) -> GestureEstimate:
    return GestureEstimate(
        timestamp_ms=timestamp_ms,
        frame_id=timestamp_ms,  # 프레임마다 다른 id — dedup이 이벤트를 구분하도록
        gesture=gesture,
        gesture_confidence=0.95,
        phase=phase,
        phase_confidence=0.9,
        uncertainty=0.05,
    )


def _locked_fusion(target: str) -> FusionEngine:
    engine = FusionEngine(
        FusionConfig(commit_threshold=0.3, min_target_confidence=0.5, min_gesture_confidence=0.5),
        AlignmentConfig(target_dwell_ms=0, target_lock_ttl_ms=2000),
    )
    engine.push_target(_target(0, target))
    engine.push_target(_target(0, target))
    return engine


def _executor(adapters: dict[str, FakeAdapter]) -> IntentExecutor:
    registry = build_default_registry()
    engine = ProtocolEngine(registry, RuntimeClock())
    return IntentExecutor(
        engine=engine,
        registry=registry,
        adapters=dict(adapters),
        capability_map=build_default_capability_map(),
    )


def test_bulb_slide_down_reaches_adapter_as_brightness_decrement() -> None:
    fusion = _locked_fusion("room.bulb")
    fusion.push_gesture(_gesture(100, GesturePhase.ONSET, "slide_two_fingers_down"))
    decision = fusion.push_gesture(_gesture(300, GesturePhase.ENDING, "slide_two_fingers_down"))
    assert decision is not None and decision.committed

    # adapter 키는 전구 프로필이 정한다 — 경로(WiZ/SmartThings)가 바뀌어도 안 깨지게 상수를 쓴다.
    bulb_adapter = FakeAdapter(BULB_ADAPTER)
    executor = _executor({BULB_ADAPTER: bulb_adapter})
    outcome = executor.execute(decision)

    assert outcome.stage == ExecutionStage.DISPATCHED
    assert outcome.executed is True
    assert len(bulb_adapter.calls) == 1
    command = bulb_adapter.calls[0]
    assert command.device_id == "room.bulb"
    assert command.capability == "brightness"
    assert command.operation == "decrement"
    # 값 자체는 config가 정한다(시연 가시성에 따라 바뀐다) — 이 테스트가 지키는 것은
    # "config의 값이 손실·변형 없이 adapter까지 도달하는가"라는 배선이므로, 숫자를
    # 베껴 두지 않고 같은 매핑에서 읽어 비교한다.
    expected = build_default_capability_map().lookup("room.bulb", "slide_two_fingers_down")
    assert expected is not None
    assert command.value == expected.value


def test_laptop_slide_down_reaches_adapter_as_scroll_decrement() -> None:
    fusion = _locked_fusion("laptop")
    fusion.push_gesture(_gesture(100, GesturePhase.ONSET, "slide_two_fingers_down"))
    decision = fusion.push_gesture(_gesture(300, GesturePhase.ENDING, "slide_two_fingers_down"))
    assert decision is not None and decision.committed

    laptop_adapter = FakeAdapter("windows")
    executor = _executor({"windows": laptop_adapter})
    outcome = executor.execute(decision)

    assert outcome.stage == ExecutionStage.DISPATCHED
    assert outcome.executed is True
    assert len(laptop_adapter.calls) == 1
    command = laptop_adapter.calls[0]
    assert command.device_id == "laptop"
    assert command.capability == "scroll"
    assert command.operation == "decrement"
    assert command.value == 3
