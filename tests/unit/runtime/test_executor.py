"""IntentExecutor: Fusion CommitDecision → 명령 dispatch 배선 단위 테스트.

실제 입력·네트워크 없이 fake adapter로 배선만 검증한다 — 각 단계(미커밋·매핑 없음·
검증 거부·dispatch)에서 올바른 stage와 executed 플래그가 나오는지 본다.
"""

from __future__ import annotations

from jarvis.contracts.messages import Command
from jarvis.gesture_fusion.fusion import CommitDecision, FusionScore
from jarvis.gesture_fusion.intent import CapabilityAction, GestureCapabilityMap
from jarvis.runtime.executor import ExecutionStage, IntentExecutor
from jarvis.runtime_protocol.adapters.base import AdapterResult, AdapterStatus
from jarvis.runtime_protocol.capture.clock import RuntimeClock
from jarvis.runtime_protocol.protocol.capability import DeviceProfile
from jarvis.runtime_protocol.protocol.engine import ProtocolEngine
from jarvis.runtime.devices import build_default_registry


class FakeAdapter:
    def __init__(self, name: str, result: AdapterResult) -> None:
        self.name = name
        self._result = result
        self.calls: list[Command] = []

    def execute(self, command: Command, profile: DeviceProfile) -> AdapterResult:
        self.calls.append(command)
        return self._result


def _score(target_confidence: float = 0.9, gesture_confidence: float = 0.9) -> FusionScore:
    return FusionScore(
        target_confidence=target_confidence,
        gesture_confidence=gesture_confidence,
        gaze_stability=0.9,
        uncertainty=0.1,
        value=0.5,
    )


def _committed(target: str, gesture: str, intent_id: str = "evt-1") -> CommitDecision:
    return CommitDecision(
        committed=True,
        reason="committed",
        target=target,
        gesture=gesture,
        score=_score(),
        timestamp_ms=1000,
        frame_id=42,
        intent_id=intent_id,
    )


def _laptop_map() -> GestureCapabilityMap:
    return GestureCapabilityMap(
        {"laptop": {"swipe_down": CapabilityAction("scroll", "decrement", 3)}}
    )


def _executor(
    capability_map: GestureCapabilityMap,
    adapters: dict[str, FakeAdapter],
    clock: RuntimeClock | None = None,
) -> IntentExecutor:
    registry = build_default_registry()
    engine = ProtocolEngine(registry, clock if clock is not None else RuntimeClock())
    return IntentExecutor(
        engine=engine,
        registry=registry,
        adapters=dict(adapters),
        capability_map=capability_map,
    )


def test_committed_with_mapping_dispatches_and_executes() -> None:
    adapter = FakeAdapter("windows", AdapterResult(AdapterStatus.ACKNOWLEDGED, "scrolled"))
    executor = _executor(_laptop_map(), {"windows": adapter})

    outcome = executor.execute(_committed("laptop", "swipe_down"))

    assert outcome.stage == ExecutionStage.DISPATCHED
    assert outcome.executed is True
    assert outcome.intent is not None and outcome.intent.capability == "scroll"
    assert len(adapter.calls) == 1
    assert adapter.calls[0].operation == "decrement" and adapter.calls[0].value == 3


def test_not_committed_makes_no_command() -> None:
    adapter = FakeAdapter("windows", AdapterResult(AdapterStatus.ACKNOWLEDGED))
    executor = _executor(_laptop_map(), {"windows": adapter})

    rejected = CommitDecision(
        committed=False,
        reason="fusion score below commit threshold",
        target="laptop",
        gesture="swipe_down",
        score=_score(),
        timestamp_ms=1000,
        frame_id=42,
    )
    outcome = executor.execute(rejected)

    assert outcome.stage == ExecutionStage.NOT_COMMITTED
    assert outcome.executed is False
    assert outcome.intent is None
    assert adapter.calls == []


def test_committed_without_mapping_is_not_executed() -> None:
    adapter = FakeAdapter("windows", AdapterResult(AdapterStatus.ACKNOWLEDGED))
    # 매핑에 없는 제스처 — 알 수 없는 조합은 실행이 아니라 거부.
    executor = _executor(_laptop_map(), {"windows": adapter})

    outcome = executor.execute(_committed("laptop", "rotate_clockwise"))

    assert outcome.stage == ExecutionStage.NO_MAPPING
    assert outcome.executed is False
    assert adapter.calls == []


def test_protocol_rejection_stops_before_adapter() -> None:
    adapter = FakeAdapter("windows", AdapterResult(AdapterStatus.ACKNOWLEDGED))
    # brightness는 step 10 격자다 — 7은 격자를 벗어나 protocol 검증에서 거부된다.
    bad_map = GestureCapabilityMap(
        {"room.bulb": {"swipe_down": CapabilityAction("brightness", "decrement", 7)}}
    )
    executor = _executor(bad_map, {"windows": adapter, "smartthings": adapter})

    outcome = executor.execute(_committed("room.bulb", "swipe_down"))

    assert outcome.stage == ExecutionStage.REJECTED
    assert outcome.executed is False
    assert outcome.rejection is not None
    assert adapter.calls == []


def test_adapter_failure_is_reported_not_executed() -> None:
    adapter = FakeAdapter("windows", AdapterResult(AdapterStatus.FAILED, "input sink error"))
    executor = _executor(_laptop_map(), {"windows": adapter})

    outcome = executor.execute(_committed("laptop", "swipe_down"))

    assert outcome.stage == ExecutionStage.DISPATCHED  # adapter까지 갔지만
    assert outcome.executed is False  # 실패는 실행으로 치지 않는다
    assert len(adapter.calls) == 1


def test_unconfigured_bulb_dispatches_but_not_executed() -> None:
    # 전구 매핑은 정상이지만 SmartThings가 미설정이면 UNCONFIGURED → 실행 아님.
    adapter = FakeAdapter("smartthings", AdapterResult(AdapterStatus.UNCONFIGURED, "no token"))
    bulb_map = GestureCapabilityMap(
        {"room.bulb": {"swipe_down": CapabilityAction("brightness", "decrement", 10)}}
    )
    executor = _executor(bulb_map, {"smartthings": adapter})

    outcome = executor.execute(_committed("room.bulb", "swipe_down"))

    assert outcome.stage == ExecutionStage.DISPATCHED
    assert outcome.executed is False
