"""Unit tests for the dispatch coordinator (routing + lifecycle mapping)."""

from __future__ import annotations

import pytest

from jarvis.contracts.messages import Command, Intent
from jarvis.runtime_protocol.adapters.base import (
    AdapterResult,
    AdapterStatus,
    DispatchCoordinator,
    UnknownAdapterError,
)
from jarvis.runtime_protocol.capture.clock import RuntimeClock
from jarvis.runtime_protocol.protocol.capability import (
    DeviceProfile,
    DeviceRegistry,
    NumberCapability,
)
from jarvis.runtime_protocol.protocol.engine import Accepted, ProtocolEngine
from jarvis.runtime_protocol.protocol.lifecycle import CommandState


class FakeTime:
    def __init__(self) -> None:
        self.value_ns = 0

    def set_ms(self, ms: int) -> None:
        self.value_ns = ms * 1_000_000

    def __call__(self) -> int:
        return self.value_ns


class FakeAdapter:
    def __init__(self, name: str, result: AdapterResult) -> None:
        self.name = name
        self._result = result
        self.calls: list[Command] = []

    def execute(self, command: Command, profile: DeviceProfile) -> AdapterResult:
        self.calls.append(command)
        return self._result


def _registry(adapter_name: str = "fake") -> DeviceRegistry:
    bulb = DeviceProfile(
        device_id="room.bulb",
        adapter=adapter_name,
        capabilities={"brightness": NumberCapability(minimum=0, maximum=100, step=10)},
    )
    return DeviceRegistry({"room.bulb": bulb})


def _intent(expires_in_ms: int = 1000) -> Intent:
    return Intent(
        intent_id="intent-1",
        target="room.bulb",
        gesture="swipe_down",
        capability="brightness",
        operation="decrement",
        value=10,
        target_confidence=0.9,
        gesture_confidence=0.9,
        expires_in_ms=expires_in_ms,
    )


def _submit(engine: ProtocolEngine) -> str:
    result = engine.submit(_intent())
    assert isinstance(result, Accepted)
    return result.command.command_id


def test_acknowledged_result_marks_command_acknowledged() -> None:
    clock = FakeTime()
    registry = _registry()
    engine = ProtocolEngine(registry, RuntimeClock(clock))
    adapter = FakeAdapter("fake", AdapterResult(AdapterStatus.ACKNOWLEDGED, "ok"))
    coordinator = DispatchCoordinator(engine, registry, {"fake": adapter})

    command_id = _submit(engine)
    report = coordinator.dispatch(command_id)

    assert report.final_state == CommandState.ACKNOWLEDGED
    assert report.adapter_status == AdapterStatus.ACKNOWLEDGED
    assert len(adapter.calls) == 1


def test_verified_result_progresses_through_acknowledged_to_verified() -> None:
    registry = _registry()
    engine = ProtocolEngine(registry, RuntimeClock(FakeTime()))
    adapter = FakeAdapter("fake", AdapterResult(AdapterStatus.VERIFIED, "confirmed"))
    coordinator = DispatchCoordinator(engine, registry, {"fake": adapter})

    report = coordinator.dispatch(_submit(engine))
    assert report.final_state == CommandState.VERIFIED


def test_unverified_result_marks_unverified() -> None:
    registry = _registry()
    engine = ProtocolEngine(registry, RuntimeClock(FakeTime()))
    adapter = FakeAdapter("fake", AdapterResult(AdapterStatus.UNVERIFIED, "no readback"))
    coordinator = DispatchCoordinator(engine, registry, {"fake": adapter})

    report = coordinator.dispatch(_submit(engine))
    assert report.final_state == CommandState.UNVERIFIED


def test_failed_result_marks_failed() -> None:
    registry = _registry()
    engine = ProtocolEngine(registry, RuntimeClock(FakeTime()))
    adapter = FakeAdapter("fake", AdapterResult(AdapterStatus.FAILED, "boom"))
    coordinator = DispatchCoordinator(engine, registry, {"fake": adapter})

    report = coordinator.dispatch(_submit(engine))
    assert report.final_state == CommandState.FAILED


def test_unconfigured_result_marks_failed() -> None:
    registry = _registry()
    engine = ProtocolEngine(registry, RuntimeClock(FakeTime()))
    adapter = FakeAdapter("fake", AdapterResult(AdapterStatus.UNCONFIGURED, "no token"))
    coordinator = DispatchCoordinator(engine, registry, {"fake": adapter})

    report = coordinator.dispatch(_submit(engine))
    assert report.final_state == CommandState.FAILED


def test_expired_command_is_not_sent_to_adapter() -> None:
    clock = FakeTime()
    registry = _registry()
    engine = ProtocolEngine(registry, RuntimeClock(clock))
    adapter = FakeAdapter("fake", AdapterResult(AdapterStatus.ACKNOWLEDGED))
    coordinator = DispatchCoordinator(engine, registry, {"fake": adapter})

    command_id = _submit(engine)  # expires_at 1000
    clock.set_ms(1500)  # TTL lapsed
    report = coordinator.dispatch(command_id)

    assert report.final_state == CommandState.EXPIRED
    assert adapter.calls == []  # safe default: never dispatched


def test_unregistered_adapter_raises() -> None:
    registry = _registry(adapter_name="ghost")
    engine = ProtocolEngine(registry, RuntimeClock(FakeTime()))
    coordinator = DispatchCoordinator(engine, registry, {})  # no "ghost" adapter

    command_id = _submit(engine)
    with pytest.raises(UnknownAdapterError):
        coordinator.dispatch(command_id)


def test_device_missing_from_registry_marks_failed() -> None:
    # Command validated against a full registry, then dispatched with an empty one.
    full = _registry()
    engine = ProtocolEngine(full, RuntimeClock(FakeTime()))
    command_id = _submit(engine)

    empty = DeviceRegistry({})
    coordinator = DispatchCoordinator(engine, empty, {"fake": FakeAdapter("fake", AdapterResult(AdapterStatus.ACKNOWLEDGED))})
    report = coordinator.dispatch(command_id)
    assert report.final_state == CommandState.FAILED
