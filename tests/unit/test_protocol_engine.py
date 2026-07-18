"""Unit tests for the protocol engine (validation + TTL + dedup + lifecycle)."""

from __future__ import annotations

import pytest

from jarvis.contracts.messages import Intent
from jarvis.runtime_protocol.capture.clock import RuntimeClock
from jarvis.runtime_protocol.protocol.capability import (
    BooleanCapability,
    CommandValue,
    DeviceProfile,
    DeviceRegistry,
    NumberCapability,
)
from jarvis.runtime_protocol.protocol.engine import (
    Accepted,
    ProtocolEngine,
    Rejected,
)
from jarvis.runtime_protocol.protocol.lifecycle import CommandState, RejectionReason


class FakeTime:
    def __init__(self) -> None:
        self.value_ns = 0

    def set_ms(self, ms: int) -> None:
        self.value_ns = ms * 1_000_000

    def __call__(self) -> int:
        return self.value_ns


def _registry() -> DeviceRegistry:
    bulb = DeviceProfile(
        device_id="room.bulb",
        adapter="smartthings",
        capabilities={
            "brightness": NumberCapability(minimum=0, maximum=100, step=10),
            "power": BooleanCapability(),
        },
    )
    return DeviceRegistry({"room.bulb": bulb})


def _intent(
    *,
    intent_id: str = "intent-1",
    target: str = "room.bulb",
    capability: str = "brightness",
    operation: str = "decrement",
    value: CommandValue = 10,
    expires_in_ms: int = 1000,
) -> Intent:
    return Intent(
        intent_id=intent_id,
        target=target,
        gesture="swipe_down",
        capability=capability,
        operation=operation,
        value=value,
        target_confidence=0.9,
        gesture_confidence=0.9,
        expires_in_ms=expires_in_ms,
    )


def _engine(time_source: FakeTime) -> ProtocolEngine:
    return ProtocolEngine(_registry(), RuntimeClock(time_source))


def test_valid_intent_becomes_validated_command() -> None:
    clock = FakeTime()
    clock.set_ms(500)
    engine = _engine(clock)

    result = engine.submit(_intent())
    assert isinstance(result, Accepted)
    command = result.command
    assert command.command_id == "cmd-intent-1"
    assert command.intent_id == "intent-1"
    assert command.device_id == "room.bulb"
    assert command.capability == "brightness"
    assert command.operation == "decrement"
    assert command.value == 10
    assert command.expires_at_ms == 1500  # now 500 + ttl 1000
    assert engine.state("cmd-intent-1") == CommandState.VALIDATED


def test_unknown_device_rejected() -> None:
    result = _engine(FakeTime()).submit(_intent(target="room.ghost"))
    assert isinstance(result, Rejected)
    assert result.reason == RejectionReason.UNKNOWN_DEVICE


def test_unknown_capability_rejected() -> None:
    result = _engine(FakeTime()).submit(_intent(capability="volume"))
    assert isinstance(result, Rejected)
    assert result.reason == RejectionReason.UNKNOWN_CAPABILITY


def test_already_expired_intent_rejected() -> None:
    result = _engine(FakeTime()).submit(_intent(expires_in_ms=0))
    assert isinstance(result, Rejected)
    assert result.reason == RejectionReason.EXPIRED


def test_unsupported_operation_rejected() -> None:
    result = _engine(FakeTime()).submit(_intent(capability="power", operation="increment"))
    assert isinstance(result, Rejected)
    assert result.reason == RejectionReason.UNSUPPORTED_OPERATION


def test_out_of_range_value_rejected() -> None:
    result = _engine(FakeTime()).submit(_intent(operation="set", value=130))
    assert isinstance(result, Rejected)
    assert result.reason == RejectionReason.INVALID_VALUE


def test_retried_intent_is_deduplicated() -> None:
    engine = _engine(FakeTime())
    first = engine.submit(_intent())
    second = engine.submit(_intent())  # same intent_id
    assert isinstance(first, Accepted)
    assert isinstance(second, Rejected)
    assert second.reason == RejectionReason.DUPLICATE


def test_dispatch_guard_passes_within_ttl() -> None:
    clock = FakeTime()
    engine = _engine(clock)
    engine.submit(_intent(expires_in_ms=1000))  # expires_at 1000
    clock.set_ms(800)
    assert engine.dispatch_guard("cmd-intent-1") == CommandState.DISPATCHED


def test_dispatch_guard_expires_after_ttl() -> None:
    clock = FakeTime()
    engine = _engine(clock)
    engine.submit(_intent(expires_in_ms=1000))  # expires_at 1000
    clock.set_ms(1500)
    assert engine.dispatch_guard("cmd-intent-1") == CommandState.EXPIRED
    assert engine.state("cmd-intent-1") == CommandState.EXPIRED


def test_dispatch_guard_unknown_command_raises() -> None:
    with pytest.raises(KeyError):
        _engine(FakeTime()).dispatch_guard("cmd-nope")


def test_full_success_lifecycle() -> None:
    clock = FakeTime()
    engine = _engine(clock)
    engine.submit(_intent())
    assert engine.dispatch_guard("cmd-intent-1") == CommandState.DISPATCHED
    assert engine.mark_acknowledged("cmd-intent-1") == CommandState.ACKNOWLEDGED
    assert engine.mark_verified("cmd-intent-1") == CommandState.VERIFIED
