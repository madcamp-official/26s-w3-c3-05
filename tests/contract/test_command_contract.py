"""Producer↔consumer contract test for the Protocol → Device boundary.

The single source of truth for this boundary is documents/interface-contract.md §4
(Command) and jarvis.contracts.messages.Command. This test pins the fields and
types both sides agree on — in particular ``device_id``, added so the dispatcher
can route a command to the right adapter — so a future change to one side that
diverges from the contract fails here instead of at integration time.
"""

from __future__ import annotations

from dataclasses import fields

from jarvis.contracts.messages import Command, Intent
from jarvis.runtime_protocol.adapters.base import (
    AdapterResult,
    AdapterStatus,
    DispatchCoordinator,
)
from jarvis.runtime_protocol.capture.clock import RuntimeClock
from jarvis.runtime_protocol.protocol.capability import (
    DeviceProfile,
    DeviceRegistry,
    NumberCapability,
)
from jarvis.runtime_protocol.protocol.engine import Accepted, ProtocolEngine

# Exactly the fields documented in interface-contract.md §4, with their types.
_EXPECTED_COMMAND_FIELDS = {
    "command_id": str,
    "intent_id": str,
    "device_id": str,
    "capability": str,
    "operation": str,
    "value": int | float | bool,
    "expires_at_ms": int,
}


def test_command_message_matches_interface_contract() -> None:
    actual = {f.name: f.type for f in fields(Command)}
    assert set(actual) == set(_EXPECTED_COMMAND_FIELDS), (
        "Command fields drifted from interface-contract.md §4"
    )


class _RecordingAdapter:
    """Consumer standing in for a real adapter; records what it received."""

    name = "fake"

    def __init__(self) -> None:
        self.received: Command | None = None

    def execute(self, command: Command, profile: DeviceProfile) -> AdapterResult:
        self.received = command
        return AdapterResult(AdapterStatus.ACKNOWLEDGED, "ok")


def test_protocol_produces_command_the_adapter_boundary_consumes() -> None:
    # Producer: the protocol engine turns an Intent into a Command.
    registry = DeviceRegistry(
        {
            "room.bulb": DeviceProfile(
                device_id="room.bulb",
                adapter="fake",
                capabilities={
                    "brightness": NumberCapability(minimum=0, maximum=100, step=10)
                },
            )
        }
    )
    engine = ProtocolEngine(registry, RuntimeClock())
    intent = Intent(
        intent_id="intent-1",
        target="room.bulb",
        gesture="swipe_down",
        capability="brightness",
        operation="decrement",
        value=10,
        target_confidence=0.9,
        gesture_confidence=0.9,
        expires_in_ms=1000,
    )
    result = engine.submit(intent)
    assert isinstance(result, Accepted)
    command = result.command

    # The producer carries the routing key derived from the intent target.
    assert command.device_id == intent.target

    # Consumer: the dispatcher routes by device_id and the adapter receives the
    # same command, proving both sides agree on the boundary shape.
    adapter = _RecordingAdapter()
    coordinator = DispatchCoordinator(engine, registry, {"fake": adapter})
    coordinator.dispatch(command.command_id)
    assert adapter.received is command
