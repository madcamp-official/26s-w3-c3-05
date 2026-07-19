"""Adapter contract and the dispatch coordinator.

An adapter is the real execution boundary for one device family (Windows,
SmartThings). It receives a validated :class:`Command` and reports honestly what
happened — it never fabricates success (development-principles 1.1). The
:class:`DispatchCoordinator` ties the protocol engine to the adapters: it guards
the TTL, routes the command to the adapter named by the device profile, and maps
the adapter's outcome onto the command lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from jarvis.contracts.messages import Command
from jarvis.runtime_protocol.protocol.capability import DeviceProfile, DeviceRegistry
from jarvis.runtime_protocol.protocol.engine import ProtocolEngine
from jarvis.runtime_protocol.protocol.lifecycle import CommandState


class AdapterStatus(StrEnum):
    """Honest outcome of an adapter execution attempt.

    ``ACKNOWLEDGED`` — the device accepted the command but the effect was not
    independently confirmed. ``VERIFIED`` — the resulting device state was read
    back and matches. ``UNVERIFIED`` — sent, but the state could not be confirmed
    (e.g. a fire-and-forget device). ``FAILED`` — the command could not be
    applied. ``UNCONFIGURED`` — the adapter is missing required configuration and
    did not attempt anything (development-principles 6.3).
    """

    ACKNOWLEDGED = "ACKNOWLEDGED"
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    FAILED = "FAILED"
    UNCONFIGURED = "UNCONFIGURED"


@dataclass(frozen=True, slots=True)
class AdapterResult:
    """What an adapter did, with a human-readable detail for trace."""

    status: AdapterStatus
    detail: str = ""


class DeviceAdapter(Protocol):
    """Executes a validated command against one device family."""

    @property
    def name(self) -> str:
        """Stable adapter key matched against ``DeviceProfile.adapter``."""
        ...

    def execute(self, command: Command, profile: DeviceProfile) -> AdapterResult: ...


class UnknownAdapterError(RuntimeError):
    """Raised when a device profile names an adapter that is not registered."""

    def __init__(self, adapter_name: str, device_id: str) -> None:
        super().__init__(
            f"device {device_id!r} needs adapter {adapter_name!r}, which is not registered"
        )
        self.adapter_name = adapter_name
        self.device_id = device_id


# Adapter status → the lifecycle transitions the coordinator applies, in order.
_STATUS_TRANSITIONS: dict[AdapterStatus, tuple[CommandState, ...]] = {
    AdapterStatus.ACKNOWLEDGED: (CommandState.ACKNOWLEDGED,),
    AdapterStatus.VERIFIED: (CommandState.ACKNOWLEDGED, CommandState.VERIFIED),
    AdapterStatus.UNVERIFIED: (CommandState.UNVERIFIED,),
    AdapterStatus.FAILED: (CommandState.FAILED,),
    AdapterStatus.UNCONFIGURED: (CommandState.FAILED,),
}


@dataclass(frozen=True, slots=True)
class DispatchReport:
    """Result of dispatching one command: final state plus the adapter detail."""

    command_id: str
    final_state: CommandState
    adapter_status: AdapterStatus | None
    detail: str


class DispatchCoordinator:
    """Guards TTL, routes to the right adapter, and drives command lifecycle."""

    def __init__(
        self,
        engine: ProtocolEngine,
        registry: DeviceRegistry,
        adapters: dict[str, DeviceAdapter],
    ) -> None:
        self._engine = engine
        self._registry = registry
        self._adapters = dict(adapters)

    def dispatch(self, command_id: str) -> DispatchReport:
        """Dispatch a validated command and record its outcome.

        Routing is resolved *before* the command is marked ``DISPATCHED``: the
        command is routed by ``device_id`` → profile → adapter name, then the TTL
        is re-checked immediately before hand-off (principle 4). Only then does it
        become ``DISPATCHED`` — so a command that can't be routed or has expired
        never carries a state implying it was sent. The adapter's honest status is
        mapped onto the lifecycle; on any failure the safe result is non-execution
        (principle 2.7).
        """
        state = self._engine.state(command_id)
        if state is None:
            raise KeyError(command_id)
        if state is not CommandState.VALIDATED:
            # Idempotent: a command dispatched once never executes again. A repeat
            # call reports the current state without touching an adapter.
            return DispatchReport(
                command_id, state, None, f"already dispatched (state {state})"
            )

        command = self._engine.command(command_id)
        assert command is not None  # state was not None, so the ledger entry exists

        profile = self._registry.get(command.device_id)
        if profile is None:
            # Target device is no longer registered: cannot route, never sent.
            final = self._engine.transition_to(command_id, CommandState.REJECTED)
            return DispatchReport(
                command_id, final, None,
                f"device {command.device_id!r} is not registered",
            )

        adapter = self._adapters.get(profile.adapter)
        if adapter is None:
            # Wiring error: the profile names an adapter that was never registered.
            # Fail loud; the command stays VALIDATED (honestly never dispatched).
            raise UnknownAdapterError(profile.adapter, command.device_id)

        guard_state = self._engine.dispatch_guard(command_id)
        if guard_state is CommandState.EXPIRED:
            return DispatchReport(
                command_id, CommandState.EXPIRED, None, "TTL lapsed before dispatch"
            )

        result = adapter.execute(command, profile)
        final_state = self._engine.state(command_id)
        for target in _STATUS_TRANSITIONS[result.status]:
            final_state = self._engine.transition_to(command_id, target)
        assert final_state is not None
        return DispatchReport(command_id, final_state, result.status, result.detail)
