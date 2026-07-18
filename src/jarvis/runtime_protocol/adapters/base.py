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

        TTL is re-checked first (principle 4); an expired command is never sent.
        The command is routed by ``device_id`` → profile → adapter name. The
        adapter's honest status is mapped onto the lifecycle; on any failure the
        safe result is non-execution (principle 2.7).
        """
        guard_state = self._engine.dispatch_guard(command_id)
        if guard_state is CommandState.EXPIRED:
            return DispatchReport(
                command_id, CommandState.EXPIRED, None, "TTL lapsed before dispatch"
            )

        command = self._engine.command(command_id)
        if command is None:  # pragma: no cover - dispatch_guard already proved it exists
            raise KeyError(command_id)

        profile = self._registry.get(command.device_id)
        if profile is None:
            state = self._engine.mark_failed(command_id)
            return DispatchReport(
                command_id, state, AdapterStatus.FAILED,
                f"device {command.device_id!r} is not registered",
            )

        adapter = self._adapters.get(profile.adapter)
        if adapter is None:
            raise UnknownAdapterError(profile.adapter, command.device_id)

        result = adapter.execute(command, profile)
        final_state = self._engine.state(command_id)
        for target in _STATUS_TRANSITIONS[result.status]:
            final_state = self._engine.transition_to(command_id, target)
        assert final_state is not None
        return DispatchReport(command_id, final_state, result.status, result.detail)
