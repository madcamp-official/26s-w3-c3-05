"""Protocol engine: Intent → validated Command, with TTL and dedup guards.

This is the safe-execution core between Fusion and the device adapters. It never
touches a real device; it decides whether a request is allowed to become a
command, stamps the command's absolute expiry on the shared clock, and re-checks
the TTL immediately before dispatch (development-principles 2, 4). A request that
fails any check is rejected with a reason instead of being executed — the safe
default is always non-execution (principle 2.7).
"""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.contracts.messages import Command, Intent
from jarvis.runtime_protocol.capture.clock import RuntimeClock
from jarvis.runtime_protocol.protocol.capability import (
    DeviceRegistry,
    validate_request,
)
from jarvis.runtime_protocol.protocol.ledger import (
    CommandLedger,
    DuplicateCommandError,
)
from jarvis.runtime_protocol.protocol.lifecycle import CommandState, RejectionReason


@dataclass(frozen=True, slots=True)
class Accepted:
    """The intent passed all checks and became a registered command."""

    command: Command


@dataclass(frozen=True, slots=True)
class Rejected:
    """The intent was refused; ``reason``/``detail`` explain why, for trace."""

    intent_id: str
    reason: RejectionReason
    detail: str


SubmitResult = Accepted | Rejected


def _command_id_for(intent_id: str) -> str:
    """Deterministic command id per intent, so retries collapse (idempotency)."""
    return f"cmd-{intent_id}"


class ProtocolEngine:
    """Validates intents into commands and guards their lifecycle."""

    def __init__(self, registry: DeviceRegistry, clock: RuntimeClock) -> None:
        self._registry = registry
        self._clock = clock
        self._ledger = CommandLedger()

    def submit(self, intent: Intent) -> SubmitResult:
        """Validate an intent and register a command, or reject with a reason."""
        profile = self._registry.get(intent.target)
        if profile is None:
            return Rejected(
                intent.intent_id,
                RejectionReason.UNKNOWN_DEVICE,
                f"device {intent.target!r} is not registered",
            )

        capability = profile.capabilities.get(intent.capability)
        if capability is None:
            return Rejected(
                intent.intent_id,
                RejectionReason.UNKNOWN_CAPABILITY,
                f"device {intent.target!r} has no capability {intent.capability!r}",
            )

        if intent.expires_in_ms <= 0:
            return Rejected(
                intent.intent_id,
                RejectionReason.EXPIRED,
                f"intent arrived already expired (expires_in_ms={intent.expires_in_ms})",
            )

        if intent.operation not in capability.operations:
            return Rejected(
                intent.intent_id,
                RejectionReason.UNSUPPORTED_OPERATION,
                f"capability {intent.capability!r} does not support {intent.operation!r}",
            )

        failure = validate_request(capability, intent.operation, intent.value)
        if failure is not None:
            return Rejected(
                intent.intent_id, RejectionReason.INVALID_VALUE, failure.detail
            )

        command = Command(
            command_id=_command_id_for(intent.intent_id),
            intent_id=intent.intent_id,
            capability=intent.capability,
            operation=intent.operation,
            value=intent.value,
            expires_at_ms=self._clock.now_ms() + intent.expires_in_ms,
        )
        # Registration is the single atomic dedup point: the ledger admits a
        # command id at most once. Relying on it (instead of a separate seen()
        # pre-check) closes the race where two concurrent submits of the same
        # intent both pass the check and one then crashes on register.
        try:
            self._ledger.register(command)
        except DuplicateCommandError:
            return Rejected(
                intent.intent_id,
                RejectionReason.DUPLICATE,
                f"intent {intent.intent_id!r} already produced a command",
            )
        return Accepted(command)

    def dispatch_guard(self, command_id: str) -> CommandState:
        """Re-check TTL just before dispatch (principle 4).

        Returns ``DISPATCHED`` if the command may go out now, or ``EXPIRED`` if
        its TTL lapsed while it waited. Raises ``KeyError`` for an unknown id.
        """
        command = self._ledger.command(command_id)
        if command is None:
            raise KeyError(command_id)
        if self._clock.now_ms() > command.expires_at_ms:
            return self._ledger.transition(command_id, CommandState.EXPIRED)
        return self._ledger.transition(command_id, CommandState.DISPATCHED)

    def mark_acknowledged(self, command_id: str) -> CommandState:
        return self._ledger.transition(command_id, CommandState.ACKNOWLEDGED)

    def mark_verified(self, command_id: str) -> CommandState:
        return self._ledger.transition(command_id, CommandState.VERIFIED)

    def mark_failed(self, command_id: str) -> CommandState:
        return self._ledger.transition(command_id, CommandState.FAILED)

    def mark_unverified(self, command_id: str) -> CommandState:
        return self._ledger.transition(command_id, CommandState.UNVERIFIED)

    def state(self, command_id: str) -> CommandState | None:
        return self._ledger.state(command_id)
