"""Command lifecycle states and legal transitions (README 10장 명령 상태).

Success path:  VALIDATED → DISPATCHED → ACKNOWLEDGED → VERIFIED
Failure paths: REJECTED (never dispatched), EXPIRED (TTL lapsed before dispatch),
               FAILED (adapter error), UNVERIFIED (dispatched but state unconfirmed).

A validated command can also become REJECTED without ever being dispatched — the
dispatcher takes this edge when it cannot route the command (the target device is
no longer registered), so an unroutable command reaches an honest terminal state
instead of a fabricated DISPATCHED.

The protocol owns VALIDATED/REJECTED/EXPIRED and the transition rules here; the
adapter layer (chunk 3) drives DISPATCHED onward but must move only along these
edges. Terminal states never transition again.
"""

from __future__ import annotations

from enum import StrEnum


class CommandState(StrEnum):
    VALIDATED = "VALIDATED"
    DISPATCHED = "DISPATCHED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"
    UNVERIFIED = "UNVERIFIED"


class RejectionReason(StrEnum):
    UNKNOWN_DEVICE = "UNKNOWN_DEVICE"
    UNKNOWN_CAPABILITY = "UNKNOWN_CAPABILITY"
    UNSUPPORTED_OPERATION = "UNSUPPORTED_OPERATION"
    INVALID_VALUE = "INVALID_VALUE"
    EXPIRED = "EXPIRED"
    DUPLICATE = "DUPLICATE"


TERMINAL_STATES: frozenset[CommandState] = frozenset(
    {
        CommandState.VERIFIED,
        CommandState.REJECTED,
        CommandState.EXPIRED,
        CommandState.FAILED,
        CommandState.UNVERIFIED,
    }
)

# Allowed forward edges. Any transition not listed here is a programming error.
_ALLOWED: dict[CommandState, frozenset[CommandState]] = {
    CommandState.VALIDATED: frozenset(
        {CommandState.DISPATCHED, CommandState.EXPIRED, CommandState.REJECTED}
    ),
    CommandState.DISPATCHED: frozenset(
        {CommandState.ACKNOWLEDGED, CommandState.FAILED, CommandState.UNVERIFIED}
    ),
    CommandState.ACKNOWLEDGED: frozenset(
        {CommandState.VERIFIED, CommandState.UNVERIFIED}
    ),
}


def can_transition(src: CommandState, dst: CommandState) -> bool:
    """Whether ``src → dst`` is a legal lifecycle edge."""
    return dst in _ALLOWED.get(src, frozenset())


class IllegalTransitionError(RuntimeError):
    """Raised when code attempts a transition outside the state machine."""

    def __init__(self, src: CommandState, dst: CommandState) -> None:
        super().__init__(f"illegal command transition {src} → {dst}")
        self.src = src
        self.dst = dst
