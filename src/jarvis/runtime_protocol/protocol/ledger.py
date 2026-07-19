"""Command ledger: dedup and lifecycle-state tracking.

Enforces development-principles 3 ("한 command_id는 최대 한 번만 실행"): a command id
is registered at most once, and its state only ever moves along legal lifecycle
edges. The ledger is the single source of truth for whether a command has already
been seen, so retries collapse instead of double-executing.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from jarvis.contracts.messages import Command
from jarvis.runtime_protocol.protocol.lifecycle import (
    CommandState,
    IllegalTransitionError,
    can_transition,
)


class DuplicateCommandError(RuntimeError):
    """Raised when a command id is registered a second time."""

    def __init__(self, command_id: str) -> None:
        super().__init__(f"command {command_id!r} already registered")
        self.command_id = command_id


@dataclass(slots=True)
class LedgerEntry:
    command: Command
    state: CommandState


class CommandLedger:
    """Thread-safe registry of commands and their lifecycle state."""

    def __init__(self) -> None:
        self._entries: dict[str, LedgerEntry] = {}
        self._lock = Lock()

    def seen(self, command_id: str) -> bool:
        with self._lock:
            return command_id in self._entries

    def register(self, command: Command) -> None:
        """Record a newly validated command. Raises on a duplicate id."""
        with self._lock:
            if command.command_id in self._entries:
                raise DuplicateCommandError(command.command_id)
            self._entries[command.command_id] = LedgerEntry(
                command=command, state=CommandState.VALIDATED
            )

    def state(self, command_id: str) -> CommandState | None:
        with self._lock:
            entry = self._entries.get(command_id)
            return entry.state if entry is not None else None

    def command(self, command_id: str) -> Command | None:
        with self._lock:
            entry = self._entries.get(command_id)
            return entry.command if entry is not None else None

    def transition(self, command_id: str, dst: CommandState) -> CommandState:
        """Move a command to ``dst`` if the edge is legal. Returns the new state.

        Raises ``KeyError`` for an unknown command and
        :class:`IllegalTransitionError` for a disallowed edge.
        """
        with self._lock:
            entry = self._entries.get(command_id)
            if entry is None:
                raise KeyError(command_id)
            if not can_transition(entry.state, dst):
                raise IllegalTransitionError(entry.state, dst)
            entry.state = dst
            return dst
