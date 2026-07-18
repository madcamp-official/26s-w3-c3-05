"""Unit tests for the command ledger (dedup + state tracking)."""

from __future__ import annotations

import pytest

from jarvis.contracts.messages import Command
from jarvis.runtime_protocol.protocol.ledger import (
    CommandLedger,
    DuplicateCommandError,
)
from jarvis.runtime_protocol.protocol.lifecycle import (
    CommandState,
    IllegalTransitionError,
)


def _command(command_id: str = "cmd-1") -> Command:
    return Command(
        command_id=command_id,
        intent_id="intent-1",
        capability="brightness",
        operation="decrement",
        value=10,
        expires_at_ms=1000,
    )


def test_register_sets_validated_state() -> None:
    ledger = CommandLedger()
    ledger.register(_command())
    assert ledger.state("cmd-1") == CommandState.VALIDATED
    assert ledger.seen("cmd-1")


def test_duplicate_registration_raises() -> None:
    ledger = CommandLedger()
    ledger.register(_command())
    with pytest.raises(DuplicateCommandError):
        ledger.register(_command())


def test_unknown_command_state_is_none() -> None:
    assert CommandLedger().state("nope") is None


def test_legal_transition_updates_state() -> None:
    ledger = CommandLedger()
    ledger.register(_command())
    assert ledger.transition("cmd-1", CommandState.DISPATCHED) == CommandState.DISPATCHED
    assert ledger.state("cmd-1") == CommandState.DISPATCHED


def test_illegal_transition_raises_and_keeps_state() -> None:
    ledger = CommandLedger()
    ledger.register(_command())
    with pytest.raises(IllegalTransitionError):
        ledger.transition("cmd-1", CommandState.VERIFIED)
    assert ledger.state("cmd-1") == CommandState.VALIDATED


def test_transition_unknown_command_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        CommandLedger().transition("nope", CommandState.DISPATCHED)


def test_command_accessor_returns_registered_command() -> None:
    ledger = CommandLedger()
    command = _command()
    ledger.register(command)
    assert ledger.command("cmd-1") == command
    assert ledger.command("nope") is None
