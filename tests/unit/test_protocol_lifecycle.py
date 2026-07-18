"""Unit tests for the command lifecycle transition rules."""

from __future__ import annotations

from jarvis.runtime_protocol.protocol.lifecycle import (
    TERMINAL_STATES,
    CommandState,
    can_transition,
)


def test_success_path_edges_are_legal() -> None:
    assert can_transition(CommandState.VALIDATED, CommandState.DISPATCHED)
    assert can_transition(CommandState.DISPATCHED, CommandState.ACKNOWLEDGED)
    assert can_transition(CommandState.ACKNOWLEDGED, CommandState.VERIFIED)


def test_validated_may_expire() -> None:
    assert can_transition(CommandState.VALIDATED, CommandState.EXPIRED)


def test_validated_may_be_rejected_when_unroutable() -> None:
    # The dispatcher takes this edge when it cannot route a validated command.
    assert can_transition(CommandState.VALIDATED, CommandState.REJECTED)
    assert CommandState.REJECTED in TERMINAL_STATES


def test_dispatched_may_fail_or_be_unverified() -> None:
    assert can_transition(CommandState.DISPATCHED, CommandState.FAILED)
    assert can_transition(CommandState.DISPATCHED, CommandState.UNVERIFIED)


def test_cannot_skip_states() -> None:
    assert not can_transition(CommandState.VALIDATED, CommandState.VERIFIED)
    assert not can_transition(CommandState.VALIDATED, CommandState.ACKNOWLEDGED)


def test_terminal_states_have_no_outgoing_edges() -> None:
    for terminal in TERMINAL_STATES:
        for dst in CommandState:
            assert not can_transition(terminal, dst)


def test_expired_and_failed_are_terminal() -> None:
    assert CommandState.EXPIRED in TERMINAL_STATES
    assert CommandState.FAILED in TERMINAL_STATES
    assert CommandState.UNVERIFIED in TERMINAL_STATES
