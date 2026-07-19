"""Unit tests for the capability model and value validation."""

from __future__ import annotations

import pytest

from jarvis.runtime_protocol.protocol.capability import (
    BooleanCapability,
    NumberCapability,
    Operation,
    validate_request,
)


def test_number_capability_rejects_nonpositive_step() -> None:
    with pytest.raises(ValueError):
        NumberCapability(minimum=0, maximum=100, step=0)


def test_number_capability_rejects_inverted_range() -> None:
    with pytest.raises(ValueError):
        NumberCapability(minimum=100, maximum=0, step=10)


def test_set_within_range_and_on_grid_is_valid() -> None:
    cap = NumberCapability(minimum=0, maximum=100, step=10)
    assert validate_request(cap, Operation.SET, 30) is None


def test_set_outside_range_is_invalid() -> None:
    cap = NumberCapability(minimum=0, maximum=100, step=10)
    failure = validate_request(cap, Operation.SET, 130)
    assert failure is not None and "outside" in failure.detail


def test_set_off_step_grid_is_invalid() -> None:
    cap = NumberCapability(minimum=0, maximum=100, step=10)
    failure = validate_request(cap, Operation.SET, 25)
    assert failure is not None and "step grid" in failure.detail


def test_number_rejects_bool_value() -> None:
    cap = NumberCapability(minimum=0, maximum=100, step=10)
    failure = validate_request(cap, Operation.SET, True)
    assert failure is not None and "numeric" in failure.detail


def test_decrement_delta_must_be_positive_multiple_of_step() -> None:
    cap = NumberCapability(minimum=0, maximum=100, step=10)
    assert validate_request(cap, Operation.DECREMENT, 10) is None
    assert validate_request(cap, Operation.DECREMENT, -10) is not None
    assert validate_request(cap, Operation.DECREMENT, 15) is not None


def test_unsupported_operation_is_flagged() -> None:
    cap = NumberCapability(
        minimum=0, maximum=100, step=10, operations=frozenset({Operation.SET})
    )
    failure = validate_request(cap, Operation.INCREMENT, 10)
    assert failure is not None and "not supported" in failure.detail


def test_boolean_set_requires_bool() -> None:
    cap = BooleanCapability()
    assert validate_request(cap, Operation.SET, True) is None
    failure = validate_request(cap, Operation.SET, 1)
    assert failure is not None and "bool" in failure.detail


def test_boolean_toggle_ignores_value() -> None:
    cap = BooleanCapability()
    assert validate_request(cap, Operation.TOGGLE, 0) is None


def test_float_step_grid_tolerates_rounding() -> None:
    cap = NumberCapability(minimum=2700, maximum=6500, step=100)
    assert validate_request(cap, Operation.SET, 3000) is None
