"""Unit tests for the bounded latest-frame queue."""

from __future__ import annotations

import pytest

from jarvis.runtime_protocol.capture.queue import BoundedLatestQueue


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError):
        BoundedLatestQueue[int](0)


def test_fifo_order_within_capacity() -> None:
    queue = BoundedLatestQueue[int](3)
    for value in (1, 2, 3):
        assert queue.put(value) is False
    assert [queue.get_nowait() for _ in range(3)] == [1, 2, 3]


def test_drops_oldest_when_full_and_keeps_latest() -> None:
    queue = BoundedLatestQueue[int](2)
    queue.put(1)
    queue.put(2)
    dropped = queue.put(3)  # full: drops 1, keeps 2 and 3

    assert dropped is True
    assert queue.dropped == 1
    assert [queue.get_nowait(), queue.get_nowait()] == [2, 3]


def test_get_nowait_returns_none_when_empty() -> None:
    queue = BoundedLatestQueue[int](1)
    assert queue.get_nowait() is None


def test_get_times_out_when_no_item_arrives() -> None:
    queue = BoundedLatestQueue[int](1)
    assert queue.get(timeout=0.01) is None


def test_len_reflects_retained_items() -> None:
    queue = BoundedLatestQueue[int](5)
    queue.put(1)
    queue.put(2)
    assert len(queue) == 2
