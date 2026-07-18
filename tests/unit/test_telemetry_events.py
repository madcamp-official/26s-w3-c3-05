"""Unit tests for trace events and the tracer."""

from __future__ import annotations

from jarvis.runtime_protocol.capture.clock import RuntimeClock
from jarvis.runtime_protocol.telemetry.events import (
    EventKind,
    InMemoryTraceSink,
    Tracer,
)


class FakeTime:
    def __init__(self) -> None:
        self.value_ns = 0

    def set_ms(self, ms: int) -> None:
        self.value_ns = ms * 1_000_000

    def __call__(self) -> int:
        return self.value_ns


def test_tracer_stamps_event_with_clock_time() -> None:
    clock = FakeTime()
    clock.set_ms(1234)
    sink = InMemoryTraceSink()
    tracer = Tracer(RuntimeClock(clock), sink)

    event = tracer.record(EventKind.INTENT_COMMIT, "intent-1", "committed")

    assert event.timestamp_ms == 1234
    assert event.kind == EventKind.INTENT_COMMIT
    assert event.correlation_id == "intent-1"
    assert event.detail == "committed"


def test_events_recorded_in_order() -> None:
    clock = FakeTime()
    sink = InMemoryTraceSink()
    tracer = Tracer(RuntimeClock(clock), sink)

    clock.set_ms(10)
    tracer.record(EventKind.LOCK_TRANSITION, "intent-1", "SEARCHING->LOCKED")
    clock.set_ms(20)
    tracer.record(EventKind.COMMAND_STATE, "cmd-1", "VALIDATED->DISPATCHED")

    events = sink.events()
    assert [e.timestamp_ms for e in events] == [10, 20]
    assert [e.kind for e in events] == [
        EventKind.LOCK_TRANSITION,
        EventKind.COMMAND_STATE,
    ]


def test_by_correlation_filters_events() -> None:
    sink = InMemoryTraceSink()
    tracer = Tracer(RuntimeClock(FakeTime()), sink)

    tracer.record(EventKind.INTENT_COMMIT, "intent-1")
    tracer.record(EventKind.INTENT_REJECT, "intent-2", "low confidence")
    tracer.record(EventKind.COMMAND_STATE, "intent-1", "DISPATCHED")

    related = sink.by_correlation("intent-1")
    assert len(related) == 2
    assert {e.kind for e in related} == {
        EventKind.INTENT_COMMIT,
        EventKind.COMMAND_STATE,
    }


def test_events_snapshot_is_a_copy() -> None:
    sink = InMemoryTraceSink()
    tracer = Tracer(RuntimeClock(FakeTime()), sink)
    tracer.record(EventKind.QUEUE_DROP, "gaze", "dropped 1")

    snapshot = sink.events()
    snapshot.clear()
    assert len(sink.events()) == 1  # mutating the snapshot does not affect the sink
