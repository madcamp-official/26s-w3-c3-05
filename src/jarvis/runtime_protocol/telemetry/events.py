"""Structured trace events for runtime observability.

Every load-bearing runtime event — tracking loss, queue drop, lock transition,
intent commit/reject, command state transition — is recorded as a
:class:`TraceEvent` carrying a correlation id and a shared-clock timestamp so a
single interaction can be reconstructed end to end (development-principles 5.5).

Traces must never contain raw frames or secrets. This module only stores the
short ``detail`` strings callers pass; callers are responsible for keeping tokens
and image data out of them (the adapters already mask secrets before this point).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import Lock
from typing import Protocol

from jarvis.runtime_protocol.capture.clock import RuntimeClock


class EventKind(StrEnum):
    """The trackable runtime events (development-principles 5.5)."""

    TRACKING_LOST = "TRACKING_LOST"
    QUEUE_DROP = "QUEUE_DROP"
    LOCK_TRANSITION = "LOCK_TRANSITION"
    INTENT_COMMIT = "INTENT_COMMIT"
    INTENT_REJECT = "INTENT_REJECT"
    COMMAND_STATE = "COMMAND_STATE"


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One recorded runtime event on the shared monotonic clock.

    ``correlation_id`` ties related events together — typically an ``intent_id``
    or ``command_id`` — so one interaction's events can be grouped.
    """

    timestamp_ms: int
    kind: EventKind
    correlation_id: str
    detail: str = ""


class TraceSink(Protocol):
    """Receives trace events. Implementations decide where they go."""

    def record(self, event: TraceEvent) -> None: ...


class InMemoryTraceSink:
    """Collects events in memory for tests, replay, and the monitoring view."""

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []
        self._lock = Lock()

    def record(self, event: TraceEvent) -> None:
        with self._lock:
            self._events.append(event)

    def events(self) -> list[TraceEvent]:
        """A snapshot copy of all recorded events, in record order."""
        with self._lock:
            return list(self._events)

    def by_correlation(self, correlation_id: str) -> list[TraceEvent]:
        with self._lock:
            return [e for e in self._events if e.correlation_id == correlation_id]


class Tracer:
    """Stamps events with the shared clock and forwards them to a sink."""

    def __init__(self, clock: RuntimeClock, sink: TraceSink) -> None:
        self._clock = clock
        self._sink = sink

    def record(
        self, kind: EventKind, correlation_id: str, detail: str = ""
    ) -> TraceEvent:
        """Record an event stamped at the current monotonic time."""
        event = TraceEvent(
            timestamp_ms=self._clock.now_ms(),
            kind=kind,
            correlation_id=correlation_id,
            detail=detail,
        )
        self._sink.record(event)
        return event
