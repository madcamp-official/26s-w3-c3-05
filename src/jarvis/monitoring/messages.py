"""System message log shown in the app's bottom panel.

A small thread-safe ring buffer of timestamped messages. The camera thread and
the UI thread both touch it, so it is guarded by a lock. Kept free of Qt so it is
unit-testable on its own.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock
from time import monotonic


class MessageLevel(StrEnum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class SystemMessage:
    timestamp_ms: int
    level: MessageLevel
    text: str


class MessageLog:
    """Bounded, thread-safe log of the most recent system messages."""

    def __init__(self, capacity: int = 200) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._messages: deque[SystemMessage] = deque(maxlen=capacity)
        self._lock = Lock()

    def add(self, level: MessageLevel, text: str) -> SystemMessage:
        message = SystemMessage(
            timestamp_ms=int(monotonic() * 1000), level=level, text=text
        )
        with self._lock:
            self._messages.append(message)
        return message

    def info(self, text: str) -> SystemMessage:
        return self.add(MessageLevel.INFO, text)

    def warn(self, text: str) -> SystemMessage:
        return self.add(MessageLevel.WARN, text)

    def error(self, text: str) -> SystemMessage:
        return self.add(MessageLevel.ERROR, text)

    def recent(self, limit: int | None = None) -> list[SystemMessage]:
        """Messages oldest→newest, optionally only the last ``limit``."""
        with self._lock:
            messages = list(self._messages)
        if limit is not None:
            return messages[-limit:]
        return messages
