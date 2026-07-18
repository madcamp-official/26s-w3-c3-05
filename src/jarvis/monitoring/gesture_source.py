"""Where the 'recognized gestures' sidebar gets its data.

The sidebar binds to a :class:`GestureSource`. Today the real Gesture module
(dev-2) does not exist, so the app uses :class:`NullGestureSource`, which honestly
reports "unavailable" and yields nothing — the sidebar shows an empty, labeled
state rather than faked detections. When dev-2 lands, a real source that adapts
``GestureEstimate`` plugs in here without changing the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RecognizedGesture:
    """One recognized gesture event for the sidebar."""

    timestamp_ms: int
    gesture: str
    confidence: float
    phase: str


class GestureSource(Protocol):
    """Yields recognized gestures. Implementations decide liveness."""

    @property
    def available(self) -> bool:
        """True when a real gesture recognizer backs this source."""
        ...

    @property
    def status_text(self) -> str:
        """Short human-readable state for the sidebar header."""
        ...

    def poll(self) -> list[RecognizedGesture]:
        """Recognized gestures since the last poll (empty if none/unavailable)."""
        ...


class NullGestureSource:
    """Placeholder for the not-yet-implemented Gesture module (dev-2)."""

    available = False
    status_text = "제스처 모듈 미구현 (2인 파트)"

    def poll(self) -> list[RecognizedGesture]:
        return []
