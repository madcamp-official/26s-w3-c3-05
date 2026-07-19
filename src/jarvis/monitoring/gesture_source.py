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


class UntrainedGestureSource:
    """Honest source for when the Gesture pipeline exists but its model is untrained.

    The dev-2 pipeline (spotting/fusion) is now implemented, but its Causal TCN
    classifier ships with random, untrained weights (``ModelMetadata.trained=
    False``). Emitting its output as a "recognized gesture" would fabricate a
    result, so this source stays ``available=False`` and yields nothing — the
    sidebar shows the honest reason instead of invented detections. When the model
    is trained, a real source that adapts ``GestureEstimate`` replaces this.
    """

    available = False
    status_text = "제스처 파이프라인 구현됨 · 모델 미학습(무작위) — 인식 비활성"

    def poll(self) -> list[RecognizedGesture]:
        return []
