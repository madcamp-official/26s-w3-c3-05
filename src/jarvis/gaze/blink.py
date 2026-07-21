"""Adaptive blink detection from MediaPipe eyelid geometry.

An absolute eye-aspect threshold alone is brittle because the normal opening
differs by user and head pose.  This detector keeps a slowly decaying personal
open-eye baseline, closes early when the lids collapse relative to that
baseline, and uses hysteresis before declaring the eyes open again.
"""

from __future__ import annotations

from jarvis.gaze.config import GazeConfig


class AdaptiveBlinkDetector:
    """Classify bilateral eye opening ratios with an adaptive baseline."""

    def __init__(self, config: GazeConfig = GazeConfig()) -> None:
        self._config = config
        self._left_open_baseline: float | None = None
        self._right_open_baseline: float | None = None
        self._closed = False

    @property
    def open_baseline(self) -> float | None:
        """Current personal open-eye reference, exposed for diagnostics/tests."""
        available = [
            value
            for value in (self._left_open_baseline, self._right_open_baseline)
            if value is not None
        ]
        return min(available) if available else None

    @property
    def eye_baselines(self) -> tuple[float | None, float | None]:
        return self._left_open_baseline, self._right_open_baseline

    def reset(self) -> None:
        self._left_open_baseline = None
        self._right_open_baseline = None
        self._closed = False

    def update(self, left_ratio: float, right_ratio: float) -> bool:
        """Return false only when both eyelids jointly indicate a blink.

        A turned face commonly makes one projected eye look much smaller. Treating
        the minimum ratio as bilateral closure froze the gaze vector during head
        turns, so each eye now has its own baseline and closure requires both.
        """
        # An unconditional absolute floor permanently classified users whose
        # normal projected eye opening was small as closed.  Keep half of the
        # configured value only as a collapsed-eye guard and learn the actual
        # open-eye threshold from the user's own baseline.
        collapsed_floor = self._config.eye_closed_ratio_threshold * 0.5

        if self._left_open_baseline is None and left_ratio > 0.0:
            self._left_open_baseline = left_ratio
        if self._right_open_baseline is None and right_ratio > 0.0:
            self._right_open_baseline = right_ratio

        left_baseline = self._left_open_baseline or collapsed_floor
        right_baseline = self._right_open_baseline or collapsed_floor
        ratio_factor = (
            self._config.blink_reopen_ratio
            if self._closed
            else self._config.blink_close_ratio
        )
        left_threshold = max(collapsed_floor, left_baseline * ratio_factor)
        right_threshold = max(collapsed_floor, right_baseline * ratio_factor)
        self._closed = left_ratio < left_threshold and right_ratio < right_threshold

        if not self._closed:
            # Track a wider opening immediately, but let the baseline decay only
            # slowly as pose changes. A partial blink therefore cannot redefine
            # itself as the new normal on the following frame.
            decay = self._config.eye_openness_baseline_decay
            if left_ratio >= left_threshold:
                self._left_open_baseline = max(
                    left_ratio,
                    left_baseline * (1.0 - decay),
                )
            if right_ratio >= right_threshold:
                self._right_open_baseline = max(
                    right_ratio,
                    right_baseline * (1.0 - decay),
                )

        return not self._closed
