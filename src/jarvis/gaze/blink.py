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
        self._open_baseline: float | None = None
        self._closed = False

    @property
    def open_baseline(self) -> float | None:
        """Current personal open-eye reference, exposed for diagnostics/tests."""
        return self._open_baseline

    def reset(self) -> None:
        self._open_baseline = None
        self._closed = False

    def update(self, left_ratio: float, right_ratio: float) -> bool:
        """Return whether both eyes are reliably open for iris estimation."""
        ratio = min(left_ratio, right_ratio)
        absolute = self._config.eye_closed_ratio_threshold

        if self._open_baseline is None:
            if ratio < absolute:
                self._closed = True
                return False
            self._open_baseline = ratio
            self._closed = False
            return True

        close_threshold = max(absolute, self._open_baseline * self._config.blink_close_ratio)
        reopen_threshold = max(
            absolute,
            self._open_baseline * self._config.blink_reopen_ratio,
        )

        if self._closed:
            self._closed = ratio < reopen_threshold
        else:
            self._closed = ratio < close_threshold

        if not self._closed:
            # Track a wider opening immediately, but let the baseline decay only
            # slowly as pose changes. A partial blink therefore cannot redefine
            # itself as the new normal on the following frame.
            decay = self._config.eye_openness_baseline_decay
            self._open_baseline = max(ratio, self._open_baseline * (1.0 - decay))

        return not self._closed
