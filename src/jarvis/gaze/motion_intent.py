"""Detect the landing direction of an iris movement without a learned model."""

from __future__ import annotations

import math
from dataclasses import dataclass

from jarvis.gaze.config import GazeConfig


@dataclass(frozen=True, slots=True)
class GazeSettleIntent:
    velocity_deg_s: tuple[float, float]
    age_ms: int


class GazeSettleTracker:
    """Remember the last meaningful iris velocity when a saccade settles.

    Fast motion arms the tracker. When its speed falls below the stop threshold,
    the most recent moving velocity becomes a short-lived overlap tie-break.
    Invalid eye frames reset the motion so a blink cannot create an intent.
    """

    def __init__(self, config: GazeConfig = GazeConfig()) -> None:
        self._config = config
        self.reset()

    def reset(self) -> None:
        self._previous_position: tuple[float, float] | None = None
        self._previous_timestamp_ms: int | None = None
        self._moving = False
        self._last_moving_velocity: tuple[float, float] | None = None
        self._intent_velocity: tuple[float, float] | None = None
        self._intent_timestamp_ms: int | None = None

    def update(
        self,
        position_deg: tuple[float, float] | None,
        timestamp_ms: int,
    ) -> GazeSettleIntent | None:
        if position_deg is None or not all(math.isfinite(value) for value in position_deg):
            self.reset()
            return None

        previous_position = self._previous_position
        previous_timestamp_ms = self._previous_timestamp_ms
        self._previous_position = position_deg
        self._previous_timestamp_ms = timestamp_ms
        if previous_position is None or previous_timestamp_ms is None:
            return self._active_intent(timestamp_ms)

        dt_ms = timestamp_ms - previous_timestamp_ms
        if dt_ms <= 0 or dt_ms > self._config.gaze_motion_max_interval_ms:
            self._moving = False
            self._last_moving_velocity = None
            return self._active_intent(timestamp_ms)

        dt_s = dt_ms / 1000.0
        velocity = (
            (position_deg[0] - previous_position[0]) / dt_s,
            (position_deg[1] - previous_position[1]) / dt_s,
        )
        speed = math.hypot(*velocity)
        if speed >= self._config.gaze_settle_start_speed_deg_s:
            self._moving = True
            self._last_moving_velocity = velocity
            self._intent_velocity = None
            self._intent_timestamp_ms = None
            return None

        if self._moving and speed <= self._config.gaze_settle_stop_speed_deg_s:
            self._moving = False
            self._intent_velocity = self._last_moving_velocity
            self._intent_timestamp_ms = timestamp_ms
            self._last_moving_velocity = None

        return self._active_intent(timestamp_ms)

    def _active_intent(self, timestamp_ms: int) -> GazeSettleIntent | None:
        if self._intent_velocity is None or self._intent_timestamp_ms is None:
            return None
        age_ms = timestamp_ms - self._intent_timestamp_ms
        if age_ms < 0 or age_ms > self._config.gaze_settle_memory_ms:
            self._intent_velocity = None
            self._intent_timestamp_ms = None
            return None
        return GazeSettleIntent(self._intent_velocity, age_ms)
