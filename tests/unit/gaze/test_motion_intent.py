from __future__ import annotations

import pytest

from jarvis.gaze.config import GazeConfig
from jarvis.gaze.motion_intent import GazeSettleTracker


def test_fast_iris_motion_emits_direction_only_when_it_settles() -> None:
    tracker = GazeSettleTracker(
        GazeConfig(
            gaze_settle_start_speed_deg_s=12.0,
            gaze_settle_stop_speed_deg_s=4.0,
            gaze_settle_memory_ms=500,
        )
    )

    assert tracker.update((0.0, 0.0), 0) is None
    assert tracker.update((2.0, 0.0), 100) is None
    intent = tracker.update((2.1, 0.0), 200)

    assert intent is not None
    assert intent.velocity_deg_s == pytest.approx((20.0, 0.0))
    assert intent.age_ms == 0


def test_settle_intent_expires_and_invalid_eye_frame_resets_motion() -> None:
    tracker = GazeSettleTracker(GazeConfig(gaze_settle_memory_ms=500))
    tracker.update((0.0, 0.0), 0)
    tracker.update((-2.0, 0.0), 100)
    intent = tracker.update((-2.1, 0.0), 200)
    assert intent is not None
    assert intent.velocity_deg_s[0] < 0.0

    assert tracker.update((-2.1, 0.0), 600) is not None
    assert tracker.update((-2.1, 0.0), 701) is None

    tracker.update((0.0, 0.0), 800)
    tracker.update((2.0, 0.0), 900)
    assert tracker.update(None, 950) is None
    assert tracker.update((2.0, 0.0), 1_000) is None
