"""Gaze configuration rejects unsafe or nonsensical thresholds."""

from __future__ import annotations

import pytest

from jarvis.gaze.config import GazeConfig


@pytest.mark.parametrize(
    "overrides",
    [
        {"smoothing_window_frames": 0},
        {"target_lock_ttl_ms": 0},
        {"minimum_probability": float("nan")},
        {"minimum_tracking_confidence": 1.1},
        {"unknown_max_angle_deg": 181.0},
    ],
)
def test_invalid_config_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        GazeConfig(**overrides)  # type: ignore[arg-type]
