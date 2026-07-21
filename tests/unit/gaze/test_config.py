"""Gaze configuration rejects unsafe or nonsensical thresholds."""

from __future__ import annotations

import pytest

from jarvis.gaze.config import GazeConfig


@pytest.mark.parametrize(
    "overrides",
    [
        {"smoothing_window_frames": 0},
        {"target_lock_ttl_ms": 0},
        {"confirmed_unknown_timeout_ms": 0},
        {"minimum_probability": float("nan")},
        {"minimum_tracking_confidence": 1.1},
        {"unknown_max_angle_deg": 181.0},
        {"iris_jump_threshold": 0.0},
        {"max_valid_eye_offset": -0.1},
        {"registration_max_area_radius_deg": 3.0},
        {"registration_max_area_radius_deg": 20.0},
        {"target_area_scale_flex": -0.1},
        {"target_area_scale_flex": 1.1},
        {"personal_gaze_feature_weight": 0.0},
        {"personal_head_feature_weight": -0.1},
        {"target_acceleration_alignment_weight": 1.1},
        {"gaze_motion_min_speed_deg_s": -0.1},
        {"gaze_motion_max_interval_ms": 0},
        {"target_match_tolerance": 0.99},
        {"target_match_tolerance": 2.01},
    ],
)
def test_invalid_config_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        GazeConfig(**overrides)  # type: ignore[arg-type]


def test_personal_classifier_prioritizes_gaze_over_head_features() -> None:
    config = GazeConfig()

    assert config.personal_feature_weights == (2.0, 2.0, 0.4, 0.4, 0.4, 0.6)
