from __future__ import annotations

import numpy as np
import pytest

from jarvis.gaze.direction import direction_to_yaw_pitch, yaw_pitch_to_direction


@pytest.mark.parametrize("yaw,pitch", [(0.0, 0.0), (25.0, -10.0), (-30.0, 15.0)])
def test_yaw_pitch_vector_round_trip(yaw: float, pitch: float) -> None:
    direction = yaw_pitch_to_direction(yaw, pitch)
    np.testing.assert_allclose(np.linalg.norm(direction), 1.0)
    actual_yaw, actual_pitch = direction_to_yaw_pitch(direction)
    assert actual_yaw == pytest.approx(yaw)
    assert actual_pitch == pytest.approx(pitch)
