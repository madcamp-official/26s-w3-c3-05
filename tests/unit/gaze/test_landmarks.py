"""Verified MediaPipe head-pose sign convention."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("mediapipe")

from jarvis.gaze.landmarks import rotation_matrix_to_euler_deg  # noqa: E402


def test_upward_head_pitch_is_positive_in_face_observation_contract() -> None:
    """MediaPipe의 upward x rotation(-20°)을 프로젝트의 up-positive로 바꾼다."""
    raw_pitch = math.radians(-20.0)
    matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, math.cos(raw_pitch), -math.sin(raw_pitch), 0.0],
            [0.0, math.sin(raw_pitch), math.cos(raw_pitch), 0.0,],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    _, pitch_deg, _ = rotation_matrix_to_euler_deg(matrix)

    assert pitch_deg == pytest.approx(20.0)
