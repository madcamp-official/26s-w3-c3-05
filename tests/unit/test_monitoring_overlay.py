"""Unit tests for the frame overlay (require OpenCV, part of the ui extra)."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from jarvis.contracts import TargetEstimate  # noqa: E402
from jarvis.gaze.features import FaceObservation, GazeVector  # noqa: E402
from jarvis.monitoring.gaze_source import GazeSnapshot  # noqa: E402
from jarvis.monitoring.overlay import draw_gaze_overlay, draw_hud, placeholder_frame  # noqa: E402


def test_draw_hud_modifies_frame() -> None:
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    before = frame.copy()
    draw_hud(frame, ["30.0 FPS", "frame #1"])
    assert not np.array_equal(before, frame)  # something was drawn
    assert frame.shape == (120, 200, 3)


def test_draw_hud_no_lines_is_noop() -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    before = frame.copy()
    draw_hud(frame, [])
    assert np.array_equal(before, frame)


def test_placeholder_frame_shape_and_content() -> None:
    frame = placeholder_frame(width=320, height=240, text="NO CAMERA")
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8
    assert frame.any()  # not all black — has text and background


def test_draw_gaze_overlay_uses_real_snapshot() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    observation = FaceObservation(
        timestamp_ms=100,
        frame_id=3,
        left_iris_relative=(0.2, -0.1),
        right_iris_relative=(0.1, -0.1),
        head_yaw_deg=10.0,
        head_pitch_deg=-3.0,
        head_roll_deg=1.0,
        eye_tracking_confidence=1.0,
        face_tracking_confidence=1.0,
        face_detected=True,
    )
    snapshot = GazeSnapshot(
        observation=observation,
        gaze_vector=GazeVector(np.array([0.2, 0.0, 0.9797959]), 1.0, 100, 3),
        estimate=TargetEstimate(100, 3, "UNKNOWN", 0.0, 0.0, 1.0),
        lock_state="SEARCHING",
    )

    draw_gaze_overlay(frame, snapshot)

    assert frame.any()
