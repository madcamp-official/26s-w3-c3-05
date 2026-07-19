"""Unit tests for the frame overlay (require OpenCV, part of the ui extra)."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from jarvis.gaze.config import GazeConfig  # noqa: E402
from jarvis.gaze.classifier import TargetClassifier  # noqa: E402
from jarvis.gaze.features import FaceObservation  # noqa: E402
from jarvis.gaze.lock import GazeLockStateMachine  # noqa: E402
from jarvis.gaze.smoothing import GazeSmoother  # noqa: E402
from jarvis.monitoring.gaze_probe import GazeSnapshot, evaluate  # noqa: E402
from jarvis.monitoring.hand_probe import HandSnapshot  # noqa: E402
from jarvis.monitoring.overlay import (  # noqa: E402
    draw_gaze_overlay,
    draw_hand_overlay,
    draw_hud,
    placeholder_frame,
)


def _hand_snapshot(*, detected: bool) -> HandSnapshot:
    points = tuple((0.4 + 0.01 * i, 0.4 + 0.01 * i) for i in range(21)) if detected else None
    return HandSnapshot(
        timestamp_ms=0,
        frame_id=0,
        hand_detected=detected,
        handedness="Right" if detected else "",
        handedness_score=0.95 if detected else 0.0,
        detection_confidence=0.9 if detected else 0.0,
        palm_scale=0.2 if detected else 0.0,
        image_points=points,
        landmark_count=21 if detected else 0,
        inference_ms=7.0,
        smoothed=True,
    )


def _snapshot(*, detected: bool) -> GazeSnapshot:
    config = GazeConfig()
    observation = FaceObservation(
        timestamp_ms=0,
        frame_id=0,
        left_iris_relative=(0.1, -0.1),
        right_iris_relative=(0.1, -0.1),
        head_yaw_deg=8.0,
        head_pitch_deg=-4.0,
        head_roll_deg=0.0,
        eye_tracking_confidence=1.0,
        face_tracking_confidence=1.0,
        face_detected=detected,
    )
    return evaluate(
        observation,
        smoother=GazeSmoother(config),
        classifier=TargetClassifier(config),
        lock=GazeLockStateMachine(config),
        config=config,
    )


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


def test_draw_gaze_overlay_draws_when_tracking() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    before = frame.copy()
    draw_gaze_overlay(frame, _snapshot(detected=True))
    assert not np.array_equal(before, frame)  # ray + HUD drawn
    assert frame.shape == (240, 320, 3)


def test_draw_gaze_overlay_shows_tracking_lost_banner() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    before = frame.copy()
    snapshot = _snapshot(detected=False)
    assert snapshot.tracking_lost is True
    draw_gaze_overlay(frame, snapshot)
    # the banner is drawn along the bottom strip
    assert not np.array_equal(before[-30:], frame[-30:])


def test_draw_hand_overlay_draws_skeleton_when_tracking() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    before = frame.copy()
    draw_hand_overlay(frame, _hand_snapshot(detected=True))
    assert not np.array_equal(before, frame)  # skeleton + HUD drawn
    assert frame.shape == (240, 320, 3)


def test_draw_hand_overlay_noop_when_no_hand() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    before = frame.copy()
    draw_hand_overlay(frame, _hand_snapshot(detected=False))
    assert np.array_equal(before, frame)  # nothing drawn for a lost hand


def test_placeholder_frame_shape_and_content() -> None:
    frame = placeholder_frame(width=320, height=240, text="NO CAMERA")
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8
    assert frame.any()  # not all black — has text and background
