"""Unit tests for the frame overlay (require OpenCV, part of the ui extra)."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from jarvis.monitoring.overlay import draw_hud, placeholder_frame  # noqa: E402


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
