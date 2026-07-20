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


def _hand_snapshot(
    *, detected: bool, tilt: float | None = 5.0, tilted: bool = False
) -> HandSnapshot:
    points = tuple((0.4 + 0.01 * i, 0.4 + 0.01 * i) for i in range(21)) if detected else None
    model = tuple((0.1 * i - 1.0, 0.1 * i - 1.0) for i in range(21)) if detected else None
    return HandSnapshot(
        timestamp_ms=0,
        frame_id=0,
        hand_detected=detected,
        handedness="Right" if detected else "",
        handedness_score=0.95 if detected else 0.0,
        detection_confidence=0.9 if detected else 0.0,
        palm_scale=0.2 if detected else 0.0,
        image_points=points,
        model_points=model,
        model_points_raw=model,
        landmark_count=21 if detected else 0,
        inference_ms=7.0,
        smoothed=True,
        wrist_velocity=(1.5, -0.5) if detected else None,
        wrist_acceleration=(0.2, 0.1) if detected else None,
        palm_tilt_degrees=tilt if detected else None,
        palm_tilted=tilted if detected else False,
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


def test_render_normalized_hand_draws_skeleton() -> None:
    from jarvis.monitoring.overlay import render_normalized_hand

    points = tuple((0.1 * i - 1.0, 0.05 * i) for i in range(21))
    canvas = render_normalized_hand(points, size=200)
    assert canvas.shape == (200, 200, 3)
    assert canvas.any()  # something drawn


def test_render_normalized_hand_handles_no_hand() -> None:
    from jarvis.monitoring.overlay import render_normalized_hand

    canvas = render_normalized_hand(None, size=120)
    assert canvas.shape == (120, 120, 3)


def test_render_normalized_hand_is_not_vertically_flipped() -> None:
    """Fingers-up (negative y in image convention) must draw ABOVE the wrist.

    Regression for a y-flip bug that rendered the hand upside down.
    """
    from jarvis.monitoring.overlay import render_normalized_hand

    size = 240
    up = tuple((0.0, -1.5) if i else (0.0, 0.0) for i in range(21))  # fingertips above wrist
    down = tuple((0.0, 1.5) if i else (0.0, 0.0) for i in range(21))  # fingertips below wrist

    def _mean_row(canvas: np.ndarray) -> float:
        mask = canvas[30:].sum(axis=2) > 120  # skip the tag row band at top
        rows = np.nonzero(mask)[0]
        return float(rows.mean())

    assert _mean_row(render_normalized_hand(up, size=size)) < _mean_row(
        render_normalized_hand(down, size=size)
    )


def test_render_vector_draws_arrow() -> None:
    from jarvis.monitoring.overlay import render_vector

    canvas = render_vector((0.3, 0.1), size=200, scale=0.3)
    assert canvas.shape == (200, 200, 3)
    assert canvas.any()  # something drawn


def test_render_vector_handles_no_signal() -> None:
    from jarvis.monitoring.overlay import render_vector

    canvas = render_vector(None, size=120, scale=1.0)
    assert canvas.shape == (120, 120, 3)


def test_render_vector_zero_scale_does_not_crash() -> None:
    """추적 첫 프레임처럼 running-max scale이 아직 0이어도 예외 없이 그려야 한다."""
    from jarvis.monitoring.overlay import render_vector

    canvas = render_vector((0.0, 0.0), size=100, scale=0.0)
    assert canvas.shape == (100, 100, 3)


def test_render_vector_length_scales_with_magnitude() -> None:
    """스케일 대비 벡터 크기가 클수록 화살표가 중심에서 더 멀리 뻗어야 한다."""
    from jarvis.monitoring.overlay import render_vector

    size = 200

    def _tip_distance_from_center(canvas: np.ndarray) -> float:
        # green arrow pixels (BGR ~ (80,200,80)) vs background/ring/text — take the
        # farthest lit pixel from center as a proxy for the arrowhead position.
        mask = (canvas[:, :, 1] > 150) & (canvas[:, :, 0] < 150)
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return 0.0
        center = size / 2
        return float(np.max(np.hypot(xs - center, ys - center)))

    small = render_vector((0.1, 0.0), size=size, scale=1.0)
    large = render_vector((0.9, 0.0), size=size, scale=1.0)
    assert _tip_distance_from_center(large) > _tip_distance_from_center(small)


def test_render_vector_mirror_flips_x_only() -> None:
    from jarvis.monitoring.overlay import render_vector

    size = 200

    def _centroid_x(canvas: np.ndarray) -> float:
        mask = (canvas[:, :, 1] > 150) & (canvas[:, :, 0] < 150)
        xs = np.nonzero(mask)[1]
        return float(xs.mean())

    normal = render_vector((0.5, 0.0), size=size, scale=1.0, mirror=False)
    mirrored = render_vector((0.5, 0.0), size=size, scale=1.0, mirror=True)
    assert _centroid_x(mirrored) < size / 2 < _centroid_x(normal)


def test_placeholder_frame_shape_and_content() -> None:
    frame = placeholder_frame(width=320, height=240, text="NO CAMERA")
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8
    assert frame.any()  # not all black — has text and background


def test_tilt_gate_state_is_visible_on_overlay() -> None:
    """게이트에 걸린 프레임은 화면이 눈에 띄게 달라져야 한다.

    거부를 조용히 무시하면 사용자는 왜 반응이 없는지 알 수 없어 손을 세울 기회조차
    얻지 못한다. 거부 표시(붉은 테두리 + 안내 문구)가 실제로 픽셀을 바꾸는지 본다.
    """
    ok = draw_hand_overlay(placeholder_frame(), _hand_snapshot(detected=True, tilt=5.0))
    rejected = draw_hand_overlay(
        placeholder_frame(), _hand_snapshot(detected=True, tilt=42.0, tilted=True)
    )
    assert not np.array_equal(ok, rejected)


def test_unknown_tilt_renders_without_crashing() -> None:
    """z를 못 내는 소스(각도 None)에서도 오버레이가 그려진다 — 게이트만 없을 뿐."""
    frame = draw_hand_overlay(
        placeholder_frame(), _hand_snapshot(detected=True, tilt=None)
    )
    assert frame is not None
