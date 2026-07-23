"""Unit tests for the frame overlay (require OpenCV, part of the ui extra)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from jarvis.gaze.config import GazeConfig  # noqa: E402
from jarvis.gaze.classifier import DeviceGazeProfile, TargetClassifier  # noqa: E402
from jarvis.gaze.features import FaceObservation  # noqa: E402
from jarvis.gaze.lock import GazeLockStateMachine  # noqa: E402
from jarvis.gaze.smoothing import GazeSmoother  # noqa: E402
from jarvis.monitoring.gaze_probe import GazeSnapshot, evaluate  # noqa: E402
from jarvis.monitoring.hand_probe import HandSnapshot  # noqa: E402
from jarvis.monitoring.overlay import (  # noqa: E402
    draw_gaze_overlay,
    draw_hand_overlay,
    draw_hud,
    draw_registration_guidance,
    draw_target_heatmap,
    draw_target_minimap,
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


def _snapshot(
    *,
    detected: bool,
    iris_relative: tuple[float, float] = (0.1, -0.1),
    head_yaw_deg: float = 8.0,
) -> GazeSnapshot:
    config = GazeConfig()
    observation = FaceObservation(
        timestamp_ms=0,
        frame_id=0,
        left_iris_relative=iris_relative,
        right_iris_relative=iris_relative,
        head_yaw_deg=head_yaw_deg,
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


def test_draw_registration_guidance_adds_panel_and_progress() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    before = frame.copy()

    draw_registration_guidance(
        frame,
        title="REGISTRATION 1/2 - CENTER",
        instruction="KEEP EYES ON CENTER",
        progress=0.5,
    )

    assert not np.array_equal(before, frame)
    assert frame.shape == (240, 320, 3)


def test_draw_gaze_overlay_draws_when_tracking() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    before = frame.copy()
    draw_gaze_overlay(frame, _snapshot(detected=True))
    assert not np.array_equal(before, frame)  # ray + HUD drawn
    assert frame.shape == (240, 320, 3)


def test_mirrored_gaze_overlay_keeps_user_facing_yaw_direction() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    snapshot = _snapshot(detected=True, iris_relative=(-0.4, 0.0), head_yaw_deg=0.0)
    assert snapshot.smoothed_gaze_direction is not None
    assert snapshot.smoothed_gaze_direction[0] > 0.0

    draw_gaze_overlay(frame, snapshot, mirror=True)

    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2
    ray_band = frame[center_y - 8 : center_y + 8]
    left_pixels = int(np.count_nonzero(ray_band[:, center_x - 80 : center_x - 8]))
    right_pixels = int(np.count_nonzero(ray_band[:, center_x + 8 : center_x + 80]))
    assert right_pixels > left_pixels


def test_draw_target_heatmap_draws_registered_target_regions() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    config = GazeConfig()
    classifier = TargetClassifier(config)
    classifier.register_profile(
        DeviceGazeProfile(
            "monitor",
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            variance=np.radians(15.0) ** 2,
        )
    )
    snapshot = evaluate(
        FaceObservation(
            timestamp_ms=0,
            frame_id=0,
            left_iris_relative=(0.0, 0.0),
            right_iris_relative=(0.0, 0.0),
            head_yaw_deg=0.0,
            head_pitch_deg=0.0,
            head_roll_deg=0.0,
            eye_tracking_confidence=1.0,
            face_tracking_confidence=1.0,
            face_detected=True,
        ),
        smoother=GazeSmoother(config),
        classifier=classifier,
        lock=GazeLockStateMachine(config),
        config=config,
    )
    before = frame.copy()

    draw_target_heatmap(frame, snapshot)

    assert not np.array_equal(before, frame)


def _registered_snapshot() -> GazeSnapshot:
    config = GazeConfig()
    classifier = TargetClassifier(config)
    classifier.register_profile(
        DeviceGazeProfile(
            "monitor",
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            variance=np.radians(15.0) ** 2,
        )
    )
    return evaluate(
        FaceObservation(
            timestamp_ms=0,
            frame_id=0,
            left_iris_relative=(0.0, 0.0),
            right_iris_relative=(0.0, 0.0),
            head_yaw_deg=0.0,
            head_pitch_deg=0.0,
            head_roll_deg=0.0,
            eye_tracking_confidence=1.0,
            face_tracking_confidence=1.0,
            face_detected=True,
        ),
        smoother=GazeSmoother(config),
        classifier=classifier,
        lock=GazeLockStateMachine(config),
        config=config,
    )


def test_draw_target_minimap_draws_inset_when_targets_registered() -> None:
    """등록 물체가 있으면 우상단 미니맵이 그려진다 — 좌하단 영역은 건드리지 않는다."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    before = frame.copy()

    draw_target_minimap(frame, _registered_snapshot())

    assert not np.array_equal(before, frame)
    # 미니맵은 우상단 보조 창이다 — 프레임 좌하단 사분면은 그대로여야 한다.
    assert np.array_equal(before[140:, :140], frame[140:, :140])


def test_draw_target_minimap_is_noop_without_registered_targets() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    before = frame.copy()
    draw_target_minimap(frame, _snapshot(detected=True))  # 등록 물체 없음
    assert np.array_equal(before, frame)


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


def test_draw_hand_overlay_can_keep_skeleton_without_detail_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """시연 영상은 스켈레톤을 남기되 상세 숫자는 오른쪽 패널에만 그린다."""
    import jarvis.monitoring.overlay as overlay_module

    text_calls: list[object] = []
    monkeypatch.setattr(
        overlay_module, "_text_block", lambda *args, **kwargs: text_calls.append(args)
    )
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    draw_hand_overlay(frame, _hand_snapshot(detected=True), show_details=False)
    assert frame.any()  # bounding boxes + landmark skeleton remain
    assert text_calls == []


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
    frame = draw_hand_overlay(placeholder_frame(), _hand_snapshot(detected=True, tilt=None))
    assert frame is not None


def test_per_class_tilt_limit_overrides_global_gate() -> None:
    """자세별 한계가 전역 한계를 이긴다.

    two_fingers는 40°까지 허용되는데 전역 게이트는 20°다. 전역 게이트를 그대로
    보여주면 실제로는 허용된 자세에 "손을 세우세요"가 떠 사용자가 혼란스러워진다
    (실사용에서 발견: 스크롤 자세가 35°에서 거부로 표시됨).
    """
    from jarvis.gesture_fusion.pose_protocol import PosePrediction

    trusted = PosePrediction(
        label="two_fingers", confidence=0.98, trusted=True, palm_tilt_degrees=35.0
    )
    snapshot = dataclasses.replace(
        _hand_snapshot(detected=True, tilt=35.0, tilted=True), pose=trusted
    )
    allowed = draw_hand_overlay(placeholder_frame(), snapshot)

    rejected_pose = PosePrediction(
        label="index_point",
        confidence=0.71,
        trusted=False,
        reason="기울기 35° > index_point 허용 20°",
        palm_tilt_degrees=35.0,
    )
    rejected = draw_hand_overlay(
        placeholder_frame(),
        dataclasses.replace(
            _hand_snapshot(detected=True, tilt=35.0, tilted=True), pose=rejected_pose
        ),
    )
    # 같은 기울기·같은 전역 게이트 상태인데 자세에 따라 화면이 달라져야 한다.
    assert not np.array_equal(allowed, rejected)
