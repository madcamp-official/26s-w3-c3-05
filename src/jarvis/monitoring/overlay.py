"""Draw a heads-up overlay on a webcam frame.

Pure frame-in → frame-out helpers used by the Live tab. Kept separate from Qt so
the drawing is unit-testable on a plain numpy array. Requires OpenCV (``ui``
extra); it is imported here, not in the package ``__init__``, so importing the
rest of ``jarvis.monitoring`` never needs cv2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from jarvis.monitoring.gaze_probe import GazeSnapshot
    from jarvis.monitoring.hand_probe import HandSnapshot

Frame = NDArray[np.uint8]

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Standard MediaPipe hand skeleton: 21 landmarks connected finger by finger.
_HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                # palm base
)

# BGR colors keyed by GazeLockState value (kept as strings to avoid importing
# the enum at runtime — overlay stays decoupled from the gaze package).
_LOCK_BGR = {
    "SEARCHING": (150, 150, 150),
    "CANDIDATE": (60, 190, 230),
    "TARGET_LOCKED": (80, 200, 80),
    "GESTURE_WAIT": (230, 180, 60),
    "EXPIRED": (100, 100, 240),
    "COMMITTED": (80, 220, 120),
}


def draw_hud(frame: Frame, lines: list[str], *, recording: bool = True) -> Frame:
    """Draw a translucent HUD panel with ``lines`` and a REC dot. Mutates ``frame``."""
    if not lines:
        return frame
    pad = 8
    line_h = 22
    panel_w = min(frame.shape[1], 8 + max(len(s) for s in lines) * 11)
    panel_h = pad * 2 + line_h * len(lines)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, dst=frame)

    for i, text in enumerate(lines):
        y = pad + line_h * (i + 1) - 6
        cv2.putText(frame, text, (pad, y), _FONT, 0.5, (235, 235, 235), 1, cv2.LINE_AA)

    if recording:
        cv2.circle(frame, (frame.shape[1] - 18, 18), 6, (60, 60, 220), thickness=-1)
    return frame


def _text_block(frame: Frame, lines: list[tuple[str, tuple[int, int, int]]], origin: tuple[int, int]) -> None:
    """Draw colored text lines with a translucent backing at ``origin`` (top-left)."""
    if not lines:
        return
    x, y = origin
    line_h = 20
    width = 8 + max(len(t) for t, _ in lines) * 10
    height = 6 + line_h * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, dst=frame)
    for i, (text, color) in enumerate(lines):
        ty = y + line_h * (i + 1) - 5
        cv2.putText(frame, text, (x + 5, ty), _FONT, 0.48, color, 1, cv2.LINE_AA)


def draw_gaze_overlay(frame: Frame, snapshot: GazeSnapshot) -> Frame:
    """Overlay the live Gaze pipeline result: gaze ray, head angles, lock state.

    Everything drawn comes from the real snapshot. Tracking loss is shown as a
    red banner instead of a stale ray — the overlay never invents a direction.
    Mutates and returns ``frame``.
    """
    h, w = frame.shape[:2]
    white = (235, 235, 235)
    grey = (170, 170, 170)

    if snapshot.tracking_lost:
        cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), thickness=-1)
        cv2.putText(
            frame, "TRACKING LOST / 얼굴 추적 손실", (10, h - 9),
            _FONT, 0.6, (90, 90, 240), 2, cv2.LINE_AA,
        )
        return frame

    state = str(snapshot.lock_state)
    ray_color = _LOCK_BGR.get(state, grey)

    # Gaze ray: use the same smoothed direction that the classifier consumes.
    left_eye = snapshot.left_eye_center_normalized
    right_eye = snapshot.right_eye_center_normalized
    if left_eye is not None and right_eye is not None:
        center = (
            int((left_eye[0] + right_eye[0]) * 0.5 * w),
            int((left_eye[1] + right_eye[1]) * 0.5 * h),
        )
    else:
        center = (w // 2, h // 2)
    if snapshot.smoothed_gaze_direction is not None:
        dx, dy, _ = snapshot.smoothed_gaze_direction
        scale = min(w, h) * 0.35
        tip = (int(center[0] + dx * scale), int(center[1] + dy * scale))
        cv2.circle(frame, center, 4, ray_color, thickness=-1)
        cv2.arrowedLine(frame, center, tip, ray_color, 2, cv2.LINE_AA, tipLength=0.2)

    stability = snapshot.smoothed_stability
    gaze_ray = (
        f"GAZE RAY yaw {snapshot.gaze_ray_yaw_deg:+5.1f}  pitch {snapshot.gaze_ray_pitch_deg:+5.1f}"
        if snapshot.gaze_ray_yaw_deg is not None and snapshot.gaze_ray_pitch_deg is not None
        else "GAZE RAY --"
    )
    nearest_profile = "PROFILE --"
    if snapshot.device_details:
        nearest = snapshot.device_details[0]
        if np.isfinite(nearest.angular_distance_deg):
            nearest_profile = (
                f"PROFILE {nearest.device_id} yaw {nearest.profile_yaw_deg:+5.1f} "
                f"pitch {nearest.profile_pitch_deg:+5.1f} err {nearest.angular_distance_deg:.1f}"
            )
    lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"LOCK  {state}", ray_color),
        (f"TARGET  {snapshot.target}  {snapshot.probability:.0%}", white),
        (gaze_ray, white),
        (nearest_profile, grey),
        (
            f"yaw {snapshot.head_yaw_deg:+5.0f}  pitch {snapshot.head_pitch_deg:+5.0f}"
            f"  roll {snapshot.head_roll_deg:+5.0f}",
            grey,
        ),
        (f"stability  {stability:.2f}" if stability is not None else "stability  --", grey),
    ]
    _text_block(frame, lines, (8, h - 6 - 20 * len(lines) - 6))
    return frame


def draw_hand_overlay(frame: Frame, snapshot: HandSnapshot) -> Frame:
    """Overlay the real hand skeleton (21 landmarks) and tracking info.

    Draws only when a hand is actually tracked; a lost frame draws nothing (never
    a stale skeleton). This is hand *tracking* — no gesture is recognized here.
    Mutates and returns ``frame``.
    """
    if not snapshot.hand_detected or snapshot.image_points is None:
        return frame
    h, w = frame.shape[:2]
    # blue for Right, orange for Left (BGR); grey if handedness unknown
    color = {"Right": (230, 180, 60), "Left": (60, 150, 230)}.get(
        snapshot.handedness, (170, 170, 170)
    )
    pts = [(int(x * w), int(y * h)) for x, y in snapshot.image_points]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for px, py in pts:
        cv2.circle(frame, (px, py), 3, (235, 235, 235), thickness=-1)

    label = snapshot.handedness or "?"
    _text_block(
        frame,
        [
            # image-space raw detection — where the hand is, not the model input
            (f"HAND  {label}  det {snapshot.detection_confidence:.0%}  [raw 검출]", color),
            (f"palm scale  {snapshot.palm_scale:.3f}", (170, 170, 170)),
        ],
        (8, 58),  # below the FPS HUD (top-left) so they do not overlap
    )
    return frame


def render_normalized_hand(
    points: tuple[tuple[float, float], ...] | None,
    *,
    size: int = 260,
    smoothed: bool = True,
) -> Frame:
    """Render normalized (wrist-origin, palm-scaled) landmarks into a square canvas.

    This is the faithful "what the model sees" view: the same normalized landmark
    coordinates the model consumes, drawn in their own space (not the webcam).
    ``points`` are the (x, y) of the normalized landmarks; ``None`` draws an empty
    canvas with a "no hand" note.
    """
    canvas: Frame = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:] = (18, 20, 26)
    color = (80, 200, 80) if smoothed else (120, 120, 120)
    tag = "모델 입력 (정규화" + ("·스무딩)" if smoothed else "·raw)")
    cv2.putText(canvas, tag, (8, 18), _FONT, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

    if points is None or len(points) != 21:
        cv2.putText(canvas, "no hand", (size // 2 - 34, size // 2), _FONT, 0.6,
                    (90, 90, 100), 1, cv2.LINE_AA)
        return canvas

    # Normalized coords keep MediaPipe's image convention (x right, y DOWN), and
    # the canvas is drawn in the same convention — so use +y (no flip). Flipping
    # would render the hand upside down relative to the webcam. The wrist (origin)
    # sits low so fingers, which have negative y (up in the image), extend upward.
    cx, cy = size // 2, int(size * 0.72)
    scale = size * 0.15
    px = [(int(cx + x * scale), int(cy + y * scale)) for x, y in points]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(canvas, px[a], px[b], color, 2, cv2.LINE_AA)
    for x, y in px:
        cv2.circle(canvas, (x, y), 3, (235, 235, 235), thickness=-1)
    cv2.circle(canvas, px[0], 5, (60, 150, 230), thickness=-1)  # wrist (origin)
    return canvas


def placeholder_frame(width: int = 640, height: int = 480, text: str = "NO CAMERA") -> Frame:
    """A solid frame with centered text, shown when no camera is available."""
    frame: Frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (28, 24, 20)
    (tw, th), _ = cv2.getTextSize(text, _FONT, 1.0, 2)
    org = ((width - tw) // 2, (height + th) // 2)
    cv2.putText(frame, text, org, _FONT, 1.0, (110, 110, 120), 2, cv2.LINE_AA)
    return frame
