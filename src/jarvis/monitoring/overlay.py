"""Draw a heads-up overlay on a webcam frame.

Pure frame-in → frame-out helpers used by the Live tab. Kept separate from Qt so
the drawing is unit-testable on a plain numpy array. Requires OpenCV (``ui``
extra); it is imported here, not in the package ``__init__``, so importing the
rest of ``jarvis.monitoring`` never needs cv2.
"""

from __future__ import annotations

import math
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
_TARGET_COLORS: tuple[tuple[int, int, int], ...] = (
    (80, 170, 255),
    (80, 220, 120),
    (230, 180, 60),
    (220, 100, 220),
    (120, 220, 220),
    (180, 120, 255),
)


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


def draw_target_heatmap(frame: Frame, snapshot: GazeSnapshot, *, mirror: bool = False) -> Frame:
    """Draw a coarse target-direction heatmap over the webcam frame.

    This is a debugging visualization, not object detection.  It projects the
    registered yaw/pitch target profiles into a rough camera FOV and colors the
    area by the nearest registered direction, dimming pixels outside the
    registered radius.
    """
    details = tuple(
        sorted(
            (d for d in snapshot.device_details if not np.isnan(d.angular_distance_deg)),
            key=lambda d: d.device_id,
        )
    )
    if not details:
        return frame

    h, w = frame.shape[:2]
    grid_w = 80
    grid_h = max(45, int(grid_w * h / max(w, 1)))
    yaw_span = 70.0
    pitch_span = 50.0
    xs = np.linspace(-yaw_span * 0.5, yaw_span * 0.5, grid_w, dtype=np.float64)
    ys = np.linspace(pitch_span * 0.5, -pitch_span * 0.5, grid_h, dtype=np.float64)
    yaw_grid, pitch_grid = np.meshgrid(xs, ys)

    heat = np.zeros((grid_h, grid_w, 3), dtype=np.float32)
    alpha = np.zeros((grid_h, grid_w), dtype=np.float32)
    for index, detail in enumerate(details[: len(_TARGET_COLORS)]):
        if np.isnan(detail.target_yaw_deg) or np.isnan(detail.target_pitch_deg):
            continue
        center_yaw = detail.target_yaw_deg
        center_pitch = detail.target_pitch_deg
        radius = max(detail.allowed_radius_deg, 1.0)
        distance = np.hypot((yaw_grid - center_yaw) / radius, (pitch_grid - center_pitch) / radius)
        influence = np.exp(-(distance**2) * 1.4)
        mask = influence > alpha
        color_bgr = _TARGET_COLORS[index % len(_TARGET_COLORS)]
        heat[mask] = np.asarray(color_bgr, dtype=np.float32)
        alpha[mask] = influence[mask]

    heat_bgr = cv2.resize(heat.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
    alpha_full = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    alpha_full = np.clip(alpha_full * 0.28, 0.0, 0.28)
    blended = (
        frame.astype(np.float32) * (1.0 - alpha_full[..., None])
        + heat_bgr.astype(np.float32) * alpha_full[..., None]
    )
    frame[:] = blended.astype(np.uint8)

    for index, detail in enumerate(details[: len(_TARGET_COLORS)]):
        color_bgr = _TARGET_COLORS[index % len(_TARGET_COLORS)]
        if np.isnan(detail.target_yaw_deg) or np.isnan(detail.target_pitch_deg):
            continue
        cx = int((detail.target_yaw_deg / yaw_span + 0.5) * w)
        cy = int((0.5 - detail.target_pitch_deg / pitch_span) * h)
        cv2.circle(
            frame,
            (cx, cy),
            max(6, int(detail.allowed_radius_deg / yaw_span * w)),
            color_bgr,
            1,
            cv2.LINE_AA,
        )
        cv2.circle(frame, (cx, cy), 4, color_bgr, thickness=-1)
        y = min(h - 10, 58 + index * 18)
        cv2.circle(frame, (14, y - 5), 5, color_bgr, thickness=-1)
        cv2.putText(frame, detail.device_id, (25, y), _FONT, 0.45, color_bgr, 1, cv2.LINE_AA)
    return frame


def draw_gaze_overlay(frame: Frame, snapshot: GazeSnapshot, *, mirror: bool = False) -> Frame:
    """Overlay the live Gaze pipeline result: gaze ray, head angles, lock state.

    Everything drawn comes from the real snapshot. Tracking loss is shown as a
    red banner instead of a stale ray — the overlay never invents a direction.
    ``mirror`` flips landmark positions to match a horizontally-flipped display
    frame. Gaze direction is already expressed in user-facing yaw coordinates, so
    its horizontal sign is not flipped again. Mutates and returns ``frame``.
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
    if snapshot.tracking_recovering:
        cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), thickness=-1)
        cv2.putText(
            frame,
            "FACE RECOVERING / last gaze hold",
            (10, h - 9),
            _FONT,
            0.6,
            (80, 190, 230),
            2,
            cv2.LINE_AA,
        )

    state = str(snapshot.lock_state)
    ray_color = _LOCK_BGR.get(state, grey)
    looking_unknown = snapshot.target == "UNKNOWN"
    looking_color = (90, 90, 240) if looking_unknown else ray_color
    looking_text = f"LOOKING AT: {snapshot.target_label}"
    (text_w, text_h), _ = cv2.getTextSize(looking_text, _FONT, 0.9, 2)
    box_x = max(8, (w - text_w) // 2 - 14)
    box_y = 44
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (box_x, box_y),
        (min(w - 8, box_x + text_w + 28), box_y + text_h + 22),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
    cv2.putText(
        frame,
        looking_text,
        (box_x + 14, box_y + text_h + 9),
        _FONT,
        0.9,
        looking_color,
        2,
        cv2.LINE_AA,
    )

    # Gaze ray: use the same smoothed direction that the classifier consumes.
    left_eye = snapshot.left_eye_center_normalized
    right_eye = snapshot.right_eye_center_normalized
    if left_eye is not None and right_eye is not None:
        cx_norm = (left_eye[0] + right_eye[0]) * 0.5
        cy_norm = (left_eye[1] + right_eye[1]) * 0.5
        center = (
            int((1.0 - cx_norm if mirror else cx_norm) * w),
            int(cy_norm * h),
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
    nearest = snapshot.device_details[0] if snapshot.device_details else None
    if nearest is None or np.isnan(nearest.angular_distance_deg):
        range_line = "range  --"
        range_color = grey
    else:
        range_line = (
            f"range {nearest.device_id} "
            f"{nearest.angular_distance_deg:.1f}/{nearest.allowed_radius_deg:.1f}deg "
            f"x{nearest.normalized_distance:.2f} {nearest.range_status}"
        )
        range_color = (80, 200, 80) if nearest.within_profile_radius else (90, 90, 240)
    lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"LOCK  {state}", ray_color),
        (f"TARGET  {snapshot.target_label}  {snapshot.probability:.0%}", white),
        (range_line, range_color),
        (
            f"yaw {snapshot.head_yaw_deg:+5.0f}  pitch {snapshot.head_pitch_deg:+5.0f}"
            f"  roll {snapshot.head_roll_deg:+5.0f}",
            grey,
        ),
        (f"stability  {stability:.2f}" if stability is not None else "stability  --", grey),
    ]
    if snapshot.camera_pose_warning:
        lines.append(("CAMERA POSE CHANGED / re-register", (60, 180, 255)))
    _text_block(frame, lines, (8, h - 6 - 20 * len(lines) - 6))
    return frame


def draw_hand_overlay(frame: Frame, snapshot: HandSnapshot, *, mirror: bool = False) -> Frame:
    """Overlay the real hand skeleton (21 landmarks) and tracking info.

    Draws only when a hand is actually tracked; a lost frame draws nothing (never
    a stale skeleton). This is hand *tracking* — no gesture is recognized here.

    Uses the One-Euro-smoothed image points when available (matching the model-input
    smoothing toggle), falling back to the raw detection. ``mirror`` flips the drawn
    x-coordinates to match a horizontally-flipped (selfie/거울상) display frame; it is
    a display concern only and leaves the underlying landmark data untouched. Text is
    drawn at un-mirrored positions so it stays readable on the flipped frame.
    Mutates and returns ``frame``.
    """
    if not snapshot.hand_detected or snapshot.image_points is None:
        return frame
    h, w = frame.shape[:2]
    # blue for Right, orange for Left (BGR); grey if handedness unknown
    color = {"Right": (230, 180, 60), "Left": (60, 150, 230)}.get(
        snapshot.handedness, (170, 170, 170)
    )
    smoothed = snapshot.image_points_smoothed
    use_smoothed = snapshot.smoothed and smoothed is not None
    src = smoothed if (snapshot.smoothed and smoothed is not None) else snapshot.image_points
    pts = [(int((1.0 - x) * w) if mirror else int(x * w), int(y * h)) for x, y in src]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for px, py in pts:
        cv2.circle(frame, (px, py), 3, (235, 235, 235), thickness=-1)

    label = snapshot.handedness or "?"
    src_tag = "스무딩" if use_smoothed else "raw"
    _text_block(
        frame,
        [
            # image-space detection (smoothed for display) — where the hand is
            (f"HAND  {label}  det {snapshot.detection_confidence:.0%}  [{src_tag} 검출]", color),
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
    mirror: bool = False,
) -> Frame:
    """Render normalized (wrist-origin, palm-scaled) landmarks into a square canvas.

    This is the faithful "what the model sees" view: the same normalized landmark
    coordinates the model consumes, drawn in their own space (not the webcam).
    ``points`` are the (x, y) of the normalized landmarks; ``None`` draws an empty
    canvas with a "no hand" note. ``mirror`` flips the drawing left↔right to match the
    selfie/거울상 webcam view — a display concern only; ``points`` are unchanged.
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
    px = [
        (int(cx - x * scale) if mirror else int(cx + x * scale), int(cy + y * scale))
        for x, y in points
    ]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(canvas, px[a], px[b], color, 2, cv2.LINE_AA)
    for x, y in px:
        cv2.circle(canvas, (x, y), 3, (235, 235, 235), thickness=-1)
    cv2.circle(canvas, px[0], 5, (60, 150, 230), thickness=-1)  # wrist (origin)
    return canvas


def render_vector(
    vector: tuple[float, float] | None,
    *,
    size: int = 200,
    scale: float,
    mirror: bool = False,
) -> Frame:
    """Draw a 2D (x, y) vector as an arrow from the canvas center.

    ``scale`` is a running max magnitude the caller maintains across frames (like
    ``render_normalized_hand``'s fixed pixel scale, but adaptive) — a fixed
    hardcoded scale would either clip a fast swipe or make a still hand's tiny
    jitter invisible, so the arrow length is always relative to the largest
    magnitude seen so far. ``vector=None`` (tracking lost / no history yet) draws
    an empty canvas with a "no signal" note rather than a stale arrow.
    ``mirror`` flips the x-component to match a horizontally-flipped (selfie/거울상)
    display frame — a display concern only, ``vector`` itself is unchanged. Y is
    drawn without flipping to match MediaPipe's y-down image convention, the same
    choice ``render_normalized_hand`` makes for the adjacent model-input canvas.
    """
    canvas: Frame = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:] = (18, 20, 26)
    center = (size // 2, size // 2)
    radius = int(size * 0.42)
    cv2.circle(canvas, center, radius, (40, 44, 52), 1, cv2.LINE_AA)
    cv2.circle(canvas, center, 3, (110, 110, 120), thickness=-1)

    if vector is None:
        cv2.putText(canvas, "no signal", (size // 2 - 42, size // 2 + 4), _FONT, 0.5,
                    (90, 90, 100), 1, cv2.LINE_AA)
        return canvas

    x, y = vector
    if mirror:
        x = -x
    magnitude = math.sqrt(x * x + y * y)
    if scale > 0.0:
        tip = (int(center[0] + (x / scale) * radius), int(center[1] + (y / scale) * radius))
    else:
        tip = center
    color = (80, 200, 80)
    cv2.arrowedLine(canvas, center, tip, color, 2, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(canvas, f"{magnitude:.3f}", (8, size - 10), _FONT, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def placeholder_frame(width: int = 640, height: int = 480, text: str = "NO CAMERA") -> Frame:
    """A solid frame with centered text, shown when no camera is available."""
    frame: Frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (28, 24, 20)
    (tw, th), _ = cv2.getTextSize(text, _FONT, 1.0, 2)
    org = ((width - tw) // 2, (height + th) // 2)
    cv2.putText(frame, text, org, _FONT, 1.0, (110, 110, 120), 2, cv2.LINE_AA)
    return frame
