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

Frame = NDArray[np.uint8]

_FONT = cv2.FONT_HERSHEY_SIMPLEX

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

    # Gaze ray: project the composed direction (x right, y up) onto the frame.
    center = (w // 2, h // 2)
    if snapshot.gaze_direction is not None:
        dx, dy, _ = snapshot.gaze_direction
        scale = min(w, h) * 0.35
        tip = (int(center[0] + dx * scale), int(center[1] - dy * scale))
        cv2.circle(frame, center, 4, ray_color, thickness=-1)
        cv2.arrowedLine(frame, center, tip, ray_color, 2, cv2.LINE_AA, tipLength=0.2)

    stability = snapshot.smoothed_stability
    lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"LOCK  {state}", ray_color),
        (f"TARGET  {snapshot.target}  {snapshot.probability:.0%}", white),
        (
            f"yaw {snapshot.head_yaw_deg:+5.0f}  pitch {snapshot.head_pitch_deg:+5.0f}"
            f"  roll {snapshot.head_roll_deg:+5.0f}",
            grey,
        ),
        (f"stability  {stability:.2f}" if stability is not None else "stability  --", grey),
    ]
    _text_block(frame, lines, (8, h - 6 - 20 * len(lines) - 6))
    return frame


def placeholder_frame(width: int = 640, height: int = 480, text: str = "NO CAMERA") -> Frame:
    """A solid frame with centered text, shown when no camera is available."""
    frame: Frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (28, 24, 20)
    (tw, th), _ = cv2.getTextSize(text, _FONT, 1.0, 2)
    org = ((width - tw) // 2, (height + th) // 2)
    cv2.putText(frame, text, org, _FONT, 1.0, (110, 110, 120), 2, cv2.LINE_AA)
    return frame
