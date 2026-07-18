"""Draw a heads-up overlay on a webcam frame.

Pure frame-in → frame-out helpers used by the Live tab. Kept separate from Qt so
the drawing is unit-testable on a plain numpy array. Requires OpenCV (``ui``
extra); it is imported here, not in the package ``__init__``, so importing the
rest of ``jarvis.monitoring`` never needs cv2.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]

_FONT = cv2.FONT_HERSHEY_SIMPLEX


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


def placeholder_frame(width: int = 640, height: int = 480, text: str = "NO CAMERA") -> Frame:
    """A solid frame with centered text, shown when no camera is available."""
    frame: Frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (28, 24, 20)
    (tw, th), _ = cv2.getTextSize(text, _FONT, 1.0, 2)
    org = ((width - tw) // 2, (height + th) // 2)
    cv2.putText(frame, text, org, _FONT, 1.0, (110, 110, 120), 2, cv2.LINE_AA)
    return frame
