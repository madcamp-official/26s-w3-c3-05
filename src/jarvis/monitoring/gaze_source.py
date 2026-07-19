"""Typed gaze snapshot passed from the camera worker to the monitor UI."""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.contracts import TargetEstimate
from jarvis.gaze.features import FaceObservation
from jarvis.gaze.smoothing import SmoothedGaze


@dataclass(frozen=True, slots=True)
class GazeSnapshot:
    observation: FaceObservation
    gaze_vector: SmoothedGaze | None
    estimate: TargetEstimate
    lock_state: str
