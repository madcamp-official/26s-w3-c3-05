"""Typed gaze snapshot passed from the camera worker to the monitor UI."""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.contracts import TargetEstimate
from jarvis.gaze.features import FaceObservation, GazeVector


@dataclass(frozen=True, slots=True)
class GazeSnapshot:
    observation: FaceObservation
    gaze_vector: GazeVector | None
    estimate: TargetEstimate
    lock_state: str
