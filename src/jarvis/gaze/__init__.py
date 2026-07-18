"""Gaze Targeting module owned by the Gaze engineer.

외부로 내보내는 값은 `jarvis.contracts.TargetEstimate`뿐이다(src/jarvis/gaze/README.md).
`jarvis.gaze.landmarks`는 mediapipe(`vision` extra)를 필요로 하므로 여기서
자동으로 import하지 않는다 — 필요한 곳에서 명시적으로 `from jarvis.gaze.landmarks
import FaceLandmarkerAdapter`를 사용한다.
"""

from jarvis.gaze.classifier import ClassificationResult, DeviceGazeProfile, TargetClassifier
from jarvis.gaze.config import DEFAULT_GAZE_CONFIG, GazeConfig
from jarvis.gaze.engine import GazeTargetingEngine
from jarvis.gaze.features import FaceObservation, GazeVector, compose_gaze_vector
from jarvis.gaze.lock import GazeLockState, GazeLockStateMachine
from jarvis.gaze.smoothing import GazeSmoother, SmoothedGaze

__all__ = [
    "DEFAULT_GAZE_CONFIG",
    "GazeConfig",
    "FaceObservation",
    "GazeVector",
    "compose_gaze_vector",
    "GazeSmoother",
    "SmoothedGaze",
    "DeviceGazeProfile",
    "ClassificationResult",
    "TargetClassifier",
    "GazeLockState",
    "GazeLockStateMachine",
    "GazeTargetingEngine",
]
