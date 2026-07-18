"""Dynamic Gesture and Intent Fusion module owned by the Gesture·Fusion engineer.

담당 범위는 documents/gesture-fusion.md를 따른다. 입력·출력은 `jarvis.contracts`
타입을 쓰고 Gaze·Runtime의 내부 구현을 직접 참조하지 않는다.

`jarvis.gesture_fusion.mediapipe_hands`는 mediapipe(`vision` extra)를 필요로 하므로
여기서 자동 import하지 않는다 — 필요한 곳에서 명시적으로 `from
jarvis.gesture_fusion.mediapipe_hands import MediaPipeHandLandmarker`를 쓴다.
"""

from jarvis.gesture_fusion.config import DEFAULT_GESTURE_CONFIG, GestureConfig
from jarvis.gesture_fusion.features import (
    FrameFeatures,
    HandFeatureExtractor,
    compute_joint_angles,
    feature_dimension,
)
from jarvis.gesture_fusion.landmarks import (
    HandLandmarkSource,
    HandObservation,
    RawHandLandmarks,
    normalize_hand,
)

__all__ = [
    "DEFAULT_GESTURE_CONFIG",
    "GestureConfig",
    "HandObservation",
    "RawHandLandmarks",
    "HandLandmarkSource",
    "normalize_hand",
    "FrameFeatures",
    "HandFeatureExtractor",
    "compute_joint_angles",
    "feature_dimension",
]
