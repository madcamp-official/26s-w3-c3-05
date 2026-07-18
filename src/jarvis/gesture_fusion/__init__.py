"""Dynamic Gesture and Intent Fusion module owned by the Gesture·Fusion engineer.

담당 범위는 documents/gesture-fusion.md를 따른다. 입력·출력은 `jarvis.contracts`
타입을 쓰고 Gaze·Runtime의 내부 구현을 직접 참조하지 않는다.

`jarvis.gesture_fusion.mediapipe_hands`(mediapipe, `vision` extra)와
`jarvis.gesture_fusion.model`(torch, `ml` extra)는 여기서 자동 import하지 않는다 —
필요한 곳에서 명시적으로 `from jarvis.gesture_fusion.mediapipe_hands import
MediaPipeHandLandmarker` / `from jarvis.gesture_fusion.model import
CausalTCNGestureModel, ModelConfig`를 쓴다. `model_protocol`은 torch 없이도 쓸 수
있으므로 여기서 바로 export한다.
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
from jarvis.gesture_fusion.model_protocol import (
    DEFAULT_GESTURE_LABELS,
    PHASE_LABELS,
    GestureModel,
    ModelMetadata,
    ModelPrediction,
    SlidingFeatureWindow,
    normalized_entropy,
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
    "DEFAULT_GESTURE_LABELS",
    "PHASE_LABELS",
    "GestureModel",
    "ModelMetadata",
    "ModelPrediction",
    "SlidingFeatureWindow",
    "normalized_entropy",
]
