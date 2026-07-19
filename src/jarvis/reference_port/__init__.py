"""참고 레포(hand-gesture-recognition-using-mediapipe) 방법론 이식 — **실험용**.

이 패키지는 프로덕션 gesture_fusion 파이프라인(동적 제스처 TCN)과 **별개**로,
Kazuhito00의 참고 레포가 쓰는 **정적 손모양 KeyPoint 분류기** 방식을 이 프로젝트의
Tasks API 랜드마크 위에 이식해 A/B 비교하기 위한 것이다.

이식 불가/가능 경계(2026-07-20 조사):
- 참고 레포의 **랜드마크 엔진**(레거시 `mp.solutions.hands`, mediapipe 0.8.4)은
  설치된 mediapipe 0.10.35(slim, solutions 없음)·Python 3.12·Apple Silicon에서
  실행 불가라 이식할 수 없다. 랜드마크는 이 프로젝트의 Tasks API를 그대로 쓴다.
- 이식 가능한 것은 참고 레포의 **방법론**뿐이다: (1) 2D max-abs 정규화
  (`pre_process_landmark`), (2) 학습된 KeyPoint MLP(42→20→10→3). 후자는 참고
  레포에 포함된 학습 가중치를 hdf5에서 추출해(`keypoint_weights.npz`) numpy로
  forward를 재구현했다 — TensorFlow·LiteRT 등 런타임 의존성 없이 순수 numpy로 돈다.

라벨: Open / Close / Pointer (참고 레포 keypoint_labels.csv 그대로).
"""

from jarvis.reference_port.keypoint_classifier import (
    ReferenceKeyPointClassifier,
    preprocess_landmark_max_abs,
)

__all__ = ["ReferenceKeyPointClassifier", "preprocess_landmark_max_abs"]
