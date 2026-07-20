"""두 랜드마크 엔진의 추출 품질을 정량 비교하는 실험 패키지.

이 프로젝트의 Tasks API `HandLandmarker`와, 참고 레포
(Kazuhito00/hand-gesture-recognition-using-mediapipe)가 쓰는 레거시
`mp.solutions.hands`를 **같은 프레임 위에서** 나란히 돌려 편차·검출 일치율·지터를
잰다. 프로덕션 `gesture_fusion` 파이프라인과는 별개인 비교 도구다.

참고 엔진 자체는 `jarvis.gesture_fusion.solutions_hands`로 본체에 이식돼 있다 —
이 패키지는 그것을 **평가**할 뿐이고, 디버그 툴(`jarvis.monitoring`)은 백엔드
선택으로 같은 엔진을 직접 쓴다.

mediapipe 0.10.14가 Tasks와 Solutions를 모두 제공하므로 두 엔진은 한 프로세스에서
돈다(예전에 쓰던 격리 venv·서브프로세스 브리지는 필요 없어져 제거했다).

- `metrics` : 편차·검출 일치율·지터 정량화 (순수 numpy — 카메라·mediapipe 불필요)
- `compare` : 좌우 A/B 시각화 CLI
"""

from jarvis.engine_port.metrics import (
    ComparisonAccumulator,
    ComparisonSummary,
    JitterTracker,
    landmark_deviation,
)

__all__ = [
    "ComparisonAccumulator",
    "ComparisonSummary",
    "JitterTracker",
    "landmark_deviation",
]
