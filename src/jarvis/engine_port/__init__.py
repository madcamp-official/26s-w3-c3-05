"""참고 레포의 랜드마크 **엔진 자체**를 이식해 기존 엔진과 비교하는 실험 패키지.

Kazuhito00/hand-gesture-recognition-using-mediapipe가 쓰는 레거시
`mp.solutions.hands`를 실제로 구동해, 이 프로젝트의 Tasks API `HandLandmarker`와
**같은 프레임 위에서** 랜드마크 품질을 A/B 비교한다. 프로덕션 `gesture_fusion`
파이프라인과는 완전히 별개인 실험용이다.

두 mediapipe 버전(0.10.35 slim vs solutions가 있는 0.10.14)은 한 프로세스에 공존할
수 없으므로 레거시 엔진은 격리 venv의 **자식 프로세스**로 돌리고 파이프로 프레임을
주고받는다. 구성:

- `protocol` : 파이프 와이어 포맷 (양쪽 venv에서 공유, 의존성 없음)
- `legacy_worker` : 격리 venv에서 도는 레거시 엔진 워커 (여기서 import하지 않는다)
- `client` : 메인 프로세스에서 워커를 부리는 동기 클라이언트
- `metrics` : 두 엔진 출력의 편차·검출 일치율·지터 정량화
- `compare` : 좌우 A/B 시각화 CLI
- `setup_legacy_env` : 격리 venv 생성 CLI

`legacy_worker`는 구버전 mediapipe를 요구하므로 이 `__init__`에서 import하지 않는다 —
메인 환경의 import·테스트·타입체크가 깨지지 않도록.
"""

from jarvis.engine_port.client import LegacyEngineClient, LegacyEngineError
from jarvis.engine_port.metrics import (
    ComparisonAccumulator,
    ComparisonSummary,
    landmark_deviation,
)
from jarvis.engine_port.protocol import LandmarkResult, ProtocolError

__all__ = [
    "ComparisonAccumulator",
    "ComparisonSummary",
    "LandmarkResult",
    "LegacyEngineClient",
    "LegacyEngineError",
    "ProtocolError",
    "landmark_deviation",
]
