# Gaze Targeting — 담당: 1인

README [7장 핵심 기능 1](../README.md)의 구현 설계/진행 상황을 기록하는 문서.
다른 모듈과 주고받는 데이터 포맷은 여기가 아니라 [interface-contract.md](interface-contract.md)에 정의한다.

## 담당 범위 (README 12장)

- Face·iris landmark
- head pose
- gaze feature 정규화
- 기기별 calibration
- target classifier
- `UNKNOWN` rejection
- gaze smoothing
- Gaze Lock
- Target Selection Accuracy 평가

## 설계 노트

구현은 `src/jarvis/gaze/`와 `src/jarvis/calibration/`에 있다. 파이프라인 순서:

```
FaceObservation (landmarks.py, MediaPipe Face Landmarker)
→ compose_gaze_vector (features.py) — 머리 회전 ⊕ 눈-머리 상대 오프셋 → 단위 벡터
→ GazeSmoother (smoothing.py) — confidence-가중 이동 평균 + stability
→ TargetClassifier (classifier.py) — 코사인 유사도 + 등록 분산 정규화 + softmax + UNKNOWN
→ GazeLockStateMachine (lock.py) — SEARCHING→CANDIDATE→TARGET_LOCKED→GESTURE_WAIT→EXPIRED/COMMITTED
→ GazeTargetingEngine.process() (engine.py) — 위 전부를 조립해 TargetEstimate 방출
```

- `jarvis.gaze.landmarks`만 mediapipe(`vision` extra)를 import한다. 나머지는 순수
  `FaceObservation` 값만 다루므로 카메라·모델 파일 없이 단위 테스트한다.
- Calibration(`src/jarvis/calibration/session.py`)은 raw 프레임을 모았다가
  `DeviceGazeProfile`(mean_direction + 각도 분산)로 축약한 뒤 버린다. 저장/불러오기는
  `src/jarvis/calibration/profiles.py`가 README 7장 JSON 포맷대로 처리한다.
- Target Selection Accuracy는 `jarvis.gaze.evaluation.compute_target_selection_accuracy`로
  계산하며 dataset_id·조건을 결과에 강제로 남긴다.
- 알려진 미검증 항목: `landmarks.py`의 head yaw/pitch/roll 부호 규약은 실제 카메라로
  검증하지 않았다 — models/README.md의 `face_landmarker.task` 항목 참고.

## 진행 상황

- [x] Face·iris landmark 어댑터 (`landmarks.py`)
- [x] gaze feature 정규화 / 단위 벡터 합성 (`features.py`)
- [x] gaze smoothing (`smoothing.py`)
- [x] 기기별 calibration (`calibration/session.py`, `calibration/profiles.py`)
- [x] target classifier (`classifier.py`)
- [x] `UNKNOWN` rejection (classifier의 `unknown_probability_threshold`)
- [x] Gaze Lock 상태 머신 (`lock.py`)
- [x] Target Selection Accuracy 평가 함수 (`evaluation.py`)
- [ ] 실제 카메라로 head pose 부호/축 검증 (Day 1 통합 테스트 필요)
- [ ] Gesture·Fusion·Runtime과의 실제 통합(코드 조립은 Runtime composition root 몫)
- [ ] 환경 변화(조명/안경/거리) 조건에서 실측 Target Selection Accuracy 수집

## 이슈 / 의사결정 필요 사항

- head yaw/pitch/roll 부호 규약이 실 카메라와 맞는지 아직 확인되지 않음 — Day 1 통합
  테스트에서 확인되면 여기와 models/README.md를 갱신할 것(있으면 [decisions.md](decisions.md)로 옮기기).
