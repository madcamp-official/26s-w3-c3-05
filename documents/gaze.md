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

- **시선 방향 벡터 합성 (2026-07-18 팀 합의)**: 머리 yaw/pitch와 눈-머리 상대 오프셋을 따로 feature로 두지 않고 하나의 시선 방향 단위 벡터로 합성한다. 등록 시(고개를 돌려 봄)와 실사용 시(고개는 그대로, 눈짓만)의 행동이 달라도 같은 방향이면 같은 벡터가 나오게 하기 위함. 기기 prototype과의 비교는 코사인 유사도(내적)로 한다. 상세는 README 7장.

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
- `UNKNOWN`은 기기 간 상대 확률(`unknown_probability_threshold`)뿐 아니라 가장 가까운
  등록 방향과의 절대 각도(`unknown_max_angle_deg`, 기본 25도)도 함께 검사한다. 등록
  기기가 하나일 때 상대 확률이 항상 1.0이 되는 경우에도 먼 시선을 거부하기 위해서다.
- Gaze Lock TTL은 마지막으로 같은 대상을 확신 있게 본 시각을 기준으로 한다. Gesture
  시작은 기존 TTL을 연장하지 않으며, gesture 시작과 commit 이벤트 모두 자신의
  `timestamp_ms`가 만료 시각에 도달했으면 `EXPIRED`로 거부한다.
- `jarvis-gaze calibrate`는 카메라 관측을 기기 프로필로 축약해 저장하고,
  `inspect-head-pose`는 실카메라 축/부호 점검값을 출력한다. `evaluate`는 정답 CSV에서
  dataset_id·환경 조건을 포함한 재현 가능한 정확도 JSON을 만든다(`tools/README.md`).
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
- [ ] 실제 카메라로 head pose 부호/축 검증 (pitch는 2026-07-18 실측에서 상하 반전 확인 후
  화면 위쪽이 양수가 되도록 수정 완료, yaw·roll 최종 확인 필요)
- [ ] Gesture·Fusion·Runtime과의 실제 통합(코드 조립은 Runtime composition root 몫)
- [ ] 환경 변화(조명/안경/거리) 조건에서 실측 Target Selection Accuracy 수집

## 이슈 / 의사결정 필요 사항

- head yaw/pitch/roll 부호 규약이 실 카메라와 맞는지 아직 확인되지 않음 — Day 1 통합
  테스트에서 확인되면 여기와 models/README.md를 갱신할 것(있으면 [decisions.md](decisions.md)로 옮기기).
## Device registration update (2026-07-19)

Demo target registration follows README section 7: the user looks at a real object for about two seconds, and the
system stores a camera-relative gaze direction profile as `mean_direction + variance`.

Implemented files:

- `src/jarvis/gaze/direction.py`: vector/yaw-pitch conversion used only at the registration/debug boundary.
- `src/jarvis/calibration/registry.py`: target add/update/rename/delete persistence, JSON auto-load, legacy profile migration,
  and conversion back to README-style `DeviceGazeProfile`.
- `src/jarvis/calibration/target_registration.py`: two-second robust sample collection with minimum frame count,
  confidence filtering, closed-eye filtering, jump filtering, median center, and robust angular variance.
- `src/jarvis/gaze/classifier.py`: registered target matching uses cosine similarity normalized by stored variance,
  then rejects `UNKNOWN` when the nearest registered direction is too far or the first/second target margin is too small.
- `src/jarvis/gaze/smoothing.py`: confidence-aware EMA smoothing is enabled before classification.
- `src/jarvis/gaze/lock.py`: 300-500 ms dwell-based lock and hysteresis are handled by the existing state machine.
- `src/jarvis/monitoring/`: debug UI can register/reregister/rename/delete targets and show the live gaze ray,
  candidate/lock state, and pipeline diagnostics without drawing artificial target-area circles.

MVP operating assumptions:

- The camera is fixed during the demo.
- User changes are announced manually, so automatic face identity/profile switching is out of MVP scope.
- Objects are registered every time they are newly added or moved.
- Real-camera yaw/pitch sign and scale still need one final fixed-camera sanity check before demo.
- The current `CalibratedGaze` is geometric yaw/pitch from the smoothed gaze vector. A learned Ridge/MLP personal correction
  can be added later when labeled calibration samples are available.
