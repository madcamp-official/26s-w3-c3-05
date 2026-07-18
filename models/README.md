# Model assets

Gaze와 Gesture 모델 파일의 로컬 위치다. 대용량 모델 바이너리는 기본적으로 Git에 직접 넣지
않으며, 각 모델의 버전·입력 feature·전처리·label·평가 결과와 확보 방법을 함께 기록한다.

## face_landmarker.task (Gaze)

`jarvis.gaze.landmarks.FaceLandmarkerAdapter`가 사용하는 MediaPipe Face Landmarker 번들
모델이다. `.gitignore`의 `models/*.task` 규칙에 따라 이 파일 자체는 커밋하지 않는다.

- **버전**: MediaPipe Tasks `face_landmarker` (float16, bundle 1) — 이 저장소는
  `mediapipe>=0.10,<0.11`(`pyproject.toml` `vision` extra)로 검증했다.
- **로컬 경로**: `models/face_landmarker.task` (커밋하지 않음, `FaceLandmarkerAdapter`
  생성 시 경로를 인자로 넘긴다).
- **확보 방법**: MediaPipe 공식 Face Landmarker 모델 카드에 문서화된 URL에서 내려받는다
  (`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`).
  팀 내부 미러가 있다면 이 항목을 갱신하고 `documents/decisions.md`에 이유를 남긴다.
- **입력**: RGB 프레임(`mediapipe.Image`, `SRGB`).
- **출력(이 프로젝트가 쓰는 것만)**: 478개 정규화 얼굴 랜드마크(홍채 포함, index 468-477)와
  `facial_transformation_matrixes`(4x4 회전+이동 행렬). `output_face_blendshapes`는 쓰지
  않는다.
- **전처리**: 프레임 단위 추론(`RunningMode.VIDEO`)만 사용한다 — 미래 프레임을 보지 않는
  causal 경로(development-principles.md 5절 3)와 일치한다.
- **label 집합**: 해당 없음(회귀 랜드마크 출력, 분류 label 없음). Gaze targeting 자체의
  "기기 label"은 이 모델이 아니라 `jarvis.gaze.classifier`의 사용자별 calibration
  (`DeviceGazeProfile`)에서 나온다.
- **평가 결과**: 이 모델 자체는 재학습하지 않으므로 별도 평가가 없다. Gaze Targeting
  Engine 전체의 정확도는 `jarvis.gaze.evaluation.compute_target_selection_accuracy`로
  측정하며(README 13장 Target Selection Accuracy ≥ 90%), 데이터셋·조건과 함께 기록해야
  재현 가능하다(development-principles.md 1절 4).
- **주의**: `landmarks.py`의 `rotation_matrix_to_euler_deg`는 표준 회전 행렬 분해식을
  가정한 값이며, 실제 카메라로 첫 통합 테스트를 할 때(README 16장 Day 1) yaw/pitch 부호나
  축이 뒤바뀌어 보이면 그 함수만 조정하면 된다 — calibration·classifier는 등록·실사용에
  동일한 변환을 쓰는 한 절대적인 부호 규약에 의존하지 않는다.

