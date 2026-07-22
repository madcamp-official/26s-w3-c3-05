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
- **머리 위치(3D 등록용, 2026-07-20 추가)**: `landmarks.py`의 `translation_from_transform`이
  같은 `facial_transformation_matrixes`의 `[:3, 3]`(이동 성분)에서 머리의 카메라 기준 3D
  위치 근사(`FaceObservation.head_position_mm`)를 추출한다. 이 값은 카메라 내부 파라미터
  보정 없이 MediaPipe의 표준 얼굴 모델 크기 가정만으로 얻은 근사 스케일이며, 실측 눈금으로
  검증하지 않았다 — `calibration/triangulation.py`의 물체 위치·유효 반경 추정과
  `documents/decisions.md`(2026-07-20)의 3D 등록 결정은 모두 이 근사가 물체 간 상대적
  거리 구분(가까운 노트북 vs 먼 전구)에는 충분하다는 가정 위에 있다. 삼각측량 품질 게이트
  (`minimum_triangulation_baseline_mm`·`minimum_triangulation_eigenvalue`·
  `maximum_triangulation_residual_mm`, `jarvis.gaze.config.GazeConfig`)를 만족하지 못하면
  항상 기존 각도 기반(mean_direction+variance) 등록으로 대체되므로, 이 근사가 부정확해도
  지어낸 3D 위치가 쓰이지는 않는다.

## hand_landmarker.task (Gesture)

`jarvis.gesture_fusion.mediapipe_hands.MediaPipeHandLandmarker`가 사용하는 MediaPipe
Hand Landmarker 번들 모델이다. `.gitignore`의 `models/*.task` 규칙에 따라 이 파일 자체는
커밋하지 않는다.

- **버전**: MediaPipe Tasks `hand_landmarker` (float16, bundle 1) — 이 저장소는
  `mediapipe>=0.10,<0.11`(`pyproject.toml` `vision` extra)로 검증했다.
- **로컬 경로**: `models/hand_landmarker.task` (커밋하지 않음, `MediaPipeHandLandmarker`
  생성 시 경로를 인자로 넘긴다).
- **확보 방법**: MediaPipe 공식 Hand Landmarker 모델 카드에 문서화된 URL에서 내려받는다
  (`https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`).
  팀 내부 미러가 있다면 이 항목을 갱신하고 `documents/decisions.md`에 이유를 남긴다.
- **입력**: RGB 프레임(`mediapipe.Image`, `SRGB`).
- **출력(이 프로젝트가 쓰는 것만)**: 손 하나당 21개 정규화 랜드마크(x, y, z)와
  handedness(Left/Right + score). MVP는 `config.num_hands=1`로 주 조작 손 하나만 쓴다.
- **전처리**: 프레임 단위 추론(`RunningMode.VIDEO`)만 사용한다 — 미래 프레임을 보지 않는
  causal 경로(development-principles.md 5절 3)와 일치한다. 손목 기준·손바닥 크기 정규화는
  모델이 아니라 `jarvis.gesture_fusion.landmarks.normalize_hand`(mediapipe 무의존, 단위
  테스트 가능)에서 수행하며, 회전은 정규화하지 않는다(손목 회전 제스처 신호 보존).
- **label 집합**: 해당 없음(회귀 랜드마크 출력). 제스처 label(swipe_down 등)은 이 모델이
  아니라 이후 추가될 gesture spotter(Causal TCN/GRU)에서 나오며, 그 모델은 별도 메타데이터로
  버전·feature·label·평가 결과를 기록한다(development-principles.md 7절 3).
- **모델/소스 교체**: landmark 소스는 `jarvis.gesture_fusion.landmarks.HandLandmarkSource`
  Protocol로 추상화되어 있어, 다른 손 검출 모델이나 원격 GPU 서버 스트리밍으로 교체해도
  downstream 코드는 바뀌지 않는다(2026-07-18 결정: 추론 위치를 교체 가능한 경계로 분리).

## Causal TCN gesture·phase classifier (Gesture)

`jarvis.gesture_fusion.model.CausalTCNGestureModel`이 구현하는 자체 학습 모델이다.
MediaPipe 번들과 달리 사전 학습된 파일을 내려받는 게 아니라, 이 저장소가 아키텍처를
정의하고 팀이 직접 데이터를 모아 학습한다.

- **버전**: **학습 완료**(2026-07-22, `models/gesture_tcn_finetuned.pt`,
  `ModelMetadata.trained=True`, `version="finetune-epoch82"`) — Jester 사전학습
  (`gesture_tcn_jester.pt`) 위에 웹캠 파인튜닝한 결과물이며 실시간 탭 인식 스트림에
  연결돼 있다. `CausalTCNGestureModel.load_weights()`로 이 `state_dict`를 불러온다.
  두 `.pt`는 `.gitignore`의 `models/*.pt`로 커밋하지 않는다(재현은 `training/README.md`).
- **아키텍처**: dilated causal 1D convolution(TCN), `jarvis.gesture_fusion.model.ModelConfig`로
  채널 수·kernel size·dropout·gesture label 집합을 조절한다(`pyproject.toml` `ml` extra:
  `torch>=2.2,<3`). 인과성(미래 프레임 미사용)은
  `tests/unit/gesture_fusion/test_model.py::test_output_is_truly_causal`로 회귀 검증한다.
- **로컬 경로**: 학습된 가중치는 `models/gesture_tcn_jester.pt`(Jester 사전학습)와
  `models/gesture_tcn_finetuned.pt`(웹캠 파인튜닝 완료본)에 각각 둔다(아직 실제 학습 실행
  전, `.gitignore`의 `models/*.pt` 규칙에 따라 커밋하지 않는다). 두 체크포인트를 따로
  보관하는 이유는 파인튜닝이 소량 데이터로 과적합할 경우 사전학습 체크포인트로 롤백할
  기준점을 남기기 위함(학습 파이프라인 인터뷰 결정).
- **학습 파이프라인**: `training/`(이 저장소 밖, `src/jarvis` 런타임 패키지와 분리 —
  `documents/decisions.md` 2026-07-19)이 Jester(20BN-Jester-v1) 사전학습 → 웹캠 소량
  파인튜닝 2단계로 이 가중치를 만든다. 실행 방법은 `training/README.md` 참고. `ModelMetadata`는
  체크포인트 파일 자체가 아니라 `<체크포인트 파일명>.metadata.json` sidecar에 기록된다 —
  `load_weights(path, metadata)`가 metadata를 파일에서 읽지 않고 호출자가 넘겨야 하므로,
  로드하는 쪽이 이 sidecar를 읽어 `ModelMetadata(**json)`으로 구성해 넘긴다.
- **입력**: `jarvis.gesture_fusion.features.HandFeatureExtractor`가 만든 feature 시퀀스
  `(window_size, feature_dim)`. `window_size`는 `ModelConfig.receptive_field`(아키텍처가
  결정), `feature_dim`은 `features.feature_dimension(GestureConfig)`(전처리 설정이 결정) —
  두 값이 어긋나면 `predict()`가 `ValueError`로 거부한다.
- **전처리**: `HandFeatureExtractor`(속도·관절 각도, task 2 — 손가락 관절 위치 가속도는
  2026-07-19에 모델 입력에서 제거했다. 손목 평행이동 가속도는 swipe 판별용 별개 신호라
  유지한다. `documents/decisions.md` 참조)와
  `jarvis.gesture_fusion.model_protocol.SlidingFeatureWindow`(causal 스트리밍 윈도우)를
  그대로 쓴다. 별도 정규화는 하지 않는다 — 입력 feature 자체가 이미 스케일 정규화됨(task 1).
- **출력**: gesture head(label 집합, 기본 `DEFAULT_GESTURE_LABELS` 10종 — none,
  rotate_clockwise/counter_clockwise, slide_two_fingers_up/down/left/right, drumming_fingers,
  doing_other_things, stop_sign)와 phase head(`IDLE`/`ONSET`/`ACTIVE`/`ENDING`, 계약상 고정 4-class).
  두 head 모두 softmax 확률의 argmax를 label로, max 확률을 confidence로 쓴다. `uncertainty`는
  gesture 확률 분포의 정규화 엔트로피([0,1], `model_protocol.normalized_entropy`).
- **label 집합**: `ModelConfig.gesture_labels`(기본 `DEFAULT_GESTURE_LABELS`) — 열린 문자열
  키(interface-contract.md 공통 규칙)라 새 제스처 추가 시 이 튜플만 확장하면 된다. Pinch·
  주먹은 README 8장이 명시한 확장 기능이라 기본 label에서 뺐다.
- **평가 결과**: 학습 전이라 없음. 학습 후에는 데이터셋·조건과 함께
  `ModelMetadata.evaluation_notes`와 이 항목에 Gesture Event Recall(README 13장, 목표
  ≥90%)을 기록한다(development-principles.md 1절 4: 재현 가능한 평가만 사용).
- **모델 교체**: `jarvis.gesture_fusion.model_protocol.GestureModel` Protocol로
  추상화되어 있어, TCN을 GRU나 다른 아키텍처, 원격 추론 서버로 바꿔도 downstream(gesture
  spotting 상태 머신, task 4)은 이 Protocol만 바라보면 된다(2026-07-18 결정: 추론 위치를
  교체 가능한 경계로 분리 — landmark 소스와 같은 설계를 모델 자체에도 적용).


## hand_pose_classifier.pt (Gesture — 정적 손 자세)

`jarvis.gesture_fusion.pose_classifier.TorchPoseClassifier`가 로드하는 정적 손 자세
분류기다. 동적 제스처(위 TCN)와 **별개 경계**로, 단일 프레임의 손 모양을 판정해 로컬에서
커서·클릭·드래그·우클릭·스크롤·바탕화면 토글을 구동한다(`documents/gesture-fusion.md`
"정적 손 자세 파이프라인" 참조). TCN과 달리 **이 파일은 커밋한다** — 10KB로 작고 데모에
바로 필요하기 때문(`.gitignore`에 `!models/hand_pose_classifier.pt` 예외).

- **아키텍처**: MLP 입력 52 → 20 → ReLU → Dropout(0.2) → 10 → ReLU → 8(8-class).
  파라미터 약 1,450개라 노트북 CPU로 수 초면 학습된다(`pyproject.toml` `ml` extra: torch).
- **입력(52차원)**: 정규화·평활된 21개 랜드마크 좌표 42 + 손끝 5점 쌍거리 10
  (`pose_protocol.pose_features()`, **학습·추론이 반드시 같이 쓰는 단일 함수**). 좌표만으론
  작은 MLP가 손끝 관계를 못 뽑아 쌍거리를 명시 추가했다(index_point 재현율 50.2%→94.0%).
- **전처리(재현성 필수)**: `normalize_hand`(손목 원점 + palm_scale) → One-Euro
  (`GestureConfig.smoothing_min_cutoff=2.0`, `beta=1.2`, `d_cutoff=1.5`), 즉 디버깅 툴
  3번 탭(손 추적)에 뜨는 좌표. z는 쓰지 않는다(`LANDMARK_DIMS=2`) — 기울기는 각도로만
  게이트에 쓴다. **저장 파일에 이 전처리 설정이 함께 들어 있고**, 로드 시 현재
  `GestureConfig`와 대조해 다르면 `PreprocessingMismatch`로 거부한다(어긋나면 예외 없이
  정확도만 조용히 떨어지는 고장 방지).
- **label 집합(8종, 2026-07-22 v2)**: index_point, **middle_point**(중지 하나 → 탭 닫기),
  pinch_index, pinch_middle, two_fingers, open_palm, fist, **none**(제어 자세가 아닌
  전이·휴지·일상 동작 배경 클래스 — 없으면 손이 보이기만 해도 명령이 나간다). 런타임은
  모델 파일에 저장된 `label_names`를 진실로 쓴다(코드 상수 `DEFAULT_POSE_LABELS`가 아님).
- **평가 결과(2026-07-22 v2, 2세션·1인, 에피소드 단위 홀드아웃)**: 전체 **94.4%**.
  자세별 재현율 index_point 99.4 / middle_point 92.5 / pinch_index 98.6 / pinch_middle 99.4 /
  two_fingers 99.2 / open_palm 94.5 / fist 95.9 / none 89.9. 이전 7-class(87.1%,
  pinch_middle 68% 정체)는 각도·에피소드 다양성 부족이 원인이었고, 처음부터 재수집으로
  pinch_middle 68→99.4% 해소. 이전 모델은 `hand_pose_classifier.7class.bak`로 보존. **1인
  데이터**라 다른 사람·조명·위치 일반화는 이 수치가 보장하지 않는다(실사용은 낮을 수 있음).
- **재학습 방법**: 수집 데이터(raw 랜드마크 + 라벨 npz)와 `training/train_pose.py`가
  있으면 누구나 같은 모델을 재현·개선할 수 있다.
  ```
  python -m training.train_pose --data <수집 npz> --out models/hand_pose_classifier.pt
  ```
  매번 처음부터 학습한다(파라미터가 적어 수 초, fine-tuning 개념 불필요). 수집기는
  누적 저장이라 새 데이터를 이어 붙인 뒤 전체를 다시 학습하면 균형이 자동으로 맞는다.
  평가는 에피소드 단위 홀드아웃(인접 프레임 누수 차단). **주의**: 수집 데이터 npz는 팀
  공유 위치에 따로 두어야 재현 가능하다(현재 로컬 `.claude/tmp/pose_frames.npz`).
- **모델 없을 때**: `pose_classifier`를 못 불러오면 `NullPoseClassifier`로 대체돼 자세를
  지어내지 않고 "자세 분류 모델 없음"을 UI에 정직하게 표시한다(크래시 없음).
