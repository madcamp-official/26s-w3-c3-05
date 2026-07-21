# Gesture & Intent Fusion — 담당: 2인

README [8장 핵심 기능 2](../README.md), [9장 핵심 기능 3](../README.md)의 구현 설계/진행 상황을 기록하는 문서.
다른 모듈과 주고받는 데이터 포맷은 여기가 아니라 [interface-contract.md](interface-contract.md)에 정의한다.

## 담당 범위 (README 12장)

- Hand landmark
- 동적 gesture spotting
- Causal TCN/GRU
- gesture phase
- 시선·제스처 temporal alignment
- fusion confidence
- safe commit
- duplicate intent 방지
- hard-negative mining

## 설계 노트

- **커서/제스처 분기 (2026-07-18 확정)**: 노트북 Lock 중 기본은 커서 모드. Gesture Spotter가 `ONSET`을 감지하면 커서 스트림을 일시정지하고 제스처 판정에 우선권을 준다. 판정이 `IDLE`로 복귀(제스처 불성립)하면 커서 모드로 돌아간다. `pointer/` 모듈과 이 신호를 주고받는 인터페이스가 필요하다.
- **추론 위치 (2026-07-18 확정)**: MVP는 로컬 추론. 단, 모델 추론 부분(landmark 시퀀스 → gesture/phase)을 교체 가능한 경계로 분리해, 나중에 keypoint를 WebSocket으로 GPU 서버에 보내는 방식으로 옮길 수 있게 한다. 서버로 옮길 경우 timestamp는 서버가 새로 찍지 않고 클라이언트 값을 그대로 반환한다([interface-contract.md](interface-contract.md) 공통 규칙).
- **커스텀 제스처 대비**: `gesture`는 열린 문자열 키다. 고정 분류기(TCN/GRU) 출력 외에, 나중에 few-shot 매처(DTW/임베딩 유사도)를 병렬로 붙이는 확장을 전제로 gesture id를 하드코딩하지 않는다.
- **좌표 차원 2D화 (2026-07-19)**: 손 랜드마크를 (x, y, z) 3D가 아니라 **x·y 2D만** 사용한다. `config.LANDMARK_DIMS=2` 상수 하나가 원시 좌표 추출(`mediapipe_hands.py`·`hand_probe.py`는 `lm.x, lm.y`만 읽음)부터 정규화(`landmarks.py`)·feature 차원(`features.py`, `_POSITION_DIMS = 21 * LANDMARK_DIMS`)까지 전 파이프라인의 좌표 차원을 결정한다. 이유: MediaPipe z(깊이)는 단안 웹캠 추정값이라 노이즈가 크고 특히 주먹류(손가락이 카메라 쪽으로 접힘) 동작에서 크게 흔들려 검출을 불안정하게 한다(참고: reference 프로젝트도 z를 버린다). 손목 평행이동(wrist translation) feature도 2D를 따른다(`_WRIST_DIMS = LANDMARK_DIMS`). **트레이드오프**: 관절각·palm_scale·속도·손목 평행이동이 모두 이미지 평면 기준이 되어, 카메라 축 방향(out-of-plane) 손가락 굽힘/밀기·손 밀기 신호는 표현할 수 없다(swipe 등 평면 동작은 영향 없음). **주의**: 학습 데이터도 반드시 같은 2D로 전처리해야 하며(모델 재현성), `LANDMARK_DIMS`를 3으로 되돌리면 재학습이 필요하다. 상세는 decisions.md(2026-07-19) 참조.
- **손가락 관절 위치 가속도 제거 (2026-07-19)**: 손가락 관절 위치의 가속도(`GestureConfig.include_acceleration`)를 모델 입력에서 완전히 제거했다. 손목 평행이동 가속도(`wrist_acceleration`, swipe 판별용 별개 신호, 위 항목·decisions.md 2026-07-19 참조)는 유지한다 — 이번 제거로 영향받지 않는다. 효과: `feature_dimension(default)`이 144→**102**(위치42+관절각14+속도42+손목속도가속도4)로 줄었다. 모니터에는 관절 위치 가속도가 표시된 적이 없어(`hand_probe.py`/`app.py`는 손목 가속도만 벡터 뷰로 노출) 모니터 UI 변경은 없다. **주의**: 학습 데이터도 같은 스키마로 전처리해야 한다(모델 재현성) — `training/`은 `feature_dimension(GestureConfig)`을 동적으로 참조하므로 코드 변경 없이 자동 반영된다.
- **랜드마크 평활화 (2026-07-19 추가, dev-3 제안 — dev-2 확인/decisions.md 기록 필요)**: 속도·가속도는 이산 미분이라 MediaPipe 프레임별 랜드마크 지터를 증폭한다. 이를 막기 위해 `HandFeatureExtractor`가 **미분 전에** 위치를 One-Euro 필터(`smoothing.py`)로 평활화한다. `GestureConfig.smooth_landmarks`(기본 True)와 `smoothing_min_cutoff`/`smoothing_beta`/`smoothing_d_cutoff`로 제어하며, 추적 손실·프레임 공백에서 필터도 함께 리셋한다. 검증: 정지한 손 ±0.01 지터에서 속도 feature 노이즈 에너지 약 98% 감소. **주의**: 학습 데이터도 같은 설정으로 전처리해야 추론과 일관되므로(모델 재현성), 학습 파이프라인 구성 시 이 값을 고정·기록한다.
  - **모니터 디버그 표시 (2026-07-19)**: 위 평활화는 모델이 소비하는 **정규화 좌표** 공간에서 일어난다. 모니터 `실시간` 탭의 웹캠 스켈레톤은 **이미지 좌표** 공간이라 이 필터를 거치지 않아 이전엔 raw 지터가 그대로 보였다. 이제 `HandProbe`가 같은 `GestureConfig` 평활 파라미터로 **표시 전용** One-Euro 필터를 이미지 좌표에도 적용해(`HandSnapshot.image_points_smoothed`) 웹캠 스켈레톤도 안정적으로 보인다. 이 표시용 필터와 `실시간`·`손 추적` 탭의 거울상(좌우 반전)은 **모두 화면 표시 전용**이며, 모델 입력·학습 데이터에는 영향을 주지 않는다. 상세 결정은 decisions.md(2026-07-19) 참조.
- **palm_scale 평활화 — 손목 평행이동 잡음 원인 발견·수정 (2026-07-19)**: 손목 위치가 정지해도 `wrist_velocity`/`wrist_acceleration`이 심하게 떨린다는 사용자 보고로 조사. 원인: `wrist_position = origin / palm_scale`의 분자(화면상 절대 위치, ~0.3~0.7)가 일반 landmark의 분자(손 안 상대적 차이, 손목 자신은 0)보다 훨씬 커서, 매 프레임 다시 계산되는 `palm_scale`의 잡음이 나눗셈에서 훨씬 크게 증폭됐다. 기존 `_wrist_smoother`(One-Euro)는 나눗셈 *이후* 값만 다뤄 이 증폭을 못 잡았다. 정지한 손 시뮬레이션 실측: 수정 전 손목 속도 잡음이 손가락 끝 속도 잡음의 약 2.5배. 시도한 대안 중 "나눗셈 전 origin만 필터링"은 오히려 더 나빠졌고(0.15배, 잡음 원인인 palm_scale 자체가 여전히 raw라 그대로 새어나감), "기존 순서 유지 + palm_scale을 별도로 평활화"가 가장 효과적이었다(3.85배 감소, 0.81→0.21 palm-width/s). 구현: `normalize_hand`는 순수 함수라 여기서 palm_scale을 평활화할 수 없으므로, `HandFeatureExtractor`가 새 `_palm_scale_smoother`로 palm_scale을 평활화하고 `wrist_position × raw_palm_scale / smoothed_palm_scale`로 재조정한다(`origin`에 직접 접근하지 않고도 `origin/smoothed_palm_scale`을 정확히 재현). `GestureConfig.smooth_palm_scale`(기본 True, `smooth_landmarks`와 함께 켜짐)과 `palm_scale_smoothing_min_cutoff`/`_beta`(기본 0 — palm_scale은 랜드마크처럼 "빠른 동작" 개념이 없어 속도 적응 불필요)/`_d_cutoff`로 제어. **주의**: 학습 데이터도 같은 설정으로 전처리해야 한다(모델 재현성).

## 정적 손 자세 파이프라인 (2026-07-20~21, 로컬 · 동적 제스처와 별개 경계)

동적 제스처(위 Task 1~9, swipe/rotate, 서버 이관 대상)와 **다른 관심사**다. 커서 이동·클릭·드래그·우클릭·스크롤·바탕화면 토글은 **단일 프레임의 손 모양**으로 판정한다. `pose_protocol.py`(순수)↔`pose_classifier.py`(torch)는 `model_protocol.py`↔`model.py`와 같은 격리 구조이며, 두 모델은 서로를 import하지 않고 입력 차원·산출물이 다르다.

흐름: `PoseClassifier`(7-class MLP) → 기울기 신뢰 판정 → `PoseStateMachine`(시간축) → `PoseControlBridge`(실제 OS 입력). 앞 두 단계는 프레임별, 뒤 두 단계가 시간 구조와 부수효과를 담당한다.

### 자세 분류기 (`pose_protocol.py`, `pose_classifier.py`, `training/train_pose.py`)

- **7-class**: `index_point`·`pinch_index`·`pinch_middle`·`two_fingers`·`open_palm`·`fist` + `none`. 참고 레포와 같은 규모의 MLP(입력→20→10→클래스), 노트북 CPU로 수 초 학습.
- **입력 feature = 정규화 좌표 42 + 손끝 쌍거리 10 = 52차원**(`pose_features()`, 학습·추론의 단일 진실). 좌표만 주면 작은 MLP가 손끝 사이 *관계*를 못 뽑는다 — 엄지-중지끝 거리는 단독으로 index_point/pinch_middle을 오류율 10.9%로 가르지만 좌표만 준 모델의 index_point 재현율은 50.2%였고, 쌍거리를 더하자 전체 82.6%→92.3%로 올랐다.
- **`none` 배경 클래스**: 소프트맥스는 반드시 한 클래스를 고르므로, 이게 없으면 손이 보이기만 해도 명령이 나간다. 전이 구간·휴지·일상 동작을 모아 학습. 효과(실측): 우클릭 직전 엄지-중지 접근이 two_fingers로 오분류돼 스크롤이 튀던 문제 0건, 헐거운 핀치 흡수. 오류가 안전한 쪽으로 이동(오발동 1.7%, 명령 간 오인 2.8%, 놓침 15.4% — 놓침은 재시도로 끝나지만 오발동은 되돌릴 수 없다).
- **전처리 재현성**: 학습·추론 모두 `normalize_hand` + One-Euro(`GestureConfig.smoothing_*`, 3번 탭 좌표)를 쓴다. 저장 파일에 전처리 설정을 넣고 로드 시 현재 `GestureConfig`와 대조 — 어긋나면 `PreprocessingMismatch`로 거부(어긋나면 예외 없이 정확도만 조용히 떨어지는 고장). 평가는 **에피소드 단위 홀드아웃**(인접 프레임 누수 차단, 실측 무작위 84.6% vs 에피소드 82%).

### 손 기울기 게이트 (`landmarks.palm_tilt_degrees`, `is_palm_tilted`, `pose_protocol.is_pose_trusted`)

- 손바닥 축(손목→중지 MCP)이 이미지 평면과 이루는 각. 카메라 쪽으로 눕히면 이 축이 2D에서 단축돼 자세 정보가 **실제로 소실**된다(구간별 정확도 0~10° 90.8%, 20~30° 47.3%, 30° 초과 37.0%). 정규화 방식 4종 비교에서 회전 민감도 30배 차이에도 정확도 동률이라 **분모 교체로는 해결 불가** — 좌표를 어떻게 정규화해도 없는 정보는 못 만든다.
- **각도만 z에서 계산**해 소스가 넘긴다(`RawHandLandmarks.palm_tilt_degrees`). 좌표는 2D 유지(`LANDMARK_DIMS=2` 불변) — z를 좌표로 되살리는 게 아니라 화면 밖 회전이라는 z만 아는 정보를 각도 하나로 요약해 굵은 임계 판정에만 쓴다. z를 못 내는 소스(`None`)에서는 게이트를 걸지 않는다(시스템 전체 정지 방지).
- **자세별 허용 각도**(`DEFAULT_POSE_TILT_LIMITS`): 손가락을 편 자세는 기울어도 실루엣이 남아 관대하다(two_fingers 40°, open_palm 30°), 나머지는 20°. **분류 뒤** 예측 자세의 한계로 판정한다 — 순환이 아니라, 먼저 분류하고 그 결과를 이 각도에서 믿어도 되는지 실측 표로 확인하는 순서다. 근거 없는 구간은 보수적으로 20°.

### 시간축 상태기계 (`pose_state.py`)

프레임별 판정만으로는 동작이 안 정해진다(같은 pinch_index라도 짧게 떼면 클릭, 유지하면 드래그). 순수 로직이라 카메라·OS 없이 테스트한다. 규칙 세 가지:

1. **진입은 느리게(dwell 120~300ms), 이탈은 빠르게 하되 관대하게**: 자세가 유지돼야 상태 진입(전이 프레임은 짧아 걸러짐). `none`이 몇 프레임 들어와도 3프레임까지는 안 끊는다(놓침 15.4%가 조작 중단으로 이어지면 안 됨).
2. **믿을 수 없는 판정(trusted=False, 기울기 게이트)은 상태를 바꾸지도 끊지도 않는다.**
3. **`none`은 자세 이력을 지우지 않는다**: 주먹→보 전이 중간이 none으로 분류되므로, "마지막 명령 자세"를 따로 기억해 전이 판정이 빈 구간을 건너뛴다(인접 상태로 보면 절대 성립 안 함).

동작: 커서 이동(index_point·pinch_index, 마우스식 상대 이동+포인터 가속+palm_scale 정규화, 참조점은 손목의 **평활된 이미지 좌표**), 클릭/드래그(핀치 유지 시간으로 분기, 진입 시점 소급), 우클릭(pinch_middle), 스크롤(two_fingers가 **가리키는 방향** — 손 이동이 아니라 MCP→끝 벡터라 손을 멈춰도 유지, 수직성 0.5 미만이면 방향 안 지어냄), 바탕화면 토글(fist→open_palm 전이). 임계는 전부 **시간**이라 투영에 흔들리지 않는다(기하학적 임계는 손 각도에 8.85배까지 흔들렸다).

### 실제 OS 제어 (`monitoring/pose_control.py`)

`PoseStateMachine` 이벤트를 `InputSink`(macOS Quartz / Windows user32)로 옮긴다. 판정(순수)과 실행(부수효과)을 분리해 상태기계를 OS 없이 테스트하고 실행을 끄고도 판정을 관찰한다. 디버깅 툴에서 토글(양 탭 공유), 드래그 중 손 놓침·제어 끄기 시 눌린 버튼을 반드시 놓는다.

- **F11 외 모든 동작이 Windows·macOS 양쪽에서 대응**된다(클릭·우클릭·드래그·스크롤·커서). `Win32InputSink`·`MacOSInputSink`가 같은 `InputSink` Protocol을 구현하고 `default_input_sink()`가 플랫폼으로 고른다.
- **드래그 실시간 이동**: macOS는 버튼을 누른 채 움직일 때 `MouseMoved`가 아니라 `LeftMouseDragged`를 받아야 창·선택 영역이 실시간으로 따라온다(그렇지 않으면 버튼을 뗄 때까지 최종 위치로만 튄다). `move_cursor(dragging=)`로 드래그 중임을 실어 이벤트 종류를 바꾼다. Windows는 `MOUSEEVENTF_MOVE`가 버튼 눌린 채면 OS가 드래그로 해석해 별도 처리가 필요 없다.
- **주먹→보 전이 키는 플랫폼별**: macOS=F11(바탕화면 표시), Windows=재생/일시정지(`_transition_key()`가 `sys.platform`으로 선택). 스크롤은 30fps 그대로면 감당이 안 돼 `SCROLL_INTERVAL_MS`(60ms)로 솎아내며, 이 스로틀은 실행·UI 기록 양쪽에 적용한다.

## 통합 규약 (배선 계층 주의)

Task 1·2는 mediapipe·카메라 없이 단위 테스트되는 순수 경계다. 실제 캡처와 붙이는 앱/배선 계층에서 아래를 지켜야 한다. 이 책임은 gesture_fusion 패키지가 아니라 배선 계층에 있다(모듈 경계 규칙: gesture_fusion은 `runtime_protocol` 내부 타입을 import하지 않는다).

- **색상 순서**: `MediaPipeHandLandmarker.process`는 **RGB**를 기대한다. OpenCV/웹캠은 기본 BGR이므로 넘기기 전 `cv2.cvtColor(frame, COLOR_BGR2RGB)`로 변환한다(Gaze의 `jarvis.gaze.cli`와 동일 규약). 어기면 예외 없이 검출 품질만 조용히 떨어진다.
- **프레임 언팩·시간축**: `capture.Frame`을 `process(frame.image, frame.timestamp_ms, frame.frame_id)`로 풀어 넘기고, `timestamp_ms`는 단일 monotonic clock 값을 그대로 전달한다(자체 시계로 재생성 금지, [interface-contract.md](interface-contract.md) 공통 규칙).
- **계약 타입 바인딩**: Task 1~3의 출력(`HandObservation`·`FrameFeatures`·`ModelPrediction`)은 모듈 내부 타입이다. Task 3은 `phase`에 `jarvis.contracts.GesturePhase`를 그대로 써 enum 재정의를 피했다(검증 완료). 모듈 경계로 나가는 최종 출력 [interface-contract.md](interface-contract.md)의 `GestureEstimate` 조립은 **Task 4(gesture spotting)**에서 원본 프레임의 `timestamp_ms`/`frame_id`를 붙여 수행한다.

## 진행 상황

- [x] **Task 1 — Hand landmark 추출·정규화** (`landmarks.py`, `mediapipe_hands.py`, `config.py`): MediaPipe 연동(교체 가능한 `HandLandmarkSource` 경계), 손목 기준·손바닥 크기 정규화(회전 보존). 좌표는 x·y 2D만 사용(`LANDMARK_DIMS=2`, z 제외 — 위 "좌표 차원 2D화" 참조). 검증 후 수정 반영: 좌표 원점을 스케일 기준과 분리(`origin_index`), `tracking_confidence`→`detection_confidence`+`handedness_score` 정정, handedness 부재 시 score 0.0 처리.
- [x] **Task 2 — Feature engineering** (`features.py`): causal 속도(monotonic `timestamp_ms` 차분)·관절 굴곡각. 추적손실·프레임 공백 시 history 리셋, 관절각 퇴화 시 NaN 대신 0. feature 그룹 on/off·차원은 `GestureConfig`로 제어. 손가락 관절 위치의 가속도는 2026-07-19에 모델 입력에서 제거(위 "설계 노트" 참조) — 손목 평행이동 가속도(`wrist_acceleration`)는 별개로 유지.
- [x] **Task 3 — Causal TCN/GRU** (`model_protocol.py`, `model.py`): dilated causal 1D conv(TCN), `GestureModel` Protocol(torch 무의존, `mediapipe_hands.py`와 같은 격리 원칙)로 아키텍처 교체 가능. `phase`는 `jarvis.contracts.GesturePhase`를 그대로 재사용(자체 enum 재정의 없음). gesture head(7-class: 6개 동적 제스처 + 배경 클래스 `"none"`) + phase head(4-class), confidence=softmax max, uncertainty=정규화 엔트로피. 진짜 인과성(미래 프레임 미사용)을 `test_output_is_truly_causal`로 회귀 검증. **모델은 아직 미학습(무작위 초기화)** — `ModelMetadata.trained=False`가 이를 명시하며, 학습 데이터 확보 전까지 fusion·safe commit에 실제 인식 결과로 쓰면 안 됨(`models/README.md` 참고). 검증 후 수정 반영: torch(`ml` extra) 미설치 환경에서도 테스트 스위트가 수집되도록 `test_model*`에 `importorskip` 가드 추가, 표준 `.[dev]`(torch 없음) 타입체크 통과를 위해 `model` 모듈 mypy `disallow_subclassing_any` 예외, `load_weights`의 `torch.load`에 `weights_only=True`(pickle 코드 실행 차단).
- [x] **Task 4 — Gesture spotting 상태 머신** (`spotting.py`): raw 모델 phase를 `min_consecutive_frames` 디바운스해 단일 프레임 노이즈를 억제하고, `IDLE→ONSET→ACTIVE→ENDING→IDLE` 외 전이(단계 건너뛰기)는 거부. ONSET 확정 시 배경 클래스(`"none"`)·낮은 gesture confidence는 게이팅으로 거부. 한 제스처당 `ENDING`은 정확히 한 프레임만 방출(방출 즉시 IDLE로 리셋) — `GestureEstimate`(계약)를 매 프레임 조립해 밀집 스트림으로 출력. 추적 손실(`prediction=None`) 시 진행 중이던 제스처를 안전하게 포기. `is_tracking_gesture` 프로퍼티로 커서/제스처 분기 신호(2026-07-18 결정) 노출 — `pointer/` 모듈 연동은 아직 미배선.
- [x] **Task 5 — 시선·제스처 temporal alignment** (`alignment.py`): `TargetLockTracker` — Gaze→Fusion 스트림(§1)에서 Fusion 자체 Target Lock을 추적(dwell 승격, TTL 슬라이딩 만료). Gaze 모듈의 자체 Gaze Lock(커서 게이팅용)과는 독립적 구현 — 모듈 경계상 Gaze 내부를 import할 수 없어 같은 원시 확률에서 별도로 계산(documents/decisions.md 기록 대상). `check_alignment`가 Commit 조건 1(lock 여부)·2(lock 이후 시작)·3(TTL 안 완료)를 판정 — 조건 6(시간 관계 유효)은 2·3의 결합으로 자연히 충족되어 별도 필드 없음. `TemporalAligner`가 Gaze·Gesture 두 비동기 스트림을 "as-of" 조인해 제스처 `ENDING` 시점에만 평가. 검증 중 `x or default` 0-falsy 함정(타임스탬프 0일 때 오작동) 발견해 명시적 `None` 체크로 수정.
- [x] **Task 6 — Fusion confidence·safe commit** (`fusion.py`): `compute_fusion_score`로 결합 점수 `S=P(target)×P(gesture)×gaze_stability×(1−uncertainty)` 계산. `FusionEngine`이 `TemporalAligner`(task 5)를 감싸 Commit 조건 4(target confidence 하한)·5(gesture confidence 하한)·threshold 판정을 더하고, 커밋 직후 `cooldown_ms` 동안 재커밋을 막는 COOLDOWN을 시간 기반 슬라이딩으로 구현. `IntentPhase`(README 9장 Intent 상태 머신, 모듈 경계를 넘지 않아 `jarvis.contracts`가 아닌 여기 정의)로 `IDLE→TARGET_CANDIDATE→TARGET_LOCKED→GESTURE_TRACKING→COOLDOWN` 관측 가능(`INTENT_CANDIDATE`·`COMMITTED`는 ENDING 처리 한 번 안에서 동기적으로 지나가는 순간 상태라 `CommitDecision`으로만 드러남). `alignment.TargetLockState`에 `candidate` 필드 추가(TARGET_CANDIDATE/IDLE 구분용). 부수 발견: `monitoring/pipeline_status.py`의 `_gesture_status`가 존재하지 않는 모듈명(`gesture_fusion.spotter`, 실제는 `spotting`)을 찾고 있어 Task 4 이후로도 계속 UNAVAILABLE을 잘못 보고하던 버그 수정. Commit 조건 7(중복 방지)·`intent_id`는 task 7, 실제 `Intent` 조립은 task 8.
- [x] **Task 7 — Duplicate intent 방지** (`dedup.py`): `generate_intent_id(frame_id)`로 `intent_id`를 결정적 생성(2026-07-18 결정: Protocol의 `command_id=cmd-{intent_id}` 결정적 생성과 대칭). `IntentDeduplicator`가 이미 커밋된 `frame_id`를 bounded LRU(기본 256개)로 기억해 Commit 조건 7을 판정. `fusion.py`(task 6)의 threshold 통과 직후·COOLDOWN 갱신 직전에 배선 — dedup 거부는 재전송/재생일 뿐 새 이벤트가 아니므로 cooldown을 새로 걸지 않음. `CommitDecision`에 `intent_id`(committed일 때만) 필드 추가.
- [x] **Task 8 — Intent 조립·출력** (`intent.py`, `configs/gesture_capability_map.json`): `GestureCapabilityMap`이 `(target device_id, gesture)` → capability/operation/value 매핑을 JSON에서 로드(코드가 아니라 config로 관리, 새 제스처·기기는 파일만 수정). 같은 제스처도 기기별로 다르게 매핑됨(노트북 swipe_down=scroll, 전구 swipe_down=brightness — README 9·15장). `assemble_intent`가 task 6·7의 `CommitDecision`을 받아 `jarvis.contracts.Intent`(계약 §3)를 최종 조립, 매핑 없는 조합은 `None`(거부). **통합 갭 발견·기록**: README 15장 필수 시나리오 "노트북 Swipe Left(창 전환)"용으로 `window_switch` capability를 매핑에 넣었으나, 3인의 `WindowsAdapter`(`_execute`)는 아직 `scroll`/`volume`/`media`만 처리하고 `window_switch`는 미구현 — dev-3과 조율 필요(documents/decisions.md 기록).
- [x] **Task 9 — Hard-negative mining** (`hard_negative_mining.py`): `jarvis.gaze.evaluation`(Target Selection Accuracy)과 같은 패턴 — `compute_wrong_actuation_rate`가 README 13장 WAR(≤1% 목표)을 `dataset_id`·`conditions`와 함께 강제 기록. `LabeledCommitAttempt`(라벨 + task 6의 실제 `CommitDecision`)로 replay trace를 표현, README 13장이 나열한 5개 오발 유형은 시나리오 라벨링 단계에서 `ground_truth_should_commit` 하나로 환원. `mine_hard_negatives`가 (1) 실제 오발(`wrong_actuation`, WAR 분자)과 (2) 결합 점수가 threshold에 근접했던 near-miss 두 종류를 채굴해 재학습·threshold 재보정 파이프라인 입력으로 제공. 실 캡처 데이터셋·모델 학습 자체는 이 저장소 범위 밖(데이터 확보 후 별도 진행) — 이 모듈은 평가·채굴 로직만 제공.

## 2인 담당 9개 작업 완료

Task 1~9(hand landmark → hard-negative mining) 전부 구현·테스트·문서화 완료. 전체 347개 테스트 통과, ruff/mypy clean. 실제 학습 데이터·모델 가중치(`ModelMetadata.trained=True`)는 별도 데이터 확보 후 진행 필요.

## 이슈 / 의사결정 필요 사항

- 캡처↔비전 모듈 간 **색상 순서(RGB/BGR)** 규약이 [interface-contract.md](interface-contract.md)에는 아직 없고 각 모듈이 배선 계층에서 개별 처리 중이다(현재 Gaze만 `cli`에서 변환). 통합 담당과 계약에 명시할지 논의 필요.
- **`window_switch` capability 미구현 (dev-3 조율 필요)**: `configs/gesture_capability_map.json`은 README 15장 필수 시나리오("노트북 Swipe Left(창 전환)")를 위해 `laptop`의 `swipe_left`/`swipe_right`를 `window_switch` capability로 매핑해 뒀지만, `src/jarvis/runtime_protocol/adapters/windows.py`의 `WindowsAdapter._execute`는 현재 `scroll`/`volume`/`media`만 처리한다. 지금은 Protocol이 `FAILED`로 정직하게 거부하지만(원칙 1.1: 성공을 가장하지 않음), 데모 시나리오를 완성하려면 dev-3이 `window_switch` 핸들러(Alt+Tab 또는 가상 데스크톱 전환)를 추가해야 한다.
