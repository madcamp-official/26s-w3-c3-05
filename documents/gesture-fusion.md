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

## 통합 규약 (배선 계층 주의)

Task 1·2는 mediapipe·카메라 없이 단위 테스트되는 순수 경계다. 실제 캡처와 붙이는 앱/배선 계층에서 아래를 지켜야 한다. 이 책임은 gesture_fusion 패키지가 아니라 배선 계층에 있다(모듈 경계 규칙: gesture_fusion은 `runtime_protocol` 내부 타입을 import하지 않는다).

- **색상 순서**: `MediaPipeHandLandmarker.process`는 **RGB**를 기대한다. OpenCV/웹캠은 기본 BGR이므로 넘기기 전 `cv2.cvtColor(frame, COLOR_BGR2RGB)`로 변환한다(Gaze의 `jarvis.gaze.cli`와 동일 규약). 어기면 예외 없이 검출 품질만 조용히 떨어진다.
- **프레임 언팩·시간축**: `capture.Frame`을 `process(frame.image, frame.timestamp_ms, frame.frame_id)`로 풀어 넘기고, `timestamp_ms`는 단일 monotonic clock 값을 그대로 전달한다(자체 시계로 재생성 금지, [interface-contract.md](interface-contract.md) 공통 규칙).
- **계약 타입 바인딩**: Task 1~3의 출력(`HandObservation`·`FrameFeatures`·`ModelPrediction`)은 모듈 내부 타입이다. Task 3은 `phase`에 `jarvis.contracts.GesturePhase`를 그대로 써 enum 재정의를 피했다(검증 완료). 모듈 경계로 나가는 최종 출력 [interface-contract.md](interface-contract.md)의 `GestureEstimate` 조립은 **Task 4(gesture spotting)**에서 원본 프레임의 `timestamp_ms`/`frame_id`를 붙여 수행한다.

## 진행 상황

- [x] **Task 1 — Hand landmark 추출·정규화** (`landmarks.py`, `mediapipe_hands.py`, `config.py`): MediaPipe 연동(교체 가능한 `HandLandmarkSource` 경계), 손목 기준·손바닥 크기 정규화(회전 보존). 검증 후 수정 반영: 좌표 원점을 스케일 기준과 분리(`origin_index`), `tracking_confidence`→`detection_confidence`+`handedness_score` 정정, handedness 부재 시 score 0.0 처리.
- [x] **Task 2 — Feature engineering** (`features.py`): causal 속도·가속도(monotonic `timestamp_ms` 차분)·관절 굴곡각. 추적손실·프레임 공백 시 history 리셋, 관절각 퇴화 시 NaN 대신 0. feature 그룹 on/off·차원은 `GestureConfig`로 제어.
- [x] **Task 3 — Causal TCN/GRU** (`model_protocol.py`, `model.py`): dilated causal 1D conv(TCN), `GestureModel` Protocol(torch 무의존, `mediapipe_hands.py`와 같은 격리 원칙)로 아키텍처 교체 가능. `phase`는 `jarvis.contracts.GesturePhase`를 그대로 재사용(자체 enum 재정의 없음). gesture head(7-class: 6개 동적 제스처 + 배경 클래스 `"none"`) + phase head(4-class), confidence=softmax max, uncertainty=정규화 엔트로피. 진짜 인과성(미래 프레임 미사용)을 `test_output_is_truly_causal`로 회귀 검증. **모델은 아직 미학습(무작위 초기화)** — `ModelMetadata.trained=False`가 이를 명시하며, 학습 데이터 확보 전까지 fusion·safe commit에 실제 인식 결과로 쓰면 안 됨(`models/README.md` 참고). 검증 후 수정 반영: torch(`ml` extra) 미설치 환경에서도 테스트 스위트가 수집되도록 `test_model*`에 `importorskip` 가드 추가, 표준 `.[dev]`(torch 없음) 타입체크 통과를 위해 `model` 모듈 mypy `disallow_subclassing_any` 예외, `load_weights`의 `torch.load`에 `weights_only=True`(pickle 코드 실행 차단).
- [x] **Task 4 — Gesture spotting 상태 머신** (`spotting.py`): raw 모델 phase를 `min_consecutive_frames` 디바운스해 단일 프레임 노이즈를 억제하고, `IDLE→ONSET→ACTIVE→ENDING→IDLE` 외 전이(단계 건너뛰기)는 거부. ONSET 확정 시 배경 클래스(`"none"`)·낮은 gesture confidence는 게이팅으로 거부. 한 제스처당 `ENDING`은 정확히 한 프레임만 방출(방출 즉시 IDLE로 리셋) — `GestureEstimate`(계약)를 매 프레임 조립해 밀집 스트림으로 출력. 추적 손실(`prediction=None`) 시 진행 중이던 제스처를 안전하게 포기. `is_tracking_gesture` 프로퍼티로 커서/제스처 분기 신호(2026-07-18 결정) 노출 — `pointer/` 모듈 연동은 아직 미배선.
- [x] **Task 5 — 시선·제스처 temporal alignment** (`alignment.py`): `TargetLockTracker` — Gaze→Fusion 스트림(§1)에서 Fusion 자체 Target Lock을 추적(dwell 승격, TTL 슬라이딩 만료). Gaze 모듈의 자체 Gaze Lock(커서 게이팅용)과는 독립적 구현 — 모듈 경계상 Gaze 내부를 import할 수 없어 같은 원시 확률에서 별도로 계산(documents/decisions.md 기록 대상). `check_alignment`가 Commit 조건 1(lock 여부)·2(lock 이후 시작)·3(TTL 안 완료)를 판정 — 조건 6(시간 관계 유효)은 2·3의 결합으로 자연히 충족되어 별도 필드 없음. `TemporalAligner`가 Gaze·Gesture 두 비동기 스트림을 "as-of" 조인해 제스처 `ENDING` 시점에만 평가. 검증 중 `x or default` 0-falsy 함정(타임스탬프 0일 때 오작동) 발견해 명시적 `None` 체크로 수정.
- [x] **Task 6 — Fusion confidence·safe commit** (`fusion.py`): `compute_fusion_score`로 결합 점수 `S=P(target)×P(gesture)×gaze_stability×(1−uncertainty)` 계산. `FusionEngine`이 `TemporalAligner`(task 5)를 감싸 Commit 조건 4(target confidence 하한)·5(gesture confidence 하한)·threshold 판정을 더하고, 커밋 직후 `cooldown_ms` 동안 재커밋을 막는 COOLDOWN을 시간 기반 슬라이딩으로 구현. `IntentPhase`(README 9장 Intent 상태 머신, 모듈 경계를 넘지 않아 `jarvis.contracts`가 아닌 여기 정의)로 `IDLE→TARGET_CANDIDATE→TARGET_LOCKED→GESTURE_TRACKING→COOLDOWN` 관측 가능(`INTENT_CANDIDATE`·`COMMITTED`는 ENDING 처리 한 번 안에서 동기적으로 지나가는 순간 상태라 `CommitDecision`으로만 드러남). `alignment.TargetLockState`에 `candidate` 필드 추가(TARGET_CANDIDATE/IDLE 구분용). 부수 발견: `monitoring/pipeline_status.py`의 `_gesture_status`가 존재하지 않는 모듈명(`gesture_fusion.spotter`, 실제는 `spotting`)을 찾고 있어 Task 4 이후로도 계속 UNAVAILABLE을 잘못 보고하던 버그 수정. Commit 조건 7(중복 방지)·`intent_id`는 task 7, 실제 `Intent` 조립은 task 8.
- [x] **Task 7 — Duplicate intent 방지** (`dedup.py`): `generate_intent_id(frame_id)`로 `intent_id`를 결정적 생성(2026-07-18 결정: Protocol의 `command_id=cmd-{intent_id}` 결정적 생성과 대칭). `IntentDeduplicator`가 이미 커밋된 `frame_id`를 bounded LRU(기본 256개)로 기억해 Commit 조건 7을 판정. `fusion.py`(task 6)의 threshold 통과 직후·COOLDOWN 갱신 직전에 배선 — dedup 거부는 재전송/재생일 뿐 새 이벤트가 아니므로 cooldown을 새로 걸지 않음. `CommitDecision`에 `intent_id`(committed일 때만) 필드 추가.
- [ ] Task 8 — Intent 조립·출력 (capability/operation/value 매핑, `jarvis.contracts.Intent` 최종 조립)
- [ ] Task 9 — hard-negative mining

## 이슈 / 의사결정 필요 사항

- 캡처↔비전 모듈 간 **색상 순서(RGB/BGR)** 규약이 [interface-contract.md](interface-contract.md)에는 아직 없고 각 모듈이 배선 계층에서 개별 처리 중이다(현재 Gaze만 `cli`에서 변환). 통합 담당과 계약에 명시할지 논의 필요.
