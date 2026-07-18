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
- **계약 타입 바인딩(Task 3 예정)**: Task 1·2의 출력(`HandObservation`·`FrameFeatures`)은 모듈 내부 타입이다. TCN/GRU가 붙어 모듈 경계로 나가는 출력은 [interface-contract.md](interface-contract.md)의 `GestureEstimate`로 매핑해야 하며, `phase`는 반드시 `jarvis.contracts.GesturePhase`(닫힌 enum)를 쓰고 자체 phase enum을 재정의하지 않는다.

## 진행 상황

- [x] **Task 1 — Hand landmark 추출·정규화** (`landmarks.py`, `mediapipe_hands.py`, `config.py`): MediaPipe 연동(교체 가능한 `HandLandmarkSource` 경계), 손목 기준·손바닥 크기 정규화(회전 보존). 검증 후 수정 반영: 좌표 원점을 스케일 기준과 분리(`origin_index`), `tracking_confidence`→`detection_confidence`+`handedness_score` 정정, handedness 부재 시 score 0.0 처리.
- [x] **Task 2 — Feature engineering** (`features.py`): causal 속도·가속도(monotonic `timestamp_ms` 차분)·관절 굴곡각. 추적손실·프레임 공백 시 history 리셋, 관절각 퇴화 시 NaN 대신 0. feature 그룹 on/off·차원은 `GestureConfig`로 제어.
- [ ] Task 3 — Causal TCN/GRU (gesture 분류 + phase, confidence·uncertainty)
- [ ] Task 4 이후 — gesture spotting, temporal alignment, fusion, duplicate 방지, intent 조립, hard-negative mining

## 이슈 / 의사결정 필요 사항

- 캡처↔비전 모듈 간 **색상 순서(RGB/BGR)** 규약이 [interface-contract.md](interface-contract.md)에는 아직 없고 각 모듈이 배선 계층에서 개별 처리 중이다(현재 Gaze만 `cli`에서 변환). 통합 담당과 계약에 명시할지 논의 필요.
