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

## 진행 상황

- [ ]

## 이슈 / 의사결정 필요 사항

(있으면 [decisions.md](decisions.md)로 옮기기)
