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

## 진행 상황

- [ ]

## 이슈 / 의사결정 필요 사항

(있으면 [decisions.md](decisions.md)로 옮기기)
